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
use kreate::docfile::{self, DocumentFile};
use kreate::{dragpreview, gpu_sync, shell};

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, Vec3, Vec4};
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
///
/// Windows only, and gated rather than left to be harmlessly ignored: these are
/// NVIDIA Optimus and AMD PowerXpress conventions, they mean nothing on macOS or
/// Linux, and an unexplained exported symbol invites someone to wonder whether
/// the Mac build needs it.
#[cfg(windows)]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;
#[cfg(windows)]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

const HELP: &str = "\
Kreate

  Navigate   MMB orbit  Shift+MMB pan  wheel zoom  F frame all
  View       Z shading  O orthographic  Shift+G grid  W wireframe
             Y auto-depth orbit (on by default)
  Add        A cube  Shift+A next primitive shape
  Edit       D duplicate (linked)  X delete  Ctrl+A select all
             Ctrl+J join selected into the active object
  Edit mode  Tab enter/leave  E extrude  X delete  Ctrl+X dissolve
             Shift+S subdivide  I inset  B bridge borders  Ctrl+R loop cut
             [ / ] after a loop cut slides it
             M merge selection  Shift+M weld by distance
             N shade smooth  Ctrl+N recalculate  Shift+N flip
             K mark sharp  Alt+K clear  Shift+K auto-smooth 30 deg
             F fill hole  Shift+F grid fill
             Y split off faces  V rip along edges  Ctrl+B bevel
             P separate the selection into a new object
             U box-project UVs (right-click for the other projections)
  UV editor  G move  R rotate  S scale   A all  Alt+A none  L linked
             U smart-project the whole thing  P repack the islands
             U / V or X / Y constrain an axis; the keys work where the
             pointer is, so G means UVs here and geometry in the viewport
             1/2/3 vert/edge/face select  L loop  Shift+L ring
             Ctrl+A all elements  Alt+A none  Ctrl+I invert
  Transform  G move  R rotate  S scale
             The key starts the tool; the mouse does the work. Move follows the
             pointer across the view, rotate follows it around the selection,
             scale is how much further from the selection it has got.
               LMB or Enter   confirm, as one undo step
               RMB or Esc     cancel, leaving no trace
               X / Y / Z      constrain to that axis; again to release
               X X            press again for the object's own axes (ZZ, YY)
               Shift (hold)   ten times finer, from where you pressed it
               Ctrl (hold)    the other way round on snapping
               type a value   2*3.5, 1.4/8, 50cm, 30deg; Backspace to correct
               wheel          proportional falloff radius, while it is on
  Settings   ` snapping on/off (Ctrl inverts it for one drag)
             Shift+` what it snaps to: increment, vertex, edge, midpoint, face
                     Element modes land the selection *on* what the cursor is
                     over; increment rounds how far it went.
             O proportional editing on/off, in edit mode
             In object mode they act on the selected objects; in edit mode on
             the selected vertices, edges or faces. Rotate and scale work about
             the selection's median, so a single vertex has nothing for them to
             do and they say so.
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

/// A digit or a sign, from a physical key.
///
/// Shared by both gesture handlers, because a number typed into a transform
/// means the same thing whichever one is running — and two copies of this
/// table is two chances for one of them to miss a key.
fn typed_character(code: KeyCode) -> Option<char> {
    match code {
        KeyCode::Digit0 | KeyCode::Numpad0 => Some('0'),
        KeyCode::Digit1 | KeyCode::Numpad1 => Some('1'),
        KeyCode::Digit2 | KeyCode::Numpad2 => Some('2'),
        KeyCode::Digit3 | KeyCode::Numpad3 => Some('3'),
        KeyCode::Digit4 | KeyCode::Numpad4 => Some('4'),
        KeyCode::Digit5 | KeyCode::Numpad5 => Some('5'),
        KeyCode::Digit6 | KeyCode::Numpad6 => Some('6'),
        KeyCode::Digit7 | KeyCode::Numpad7 => Some('7'),
        KeyCode::Digit8 | KeyCode::Numpad8 => Some('8'),
        KeyCode::Digit9 | KeyCode::Numpad9 => Some('9'),
        KeyCode::Period | KeyCode::NumpadDecimal => Some('.'),
        KeyCode::Minus | KeyCode::NumpadSubtract => Some('-'),
        KeyCode::Slash | KeyCode::NumpadDivide => Some('/'),
        KeyCode::NumpadMultiply => Some('*'),
        _ => None,
    }
}

/// The gesture's parameters, with the mode the operator needs to read them.
///
/// The gesture reports its own numbers; the mode says which group of them is
/// the live one. Set here rather than inside the gesture because it is a
/// property of the *operator's* interface, and the gesture is deliberately
/// ignorant of which operator it drives.
fn mode_params(kind: kreate::uvmodal::Kind, gesture: &kreate::uvmodal::UvGesture) -> PropertyMap {
    let mut params = gesture.params();
    let mode = match kind {
        kreate::uvmodal::Kind::Move => 0,
        kreate::uvmodal::Kind::Rotate => 1,
        kreate::uvmodal::Kind::Scale => 2,
    };
    params.set("mode", PropertyValue::Enum(mode));
    params
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

    /// The format egui samples this texture through.
    ///
    /// Not a detail. egui's fragment shader says of its input: "we expect
    /// normal textures that are NOT sRGB-aware" — it treats what it samples as
    /// already gamma-encoded, and for an sRGB framebuffer converts it *down* to
    /// linear so the hardware can encode it back. Sampling through the sRGB
    /// view decodes once more before egui ever sees it, so the image is
    /// decoded twice and composited far too dark: the viewport background
    /// arrived on screen at 4/255 instead of 28/255.
    ///
    /// The scene still renders *into* the sRGB view, so the shaders keep
    /// writing linear and the hardware keeps encoding. Only the sampling side
    /// changes.
    const SAMPLED_AS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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
            view_formats: &[Self::SAMPLED_AS],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampled = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(Self::SAMPLED_AS),
            ..Default::default()
        });
        let id = renderer.register_native_texture(device, &sampled, wgpu::FilterMode::Linear);
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

/// Viewport pixels one wheel notch pans.
///
/// A notch is a coarse unit and a pan is a fine one, so this is smaller than it
/// looks: about a third of a line of text, which is what a sideways swipe on a
/// trackpad feels like elsewhere.
const WHEEL_PAN_PIXELS: f64 = 40.0;

/// Notches one unit of pinch is worth.
///
/// A pinch reports a fraction of the distance between the fingers, so a full
/// pinch is around one — several notches of zoom, which is what the gesture
/// means on the platforms that send it.
const PINCH_NOTCHES: f64 = 6.0;

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

/// A transform being previewed on the GPU.
///
/// Held on `Gfx` rather than beside `self.gesture` because what it owns is GPU
/// resources whose lifetime is the drag: bind groups naming the previewed
/// mesh's own position and normal buffers, a rest copy of both, and the tables
/// the normal passes walk. When this is dropped they go with it.
///
/// It deliberately does not hold the mesh it was built against. Nothing writes
/// the document while a preview exists, so there is no drift to detect; and if
/// something else changes the scene, `dragpreview::preview_survives` says so
/// from the sync plan and the preview ends. A handle kept here would only
/// invite a staleness check that cannot fire.
struct LivePreview {
    deform: kreate_render::Deform,
    /// Which operator is being previewed, so the params can be read without
    /// asking the modal session for its id.
    op_id: &'static str,
    /// Which mesh vertex each render vertex came from, kept so that changing
    /// the proportional radius mid-drag does not re-extract the render mesh.
    ///
    /// Four bytes a render vertex. The whole `RenderMeshCpu` is about a hundred,
    /// and the other ninety-six are on the GPU already.
    render_to_vert: Vec<u32>,
    /// The document mesh's vertex count, which the falloff is indexed by.
    mesh_verts: usize,
    /// Which object's space the mesh is in, and the world point a rotation or a
    /// scale turns about.
    ///
    /// Both, because the shader needs both and they are found together: the
    /// pivot is the selection's median in world coordinates, which can only be
    /// worked out with the object's transform in hand.
    place: kreate_render::Placement,
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
    /// The three compute pipelines a drag runs. Built once; a `Deform` borrows
    /// them per drag.
    deformer: kreate_render::Deformer,
    /// The drag being previewed on the GPU, if there is one.
    preview: Option<LivePreview>,

    egui_renderer: egui_wgpu::Renderer,
    target: ViewportTarget,
}

impl Gfx {
    /// Puts the previewed mesh back where the document says it is, and forgets
    /// the preview.
    ///
    /// The identity gesture rather than a re-upload: the shader recomputes from
    /// the rest copy every frame, so running it with nothing to apply writes
    /// exactly the buffers `upload` wrote. A device test asserts that, which is
    /// what makes this two dispatches instead of an eighty-millisecond
    /// extraction.
    ///
    /// Needed because a cancelled drag leaves the document holding the very
    /// handle the cache already has — that is what makes undo cheap — so
    /// `gpu_sync` finds nothing to do and would leave the deformed positions on
    /// screen with the document at rest.
    fn restore_preview(&mut self) -> bool {
        let Some(preview) = self.preview.take() else {
            return false;
        };
        preview
            .deform
            .set(&self.gpu.queue, &kreate_render::Gesture::IDENTITY);
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview restore"),
            });
        preview.deform.dispatch(&mut encoder, &self.deformer);
        self.gpu.queue.submit([encoder.finish()]);
        true
    }
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
    /// The UV transform being dragged, and its uncommitted history entry.
    ///
    /// Separate from `gesture` rather than a variant of it: the two cannot run
    /// at once — one needs the pointer over the viewport and the other over
    /// the UV panel — but they carry different state, and a shared enum would
    /// mean every use site matching on which one it had.
    uv_gesture: Option<(kreate::uvmodal::UvGesture, kreate_ops::ModalSession)>,
    /// Which element type edit-mode selection acts on. 0 vert, 1 edge, 2 face.
    select_mode: u32,
    /// The edit structure, held while edit mode is open.
    ///
    /// Opened on entering edit mode and dropped on leaving, so the decode
    /// happens once rather than once per operator. Lives in the application
    /// rather than the document: they are a staging area for edits, and a
    /// document carrying one would snapshot it into every undo step.
    ///
    /// One per mesh, because edit mode is a set of meshes. Without that, every
    /// operator on anything but the active mesh would decode and re-encode it —
    /// which is the cost a session exists to avoid, put back for every object
    /// after the first.
    edit: kreate_ops::EditSessions,
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
    /// What each held button's drag is doing, decided when it was pressed.
    ///
    /// Latched rather than resolved per motion, for three reasons, and the first
    /// two are bugs that were in the per-motion version:
    ///
    /// - the navigation gizmo's "the pointer is over me" flag is only recomputed
    ///   when the interface runs, so between frames a drag consulted a stale one
    ///   and navigation stopped working until something forced a redraw;
    /// - releasing a modifier mid-drag turned an orbit into a pan halfway
    ///   through;
    /// - a drag is one gesture, and what it does is settled by how it began.
    drag_ops: [Option<String>; 3],
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
    /// Where the document lives, and everything about saving it.
    file: DocumentFile,
    /// Autosaves from sessions that did not exit cleanly, offered once.
    ///
    /// Collected at startup rather than polled: a directory becoming stale while
    /// this instance runs means another instance crashed, and interrupting the
    /// user's work to say so is not this application's business.
    recoverable: Vec<docfile::Recoverable>,
    /// Snapping, as the user has it set. `Ctrl` inverts it for one drag.
    ///
    /// Held on the application rather than inside a gesture because it is a
    /// standing preference: it survives between transforms, and the next one
    /// starts from it.
    snap: kreate::modal::Snap,
    /// The proportional editing radius, or `None` when it is off.
    ///
    /// §2.2 calls this unexpectedly central to how a modelling tool feels, and
    /// it is: without it every vertex move is a spike, and shaping a surface
    /// means moving each vertex individually.
    proportional: Option<f64>,
    /// The interactive transform in progress, if any.
    ///
    /// While this is `Some`, the gesture owns the keyboard and the pointer:
    /// §2.1's pipeline puts the modal stack above the keymap for exactly this
    /// reason, so that `X` means "constrain to X" rather than "delete" for as
    /// long as a transform is running.
    gesture: Option<(kreate::modal::Gesture, kreate_ops::ModalSession)>,
    /// The last title pushed to the window, so it is only set when it changes.
    ///
    /// Setting a window title is a system call that can repaint the frame, and
    /// doing it every frame is a visible flicker on some window managers.
    shown_title: String,
}

impl App {
    fn new(doc: Document, locale: &str) -> Self {
        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        let mut app = Self::empty(doc, locale, registry);
        app.load_keymap();
        app.load_workspaces();
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
            uv_gesture: None,
            select_mode: 2,
            edit: kreate_ops::EditSessions::new(),
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
            drag_ops: [const { None }; 3],
            viewport_px: None,
            ui: ShellState::with_catalog(shell::load_catalog(locale)),
            wanted_viewport: (0, 0),
            next_shape: 1,
            snap: kreate::modal::Snap::default(),
            proportional: None,
            gesture: None,
            file: DocumentFile::new(),
            // Collected once, at startup. This is the only moment a recovery
            // offer is not an interruption.
            recoverable: docfile::find_recoverable(),
            shown_title: String::new(),
        }
    }

    /// Dispatches an operator. The only way this application changes the document.
    fn run_op(&mut self, op_id: &str, params: PropertyMap) {
        // Recorded before it runs, not after. The operator that panics is the one
        // most worth having in the report, and an entry written on success would
        // be missing exactly that one.
        //
        // The registry's id rather than the caller's `&str`, because the ring
        // buffer holds `&'static str` and the registry owns those for the life
        // of the program. An id that names no operator is not recorded — the
        // dispatch below will refuse it anyway.
        if let Some(info) = self.registry.get(op_id) {
            kreate::breadcrumbs::record(info.id);
        }
        // Through the edit session when edit mode is open. An operator that can
        // use it works in place; one that cannot — or that runs outside edit
        // mode — converts for itself, which is the same code path as before.
        match self.history.dispatch_in_session(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            op_id,
            &params,
        ) {
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

    /// Loads the named layouts, or falls back to the ones Kreate ships with.
    ///
    /// Anything wrong with the file is logged and the rest still loads, on the
    /// same grounds as the keymap: a broken preferences file must cost a
    /// setting, not a session.
    fn load_workspaces(&mut self) {
        let Some(path) = kreate::prefs::workspace_path() else {
            return;
        };
        let (spaces, problems) = kreate::prefs::load_workspaces(&path);
        for problem in problems {
            tracing::warn!("{problem}");
            println!("workspaces: {problem}");
        }
        self.ui.workspaces = spaces;
    }

    /// Writes the named layouts, if they have changed and settled.
    fn save_workspaces(&mut self) {
        if !self.ui.layout_needs_saving() {
            return;
        }
        // Marked saved either way. A failure that left the flag set would retry
        // on the next frame, for ever, on a directory that is not going to
        // become writable.
        self.ui.layout_saved();
        let Some(path) = kreate::prefs::workspace_path() else {
            return;
        };
        if let Err(e) = kreate::prefs::save_workspaces(&path, &self.ui.workspaces) {
            // Reported, never fatal: a read-only home directory is not a reason
            // to stop working.
            println!("workspaces: could not save to {}: {e}", path.display());
        }
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
        // What `Tab` means, and it is not expressible as "set this mode": the
        // user means "in or out of editing whatever this is". With more modes it
        // is still a two-way gesture, between object mode and the last one they
        // were editing in.
        self.set_mode(kreate::mode::toggled(self.op_ctx.mode));
    }

    /// Switches mode, if the mode can be entered.
    ///
    /// One door, so that everything a mode change has to bring with it happens
    /// once rather than at each caller: the selection sweep, the edit session,
    /// the interface's copy of the mode, the tooltips and a redraw. There were
    /// three callers before this and each remembered a different subset.
    fn set_mode(&mut self, mode: OpMode) {
        // The decision is `kreate::mode::plan`, where it can be tested: this
        // function needs a window, a GPU and an event loop, so nothing written
        // inside it can be. A sweep proved the point — the guard that used to
        // live here, and the selection sweep below, could both be removed
        // without a single test noticing.
        let target = kreate_ops::ops::mesh::target_mesh(&self.doc);
        let Some(change) = kreate::mode::plan(self.op_ctx.mode, mode, target.is_some()) else {
            return;
        };
        self.op_ctx.mode = change.to;

        self.run_op_with(
            "mesh.select",
            &[("action", PropertyValue::Enum(u32::from(!change.select_all)))],
        );

        // Opened after the selection change rather than before it, so the
        // sessions start current with the document instead of immediately
        // stale.
        //
        // One per mesh in edit mode, in `edit_targets` order — the same order
        // the operators walk them in, so a multi-mesh edit is deterministic.
        if change.open_session {
            let meshes: Vec<_> = kreate_ops::ops::mesh::edit_targets(&self.doc)
                .into_iter()
                .map(|t| t.mesh)
                .collect();
            self.edit.open_for(&self.doc, &meshes);
        } else {
            self.edit.clear();
        }

        // The mode decides which context a shortcut resolves in, so the
        // tooltips change with it: `X` deletes an object in one mode and
        // constrains an axis in the other.
        self.refresh_command_info();
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// What a command looks like to the user: name, sentence, and shortcut.
    ///
    /// The shortcut comes from the *keymap*, resolved in the context the command
    /// would actually run in. A hardcoded table would be a second place the keys
    /// are written down, and it would go stale the moment anyone rebound one —
    /// which §2.5 makes a first-class thing to do.
    fn command_info(
        &self,
        op_id: &str,
        params: &PropertyMap,
    ) -> kreate_ui::command_info::CommandInfo {
        let label = self
            .registry
            .get(op_id)
            .map(|info| info.label)
            .unwrap_or(op_id);
        let context = if self.op_ctx.mode == OpMode::EditMesh {
            kreate_input::Context::EditMesh
        } else {
            kreate_input::Context::ObjectMode
        };
        kreate_ui::command_info::describe(
            &self.ui.catalog,
            &kreate_ui::command_info::variant_id(op_id, params),
            label,
            self.keymap.shortcut_for_action(op_id, params, context),
        )
    }

    /// The commands the right-click menu offers, for the current mode.
    ///
    /// Filtered by each operator's own `poll`, so nothing unusable is offered.
    /// Greyed-out rows are the alternative and are worse in a pop-up: a menu
    /// opened *at* something, where most rows are dead, makes the user hunt for
    /// the live ones. The menu bar can afford that; a menu under the cursor
    /// cannot.
    /// How long ago an autosave was written, in words.
    ///
    /// Formatted here rather than in the panel because it is prose, and
    /// because the panel has no business knowing about clocks. It matters more
    /// than it looks: a crashed run leaves one autosave per session, they are
    /// all called whatever the document was called, and the time is the only
    /// thing that tells the user which one they want back.
    fn describe_age(&self, modified: Option<std::time::SystemTime>) -> String {
        let t = |key: &str, n: u64| {
            self.ui
                .catalog
                .format(key, kreate_i18n::TextKind::Ui, &[("n", &n.to_string())])
        };
        let Some(modified) = modified else {
            // No timestamp is not an error worth a message of its own; an
            // unknown age reads as the oldest thing there is, which is also
            // where the list puts it.
            return String::new();
        };
        let Ok(elapsed) = modified.elapsed() else {
            // A file dated in the future — a clock change, or a copy from
            // another machine. "Just now" is the least misleading answer.
            return self
                .ui
                .catalog
                .format("recover-age-now", kreate_i18n::TextKind::Ui, &[]);
        };
        let seconds = elapsed.as_secs();
        match seconds {
            0..=59 => self
                .ui
                .catalog
                .format("recover-age-now", kreate_i18n::TextKind::Ui, &[]),
            60..=3599 => t("recover-age-minutes", seconds / 60),
            3600..=86_399 => t("recover-age-hours", seconds / 3600),
            _ => t("recover-age-days", seconds / 86_400),
        }
    }

    /// Publishes the UV map for the editor panel.
    ///
    /// Read from the *edit structure*, not from the document, and that is not
    /// a convenience: the UV operators take corner slots, which are arena
    /// slots with gaps in them, while the document's corners are dense. A
    /// panel that published dense indices would select the wrong corners the
    /// moment anything had been deleted — silently, and only on meshes that
    /// had been edited.
    ///
    /// Rebuilt each frame the panel is open rather than cached against a
    /// version: a UV drag changes it sixty times a second, and a cache that
    /// every operator has to remember to invalidate is a cache one of them
    /// will forget.
    fn publish_uv_view(&mut self) {
        if !self.ui.has_editor(kreate_ui::EditorKind::UvEditor) {
            self.ui.uv = None;
            return;
        }
        // The target's, because a UV editor shows one layout. Which mesh in
        // edit mode that is follows the same rule everything else does: the
        // active object, or the first selected one.
        let session =
            kreate_ops::ops::mesh::target_mesh(&self.doc).and_then(|mesh| self.edit.get(mesh));
        let Some(session) = session else {
            // Outside edit mode there is no UV map to show, which the panel
            // says out loud rather than drawing an empty square.
            self.ui.uv = None;
            return;
        };
        // `Some(empty)` when the mesh has no UV layer, so the panel can tell
        // "no mesh" from "no UVs" and say whichever is useful. The extraction
        // itself lives in `uvview`, where it can be tested.
        self.ui.uv = Some(kreate::uvview::from_bmesh(session.bmesh()).unwrap_or_default());
    }

    fn build_context_menu(&mut self) {
        use kreate_ui::context_menu::{Entry, Group};

        // `(operator, parameters)` rather than an operator alone, because a
        // menu row is a *command* and several rows can reach one operator:
        // `mesh.normals` recalculates or flips depending on `action`, and
        // `mesh.shade` smooths or facets depending on `smooth`. Carrying the
        // parameters is also what lets each row find its own shortcut instead of
        // borrowing the other variant's.
        fn group(
            title: &str,
            rows: &[(&str, &[(&'static str, PropertyValue)])],
            app: &App,
        ) -> Group {
            let entries = rows
                .iter()
                .filter(|(id, _)| {
                    app.registry
                        .get(id)
                        .is_some_and(|info| info.poll(&app.op_ctx, &app.doc))
                })
                .map(|(id, pairs)| {
                    let mut params = PropertyMap::new();
                    for (name, value) in *pairs {
                        params.set(name, value.clone());
                    }
                    let info = app.command_info(id, &params);
                    Entry::new(*id, info).with_params(params)
                })
                .collect();
            Group {
                title: title.to_string(),
                entries,
            }
        }

        let t = |key: &str| self.ui.catalog.get(key);
        let groups = if self.op_ctx.mode == OpMode::EditMesh {
            vec![
                group(
                    &t("menu-group-transform"),
                    &[
                        ("mesh.translate", &[]),
                        ("mesh.rotate", &[]),
                        ("mesh.scale", &[]),
                    ],
                    self,
                ),
                group(
                    &t("menu-group-shape"),
                    &[
                        ("mesh.extrude_region", &[]),
                        ("mesh.inset", &[]),
                        ("mesh.bevel", &[]),
                        ("mesh.subdivide", &[]),
                        ("mesh.loop_cut", &[]),
                        ("mesh.bridge", &[]),
                        ("mesh.fill", &[]),
                        ("mesh.spin", &[]),
                    ],
                    self,
                ),
                group(
                    &t("menu-group-remove"),
                    &[
                        ("mesh.delete", &[]),
                        ("mesh.dissolve", &[]),
                        ("mesh.merge", &[]),
                        ("mesh.split", &[]),
                        ("mesh.separate", &[]),
                    ],
                    self,
                ),
                {
                    // Project-from-view is the one row whose parameters the
                    // *application* has to supply: the operator takes a basis
                    // rather than a camera, on purpose — see `ops::uv` — so
                    // the view-dependence lives here, at the one place that
                    // legitimately has a view, and not in the keymap or the
                    // kernel.
                    let right = self.camera.right();
                    let up = self.camera.up();
                    group(
                        &t("menu-group-uv"),
                        &[
                            (
                                "uv.project",
                                &[
                                    ("mode", PropertyValue::Enum(0)),
                                    ("right", PropertyValue::Vec3([right.x, right.y, right.z])),
                                    ("up", PropertyValue::Vec3([up.x, up.y, up.z])),
                                ],
                            ),
                            ("uv.project", &[("mode", PropertyValue::Enum(1))]),
                            ("uv.project", &[("mode", PropertyValue::Enum(2))]),
                            ("uv.project", &[("mode", PropertyValue::Enum(3))]),
                            ("uv.smart_project", &[]),
                            ("uv.pack", &[]),
                        ],
                        self,
                    )
                },
                group(
                    &t("menu-group-shading"),
                    &[
                        ("mesh.shade", &[("smooth", PropertyValue::Bool(true))]),
                        ("mesh.shade", &[("smooth", PropertyValue::Bool(false))]),
                        ("mesh.mark_sharp", &[]),
                        ("mesh.normals", &[("action", PropertyValue::Enum(0))]),
                        ("mesh.normals", &[("action", PropertyValue::Enum(2))]),
                    ],
                    self,
                ),
            ]
        } else {
            vec![
                group(
                    &t("menu-group-transform"),
                    &[
                        ("object.translate", &[]),
                        ("object.rotate", &[]),
                        ("object.scale", &[]),
                    ],
                    self,
                ),
                group(
                    &t("menu-group-object"),
                    &[
                        ("object.add_primitive", &[]),
                        (
                            "object.duplicate",
                            &[("linked", PropertyValue::Bool(false))],
                        ),
                        ("object.duplicate", &[("linked", PropertyValue::Bool(true))]),
                        ("object.join", &[]),
                        ("object.delete", &[]),
                    ],
                    self,
                ),
            ]
        };
        self.ui.context_menu.groups = groups;
    }

    /// Opens the right-click menu at the cursor.
    fn open_context_menu(&mut self) {
        let Some((rx, ry, _, _)) = self.viewport_px else {
            return;
        };
        let Some((x, y)) = self.input.cursor else {
            return;
        };
        let scale = self.egui_ctx.pixels_per_point().max(0.01);
        self.build_context_menu();
        // Interface points, not physical pixels: everything egui places is in
        // points, and passing pixels puts the menu at a fraction of the right
        // position on any display that is not at 100%.
        self.ui.context_menu.at = Some(egui::pos2(
            (x as f32 - rx) / scale + rx / scale,
            (y as f32 - ry) / scale + ry / scale,
        ));
        self.redraw.set(RedrawReason::UI);
    }

    /// Refreshes the tooltips the top row shows.
    ///
    /// Rebuilt when the mode changes rather than every frame: it walks the
    /// keymap once per command, and the answer only moves when the mode or the
    /// bindings do.
    fn refresh_command_info(&mut self) {
        // `(operator, parameters)` for the same reason the menu uses it: two
        // buttons can run one operator, and each needs its own shortcut.
        const SHOWN: &[(&str, &[(&str, PropertyValue)])] = &[
            ("object.add_primitive", &[]),
            ("object.duplicate", &[]),
            ("object.delete", &[]),
            ("object.translate", &[]),
            ("object.rotate", &[]),
            ("object.scale", &[]),
            ("mesh.translate", &[]),
            ("mesh.rotate", &[]),
            ("mesh.scale", &[]),
            ("mesh.extrude_region", &[]),
            ("mesh.subdivide", &[]),
            ("mesh.dissolve", &[]),
            ("mesh.delete", &[]),
            ("mesh.inset", &[]),
            ("mesh.bridge", &[]),
            ("mesh.loop_cut", &[]),
            ("mesh.mark_sharp", &[]),
            ("mesh.fill", &[]),
            ("mesh.spin", &[]),
            ("mesh.bevel", &[]),
            ("mesh.merge", &[]),
            ("mesh.split", &[]),
        ];
        self.ui.command_info = SHOWN
            .iter()
            .map(|(id, pairs)| {
                let mut params = PropertyMap::new();
                for (name, value) in *pairs {
                    params.set(name, value.clone());
                }
                (id.to_string(), self.command_info(id, &params))
            })
            .collect();
    }

    /// Hands the recovery list to the interface.
    ///
    /// Copied across rather than shared, because the interface layer reads a view
    /// model and never reaches into application state — the same rule that keeps
    /// the renderer off the document.
    fn sync_recoverable(&mut self) {
        if self.ui.recoverable.len() != self.recoverable.len() {
            self.ui.recoverable = self
                .recoverable
                .iter()
                .map(|r| shell::RecoveryOffer {
                    path: r.path.clone(),
                    name: r.name.clone(),
                    age: self.describe_age(r.modified),
                })
                .collect();
        }
    }

    fn set_select_mode(&mut self, mode: u32) {
        self.select_mode = mode;
        self.ui.select_mode = mode;
        self.redraw.set(RedrawReason::UI);
    }

    // --- the interactive transform -------------------------------------------

    /// Where the selection is, in world space.
    ///
    /// The median of what is selected: the objects' origins in object mode, the
    /// affected vertices in edit mode. This is what a rotation turns about and
    /// what a scale grows from, and it is computed here rather than inside the
    /// gesture because what counts as "the selection" is a question about the
    /// document.
    fn selection_pivot(&self) -> Option<DVec3> {
        if self.op_ctx.mode == OpMode::EditMesh {
            // The operator's own answer, not a second calculation of it. The
            // gesture measures the pointer's angle and its scale factor from
            // this point, so a pivot that disagreed with the one the transform
            // turns about would make the mesh lag the cursor — and with several
            // objects in edit mode the joint median is nowhere near any one
            // object's own.
            //
            // Zero radius on purpose: proportional editing widens what *moves*
            // and not what the selection's median is.
            return kreate_ops::ops::mesh::weigh_edit_selection(&self.doc, 0.0)
                .map(|weighing| weighing.pivot);
        }
        let objects = &self.doc.selection.objects;
        if objects.is_empty() {
            return None;
        }
        let mut total = DVec3::ZERO;
        for id in objects {
            total += self.doc.world_transform(*id).translation;
        }
        Some(total / objects.len() as f64)
    }

    /// The cursor in viewport pixels, **without** the bounds check.
    ///
    /// A gesture keeps tracking when the pointer leaves the viewport, which is
    /// why this exists beside `cursor_viewport_px`. Someone dragging a long way
    /// runs off the edge of a docked viewport routinely, and a transform that
    /// froze there — or worse, snapped back — would be unusable. What is outside
    /// the viewport is still a real pointer position; only *picking* needs it to
    /// be inside.
    fn cursor_viewport_unclamped(&self) -> Option<(f32, f32)> {
        let (x, y, _, _) = self.viewport_px?;
        let (cx, cy) = self.input.cursor?;
        Some((cx as f32 - x, cy as f32 - y))
    }

    /// The selection's own axes, for a local-orientation constraint.
    ///
    /// The active object's rotation. In edit mode that is the object the
    /// elements belong to, which is the right answer: mesh elements have no
    /// orientation of their own, and what a modeller means by "local" while
    /// editing a mesh is the object it sits in.
    ///
    /// Falls back to the world axes when there is nothing to ask, so a local
    /// constraint always has *some* meaning rather than collapsing to zero.
    fn selection_basis(&self) -> [Vec3; 3] {
        let world = self
            .doc
            .selection
            .active
            .or_else(|| self.doc.selection.objects.first().copied())
            .map(|id| self.doc.world_transform(id));
        let Some(world) = world else {
            return [Vec3::X, Vec3::Y, Vec3::Z];
        };
        let m = world.matrix3;
        [
            Vec3::new(m.x_axis.x as f32, m.x_axis.y as f32, m.x_axis.z as f32),
            Vec3::new(m.y_axis.x as f32, m.y_axis.y as f32, m.y_axis.z as f32),
            Vec3::new(m.z_axis.x as f32, m.z_axis.y as f32, m.z_axis.z as f32),
        ]
    }

    /// The geometry a drag may snap to: everything that is not about to move.
    ///
    /// The exclusion is the whole point, and it is why this cannot be a general
    /// scene index reused between tools. A selection that can snap to itself
    /// locks solid the moment the drag starts: the nearest vertex to a moving
    /// vertex is itself, at zero distance, always.
    ///
    /// Built from the *evaluated* scene, so a drag snaps to what is on screen —
    /// including geometry a modifier produced, which is what the user is
    /// looking at and aiming for.
    fn build_snap_world(&self) -> kreate::snap::SnapWorld {
        let mut world = kreate::snap::SnapWorld::default();
        if !self.snap.mode.needs_geometry() {
            // Increment snapping needs no candidates, and gathering a scene's
            // worth of them to ignore would be the whole cost for none of the
            // benefit.
            return world;
        }

        let edited = (self.op_ctx.mode == OpMode::EditMesh)
            .then(|| kreate_ops::ops::mesh::target_mesh(&self.doc))
            .flatten();

        for (id, mesh) in &self.evaluated {
            let object = self.doc.object(*id);
            let Some(object) = object else { continue };
            if !object.visible {
                continue;
            }
            let to_world = self.doc.world_transform(*id);

            let moving: Vec<u32> = if self.op_ctx.mode == OpMode::EditMesh {
                // In edit mode only the selected elements move, and only within
                // the mesh being edited. Every other object is a candidate whole.
                if object.mesh() == edited {
                    kreate_ops::ops::mesh::selection_vertices(mesh)
                } else {
                    Vec::new()
                }
            } else if self.doc.selection.objects.contains(id) {
                // In object mode a selected object moves entirely, so none of it
                // is a candidate.
                continue;
            } else {
                Vec::new()
            };

            world.add_mesh(mesh, &to_world, &moving);
        }
        world
    }

    /// Starts an interactive transform.
    fn begin_gesture(&mut self, kind: kreate::modal::Kind) {
        // One at a time. Starting a second would abandon the first's transaction
        // and leave the document part-transformed with nothing recording it.
        if self.gesture.is_some() {
            return;
        }
        // Bounds-checked at the start: a transform begun with the pointer over
        // a panel has no sensible reference position, and the reference is what
        // every later frame is measured against.
        if self.cursor_viewport_px().is_none() {
            self.status("point at the viewport to transform");
            return;
        }
        let Some(cursor) = self.cursor_viewport_unclamped() else {
            return;
        };
        let Some((_, _, w, h)) = self.viewport_px else {
            return;
        };
        let Some(pivot) = self.selection_pivot() else {
            self.status("nothing selected");
            return;
        };

        let mode = self.op_ctx.mode;
        let mut gesture = kreate::modal::Gesture::begin(
            kind,
            cursor,
            pivot,
            &self.camera,
            (w, h),
            self.selection_basis(),
            self.snap,
            self.build_snap_world(),
        );
        // Ctrl inverts snapping, and it may already be held when the gesture
        // starts. Read once here rather than waiting for the next modifier
        // event, which might never come.
        gesture.set_snap_inverted(self.input.ctrl_down);
        // Proportional editing is meaningless in object mode: there is no
        // neighbouring geometry to fall off towards, only other objects, and
        // dragging one object part-way because another is near it is not
        // something anyone asks for.
        // Asked of the mode, because that is the claim: proportional editing
        // falls off towards neighbouring *geometry*, and only a mode that has
        // the elements on screen has any.
        if mode.shows_elements() {
            gesture.set_proportional(self.proportional);
        }
        let op_id = kind.op_id(mode);
        // Started with the identity parameters, so pressing the key alone moves
        // nothing. The first pointer motion is what does anything.
        match self.history.begin_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            op_id,
            &gesture.params(),
        ) {
            Ok(Some(session)) => {
                let status = gesture.status();
                self.gesture = Some((gesture, session));
                // After the session, not before: a preview for a gesture the
                // operator declined would deform a mesh nothing is going to
                // transform.
                self.begin_preview(op_id);
                self.status(&status);
            }
            // The operator declined. Nothing was started, and nothing must be:
            // driving a gesture whose `begin_modal` returned `None` would retune
            // whatever entry happened to be on top of the history.
            Ok(None) => self.status(&format!("{} is not possible here", kind.label())),
            Err(e) => self.status(&format!("{}: {e}", kind.label())),
        }
    }

    /// Starts previewing the drag on the GPU, if this one can be.
    ///
    /// Everything about it is best-effort: every `None` below falls through to
    /// the CPU path, which is the path that ran before any of this existed and
    /// still produces the authoritative answer at confirm. So a mesh with a
    /// modifier stack, an operator that changes topology, an empty selection or
    /// a missing GPU all simply drag the old way.
    fn begin_preview(&mut self, op_id: &'static str) {
        if !dragpreview::is_previewable(op_id) {
            return;
        }
        let Some(mesh_id) = kreate_ops::ops::mesh::target_mesh(&self.doc) else {
            return;
        };
        let Some(doc_mesh) = self.doc.mesh_arc(mesh_id).cloned() else {
            return;
        };
        let active = self
            .doc
            .selection
            .active
            .or_else(|| self.doc.selection.objects.first().copied());
        let Some(active) = active else { return };
        // A modifier stack means the mesh on the GPU is not the mesh the
        // operator transforms, and deforming the stack's output with weights
        // measured on its input is not an approximation of anything — the two
        // need not even have the same vertex count. So a stack means the CPU
        // path, which re-evaluates properly every frame.
        //
        // Asked of the object rather than inferred by comparing handles,
        // because that comparison cannot be made here: see below.
        if self
            .doc
            .object(active)
            .is_none_or(|o| !o.modifiers.is_empty())
        {
            return;
        }
        // The slot this object's geometry is in, found by the handle the cache
        // was built with — which is *last frame's*, not the document's. The
        // identity parameters `begin_modal` just applied replaced the document's
        // handle, since `mesh_mut` is `Arc::make_mut` and the cache holds a
        // clone.
        let Some((_, stale)) = self.evaluated.iter().find(|(id, _)| *id == active) else {
            return;
        };
        let stale = stale.clone();

        // The operator's own falloff, not a second implementation of it: the
        // preview is what the user shapes the mesh against and the operator is
        // what runs when the pointer comes up, so any divergence would show as
        // the mesh moving at the end of every drag.
        let radius = self.proportional.unwrap_or(0.0);
        let to_world = self.doc.world_transform(active);
        let Some(weighing) = kreate_ops::ops::mesh::weigh_edit_selection(&self.doc, radius) else {
            return;
        };
        // One `Deform` covers one mesh, so a drag reaching several takes the CPU
        // path. Not a placeholder: previewing one of them and leaving the rest
        // to the operator would put two different transforms on screen at once,
        // which is worse than a slow frame — and the pivot the preview would use
        // is the joint one, so the previewed mesh would be visibly turning about
        // a point the others have not agreed to yet.
        let [part] = &weighing.parts[..] else {
            return;
        };
        if part.target.mesh != mesh_id {
            return;
        }
        // An object flattened by a zero scale has no way back from world space.
        // The operator skips the same case, so this mesh would not move anyway.
        let Some(place) = kreate_render::Placement::new(to_world, weighing.pivot) else {
            return;
        };
        // The one expensive thing a preview costs, and it is paid once per drag
        // instead of once per frame — which is the entire point. What it buys is
        // the tables the normal passes walk, and they depend only on topology,
        // which a transform does not change.
        //
        // Measured at 190 ms for two hundred thousand vertices, against 0.3 ms
        // for a previewed frame. It is the obvious next thing to move: the
        // tables belong beside the resident mesh, and then a drag begins for
        // free. Until then, pressing the key costs what every *frame* used to.
        let cpu = RenderMeshCpu::extract(&doc_mesh);
        let weights = dragpreview::spread_weights(
            &cpu.render_to_vert,
            doc_mesh.num_verts() as usize,
            &part.indices,
            &part.weights,
        );
        let Some(gfx) = self.gfx.as_mut() else { return };
        let Some((held, gpu_mesh)) = gfx.meshes.iter_mut().find(|(m, _)| Arc::ptr_eq(m, &stale))
        else {
            return;
        };
        // Bringing the slot up to date out of the extraction that was needed
        // anyway. Without this the next `sync_gpu` would find a positions-only
        // change and extract the mesh a second time, on the frame the user
        // pressed the key — and adopting the handle is what makes that sync find
        // nothing to do, which is in turn what lets the preview survive it.
        //
        // Refused rather than forced if the topology moved: `update_positions`
        // has no size check of its own, and a mismatch would write past the end
        // of a buffer.
        if !stale.only_positions_differ(&doc_mesh) && !Arc::ptr_eq(&stale, &doc_mesh) {
            return;
        }
        if cpu.positions.len() as u32 != gpu_mesh.render_verts {
            return;
        }
        gpu_mesh.update_positions(&gfx.gpu.queue, &cpu);
        *held = doc_mesh.clone();

        let mut encoder = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview begin"),
            });
        let deform = kreate_render::Deform::begin(
            &gfx.gpu.device,
            &mut encoder,
            &gfx.deformer,
            gpu_mesh,
            &cpu,
            &weights,
        );
        let Some(deform) = deform else { return };
        gfx.gpu.queue.submit([encoder.finish()]);
        gfx.preview = Some(LivePreview {
            deform,
            op_id,
            render_to_vert: cpu.render_to_vert,
            mesh_verts: doc_mesh.num_verts() as usize,
            place,
        });
    }

    /// Shows this frame of the drag on the GPU. `true` when it did.
    ///
    /// A false return means the caller has to write the document instead, and
    /// the preview has already been given up and the mesh put back — so the
    /// fallback is a frame late at worst, never a frame wrong.
    fn drive_preview(&mut self, params: &kreate_reflect::PropertyMap) -> bool {
        let Some(gfx) = self.gfx.as_mut() else {
            return false;
        };
        let Some(preview) = gfx.preview.as_ref() else {
            return false;
        };
        let Some(gesture) = dragpreview::gesture_for(preview.op_id, params, preview.place) else {
            // The params were not the shape this operator's preview reads.
            // Nothing here should produce that, which is exactly why it is
            // handled rather than unwrapped: showing a stale frame while the
            // pointer moves is the one failure this path must not have.
            gfx.restore_preview();
            return false;
        };
        preview.deform.set(&gfx.gpu.queue, &gesture);
        let mut encoder = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview frame"),
            });
        preview.deform.dispatch(&mut encoder, &gfx.deformer);
        gfx.gpu.queue.submit([encoder.finish()]);
        true
    }

    /// Rewrites the falloff a live preview is using.
    ///
    /// Called when the wheel changes the proportional radius mid-drag. The
    /// weights are the only thing that changes: the rest copy, the tables and
    /// the split all depend on topology and the radius is not part of it.
    fn reweigh_preview(&mut self) {
        if self.gfx.as_ref().is_none_or(|g| g.preview.is_none()) {
            return;
        }
        let radius = self.proportional.unwrap_or(0.0);
        let to_world = kreate_ops::ops::mesh::target_object(&self.doc)
            .map(|id| self.doc.world_transform(id))
            .unwrap_or(kreate_math::glam::DAffine3::IDENTITY);
        let weighing = kreate_ops::ops::mesh::weigh_edit_selection(&self.doc, radius);
        let Some(gfx) = self.gfx.as_mut() else { return };
        // A wider radius can reach a *second* mesh, and one `Deform` cannot
        // follow it there. Giving the preview up is the honest answer: the drag
        // finishes on the CPU path, which does reach.
        let single = weighing.as_ref().filter(|w| w.parts.len() == 1);
        let Some(weighing) = single else {
            gfx.restore_preview();
            return;
        };
        let part = &weighing.parts[0];
        // The radius does not move the pivot — the median is the selection's and
        // the selection has not changed — but it is free to carry the fresh one,
        // and pinning it to what the falloff was measured with is one fewer
        // thing that can drift apart.
        let place = kreate_render::Placement::new(to_world, weighing.pivot);
        let Some(preview) = gfx.preview.as_mut() else {
            return;
        };
        let weights = dragpreview::spread_weights(
            &preview.render_to_vert,
            preview.mesh_verts,
            &part.indices,
            &part.weights,
        );
        if let Some(place) = place {
            preview.place = place;
        }
        if !preview.deform.set_weights(&gfx.gpu.queue, &weights) {
            // A refused length means the render mesh is not the one this preview
            // was built against, which should be impossible while the document
            // is untouched. Put the mesh back rather than leave a falloff that
            // is right at one end of the buffer and stale at the other.
            gfx.restore_preview();
        }
    }

    /// Feeds the pointer to the gesture and re-applies the operator.
    fn drive_gesture(&mut self) {
        let Some(cursor) = self.cursor_viewport_unclamped() else {
            return;
        };
        // Copied out before the mutable borrow of `self.gesture` below.
        let camera = self.camera;
        let viewport_px = self.viewport_px;
        // Scoped, because the preview below needs `&mut self` and the params are
        // all it needs from here.
        let (params, status) = {
            let Some((gesture, _)) = &mut self.gesture else {
                return;
            };
            // The camera first: orbiting during a transform is allowed, and a
            // snap resolved against a stale view would catch on whatever used to
            // be under the cursor.
            if let Some((_, _, w, h)) = viewport_px {
                gesture.set_view(&camera, (w, h));
            }
            gesture.track(cursor);
            (gesture.params(), gesture.status())
        };

        // The GPU preview, when there is one: a 64-byte uniform and three
        // dispatches, and the document is not written at all.
        //
        // Which is the whole optimisation. The CPU path below writes the
        // positions array, which makes a new `MeshData`, which `gpu_sync`
        // recognises as positions-only and repositions — and `update_positions`
        // calls `RenderMeshCpu::extract` to find out what to write. That
        // extraction is 83 ms at two hundred thousand vertices against a budget
        // of one, and almost none of it is work a move changes.
        //
        // §1.7 already said the document should not be written until the
        // gesture is confirmed. This is what it costs not to.
        if self.drive_preview(&params) {
            self.status(&status);
            self.redraw.set(RedrawReason::SCENE_CHANGED);
            return;
        }

        // Restore and re-apply from the original — §2.2's rule. `update_modal`
        // owns that; nothing here is incremental.
        let Some((_, session)) = &mut self.gesture else {
            return;
        };
        if let Err(e) = self.history.update_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            session,
            &params,
        ) {
            tracing::warn!(error = %e, "transform failed");
        }
        self.status(&status);
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Confirms the gesture: one history entry, carrying the final parameters.
    fn confirm_gesture(&mut self) {
        let Some((gesture, mut session)) = self.gesture.take() else {
            return;
        };
        // A previewed drag never wrote the document, so the operator has to run
        // now — once, with the final parameters. Which is exactly the shape
        // §1.6 wants anyway: the history entry, the adjust panel and the script
        // line all come from one `(op_id, params)` call, and the pointer's whole
        // journey leaves no trace beyond where it ended up.
        //
        // The preview is dropped rather than restored: this run rewrites the
        // positions, which makes a new handle, which `gpu_sync` repositions. One
        // extraction, at mouse-up, instead of one per frame.
        if self
            .gfx
            .as_mut()
            .is_some_and(|gfx| gfx.preview.take().is_some())
        {
            if let Err(e) = self.history.update_modal(
                &self.registry,
                &mut self.doc,
                Some(&mut self.edit),
                &self.op_ctx,
                &mut session,
                &gesture.params(),
            ) {
                tracing::warn!(error = %e, "transform failed");
            }
        }
        let call = session.params().to_script_call(session.op_id());
        self.history.commit_modal(session);
        // Printed like any other operator, because it *is* like any other
        // operator: the line reproduces the drag exactly.
        println!("{call}");
        self.status(&format!("{} done", gesture.kind.label()));
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Abandons the gesture. The document goes back and nothing is recorded.
    fn cancel_gesture(&mut self) {
        let Some((gesture, session)) = self.gesture.take() else {
            return;
        };
        // The preview has to be undone explicitly, and this is the one place
        // where that is not obvious. A cancel puts the document back to the very
        // handle the GPU cache is already holding — that is what makes it free
        // — so `gpu_sync` sees nothing to do and would leave the deformed
        // positions on screen with the document at rest.
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.restore_preview();
        }
        History::cancel_modal(session, &mut self.doc);
        self.status(&format!("{} cancelled", gesture.kind.label()));
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Grows or shrinks the proportional radius during a gesture.
    ///
    /// The wheel, because that is the hand that is free: the pointer is driving
    /// the transform and the other hand is on the modifiers. Multiplicative
    /// rather than additive, so one notch feels the same whether the radius is a
    /// centimetre or ten metres.
    fn adjust_proportional(&mut self, notches: f32) -> bool {
        if self.gesture.is_none() || self.proportional.is_none() {
            return false;
        }
        let radius = self.proportional.unwrap_or(1.0) * 1.15f64.powf(notches as f64);
        let radius = radius.clamp(1e-4, 1e4);
        self.proportional = Some(radius);
        if let Some((gesture, _)) = &mut self.gesture {
            gesture.set_proportional(Some(radius));
        }
        // Before the frame, not after: the falloff is what the frame is about to
        // be drawn with, and a frame drawn with the old one would show the wheel
        // lagging a notch behind the hand.
        self.reweigh_preview();
        self.drive_gesture();
        true
    }

    /// Engages or releases precision on a running gesture.
    ///
    /// Measured from where the pointer is *now*, which is what stops the
    /// selection jumping when the modifier goes down or comes up. §2.2 is
    /// explicit about that, and it is the difference between a precision mode
    /// that is usable and one that surprises the user twice per drag.
    fn set_gesture_precision(&mut self, on: bool) {
        let Some(cursor) = self.cursor_viewport_unclamped() else {
            return;
        };
        let Some((gesture, _)) = &mut self.gesture else {
            return;
        };
        gesture.set_precise(on, cursor);
        self.drive_gesture();
    }

    /// A keystroke while a gesture runs. `true` when the gesture consumed it.
    ///
    /// The modal stack sits above the keymap in §2.1's pipeline, so this is
    /// checked first and the keymap never sees these keys during a transform.
    /// Without that, `X` would delete the selection halfway through moving it.
    fn gesture_key(&mut self, code: KeyCode) -> bool {
        use kreate::modal::Axis;
        if self.gesture.is_none() {
            return false;
        }
        let axis = match code {
            KeyCode::KeyX => Some(Axis::X),
            KeyCode::KeyY => Some(Axis::Y),
            KeyCode::KeyZ => Some(Axis::Z),
            _ => None,
        };
        if let Some(axis) = axis {
            if let Some((gesture, _)) = &mut self.gesture {
                gesture.cycle_axis(axis);
            }
            self.drive_gesture();
            return true;
        }
        match code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                self.confirm_gesture();
                true
            }
            KeyCode::Escape => {
                self.cancel_gesture();
                true
            }
            KeyCode::Backspace => {
                if let Some((gesture, _)) = &mut self.gesture {
                    gesture.backspace();
                }
                self.drive_gesture();
                true
            }
            _ => {
                let typed = typed_character(code);
                match typed {
                    Some(c) => {
                        if let Some((gesture, _)) = &mut self.gesture {
                            gesture.type_character(c);
                        }
                        self.drive_gesture();
                        true
                    }
                    // Everything else is swallowed rather than passed on. A
                    // gesture is a mode: while one is running, a stray key must
                    // not run an unrelated operator on a half-transformed
                    // selection.
                    None => true,
                }
            }
        }
    }

    /// Applies the commands the panels queued this frame.
    fn drain_ui_commands(&mut self) {
        for command in self.ui.queue.drain() {
            match command {
                Command::RunOperator { op_id, params } => self.run_op(&op_id, params),
                Command::ToggleEditMode => self.toggle_edit_mode(),
                // An unknown tag is ignored, which is what a script or a keymap
                // written against a later version should get.
                Command::SetMode(tag) => {
                    if let Some(mode) = OpMode::from_tag(&tag) {
                        self.set_mode(mode);
                    }
                }
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
                Command::NewDocument => self.new_document(),
                Command::OpenDocument => self.open_document(),
                Command::SaveDocument => self.save_document(false),
                Command::SaveDocumentAs => self.save_document(true),
                Command::RecoverDocument { path } => {
                    self.load_from(&path);
                    // The recovered document deliberately keeps *no* path: it
                    // came from an autosave, and inheriting that path would make
                    // the next Ctrl+S write back into the autosave directory
                    // instead of asking where the document belongs.
                    self.file = DocumentFile::new();
                    docfile::forget_recoverable(&path);
                    self.recoverable.retain(|r| r.path != path);
                }
                Command::DiscardRecovery { path } => {
                    docfile::forget_recoverable(&path);
                    self.recoverable.retain(|r| r.path != path);
                    self.redraw.set(RedrawReason::UI);
                }
            }
        }
    }

    /// Replaces the document with an empty one.
    fn new_document(&mut self) {
        if !self.confirm_discarding_changes("start a new document") {
            return;
        }
        self.adopt(load_document(None));
        self.file = DocumentFile::new();
        self.status("new document");
    }

    fn open_document(&mut self) {
        if !self.confirm_discarding_changes("open another document") {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Kreate document", &["kreate"])
            .set_title("Open")
            .pick_file()
        else {
            return;
        };
        self.load_from(&path);
    }

    fn load_from(&mut self, path: &std::path::Path) {
        match DocumentFile::open(path) {
            Ok((document, file)) => {
                self.adopt(document);
                self.file = file;
                self.status(&format!("opened {}", path.display()));
            }
            // Reported, never fatal. A file that will not open is the most
            // ordinary thing a user can hand an application, and losing the
            // session they already have over it would be indefensible.
            Err(e) => self.status(&format!("cannot open {}: {e}", path.display())),
        }
    }

    fn save_document(&mut self, ask: bool) {
        let existing = self.file.path().map(|p| p.to_path_buf());
        let path = match (&existing, ask) {
            (Some(path), false) => path.clone(),
            _ => {
                let mut dialog = rfd::FileDialog::new()
                    .add_filter("Kreate document", &["kreate"])
                    .set_title(if ask { "Save As" } else { "Save" });
                if let Some(current) = &existing {
                    if let Some(parent) = current.parent() {
                        dialog = dialog.set_directory(parent);
                    }
                    if let Some(name) = current.file_name() {
                        dialog = dialog.set_file_name(name.to_string_lossy().to_string());
                    }
                } else {
                    dialog = dialog.set_file_name("untitled.kreate");
                }
                match dialog.save_file() {
                    Some(path) => path,
                    None => return,
                }
            }
        };
        // The dialog may hand back a name with no extension, and a document
        // called `model` is one the shell will not associate with anything.
        let path = if path.extension().is_none() {
            path.with_extension("kreate")
        } else {
            path
        };
        match self.file.save(&path, &self.doc) {
            Ok(()) => self.status(&format!("saved {}", path.display())),
            Err(e) => self.status(&format!("cannot save {}: {e}", path.display())),
        }
    }

    /// Asks before throwing away unsaved work. `true` to proceed.
    ///
    /// A modal dialog, and deliberately: this is the one moment where carrying
    /// on regardless destroys something the user cannot get back, and a
    /// non-blocking notification they can miss is not consent.
    fn confirm_discarding_changes(&self, what: &str) -> bool {
        if !self.file.is_modified(&self.doc) {
            return true;
        }
        let name = self
            .file
            .path()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "This document".to_string());
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Unsaved changes")
            .set_description(format!(
                "{name} has changes that have not been saved.\n\nDiscard them and {what}?"
            ))
            .set_buttons(rfd::MessageButtons::YesNo)
            .show()
            == rfd::MessageDialogResult::Yes
    }

    /// Replaces the document and everything derived from it.
    ///
    /// The caches are not optional to clear. `pick`, `modifiers` and `evaluated`
    /// are all keyed on the document that produced them, and a new document with
    /// the old caches shows the previous file's geometry — which reads as the
    /// open having silently failed.
    fn adopt(&mut self, document: Document) {
        self.doc = document;
        self.history = History::new();
        self.edit.clear();
        self.op_ctx.mode = OpMode::Object;

        self.pick = kreate::pick::PickCache::default();
        self.modifiers = kreate_scene::ModifierCache::new();
        self.evaluated.clear();
        self.hover = None;
        self.drag_select = None;
        self.ui.adjust_params = None;
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Puts a line in the log and on the status bar.
    fn status(&mut self, message: &str) {
        tracing::info!("{message}");
        self.ui.status = Some(message.to_string());
        self.redraw.set(RedrawReason::UI);
    }

    /// Writes an autosave if one is due, and keeps the window title current.
    ///
    /// Called once a frame. Both halves are almost always two comparisons and
    /// nothing else, which is why they can live on the frame path at all.
    fn tick_document(&mut self) {
        self.file.maybe_autosave(&self.doc);
        let title = self.file.title(&self.doc);
        if title != self.shown_title {
            if let Some(window) = &self.window {
                window.set_title(&title);
            }
            self.shown_title = title;
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
        // Past that return, so something other than the drag changed the scene
        // and buffers are about to be rebuilt or dropped. A live preview's bind
        // groups name the ones it was built against; see
        // `dragpreview::preview_survives` for why the answer is the whole plan
        // rather than the previewed slot.
        if !dragpreview::preview_survives(&plan, held.len()) && gfx.restore_preview() {
            tracing::debug!("the scene changed under a GPU preview; finishing the drag on the CPU");
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

        // A slot the plan wants re-flagged keeps its geometry and adopts the new
        // handle: three small buffer writes instead of re-uploading a mesh
        // because one face was clicked.
        //
        // Adopting the handle is not bookkeeping — it is the whole thing. The
        // draw loop finds a slot with `Arc::ptr_eq` against the evaluated mesh,
        // so a slot left holding the *previous* handle is a slot nothing
        // matches, and the object silently stops being drawn. That was the bug:
        // pressing Tab selects every element, which is a selection-only change,
        // which took this path — and the object vanished.
        for (slot, mesh) in &plan.reflags {
            let Some(Some((held, gpu_mesh))) = old.get_mut(*slot) else {
                continue;
            };
            // The flags alone. Extracting the whole render mesh to reach
            // them cost 922 ms on a million faces, and a click changes none of
            // it: not the triangulation, not the normals, not the split.
            let flags = kreate_render::ElementFlags::extract(mesh);
            if gpu_mesh.update_flags(&gfx.gpu.queue, &flags) {
                *held = mesh.clone();
            } else {
                // The element counts moved, so this was not a selection change
                // after all and the flag buffers are the wrong size. Dropping the
                // slot makes the plan fall through to a full upload rather than
                // leaving a mesh drawn with someone else's flags.
                old[*slot] = None;
            }
        }

        // And a slot the plan wants moved keeps everything but its positions
        // and normals: two writes instead of rebuilding every buffer
        // because the user dragged the pointer one pixel.
        //
        // `update_positions` has no size check to fail, unlike
        // `update_flags` — the render vertex count follows from the
        // topology, and `only_positions_differ` has already established
        // that the topology is identical. It is asserted rather than
        // trusted, because a wrong answer here writes past the end of
        // somebody's buffer.
        for (slot, mesh) in &plan.repositions {
            let Some(Some((held, gpu_mesh))) = old.get_mut(*slot) else {
                continue;
            };
            let cpu = RenderMeshCpu::extract(mesh);
            if cpu.positions.len() as u32 == gpu_mesh.render_verts {
                gpu_mesh.update_positions(&gfx.gpu.queue, &cpu);
                *held = mesh.clone();
            } else {
                old[*slot] = None;
            }
        }

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

    /// Whether the cursor is inside the viewport rectangle.
    ///
    /// Window coordinates only — deliberately not [`App::cursor_viewport_px`],
    /// which converts into the offscreen texture's pixels and so needs the
    /// graphics state to exist. Whether a press belongs to the scene is a
    /// question about the layout, not about the renderer, and tying the two
    /// together makes it untestable and answers `false` before the first frame
    /// for a reason that has nothing to do with the layout.
    fn cursor_in_viewport(&self) -> bool {
        let (Some((x, y)), Some((rx, ry, w, h))) = (self.input.cursor, self.viewport_px) else {
            return false;
        };
        let (u, v) = ((x as f32 - rx) / w, (y as f32 - ry) / h);
        (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)
    }

    /// Whether a press just now belongs to the viewport rather than the
    /// interface.
    ///
    /// Decided by geometry — is the pointer in the viewport, and outside the
    /// navigation gizmo — rather than by egui's `consumed`, which answers a
    /// different question and answers it wrongly here.
    ///
    /// `egui_wants_pointer_input` is `is_pointer_over_egui() && !any_button_down`,
    /// and the viewport lives inside a central panel, so "over egui" is true
    /// everywhere in it. The result is that *every* press over the scene came
    /// back consumed and every release did not — the exact inverse of what a
    /// viewport needs, and why navigation appeared to work only sometimes.
    ///
    /// The gizmo is excluded by its own circle rather than by a hover flag,
    /// because a flag is only recomputed when the interface runs and a press can
    /// arrive between frames.
    fn press_belongs_to_viewport(&self) -> bool {
        if !self.cursor_in_viewport() {
            return false;
        }
        let Some((x, y)) = self.input.cursor else {
            return false;
        };
        let scale = self.egui_ctx.pixels_per_point() as f64;
        let at = egui::pos2((x / scale) as f32, (y / scale) as f32);
        // Anything drawn over the scene takes the pointer: the sidebar's
        // coordinate fields above all, where a drag would otherwise type a
        // number *and* orbit the camera.
        if self.ui.viewport_blockers.iter().any(|r| r.contains(at)) {
            return false;
        }
        let Some((centre, radius)) = self.ui.gizmo_circle else {
            return true;
        };
        let dx = x / scale - centre.x as f64;
        let dy = y / scale - centre.y as f64;
        // The gizmo takes its own circle, and the viewport everything else. One
        // drag must not both turn the gizmo and turn the scene.
        dx * dx + dy * dy > (radius as f64) * (radius as f64)
    }

    /// Applies a scroll, in notches.
    ///
    /// Horizontal scroll pans. It used to be discarded, so a sideways two-finger
    /// swipe on a trackpad — and a tilt wheel — did nothing at all.
    ///
    /// The vertical component asks the keymap, so the wheel is rebindable like
    /// everything else. Unbound combinations fall back to zooming rather than
    /// doing nothing: a wheel that is inert for no visible reason reads as the
    /// application being broken, and a trackpad pinch arrives here as
    /// Ctrl+wheel on Windows.
    fn apply_wheel(&mut self, x: f64, y: f64) {
        if x == 0.0 && y == 0.0 {
            return;
        }
        self.input.note_event();
        if x != 0.0 {
            self.input.pan_delta.0 += x * WHEEL_PAN_PIXELS;
        }
        if y == 0.0 {
            return;
        }
        let action = self.keymap.resolve(
            KeyContext::Viewport,
            &KeyInput::Wheel,
            self.modifiers(),
            EventKind::Press,
        );
        match action.as_ref().map(|a| a.op_id.as_str()) {
            // Sideways, not up and down: this is the horizontal-scroll
            // convention, and a vertical pan on the wheel would fight the zoom
            // sharing it.
            Some("view.pan") => self.input.pan_delta.0 += y * WHEEL_PAN_PIXELS,
            Some("view.orbit") => self.input.orbit_delta.1 += y * WHEEL_PAN_PIXELS,
            _ => self.input.scroll += y,
        }
    }

    /// Ends every drag in progress, wherever its release went.
    ///
    /// Modifier state is cleared too: a drag that ended with Alt held leaves the
    /// application believing Alt is still down, because the key-up went to
    /// whatever took focus.
    fn cancel_all_drags(&mut self) {
        self.buttons = [false; 3];
        self.drag_ops = [const { None }; 3];
        // Dropped rather than committed. A rubber band whose release was never
        // seen has no end point worth trusting, and committing a selection the
        // user did not finish is worse than losing one they did.
        self.drag_select = None;
        self.input.shift_down = false;
        self.input.ctrl_down = false;
        self.input.alt_down = false;
    }

    /// Feeds a pointer movement to whatever drags are in progress.
    ///
    /// What each drag does was settled when it began — see [`App::drag_ops`] —
    /// so this never consults the keymap. Which navigation a button performs
    /// still comes from the keymap, just once: the Maya preset's Alt+LMB orbit
    /// needs no code here at all.
    ///
    /// Separate from the event handler so it can be tested. A drag that quietly
    /// changes what it is doing partway through is the kind of thing only a test
    /// catches, because using it feels like the application being unreliable
    /// rather than like a bug with a shape.
    fn apply_drag_motion(&mut self, delta: (f64, f64)) {
        for slot in 0..self.drag_ops.len() {
            let Some(op) = self.drag_ops[slot].clone() else {
                continue;
            };
            self.input.note_event();
            match op.as_str() {
                "view.orbit" => {
                    self.input.orbit_delta.0 += delta.0;
                    self.input.orbit_delta.1 += delta.1;
                }
                "view.pan" => {
                    self.input.pan_delta.0 += delta.0;
                    self.input.pan_delta.1 += delta.1;
                }
                // Vertical drag zooms, at a gentler rate than the wheel's
                // notches.
                "view.dolly" => self.input.scroll -= delta.1 * 0.02,
                _ => {}
            }
        }
    }

    /// Decides what a mouse button's drag will do, at the moment it is pressed.
    ///
    /// Everything about the gesture is settled here: whether it navigates and
    /// how, or whether it selects. Nothing downstream re-asks, so a modifier
    /// released mid-drag cannot turn an orbit into a pan, and a pointer that
    /// wanders over a panel cannot stop it.
    fn begin_mouse_drag(&mut self, slot: usize, button: KeyMouse) {
        let action = self.drag_action(button);
        if let Some(action) = action {
            // On the press, not on the first motion: by the time the pointer has
            // moved the view has already turned once about the old pivot, and
            // that first frame is exactly the jump auto-depth exists to remove.
            if action.op_id == "view.orbit" {
                self.set_auto_depth_pivot();
            }
            self.drag_ops[slot] = Some(action.op_id);
            return;
        }

        // Unbound, so a left press is a selection gesture — a click or a box
        // depending on whether the pointer moves, which cannot be told apart
        // until it does. Nothing is committed here.
        //
        // Shift extends and Ctrl removes, read from the live modifier state
        // rather than from the keymap: this is one gesture rather than several
        // bindings, and a preset that rebinds selection has to move its variants
        // with it.
        if button == KeyMouse::Left {
            self.begin_drag_select();
        }
    }

    /// Ends whatever that button was doing.
    ///
    /// Runs even when the interface claimed the release — over a panel, a menu,
    /// the navigation gizmo — because a button that went down has to come up. It
    /// not doing so was what made navigation stick on and then appear broken.
    fn end_mouse_drag(&mut self, slot: usize) {
        let was_navigating = self.drag_ops[slot].take().is_some();
        if !was_navigating && slot == 0 {
            self.finish_drag_select();
        }
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
                        // Which object the click landed on, so a second mesh in
                        // edit mode can be selected in at all. Slot plus one,
                        // because slot zero is a real object and the operator
                        // reads zero as "not stated".
                        params.set("object", PropertyValue::Int(hit.object as i64 + 1));
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
        // Before the interface borrows anything: the panel reads this, and it
        // has to be settled while `self` is still free.
        self.publish_uv_view();
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
        // Rebuilt every frame rather than pushed on change: a gesture's numbers
        // move continuously, and a header a frame behind the drag would be
        // worse than no header.
        self.ui.modal = self
            .gesture
            .as_ref()
            .map(|(g, _)| shell::ModalHeader {
                text: g.status(),
                axis: g.axis_index(),
            })
            .or_else(|| {
                // The UV gesture wears the same header, over the same
                // viewport: one place the tool reports itself, whichever panel
                // started it. U and V take the X and Y axis colours, because
                // that is what they are — the map's two directions, in the
                // same order.
                self.uv_gesture.as_ref().map(|(g, _)| shell::ModalHeader {
                    text: g.status(),
                    axis: g.axis().map(|a| match a {
                        kreate::uvmodal::Axis::U => 0,
                        kreate::uvmodal::Axis::V => 1,
                    }),
                })
            });

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
            // Handed the mode rather than the interface keeping a copy of it. A
            // duplicate of a value the application already owns is a value that
            // can disagree with it, and the way that showed was a sweep removing
            // the line that refreshed the copy: nothing failed, and the header
            // would have gone on naming the mode the user had left.
            let mode = self.op_ctx.mode;
            self.egui_ctx.run_ui(raw_input, |ui| {
                viewport_rect = shell::draw(
                    ui, ui_state, doc, registry, history, keymap, preset, overrides, mode, image,
                );
            })
        };
        state.handle_platform_output(&window, output.platform_output.clone());

        // The single input gate. Scattered `if !typing` checks are how a rename
        // field ends up firing tool shortcuts again.
        self.ui.egui_wants_keyboard = self.egui_ctx.egui_wants_keyboard_input();
        self.ui.ime_active = output.platform_output.ime.is_some();

        self.apply_keymap_edits();
        self.save_workspaces();
        self.drain_ui_commands();
        // After the commands, so a save or an open is reflected in the title on
        // the same frame it happened rather than the next one, and so the
        // autosave timer sees this frame's edits.
        self.file.touch(&self.doc);
        self.tick_document();
        self.sync_recoverable();
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
                        .preview(Some(*id) == preview_target)
                        // Exempt from screen-size simplification. In edit mode
                        // the object-level `selected` flag above is deliberately
                        // false, so without this the mesh actually being edited
                        // would be the one thing eligible to have its overlays
                        // faded — precisely backwards.
                        .editing(
                            self.op_ctx.mode == OpMode::EditMesh
                                && self.doc.selection.contains(*id),
                        ),
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
        // The overlay set, asked of the mode. `== EditMesh` said the wrong
        // thing: sculpt and weight-paint modes will also be editing a mesh
        // and will also not want element dots.
        gfx.viewport.show_elements = self.op_ctx.mode.shows_elements();
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
            .prepare(&gfx.gpu.device, &gfx.gpu.queue, &self.camera, &mut draws);
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
        // The UV editor first, and by *where the pointer is* rather than by a
        // focus flag: `G` means move-UVs over that panel and move-geometry
        // over the viewport, which is what the user already believes and needs
        // no third state to remember.
        if self.pointer_in_uv_panel() {
            return KeyContext::UvEditor;
        }
        match self.op_ctx.mode {
            OpMode::EditMesh => KeyContext::EditMesh,
            OpMode::Object => KeyContext::ObjectMode,
        }
    }

    /// Whether the pointer is over the open UV panel.
    fn pointer_in_uv_panel(&self) -> bool {
        self.uv_panel_cursor().is_some()
    }

    /// The pointer in UV units, if it is over the panel.
    ///
    /// `None` covers three different situations that all mean the same thing
    /// here — the panel is closed, it has not drawn yet, or the pointer is
    /// elsewhere — so they are deliberately not distinguished.
    fn uv_panel_cursor(&self) -> Option<glam::Vec2> {
        let (x, y) = self.input.cursor?;
        // Physical pixels to interface points: the pointer is tracked in the
        // window's pixels and the areas laid out in points.
        let scale = f64::from(self.egui_ctx.pixels_per_point().max(0.01));
        let at = egui::pos2((x / scale) as f32, (y / scale) as f32);
        // Whichever UV editor the pointer is over. The layout permits more than
        // one, and each has its own view, so the answer depends on which.
        let (_, view) = self.ui.uv_at(at)?;
        let rect = view.rect?;
        let uv = kreate_ui::uv_editor::to_uv(view, rect, at);
        Some(glam::Vec2::new(uv.x, uv.y))
    }

    /// Starts a UV transform, if there is a selection to transform.
    fn begin_uv_gesture(&mut self, kind: kreate::uvmodal::Kind) {
        use kreate_mesh::edit::uv_edit;

        // Any gesture already running is confirmed rather than abandoned:
        // pressing `S` during a move means "and now scale it", which is what
        // the viewport's transform does too.
        self.confirm_uv_gesture();
        self.confirm_gesture();

        let Some(cursor) = self.uv_panel_cursor() else {
            return;
        };
        // The target's, because a UV editor shows one layout. Which mesh in
        // edit mode that is follows the same rule everything else does: the
        // active object, or the first selected one.
        let session =
            kreate_ops::ops::mesh::target_mesh(&self.doc).and_then(|mesh| self.edit.get(mesh));
        let Some(session) = session else {
            return;
        };
        let bm = session.bmesh();
        let corners = uv_edit::selected(bm);
        if corners.is_empty() {
            self.status("nothing selected in the UV editor");
            return;
        }
        let pivot = uv_edit::median(bm, &corners, kreate_mesh::attribute::names::UV);

        let mut gesture = kreate::uvmodal::UvGesture::begin(kind, cursor, pivot);
        gesture.set_precise(self.input.shift_down);
        let params = mode_params(kind, &gesture);
        match self.history.begin_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            "uv.transform",
            &params,
        ) {
            Ok(Some(session)) => {
                let status = gesture.status();
                self.uv_gesture = Some((gesture, session));
                self.status(&status);
            }
            Ok(None) => self.status("that is not possible here"),
            Err(e) => self.status(&format!("{e}")),
        }
    }

    /// Feeds the pointer to the UV gesture and re-applies the operator.
    fn drive_uv_gesture(&mut self) {
        let Some(cursor) = self.uv_panel_cursor() else {
            return;
        };
        let Some((gesture, session)) = &mut self.uv_gesture else {
            return;
        };
        gesture.track(cursor);
        let kind = gesture.kind();
        let params = mode_params(kind, gesture);
        let status = gesture.status();
        // Restored and re-applied from the original every frame — §2.2's rule.
        // `update_modal` owns that; nothing here is incremental.
        if let Err(e) = self.history.update_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            session,
            &params,
        ) {
            tracing::warn!(error = %e, "uv transform failed");
        }
        self.status(&status);
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Confirms the UV gesture, if one is running.
    fn confirm_uv_gesture(&mut self) {
        let Some((_, session)) = self.uv_gesture.take() else {
            return;
        };
        let call = session.params().to_script_call(session.op_id());
        self.history.commit_modal(session);
        println!("{call}");
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Abandons the UV gesture. The document goes back and nothing is recorded.
    fn cancel_uv_gesture(&mut self) {
        let Some((_, session)) = self.uv_gesture.take() else {
            return;
        };
        kreate_ops::History::cancel_modal(session, &mut self.doc);
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Keys that belong to a running UV gesture.
    ///
    /// Consulted before the keymap, for the same reason the viewport's are:
    /// while a transform runs, `U` means "constrain to U" and `Escape` means
    /// "cancel this", not "unwrap" and not "quit".
    fn uv_gesture_key(&mut self, code: KeyCode) -> bool {
        use kreate::uvmodal::Axis;
        if self.uv_gesture.is_none() {
            return false;
        }
        // `X` and `Y` as well as `U` and `V`: the keys are where the hand
        // already is from the viewport, and the letters are what the panel
        // labels its axes. Both, rather than making the user pick.
        let axis = match code {
            KeyCode::KeyU | KeyCode::KeyX => Some(Axis::U),
            KeyCode::KeyV | KeyCode::KeyY => Some(Axis::V),
            _ => None,
        };
        if let Some(axis) = axis {
            if let Some((gesture, _)) = &mut self.uv_gesture {
                gesture.constrain(axis);
            }
            self.drive_uv_gesture();
            return true;
        }
        match code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                self.confirm_uv_gesture();
                true
            }
            KeyCode::Escape => {
                self.cancel_uv_gesture();
                true
            }
            KeyCode::Backspace => {
                if let Some((gesture, _)) = &mut self.uv_gesture {
                    gesture.backspace();
                }
                self.drive_uv_gesture();
                true
            }
            _ => {
                let typed = typed_character(code);
                match typed {
                    Some(c) => {
                        let taken = self
                            .uv_gesture
                            .as_mut()
                            .is_some_and(|(g, _)| g.type_char(c));
                        if taken {
                            self.drive_uv_gesture();
                        }
                        taken
                    }
                    None => false,
                }
            }
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
        // The modal stack, first. §2.1 puts it at the top of the pipeline and
        // this is why: while a transform runs, `X` means "constrain to X" and
        // `Escape` means "cancel the transform" — not "delete" and not "quit".
        // Checked before the escape hatches below, because a gesture is exactly
        // the state a user most needs `Escape` to get them out of.
        if self.uv_gesture_key(code) {
            return;
        }
        if self.gesture_key(code) {
            return;
        }
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
            // The three transforms start a gesture rather than running an
            // operator. The keymap still names the fixed-step operator ids, so a
            // script or a rebound key can invoke them directly with parameters —
            // the interactive path is a *way of arriving at* those parameters,
            // not a different operation.
            // The UV transform arrives with the mode its binding named,
            // which is how one operator serves three keys.
            "uv.transform" => {
                let mode = action
                    .params
                    .get("mode")
                    .and_then(|v| v.as_int())
                    .unwrap_or(0);
                self.begin_uv_gesture(match mode {
                    1 => kreate::uvmodal::Kind::Rotate,
                    2 => kreate::uvmodal::Kind::Scale,
                    _ => kreate::uvmodal::Kind::Move,
                });
            }
            "object.translate" | "mesh.translate" => self.begin_gesture(kreate::modal::Kind::Move),
            "object.rotate" | "mesh.rotate" => self.begin_gesture(kreate::modal::Kind::Rotate),
            "object.scale" | "mesh.scale" => self.begin_gesture(kreate::modal::Kind::Scale),
            "view.cycle_snap_mode" => {
                self.snap.mode = self.snap.mode.next();
                let label = self.snap.mode.label();
                self.status(&format!("snap to {label}"));
            }
            "view.toggle_snap" => {
                self.snap.enabled = !self.snap.enabled;
                let on = self.snap.enabled;
                self.status(if on { "snapping on" } else { "snapping off" });
            }
            "view.toggle_proportional" => {
                self.proportional = match self.proportional {
                    Some(_) => None,
                    // A radius rather than a flag, so the first press has an
                    // answer for "how far". One metre is a scene-scale default;
                    // the wheel adjusts it during a drag.
                    None => Some(1.0),
                };
                let on = self.proportional.is_some();
                self.status(if on {
                    "proportional editing on"
                } else {
                    "proportional editing off"
                });
            }
            "file.new" => self.new_document(),
            "file.open" => self.open_document(),
            "file.save" => self.save_document(false),
            "file.save_as" => self.save_document(true),
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
            "view.toggle_sidebar" => {
                self.ui.toggle_sidebar();
                self.redraw.set(RedrawReason::UI);
            }
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

/// Writes an emergency save if the process is unwinding from a panic.
///
/// Here rather than in the panic hook, and that is not a detail. The hook runs
/// from a static closure that cannot be handed the document, and reaching it
/// through a global lock would risk the hook deadlocking against a panic that
/// happened while that lock was held — the one case a crash reporter has to
/// survive. `Drop` runs during the same unwind with `&mut self` in hand and the
/// document intact, so the save is both possible and safe.
///
/// It goes beside the autosave, never over the user's file: a process that has
/// just panicked may hold exactly the state that caused the panic.
impl Drop for App {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        if !self.file.is_modified(&self.doc) {
            // Nothing to rescue, and writing anyway would leave a recovery offer
            // for work that is already on disk.
            return;
        }
        match self.file.emergency_save(&self.doc) {
            Some(path) => eprintln!("unsaved work written to {}", path.display()),
            None => eprintln!("unsaved work could not be written"),
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
        // After the fonts, because the theme is built on whatever style the
        // context already carries and installing it the other way round would
        // be a silent ordering trap for the next person.
        kreate_ui::install_theme(&self.egui_ctx);
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
        let mut viewport = Viewport::new(
            &gpu.device,
            ViewportTarget::FORMAT,
            target.size.0,
            target.size.1,
        );
        // Pay for shader compilation now, while the window is still coming up,
        // rather than on the frame where the user first presses Tab or moves the
        // mouse over the scene. Creating a pipeline and compiling it are not the
        // same thing: the driver defers the second until a draw arrives, and the
        // overlay and picking pipelines are not touched by an ordinary first
        // frame. Measured cold, that first touch costs around ten milliseconds —
        // a dropped frame, landing exactly when the user starts interacting.
        viewport.warm_up(&gpu.device, &gpu.queue);

        let pick_target = kreate_render::PickTarget::new(&gpu.device, target.size.0, target.size.1);
        let region = kreate_render::RegionSelect::new(&gpu.device);
        // Built with the rest of the pipelines rather than when the first drag
        // starts, for the same reason `warm_up` exists above: a driver compiles
        // a compute pipeline on first dispatch, and paying for three of them on
        // the frame the user presses `G` puts the cost exactly where it is most
        // visible.
        let deformer = kreate_render::Deformer::new(&gpu.device);
        self.gfx = Some(Gfx {
            surface,
            config,
            gpu,
            viewport,
            meshes: Vec::new(),
            pick_target,
            region,
            deformer,
            preview: None,
            egui_renderer,
            target,
        });
        self.window = Some(window);
        self.frame_all();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // The interface sees every event, and says whether it needs a repaint.
        //
        // Its `consumed` flag is deliberately ignored. It reports
        // `is_pointer_over_egui() && !any_button_down`, and the viewport lives
        // inside a central panel — so "over egui" is true across the whole scene
        // and *every* press and every wheel notch over the model came back
        // consumed while every release did not. Acting on it dropped exactly the
        // input the viewport exists to receive.
        //
        // Pointer ownership is decided by geometry instead: the viewport takes
        // what lands in it, the navigation gizmo takes its own circle, and the
        // panels keep everything outside. Keyboard input is a different question
        // with a different answer — see `ShellState::keymap_enabled`, which asks
        // whether a text field has focus.
        if let (Some(state), Some(window)) = (self.egui_state.as_mut(), self.window.as_ref()) {
            let response = state.on_window_event(window, &event);
            if response.repaint {
                self.redraw.set(RedrawReason::UI);
            }
        }

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

            // A button that went down has to come up, wherever it comes up.
            // Guarding this arm on the interface's `consumed` flag was a real
            // bug: a drag released over a panel had its release eaten, the button
            // stayed held, and from then on the view turned whenever the pointer
            // moved with nothing pressed.
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let slot = match button {
                    MouseButton::Left => Some(0),
                    MouseButton::Middle => Some(1),
                    MouseButton::Right => Some(2),
                    _ => None,
                };
                // A running transform owns the buttons: left confirms, right
                // cancels, and neither may also start a selection or an orbit.
                // Handled before the drag machinery for the same reason the
                // modal stack sits above the keymap.
                if self.uv_gesture.is_some() && pressed {
                    match button {
                        MouseButton::Left => {
                            self.confirm_uv_gesture();
                            return;
                        }
                        MouseButton::Right => {
                            self.cancel_uv_gesture();
                            return;
                        }
                        // The middle button pans the UV view, which is allowed
                        // during a transform for the same reason orbiting is:
                        // looking at what you are doing is not a conflict.
                        _ => {}
                    }
                }
                if self.gesture.is_some() && pressed {
                    match button {
                        MouseButton::Left => {
                            self.confirm_gesture();
                            return;
                        }
                        MouseButton::Right => {
                            self.cancel_gesture();
                            return;
                        }
                        // The middle button is navigation, which is allowed to
                        // run during a transform: orbiting to see what you are
                        // doing is not a conflict.
                        _ => {}
                    }
                }
                // A plain right-click on the viewport opens the menu. Only a
                // plain one: `Alt+RMB` is a navigation binding in the Maya
                // preset, and taking every right-click would break it.
                if button == MouseButton::Right
                    && pressed
                    && !self.input.alt_down
                    && !self.input.ctrl_down
                    && !self.input.shift_down
                    && self.gesture.is_none()
                    && self.press_belongs_to_viewport()
                {
                    self.open_context_menu();
                    return;
                }
                if let Some(slot) = slot {
                    let mapped = [KeyMouse::Left, KeyMouse::Middle, KeyMouse::Right][slot];
                    if pressed && !self.buttons[slot] {
                        // Ownership is decided by where the press landed, not by
                        // asking the interface whether it wants it. Asked once,
                        // at the press, so the rest of the gesture never has to
                        // ask again and cannot change its mind halfway through.
                        if self.press_belongs_to_viewport() {
                            self.begin_mouse_drag(slot, mapped);
                        }
                    } else if !pressed && self.buttons[slot] {
                        self.end_mouse_drag(slot);
                    }
                    self.buttons[slot] = pressed;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.track_drag_select();
                let p = (position.x, position.y);
                if let Some(prev) = self.input.cursor {
                    self.apply_drag_motion((p.0 - prev.0, p.1 - prev.1));
                }
                self.input.cursor = Some(p);
                // After the cursor is recorded, because the gesture reads it
                // from there rather than from the event — one source of truth
                // for where the pointer is.
                if self.gesture.is_some() {
                    self.drive_gesture();
                }
                if self.uv_gesture.is_some() {
                    self.drive_uv_gesture();
                }
            }

            // Owned by geometry, like a press. `consumed` is worse than useless
            // here: with no button held it is *always* true over the viewport,
            // so the wheel was being dropped exactly where zooming is wanted.
            // Over a panel the cursor is outside the viewport and the interface
            // keeps its own scrolling.
            WindowEvent::MouseWheel { delta, .. } if self.cursor_in_viewport() => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x as f64, y as f64),
                    // A precision trackpad reports pixels. Scaled to notches so
                    // one unit means the same thing whichever device sent it.
                    MouseScrollDelta::PixelDelta(p) => (p.x / 60.0, p.y / 60.0),
                };
                // A running gesture with proportional editing on takes the
                // wheel for its radius. Zooming mid-transform would move the
                // camera the selection is being dragged relative to, which is
                // not what a hand reaching for the wheel means here.
                if !self.adjust_proportional(y as f32) {
                    self.apply_wheel(x, y);
                }
            }

            // Pinch to zoom, which winit reports only on macOS and iOS. On
            // Windows and Linux a trackpad pinch arrives as a wheel event
            // instead, which the arm above already handles — there is no
            // gesture to miss, only a different name for it.
            WindowEvent::PinchGesture { delta, .. } if self.cursor_in_viewport() => {
                self.apply_wheel(0.0, delta * PINCH_NOTCHES);
            }

            // The last way a held button can be left held: the release goes
            // somewhere else. Alt+Tab mid-drag, a notification stealing focus, a
            // screen locking — the window never sees the button come up, and
            // without this the view would turn on every pointer movement until
            // the user happened to press and release again.
            //
            // The same class of bug as a release the interface claimed, and worth
            // fixing in the same place: whatever the reason, a gesture whose end
            // was not observed has ended.
            WindowEvent::Focused(false) => self.cancel_all_drags(),

            WindowEvent::ModifiersChanged(m) => {
                self.input.shift_down = m.state().shift_key();
                self.input.ctrl_down = m.state().control_key();
                self.input.alt_down = m.state().alt_key();
                // Shift is precision while a transform runs. Handled from the
                // modifier event rather than from a key press, because it is a
                // *hold*: it has to engage and release, and a press-only path
                // would leave a gesture stuck in precision after the key came
                // back up.
                self.set_gesture_precision(self.input.shift_down);
                // Ctrl inverts snapping rather than enabling it: someone who has
                // snapping on wants Ctrl to let them off it, and someone who has
                // it off wants Ctrl to turn it on. One modifier, always meaning
                // "the other way".
                if let Some((gesture, _)) = &mut self.gesture {
                    gesture.set_snap_inverted(self.input.ctrl_down);
                }
                if self.gesture.is_some() {
                    self.drive_gesture();
                }
                // Precision, on the same modifier and measured from where it
                // was pressed. There is nothing to snap to in UV space, so
                // Ctrl means nothing here — and pretending otherwise would be
                // worse than leaving it alone.
                if let Some((gesture, _)) = &mut self.uv_gesture {
                    gesture.set_precise(self.input.shift_down);
                }
                if self.uv_gesture.is_some() {
                    self.drive_uv_gesture();
                }
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
/// Loads the document named on the command line, or a starter scene.
///
/// Returns the file state alongside it, so a `.kreate` opened from the command
/// line knows where it came from and Ctrl+S writes back to it. Without that, the
/// most natural way to open a document — dragging it onto the executable — would
/// produce a document that cannot be saved without a dialog.
fn load_document_and_file(path: Option<&str>) -> (Document, DocumentFile) {
    if let Some(path) = path {
        let path = std::path::Path::new(path);
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("kreate"))
        {
            match DocumentFile::open(path) {
                Ok(pair) => return pair,
                // Reported and then carried on with an empty document, rather
                // than refusing to start. An application that will not open
                // because of the file it was asked to open leaves the user with
                // no way to open a different one.
                Err(e) => {
                    tracing::error!("cannot open {}: {e}", path.display());
                    return (load_document(None), DocumentFile::new());
                }
            }
        }
    }
    (load_document(path), DocumentFile::new())
}

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

    // Before anything else, and before the window: this process may be the dump
    // writer for another one, in which case it must not open a window, load a
    // document, or claim a session.
    if let Some(socket) = kreate::crashreport::reporter_socket(&args) {
        kreate::crashreport::run_reporter(&socket);
    }

    install_panic_hook();
    // Held for the life of `main`. Dropping it detaches the handler and lets the
    // reporter exit, which is what should happen on a clean shutdown.
    let _crash_reporter = kreate::crashreport::install();

    // After the reporter is attached, so the crash it causes is one the reporter
    // can see. Debug builds only.
    #[cfg(debug_assertions)]
    if args.iter().any(|a| a == "--crash-now") {
        kreate::crashreport::crash_on_purpose();
    }

    println!("{HELP}\n");
    let (doc, file) = load_document_and_file(path.as_deref());
    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App::new(doc, &locale);
    app.file = file;
    event_loop.run_app(&mut app).expect("run");
}

/// Writes a crash report before the process unwinds.
///
/// Tier one of the two §3.4 describes. This catches a Rust panic, which is the
/// common case and the one where the process is still healthy enough to write a
/// file. The other tier — a GPU fault, a stack overflow, an abort — needs a
/// separate process to survive it, because this hook does not run at all for
/// those.
///
/// The report exists because the useful facts about a crash are the ones nobody
/// thinks to ask for. A backtrace alone rarely identifies a mesh bug; the
/// sequence of operators that led to it usually does, which is why the ring
/// buffer of recent operator ids is the most valuable field in the file.
fn install_panic_hook() {
    let Some(dir) = docfile::autosave_root() else {
        return;
    };
    // Chained rather than replaced: the default hook is what prints the panic to
    // the terminal, and a developer running from a shell should still see it.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Written first and separately from anything that could itself panic, so
        // a failure while assembling the report does not lose the report.
        let report = dir.join("crash.txt");
        let _ = std::fs::create_dir_all(&dir);
        let mut text = String::new();
        text.push_str(&format!("kreate {}\n", env!("CARGO_PKG_VERSION")));
        text.push_str(&format!("panic: {info}\n"));
        text.push_str(&format!(
            "backtrace:\n{}\n",
            std::backtrace::Backtrace::force_capture()
        ));
        text.push_str("recent operators (oldest first):\n");
        for id in kreate::recent_operators() {
            text.push_str(&format!("  {id}\n"));
        }
        let _ = std::fs::write(&report, text);
        eprintln!("crash report written to {}", report.display());
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An application with no window, which is all the input state machine needs.
    fn app() -> App {
        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        let mut app = App::empty(Document::new(), "en-US", registry);
        app.keymap = presets::load(Preset::Kreate);
        app
    }

    /// The middle button, which the default preset orbits with.
    const ORBIT: usize = 1;

    #[test]
    fn a_middle_drag_navigates() {
        let mut app = app();
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        assert_eq!(
            app.drag_ops[ORBIT].as_deref(),
            Some("view.orbit"),
            "the default preset orbits on the middle button"
        );
    }

    #[test]
    fn a_release_ends_the_drag() {
        let mut app = app();
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        app.end_mouse_drag(ORBIT);
        assert!(app.drag_ops[ORBIT].is_none());
    }

    #[test]
    fn a_release_the_interface_claimed_still_ends_the_drag() {
        // The bug this exists for, reported as "navigation works, then does not".
        //
        // A drag begun over the viewport and released over a panel had its
        // release eaten by the interface, so the button stayed held. From then on
        // the view turned whenever the pointer moved with nothing pressed, until
        // some later press and release happened to clear it.
        //
        // The release path now runs whether or not the interface claimed the
        // event: a button that went down has to come up.
        let mut app = app();
        app.buttons[ORBIT] = true;
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        assert!(app.drag_ops[ORBIT].is_some());

        // What the event handler does for a release, consumed or not.
        app.end_mouse_drag(ORBIT);
        app.buttons[ORBIT] = false;

        assert!(
            app.drag_ops[ORBIT].is_none(),
            "the drag outlived the button that started it"
        );
    }

    #[test]
    fn a_drag_keeps_doing_what_it_started_doing() {
        // Releasing a modifier partway through used to turn a pan into an orbit,
        // because the binding was resolved on every motion rather than once.
        //
        // Checked through the motion path rather than by reading the latch: the
        // latch not changing proves nothing unless what consumes it agrees.
        let mut app = app();
        app.input.shift_down = true;
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        assert_eq!(app.drag_ops[ORBIT].as_deref(), Some("view.pan"));

        app.apply_drag_motion((10.0, 0.0));
        app.input.shift_down = false;
        app.apply_drag_motion((10.0, 0.0));

        assert_eq!(app.input.pan_delta.0, 20.0, "both movements should pan");
        assert_eq!(
            app.input.orbit_delta,
            (0.0, 0.0),
            "the gesture changed underneath the user"
        );
    }

    #[test]
    fn losing_focus_ends_every_drag() {
        // Alt+Tab mid-drag: the release goes to whatever took focus, and the
        // window never sees the button come up. The same class of bug as a
        // release the interface claimed.
        let mut app = app();
        app.input.alt_down = true;
        app.begin_mouse_drag(0, KeyMouse::Left);
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        app.buttons = [true, true, false];

        app.cancel_all_drags();

        app.apply_drag_motion((30.0, 30.0));
        assert_eq!(app.input.orbit_delta, (0.0, 0.0));
        assert_eq!(app.input.pan_delta, (0.0, 0.0));
        assert_eq!(app.buttons, [false; 3]);
        assert!(
            !app.input.alt_down,
            "a modifier held when focus was lost never gets its key-up"
        );
    }

    #[test]
    fn losing_focus_abandons_a_selection_drag_rather_than_committing_it() {
        // A rubber band whose release was never seen has no end point worth
        // trusting. Committing a selection the user did not finish is worse than
        // losing one they did.
        let mut app = app();
        app.begin_mouse_drag(0, KeyMouse::Left);
        app.drag_select = Some(DragSelect {
            origin: (10.0, 10.0),
            current: (90.0, 90.0),
            moved: true,
            extend: false,
            deselect: false,
        });

        app.cancel_all_drags();
        assert!(app.drag_select.is_none());
    }

    /// Puts a viewport on screen and the cursor in the middle of it.
    ///
    /// `viewport_px` is what the application knows about where the scene is
    /// drawn; the render size comes from the graphics state, which a headless
    /// test has none of, so ownership is tested through the parts that do not
    /// need one.
    fn with_viewport(app: &mut App) {
        app.viewport_px = Some((100.0, 50.0, 800.0, 600.0));
        app.input.cursor = Some((500.0, 350.0));
    }

    #[test]
    fn a_press_outside_the_viewport_belongs_to_the_interface() {
        // Over a panel: the interface keeps it, and the view must not move.
        let mut app = app();
        with_viewport(&mut app);
        app.input.cursor = Some((10.0, 10.0));
        assert!(!app.press_belongs_to_viewport());
    }

    #[test]
    fn a_press_with_no_viewport_belongs_to_nobody() {
        // Before the first frame there is no scene to click into.
        let mut app = app();
        app.input.cursor = Some((500.0, 350.0));
        assert!(!app.press_belongs_to_viewport());
    }

    #[test]
    fn a_press_over_the_navigation_gizmo_is_left_to_it() {
        // One drag must not both turn the gizmo and turn the scene. Excluded by
        // the gizmo's own circle rather than by a hover flag, because a flag is
        // only recomputed when the interface runs and a press can arrive between
        // frames.
        let mut app = app();
        with_viewport(&mut app);
        app.ui.gizmo_circle = Some((egui::pos2(500.0, 350.0), 40.0));
        assert!(
            !app.press_belongs_to_viewport(),
            "the press landed inside the gizmo"
        );

        app.input.cursor = Some((500.0, 600.0));
        assert!(
            app.press_belongs_to_viewport(),
            "well clear of the gizmo, so the scene takes it"
        );
    }

    #[test]
    fn the_wheel_zooms() {
        let mut app = app();
        app.apply_wheel(0.0, 3.0);
        assert_eq!(app.input.scroll, 3.0);
        assert_eq!(app.input.pan_delta, (0.0, 0.0));
    }

    #[test]
    fn horizontal_scroll_pans_rather_than_being_discarded() {
        // A sideways two-finger swipe on a trackpad, and a tilt wheel, used to do
        // nothing: only the vertical component was read.
        let mut app = app();
        app.apply_wheel(2.0, 0.0);
        assert!(
            app.input.pan_delta.0 != 0.0,
            "the sideways swipe was dropped"
        );
        assert_eq!(app.input.scroll, 0.0, "it must not also zoom");
    }

    #[test]
    fn shift_and_the_wheel_scroll_sideways() {
        // The convention nearly everywhere, and the only horizontal pan available
        // on a mouse with no tilt wheel and no trackpad.
        let mut app = app();
        app.input.shift_down = true;
        app.apply_wheel(0.0, 2.0);
        assert!(app.input.pan_delta.0 != 0.0);
        assert_eq!(app.input.scroll, 0.0);
    }

    #[test]
    fn an_unbound_modifier_still_zooms() {
        // A trackpad pinch arrives as Ctrl+wheel on Windows, and a wheel that is
        // inert for no visible reason reads as the application being broken.
        let mut app = app();
        app.input.ctrl_down = true;
        app.apply_wheel(0.0, 2.0);
        assert_eq!(app.input.scroll, 2.0);
    }

    #[test]
    fn an_empty_scroll_does_nothing_at_all() {
        // Trackpads emit these at the end of a gesture. Waking the frame loop for
        // one is how an idle window stops being idle.
        let mut app = app();
        app.apply_wheel(0.0, 0.0);
        assert!(!app.input.has_pending());
        assert!(app.input.first_event_at.is_none());
    }

    #[test]
    fn motion_without_a_button_moves_nothing() {
        // The symptom of the release bug, stated directly: with no drag in
        // progress, moving the pointer must not turn the view.
        let mut app = app();
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        app.end_mouse_drag(ORBIT);
        app.apply_drag_motion((25.0, 25.0));

        assert_eq!(app.input.orbit_delta, (0.0, 0.0));
        assert_eq!(app.input.pan_delta, (0.0, 0.0));
        assert_eq!(app.input.scroll, 0.0);
    }

    #[test]
    fn a_middle_drag_actually_turns_the_view() {
        // The whole chain, from press to accumulated delta.
        let mut app = app();
        app.begin_mouse_drag(ORBIT, KeyMouse::Middle);
        app.apply_drag_motion((12.0, -4.0));
        assert_eq!(app.input.orbit_delta, (12.0, -4.0));
    }

    #[test]
    fn an_unbound_left_press_starts_a_selection_rather_than_navigation() {
        let mut app = app();
        app.begin_mouse_drag(0, KeyMouse::Left);
        assert!(
            app.drag_ops[0].is_none(),
            "a plain left drag is the selection rubber band"
        );
    }

    #[test]
    fn a_bound_left_press_navigates_instead_of_selecting() {
        // Alt+LMB orbits in every preset, so that navigation works on a mouse
        // without a usable middle button. It must not also begin a selection.
        let mut app = app();
        app.input.alt_down = true;
        app.begin_mouse_drag(0, KeyMouse::Left);
        assert_eq!(app.drag_ops[0].as_deref(), Some("view.orbit"));
        assert!(
            app.drag_select.is_none(),
            "one press must not start two gestures"
        );
    }

    #[test]
    fn every_button_tracks_its_own_drag() {
        // Holding one button while pressing another is ordinary, and the two must
        // not share state.
        let mut app = app();
        app.input.alt_down = true;
        app.begin_mouse_drag(0, KeyMouse::Left);
        app.begin_mouse_drag(2, KeyMouse::Right);
        assert_eq!(app.drag_ops[0].as_deref(), Some("view.orbit"));
        assert_eq!(app.drag_ops[2].as_deref(), Some("view.dolly"));

        app.end_mouse_drag(0);
        assert!(app.drag_ops[0].is_none());
        assert_eq!(
            app.drag_ops[2].as_deref(),
            Some("view.dolly"),
            "releasing one button ended another's drag"
        );
    }
}
