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

// A window application: no console window opens beside it on Windows. What
// it prints still reaches a terminal or a pipe it is started from with its
// output redirected; run from Explorer, there is nowhere for it to go, and
// the crash report and autosave are files, not console output.
#![cfg_attr(windows, windows_subsystem = "windows")]

// The shell lives in this package's library target so tests can drive it.
use kreate::docfile::{self, DocumentFile};
use kreate::{dragpreview, gpu_sync, scenelights, shell};

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
use kreate_scene::document::ObjectId;
use kreate_scene::Document;
use kreate_ui::Command;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

mod collab;
mod connector;
mod panels;
mod render_jobs;
mod walking;

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
               B              snap to the vertex, edge or face under the pointer
               type a value   2*3.5, 1.4/8, 50cm, 30deg; Backspace to correct
               wheel          proportional falloff radius, while it is on
  Settings   ` snapping on/off (Ctrl inverts it for one drag)
             Shift+` what it snaps to: increment, vertex/edge/face, vertex, edge, midpoint, face
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
  Quit       Ctrl+Q or close the window  (F1 for this list; Esc only cancels)

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
    /// The animation is playing, and the next frame is a new one.
    const PLAYING: u32 = 1 << 6;
    /// A still view refining itself: the next frame adds a sample to the
    /// mean. Stops by itself when the mean is full.
    const ACCUMULATE: u32 = 1 << 7;

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
    /// The platform's command key: Cmd on macOS, Ctrl elsewhere. What copy
    /// and paste are held with, which on a Mac is not Ctrl.
    command_down: bool,
    alt_down: bool,
    /// How hard a pen is pressing, 0 to 1, while one is down; `None` for a
    /// mouse.
    pen_pressure: Option<f32>,
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
    /// Kept for the connector's capture, which copies the picture out.
    texture: wgpu::Texture,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[Self::SAMPLED_AS],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampled = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(Self::SAMPLED_AS),
            ..Default::default()
        });
        let id = renderer.register_native_texture(device, &sampled, wgpu::FilterMode::Linear);
        Self {
            view,
            texture,
            id,
            size,
        }
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
/// A weight-paint stroke in progress: its modal session, what it has
/// painted and how much, and the surface's vertex normals in the world — for
/// Front Faces Only, worked out once because a stroke does not move the mesh.
struct PaintStroke {
    session: kreate_ops::ModalSession,
    touched: std::collections::BTreeMap<u32, f32>,
    normals: Vec<DVec3>,
}

/// A sculpt stroke in progress: the dabs laid so far in the mesh's space, the
/// brush it started as, and — once anything has changed — its modal session.
struct SculptStroke {
    object: kreate_scene::document::ObjectId,
    brush: u32,
    invert: bool,
    dabs: Vec<f64>,
    /// The pen's pressure at each dab.
    pressures: Vec<f64>,
    /// The face set a face set stroke paints: one past the highest there is.
    face_set: i32,
    /// With dynamic topology, the surface as the stroke found it: what the
    /// brush finds its way onto. The live mesh changes its triangles every
    /// frame, and a tree rebuilt for each of them costs more than the
    /// stroke; the surface the brush meets moves only as far as it has
    /// pushed it.
    pick_scene: Option<Vec<(kreate_scene::document::ObjectId, Arc<MeshData>)>>,
    /// Where the last dab went, in the world, to space the next one.
    last: DVec3,
    radius_world: f64,
    /// For a grab: where it took hold, and the plane facing the view through it.
    hold: DVec3,
    /// What the operator last ran with — the dab count and, for a grab, the
    /// drag — so a frame with nothing new does not run it again.
    sent: Option<(usize, DVec3)>,
    session: Option<kreate_ops::ModalSession>,
}

/// Which place of a curve point is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CurvePart {
    Point,
    Left,
    Right,
}

/// A curve point or handle being dragged across the view.
struct CurveDrag {
    object: ObjectId,
    spline: u32,
    index: u32,
    part: CurvePart,
    /// The point as it was taken.
    point: kreate_scene::curve::CurvePoint,
    /// The plane the drag moves in: facing the view, through what was
    /// grabbed, in the world.
    plane: DVec3,
    /// Where on that plane the pointer began.
    from: DVec3,
    session: Option<kreate_ops::ModalSession>,
}

/// A physics bake on its own thread.
struct BakeJob {
    token: u32,
    /// Which simulations it bakes: `physics.bake`'s `only`.
    only: u32,
    /// Frames done in the high half, of how many in the low.
    progress: Arc<std::sync::atomic::AtomicU64>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// The result once it is done; `Some(None)` when it was stopped.
    result: Arc<std::sync::Mutex<Option<Option<BakeResult>>>>,
}

type BakeResult = (
    kreate_scene::physics::PhysicsCache,
    kreate_scene::physics::bake::BakeSummary,
);

/// A texture paint stroke in progress: the dabs laid so far in the image's
/// UV square, and — once anything has changed — its modal session.
struct TextureStroke {
    object: kreate_scene::document::ObjectId,
    /// The image painted, by identity, as the operator names it.
    image: String,
    dabs: Vec<f64>,
    /// Where the last dab went, in the world, to space the next one.
    last: DVec3,
    radius_world: f64,
    session: Option<kreate_ops::ModalSession>,
}

/// Blender's circle select: a brush that selects what it passes over.
struct CircleSelect {
    /// Its radius in the viewport's own pixels.
    radius: f32,
    /// The stroke in progress, while the button is down.
    stroke: Option<CircleStroke>,
}

/// One stroke of the brush: everything it has passed over so far, and — in
/// edit mode — the operator running live, so the stroke is one undo step.
struct CircleStroke {
    deselect: bool,
    elements: std::collections::BTreeSet<u32>,
    session: Option<kreate_ops::ModalSession>,
    /// The object selection it started from, for object mode.
    before: Vec<u32>,
}

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
    /// The lasso's path, when the tool is Lasso; a box otherwise.
    lasso: Option<Vec<(f32, f32)>>,
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
        match &self.lasso {
            Some(path) if path.len() >= 3 => kreate_render::SelectShape::lasso(path),
            _ => kreate_render::SelectShape::Box {
                from: self.origin,
                to: self.current,
            },
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
    /// Objects skinned on the GPU: each keeps a mesh uploaded once from its
    /// rest shape, and a pose is the bones' matrices. See `kreate_render::skin`.
    skinned: Vec<SkinnedGpu>,
    /// The id buffer and the ring of readback buffers behind it.
    pick_target: kreate_render::PickTarget,
    /// The compute reduction behind box, circle and lasso selection.
    region: kreate_render::RegionSelect,
    /// The three compute pipelines a drag runs. Built once; a `Deform` borrows
    /// them per drag.
    deformer: kreate_render::Deformer,
    /// The drag being previewed on the GPU, if there is one.
    preview: Option<LivePreview>,
    /// The mesh a sculpt stroke is reshaping in place. See `SculptView`.
    sculpt_view: Option<SculptView>,

    egui_renderer: egui_wgpu::Renderer,
    target: ViewportTarget,
    /// The 3D viewports after the first, by their area: each a renderer, a
    /// picture and a camera of its own.
    extra: std::collections::BTreeMap<kreate_ui::AreaId, ExtraView>,
}

/// A viewport after the first. It draws the same scene through its own
/// camera; selecting and editing happen in the first one.
struct ExtraView {
    viewport: Viewport,
    target: ViewportTarget,
    camera: kreate_render::Camera,
    /// Where it is on the window, in physical pixels: x, y, width, height.
    rect_px: (f32, f32, f32, f32),
    /// The picture's size wanted, in pixels.
    size: (u32, u32),
    /// Made since the scene's sky, materials and traced scene were last
    /// handed over: they are handed to it whole on the next lit frame.
    fresh: bool,
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

/// The UV at `p` on `face` of `mesh` (both in the mesh's own space), and how
/// many UV units one unit of the mesh's length is there — what turns a
/// brush's size on the surface into its size on the texture.
///
/// The face's fan of triangles, and the one `p` falls in — or, for a point
/// just off every one of them, the nearest. `None` for a mesh with no UVs.
fn uv_on_face(mesh: &kreate_mesh::MeshData, face: u32, p: glam::Vec3) -> Option<(glam::Vec2, f64)> {
    let uvs = mesh
        .attributes()
        .get_f32x2(kreate_mesh::attribute::names::UV)?;
    if face >= mesh.num_faces() {
        return None;
    }
    let corners: Vec<usize> = mesh.face_corner_range(face).collect();
    let positions = mesh.positions();
    let verts = mesh.corner_verts();
    let mut best: Option<(f32, glam::Vec2, f64)> = None;
    for i in 1..corners.len().saturating_sub(1) {
        let (c0, c1, c2) = (corners[0], corners[i], corners[i + 1]);
        let (a, b, c) = (
            positions[verts[c0] as usize],
            positions[verts[c1] as usize],
            positions[verts[c2] as usize],
        );
        let (v0, v1, v2) = (b - a, c - a, p - a);
        let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
        let (d20, d21) = (v2.dot(v0), v2.dot(v1));
        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1e-20 {
            continue;
        }
        let wb = (d11 * d20 - d01 * d21) / denom;
        let wc = (d00 * d21 - d01 * d20) / denom;
        let wa = 1.0 - wb - wc;
        let off = (-wa).max(-wb).max(-wc).max(0.0);
        let (ua, ub, uc) = (uvs[c0], uvs[c1], uvs[c2]);
        let uv = ua * wa + ub * wb + uc * wc;
        let world_area = v0.cross(v1).length() as f64 * 0.5;
        let uv_area = ((ub - ua).perp_dot(uc - ua) * 0.5).abs() as f64;
        let scale = if world_area > 1e-20 {
            (uv_area / world_area).sqrt()
        } else {
            0.0
        };
        if best.as_ref().is_none_or(|(o, _, _)| off < *o) {
            best = Some((off, uv, scale));
        }
    }
    best.map(|(_, uv, scale)| (uv, scale))
}

/// Where each edited object's elements start in a region select's bitset,
/// by object key, and each one's (key, base, count). See `App::edit_bases`.
type EditBases = (Vec<u32>, Vec<(u32, u32, u32)>);

/// One object skinned on the GPU. See `Gfx::skinned`.
/// A mesh being sculpted, drawn by writing its new positions into the GPU
/// mesh it was uploaded as and recomputing the normals there — not by
/// extracting it again every frame of the stroke, which at a hundred thousand
/// vertices is most of the frame. See `kreate_render::Reshape`.
struct SculptView {
    /// The mesh the GPU buffers now show.
    held: Arc<MeshData>,
    reshape: kreate_render::Reshape,
}

/// The Measure tool: the points clicked so far, where the next would go,
/// and everything it can catch on — all of the scene, gathered once.
struct MeasureTool {
    world: kreate::snap::SnapWorld,
    points: Vec<DVec3>,
    hover: Option<DVec3>,
    /// Ctrl was held on the second click: the second point is the corner of
    /// an angle, and a third is wanted.
    angle: bool,
}

struct SkinnedGpu {
    object: ObjectId,
    /// The mesh as it was before the bones: what `mesh` was uploaded from.
    rest: Arc<MeshData>,
    /// The CPU's skinned mesh this frame's pose matches — the handle the
    /// draw loop finds this by, and the test for whether the pose moved.
    shown: Arc<MeshData>,
    mesh: GpuMesh,
    skin: kreate_render::Skin,
}

struct App {
    /// The connector's socket. See `kreate::bridge`.
    bridge: Option<kreate::bridge::Bridge>,
    /// The release the update check downloaded, once it has; said in the
    /// status bar, then forgotten. See `kreate::update`.
    update_ready: Option<Arc<std::sync::Mutex<Option<String>>>>,
    /// The Claude panel's conversation. See `kreate::claude`.
    chat: Option<kreate::claude::Chat>,
    /// Whether the last chat entry is a reply still streaming in.
    chat_streaming: bool,
    /// Each answer's changes, by the history it began and ended at — what
    /// the chat panel's "undo this answer" takes back.
    chat_turns: Vec<ChatTurn>,
    /// Where and how big an asset on its way is to land, and the connector
    /// waiting to hear it has — by the asset's id while it downloads, then
    /// by its file while it is read.
    asset_wishes: std::collections::HashMap<String, ImportWish>,
    import_wishes: std::collections::HashMap<std::path::PathBuf, ImportWish>,
    /// What goes with the next chat message: files picked or pasted. The
    /// panel shows `ui.chat_attachments`, made from this.
    chat_attachments: Vec<kreate::attach::Attachment>,
    /// Help ▸ Contact's messages on their way. See `kreate::contact`.
    outbox: Option<kreate::contact::Outbox>,
    /// Scene files read on threads of their own, and what wakes the loop
    /// when one is ready. See `import_into`.
    imports: (
        std::sync::mpsc::Sender<ImportReady>,
        std::sync::mpsc::Receiver<ImportReady>,
    ),
    import_wake: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
    /// The reads under way, for the progress card and its Cancel.
    importing: Vec<Importing>,
    next_import: u64,
    /// The asset browser's lists, pictures and downloads.
    assets: Option<kreate::assets::Library>,
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
    /// The materials the viewport is holding, so a frame can tell a colour
    /// change from a material being added.
    ///
    /// Here rather than in the viewport because it is a fact about the
    /// *document*, and `kreate-render` does not see one — the same reason the
    /// environment's key lives here.
    materials: Option<kreate::materials::Materials>,
    /// What the path tracer's scene was last built from; a different scene
    /// is built again.
    path_trace_key: Option<u64>,
    /// Which environment the viewport is holding, so a frame does not rebuild
    /// six faces and their mips to find out that nothing changed.
    ///
    /// Here rather than in the viewport because it is a fact about the
    /// *document*, and `kreate-render` does not see one.
    environment: Option<kreate::environment::EnvironmentKey>,
    /// Whether the viewports hold the studio's room as their environment.
    studio_environment: bool,
    /// Whether the scene's own sky must be rebuilt, the studio having held
    /// the environment in the meantime.
    environment_stale: bool,
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
    /// When the scene is.
    ///
    /// Nothing reads it yet — no modifier depends on the clock and there is no
    /// timeline to move it — but it is the input every evaluation now takes,
    /// and threading that through afterwards is the change §5 calls fatal.
    ///
    /// Held here rather than in the document, for now. The frame a file opens
    /// at is a scene setting and belongs in the document eventually; what it is
    /// *not* is part of the undo snapshot, because scrubbing is not an edit and
    /// undoing a move must not also move the playhead. Sorting out which half
    /// goes where is Phase 10's, along with the timeline that would make it
    /// visible.
    time: kreate_math::SceneTime,
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
    /// How far each button has dragged since it went down, in pixels: an
    /// Alt+click that did not move is a loop select, not an orbit.
    drag_travel: [f64; 3],
    /// The extra viewport a navigation in progress turns, if it is one.
    nav_view: Option<kreate_ui::AreaId>,
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
    /// A still is being rendered: the viewport draws the picture alone, with
    /// no lamps' or cameras' glyphs and no element overlays.
    rendering_still: bool,
    /// The render under way, F12's or Ctrl+F12's. See `RenderJob`.
    render_job: Option<RenderJob>,
    /// The last render's picture, for View Render.
    last_render: Option<(u32, u32, Vec<u8>, String)>,
    /// The files the last render wrote, in order.
    last_render_files: Vec<std::path::PathBuf>,
    /// A capture or a render is waiting on the viewport: draw it even when
    /// the window is hidden and there is no frame to show it in.
    capturing: bool,
    /// Which primitive Shift+A adds next.
    next_shape: u32,
    /// Set for the one `begin_gesture` that Alt+E ▸ Along Normals starts.
    extrude_along_normals: bool,
    /// Where the document lives, and everything about saving it.
    file: DocumentFile,
    /// Autosaves from sessions that did not exit cleanly, offered once.
    ///
    /// Collected at startup rather than polled: a directory becoming stale while
    /// this instance runs means another instance crashed, and interrupting the
    /// user's work to say so is not this application's business.
    recoverable: Vec<docfile::Recoverable>,
    /// What Ctrl+C took: the selection as a fragment's text, for Ctrl+V.
    ///
    /// The application's own rather than the system clipboard, for now — the
    /// same as Blender's, which is per process. The text is what the system
    /// clipboard would carry, so moving to it is a matter of where the string
    /// is kept.
    clipboard: Option<String>,
    /// The key name of the keystroke being dispatched, while it is.
    ///
    /// A pie closes when the key that opened it is let go, so the action that
    /// opens one has to know which key that was — and an action is a keymap
    /// entry, which says what to do and not how it was asked for.
    dispatching_key: Option<&'static str>,
    /// Whether the animation is playing, and when the last frame was stepped.
    ///
    /// Application state, not the document's: a file saved mid-play opens
    /// stopped, the way every animation tool's does.
    playing: Option<std::time::Instant>,
    /// When the interface asked to be drawn again, if it did.
    ///
    /// egui says so after a frame: a menu that opened on a click is measured
    /// on one frame and shown on the next, a hover fades in, a tooltip waits
    /// its delay. Ignored, the window sleeps until the next event — so a
    /// clicked menu appeared only when the mouse moved.
    ui_repaint_at: Option<Instant>,
    /// The weight-paint brush stroke in progress: its modal session and the
    /// vertices it has crossed so far.
    paint_stroke: Option<PaintStroke>,
    /// A sculpt stroke in progress.
    sculpt_stroke: Option<SculptStroke>,
    /// A texture paint stroke in progress.
    texture_stroke: Option<TextureStroke>,
    /// The pointer moved during a brush stroke since the stroke was last
    /// driven. A mouse reports hundreds of moves a second and each drive is
    /// a raycast and a re-run of the stroke, so they are driven once a frame,
    /// not once a move.
    stroke_moved: bool,
    /// The painted mesh as the screen shows it in weight paint mode, with the
    /// active group's weights where the renderer's ramp reads them — kept, so
    /// an unchanged mesh is the same handle and is not uploaded again. Keyed
    /// on the evaluated mesh it was made from and the group.
    weight_display: Option<(Arc<MeshData>, Option<String>, Arc<MeshData>)>,
    /// The sculpted mesh with its mask and face sets where the overlay reads
    /// them. See `kreate::sculptview`.
    sculpt_display: kreate::sculptview::SculptDisplay,
    /// The studio and flat modes show base-colour images. On by default: a
    /// texture that was mapped is what the user expects to see.
    solid_textures: bool,
    /// A physics bake running off the main thread.
    bake_job: Option<BakeJob>,
    /// The tree a dynamic-topology stroke finds the surface in: see
    /// `SculptStroke::pick_scene`.
    stroke_pick: kreate::pick::PickCache,
    /// The token the next staged bake is handed over by.
    bake_tokens: u32,
    /// The images the viewport holds for them, and each one's revision, so
    /// they go up again only when they change.
    solid_images: Vec<(kreate_scene::document::ImageId, u64)>,
    /// Meshes with hidden elements, as drawn outside edit mode: without the
    /// hide layers. Input handle and the copy.
    unhidden: Vec<(Arc<MeshData>, Arc<MeshData>)>,
    /// `Ctrl+R` looking for an edge: the one under the pointer, if any, and
    /// how many loops the wheel has asked for. Blender's loop cut hover.
    loop_cut_hover: Option<(Option<u32>, u32)>,
    /// The edge and loop count a hover's click hands to the slide that follows.
    loop_cut_seed: Option<(u32, u32)>,
    /// `K` drawing a cut: the points clicked so far, in the world, and where
    /// the next click would put one.
    knife: Option<(Vec<DVec3>, Option<DVec3>)>,
    /// The sketch tools' state: see `kreate::sketching`.
    sketching: kreate::sketching::SketchUi,
    /// Sketch mode was asked for with no sketch to draw in: the plane menu
    /// is open, and choosing a plane starts one and enters.
    sketch_waiting: bool,
    /// The conversation being kept on disk: see `kreate::chatlog`.
    chat_record: Option<kreate::chatlog::Record>,
    /// Waiting for a click on a flat face, to start a sketch on it.
    sketch_face_pick: bool,
    /// Waiting for a click on a surface, or the floor, to put text there.
    text_place: bool,
    /// The sketch the camera was last turned to face, so entering turns it
    /// once and leaves the user's own orbit alone after.
    sketch_faced: Option<kreate_scene::Uid>,
    /// Which points of the sketch are pinned, and what that was worked out
    /// for: the analysis is a matrix reduction, not a per-frame cost.
    sketch_pinned: (u64, Vec<bool>),
    /// The Measure tool, while it is out. See `begin_measure`.
    measure: Option<MeasureTool>,
    /// Walk Navigation, while it runs, and when it last moved the view.
    /// See `walking.rs`.
    walk: Option<kreate::walk::Walk>,
    walk_clock: Option<Instant>,
    /// A question is on screen (save before leaving?), so a second close
    /// waits for its answer rather than asking again over it.
    asking: bool,
    /// The user has said to close: closes still queued close at once.
    quitting: bool,
    /// Align Faces, while it is out: the first face clicked, once it has
    /// been — the part that moves, its face's middle and normal.
    mate: Option<Option<(ObjectId, DVec3, DVec3)>>,
    /// The active object's mass properties, and the evaluated mesh and
    /// placement they were measured for: measured again only when either
    /// changes.
    mass_cache: Option<(usize, [f64; 12], kreate_mesh::measure::MassProperties)>,
    /// The section view's cut meshes, by the evaluated mesh and the plane in
    /// its space they were cut from.
    sectioned: Vec<(Arc<MeshData>, [f64; 7], Arc<MeshData>)>,
    /// The brush size circle select last used, so the next one starts there.
    circle_radius: f32,
    /// Local view (`/`): the objects it set aside — everything not selected
    /// when it was entered — and the view to go back to. What is set aside
    /// rather than what is shown, so what is made while in it is shown, as
    /// in Blender: a cube added in local view was there and not drawn.
    local_view: Option<(Vec<kreate_scene::document::ObjectId>, kreate_render::Camera)>,
    /// Looking through the scene camera (Numpad 0): the view to go back to.
    camera_view: Option<kreate_render::Camera>,
    /// The camera object the view is through, while it is: its lens is the
    /// path tracer's.
    camera_view_object: Option<kreate_scene::document::ObjectId>,
    /// Where the camera view last put the viewport. The view follows the
    /// camera while the viewport is still there; once something else has
    /// moved it — the user orbiting, a framing — the camera view is left.
    camera_view_pose: Option<kreate_render::Camera>,
    /// `C`: the selection brush, while it is on.
    circle: Option<CircleSelect>,
    /// The document version the drivers last ran against. They run again
    /// whenever it moves — an edit, an undo, a frame — so a driven property
    /// follows its variables while they are being changed.
    driven_at: u64,
    /// The scene as drawing in a sketch snaps to it, and the document
    /// version and snap mode it was gathered for.
    sketch_snap_cache: Option<(u64, kreate::snap::Mode, Arc<kreate::snap::SnapWorld>)>,
    /// The extrude's automatic operation and target, and what they were
    /// worked out for: the scene, the regions, the distance, symmetric.
    #[allow(clippy::type_complexity)]
    solid_auto: Option<((u64, Vec<[u64; 2]>, u64, bool), (u32, String))>,
    /// A sketch drag's step, live while the button is down.
    sketch_drag_session: Option<kreate_ops::ModalSession>,
    /// A point or handle of the active curve being dragged in the viewport.
    curve_drag: Option<CurveDrag>,
    /// The object Auto dynamic topology last said it would not cut.
    dyntopo_told: Option<ObjectId>,
    /// File ▸ Shared Session, while in one.
    session: Option<kreate_collab::Session>,
    /// The objects whose inside this participant holds the session's lock on.
    collab_locks: Vec<u64>,
    /// Blender's record button: while on, every object transform keys what it
    /// changed at the current frame.
    autokey: bool,
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
    /// Its falloff curve, one of `kreate_ops::ops::mesh::FALLOFF_CURVES`.
    proportional_falloff: u32,
    /// Whether circle select is on because the header's tool is Circle, and
    /// so goes off when the tool changes.
    circle_from_tool: bool,
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
        // File ▸ Open Recent, as the last session left it.
        app.ui.resolution_scale = kreate::prefs::viewport_path()
            .map_or(1.0, |p| kreate::prefs::load_resolution_scale(&p));
        app.ui.punchy =
            kreate::prefs::viewport_path().is_none_or(|p| kreate::prefs::load_punchy(&p));
        app.ui.trace_when_still =
            kreate::prefs::viewport_path().is_none_or(|p| kreate::prefs::load_trace_when_still(&p));
        app.ui.appearance = kreate::prefs::interface_path()
            .map(|p| kreate::prefs::load_appearance(&p))
            .unwrap_or_default();
        app.ui.recent_files = kreate::prefs::recent_path()
            .map(|p| kreate::prefs::load_recent(&p))
            .unwrap_or_default();
        app.load_keymap();
        app.load_workspaces();
        app
    }

    fn empty(doc: Document, locale: &str, registry: OperatorRegistry) -> Self {
        Self {
            bridge: None,
            update_ready: None,
            chat: None,
            chat_streaming: false,
            chat_turns: Vec::new(),
            asset_wishes: std::collections::HashMap::new(),
            import_wishes: std::collections::HashMap::new(),
            chat_attachments: Vec::new(),
            outbox: None,
            imports: std::sync::mpsc::channel(),
            import_wake: None,
            importing: Vec::new(),
            next_import: 0,
            assets: None,
            window: None,
            gfx: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            fonts_warmed: false,
            camera: kreate::startup::starting_camera(),
            input: Input::default(),
            redraw: RedrawReason(RedrawReason::SCENE_CHANGED),
            latency: LatencyStats::default(),
            doc,
            materials: None,
            path_trace_key: None,
            environment: None,
            studio_environment: false,
            environment_stale: false,
            registry,
            history: History::new(),
            op_ctx: OpContext::default(),
            uv_gesture: None,
            select_mode: 2,
            edit: kreate_ops::EditSessions::new(),
            time: kreate_math::SceneTime::default(),
            pick: kreate::pick::PickCache::new(),
            modifiers: kreate_scene::ModifierCache::new(),
            evaluated: Vec::new(),
            auto_depth: true,
            hover: None,
            xray: false,
            drag_select: None,
            motion: kreate_render::CameraMotion::new(kreate::startup::starting_camera()),
            last_frame: None,
            keymap: Keymap::default(),
            preset: Preset::default(),
            key_overrides: std::collections::BTreeMap::new(),
            buttons: [false; 3],
            drag_ops: [const { None }; 3],
            drag_travel: [0.0; 3],
            nav_view: None,
            viewport_px: None,
            ui: ShellState::with_catalog(shell::load_catalog(locale)),
            wanted_viewport: (0, 0),
            rendering_still: false,
            render_job: None,
            last_render: None,
            last_render_files: Vec::new(),
            capturing: false,
            next_shape: 1,
            extrude_along_normals: false,
            snap: kreate::modal::Snap::default(),
            proportional: None,
            proportional_falloff: 0,
            circle_from_tool: false,
            gesture: None,
            file: DocumentFile::new(),
            // Collected once, at startup. This is the only moment a recovery
            // offer is not an interruption.
            recoverable: docfile::find_recoverable(),
            clipboard: None,
            dispatching_key: None,
            playing: None,
            ui_repaint_at: None,
            driven_at: u64::MAX,
            sketch_snap_cache: None,
            sketch_drag_session: None,
            curve_drag: None,
            solid_auto: None,
            dyntopo_told: None,
            session: None,
            collab_locks: Vec::new(),
            paint_stroke: None,
            sculpt_stroke: None,
            sculpt_display: Default::default(),
            solid_textures: true,
            bake_job: None,
            stroke_pick: kreate::pick::PickCache::new(),
            bake_tokens: 0,
            solid_images: Vec::new(),
            stroke_moved: false,
            texture_stroke: None,
            weight_display: None,
            unhidden: Vec::new(),
            loop_cut_hover: None,
            loop_cut_seed: None,
            knife: None,
            sketching: Default::default(),
            sketch_waiting: false,
            chat_record: None,
            sketch_face_pick: false,
            text_place: false,
            sketch_faced: None,
            sketch_pinned: (0, Vec::new()),
            measure: None,
            walk: None,
            walk_clock: None,
            asking: false,
            quitting: false,
            mate: None,
            mass_cache: None,
            sectioned: Vec::new(),
            circle_radius: 40.0,
            local_view: None,
            camera_view: None,
            camera_view_object: None,
            camera_view_pose: None,
            circle: None,
            autokey: false,
            shown_title: String::new(),
        }
    }

    /// Dispatches an operator. The only way this application changes the document.
    /// The frame auto-keying puts a key on, or `None` while it is off.
    fn autokey_frame(&self) -> Option<f64> {
        self.autokey.then(|| self.doc.playback.frame.round())
    }

    fn run_op(&mut self, op_id: &str, mut params: PropertyMap) {
        // A new shape on the cursor lands inside whatever is already there —
        // on a new scene, inside the cube — and nothing seems to happen. Set
        // beside it instead, and said, when the spot is taken.
        if matches!(
            op_id,
            "object.add_primitive"
                | "object.add_empty"
                | "object.add_armature"
                | "object.add_gear"
                | "object.add_text"
                | "object.add_wall"
                | "object.add_door"
                | "object.add_window"
                | "object.add_stairs"
                | "object.add_column"
                | "object.add_railing"
                | "object.add_slab"
                | "object.add_roof"
        ) && params.get("location").is_none()
        {
            let spot = self.free_spot(self.doc.cursor);
            if spot != self.doc.cursor {
                params.set("location", PropertyValue::Vec3(spot.to_array()));
                let message = self.ui.catalog.get("add-beside");
                self.status(&message);
            }
        }
        let params = self.app_params().apply(op_id, &params);
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
                    println!("{}", entry.to_script_preview(kreate::shell::SCRIPT_PREVIEW));
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
        //
        // A sketch is held as a mesh of points and lines, and editing it is
        // the sketch tools, not its vertices: Tab on one goes into it.
        if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Objects
            && kreate_ops::ops::sketch_draw::active_sketch(&self.doc).is_some()
        {
            self.set_mode(OpMode::Sketch);
            return;
        }
        let has = kreate_ops::mode::Has::of(&self.doc);
        self.set_mode(kreate::mode::toggled(self.op_ctx.mode, has));
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
        if mode == OpMode::Sketch
            && kreate_ops::ops::sketch_draw::active_sketch(&self.doc).is_none()
        {
            self.sketch_waiting = true;
            self.open_popup_menu("sketch_plane");
            return;
        }
        // A body made from sketches holds no mesh of its own — it is built
        // again from its features — so editing its vertices means making it
        // a plain mesh first, as Fusion does when design history is turned
        // off. One undo step brings the features back.
        if mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements
            && self.op_ctx.mode.acts_on() != kreate_ops::mode::ActsOn::MeshElements
            && self
                .doc
                .selection
                .active
                .and_then(|id| self.doc.object(id))
                .is_some_and(|o| !o.features.is_empty())
        {
            self.run_op_with("feature.to_mesh", &[]);
            self.status(&self.ui.catalog.get("feature-made-mesh"));
        }
        let has = kreate_ops::mode::Has::of(&self.doc);
        let Some(change) = kreate::mode::plan(self.op_ctx.mode, mode, has) else {
            return;
        };
        if !self.collab_may_enter(change.to) {
            return;
        }
        // Only a change into or out of a mode with element selections sweeps
        // them. Entering pose mode is not an edit of any mesh, and a history
        // step for a selection nobody can see would be one more thing to undo.
        let sweeps = self.op_ctx.mode.edits_elements() || change.to.edits_elements();
        let meshes = |m: OpMode| m.acts_on() == kreate_ops::mode::ActsOn::MeshElements;
        let (entering, leaving) = (
            !meshes(self.op_ctx.mode) && meshes(change.to),
            meshes(self.op_ctx.mode) && !meshes(change.to),
        );
        self.op_ctx.mode = change.to;
        // With a shape key active, edit mode edits that key: its shape is
        // traded into the mesh before the session reads it.
        if entering {
            self.run_op_with(
                "object.shape_key_edit",
                &[("enter", PropertyValue::Bool(true))],
            );
        }

        if sweeps {
            self.run_op_with(
                "mesh.select",
                &[("action", PropertyValue::Enum(u32::from(!change.select_all)))],
            );
        }

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
        // And traded back on the way out, once the session has let go.
        if leaving {
            self.run_op_with(
                "object.shape_key_edit",
                &[("enter", PropertyValue::Bool(false))],
            );
        }

        // The mode decides which context a shortcut resolves in, so the
        // tooltips change with it: `X` deletes an object in one mode and
        // constrains an axis in the other.
        // A brush moves vertices, so a mesh with few of them barely changes
        // under one — the default cube has eight. Said once, on the way in,
        // rather than left to look like a brush that does nothing.
        // A texture shows only in the rendered view, so painting one starts
        // there — as Blender's texture paint shows the texture.
        if change.to.acts_on() == kreate_ops::mode::ActsOn::Texture {
            if let Some(gfx) = self.gfx.as_mut() {
                gfx.viewport.shading = kreate_render::ShadingMode::Rendered;
            }
            if self.paint_target().is_none() {
                let message = self.ui.catalog.get("texture-paint-no-image");
                self.status(&message);
            }
        }
        if change.to.acts_on() == kreate_ops::mode::ActsOn::Sculpt {
            let verts = self
                .doc
                .selection
                .active
                .and_then(|id| self.doc.object(id))
                .and_then(|o| o.mesh())
                .and_then(|m| self.doc.mesh(m))
                .map(|m| m.num_verts())
                .unwrap_or(0);
            let cuts = self.doc.selection.active.is_some_and(|id| {
                self.dyntopo_loses(id).is_empty()
                    && self.ui.sculpt.dyntopo != kreate::shell::DYNTOPO_OFF
            }) || self.ui.sculpt.dyntopo == kreate::shell::DYNTOPO_ON;
            // With dynamic topology the brush adds what it needs as it goes.
            if verts < SCULPT_DENSE_ENOUGH && !cuts {
                let message = self
                    .ui
                    .catalog
                    .get_args("sculpt-coarse", &[("count", &verts.to_string())]);
                self.status(&message);
            }
        }
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
        let context = mode_key_context(self.op_ctx.mode);
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
        if !self.ui.has_editor(kreate_ui::EditorId::UV_EDITOR) {
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
        // Bones first: pose mode and armature edit mode act on neither
        // objects nor mesh elements, and their menu is their own.
        let bone_groups = match self.op_ctx.mode.acts_on() {
            kreate_ops::mode::ActsOn::Bones => Some(vec![
                group(
                    &t("menu-group-transform"),
                    &[
                        ("pose.translate", &[]),
                        ("pose.rotate", &[]),
                        ("pose.scale", &[]),
                    ],
                    self,
                ),
                group(
                    &t("menu-group-keys"),
                    &[("anim.keyframe_insert", &[]), ("anim.keyframe_delete", &[])],
                    self,
                ),
                group(
                    &t("menu-group-clear"),
                    &[
                        ("pose.clear_transform", &[("what", PropertyValue::Enum(1))]),
                        ("pose.clear_transform", &[("what", PropertyValue::Enum(2))]),
                        ("pose.clear_transform", &[("what", PropertyValue::Enum(3))]),
                        ("pose.clear_transform", &[("what", PropertyValue::Enum(0))]),
                    ],
                    self,
                ),
            ]),
            kreate_ops::mode::ActsOn::BoneJoints => Some(vec![
                group(
                    &t("menu-group-transform"),
                    &[
                        ("armature.translate", &[]),
                        ("armature.rotate", &[]),
                        ("armature.scale", &[]),
                    ],
                    self,
                ),
                group(
                    &t("menu-group-bones"),
                    &[
                        ("armature.extrude", &[]),
                        ("armature.bone_add", &[]),
                        ("armature.delete", &[]),
                    ],
                    self,
                ),
            ]),
            _ => None,
        };
        let groups = if let Some(groups) = bone_groups {
            groups
        } else if self.op_ctx.mode.edits_elements() {
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
                        ("mesh.edges_to_sketch", &[]),
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
                            ("uv.unwrap", &[]),
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
                {
                    let mut object = group(
                        &t("menu-group-object"),
                        &[
                            (
                                "object.duplicate",
                                &[("linked", PropertyValue::Bool(false))],
                            ),
                            ("object.duplicate", &[("linked", PropertyValue::Bool(true))]),
                            ("object.join", &[]),
                            ("object.boolean", &[("operation", PropertyValue::Enum(0))]),
                            ("object.boolean", &[("operation", PropertyValue::Enum(1))]),
                            ("object.boolean", &[("operation", PropertyValue::Enum(2))]),
                            ("object.delete", &[]),
                        ],
                        self,
                    );
                    // Add opens the Add menu, as Shift+A does — a shape, a
                    // lamp, a camera. The row used to add a cube with nothing
                    // said, which lands on the 3D cursor: on a new scene that
                    // is inside the cube already there, and nothing appeared
                    // to happen.
                    let mut params = PropertyMap::new();
                    params.set("menu", PropertyValue::Text("add".into()));
                    let mut info = self.command_info("app.menu", &params);
                    info.name = format!("{}...", t("menu-add"));
                    object
                        .entries
                        .insert(0, Entry::new("app.menu", info).with_params(params));
                    object
                },
            ]
        };
        self.ui.context_menu.groups = groups;
    }

    /// Opens a named menu under the pointer, in the right-click menu's place:
    /// `add` (Shift+A) or `apply` (Ctrl+A). Each row is an operator with its
    /// parameters, named as Blender names it.
    fn open_popup_menu(&mut self, name: &str) {
        use kreate_ui::context_menu::{Entry, Group};
        let t = |key: &str| self.ui.catalog.get(key);
        let row = |app: &App, label: String, id: &str, pairs: &[(&'static str, PropertyValue)]| {
            let mut params = PropertyMap::new();
            for (k, v) in pairs {
                params.set(k, v.clone());
            }
            let mut info = app.command_info(id, &params);
            info.name = label;
            Entry::new(id, info).with_params(params)
        };
        let groups = match name {
            "add" => {
                let shapes = [
                    "menu-add-plane",
                    "menu-add-cube",
                    "menu-add-grid",
                    "menu-add-uv-sphere",
                    "menu-add-ico-sphere",
                    "menu-add-cylinder",
                    "menu-add-cone",
                    "menu-add-torus",
                ];
                let mesh = shapes
                    .iter()
                    .enumerate()
                    .map(|(i, key)| {
                        row(
                            self,
                            t(key),
                            "object.add_primitive",
                            &[
                                ("shape", PropertyValue::Enum(i as u32)),
                                ("on_floor", PropertyValue::Bool(true)),
                            ],
                        )
                    })
                    .collect();
                let lights = [
                    "menu-add-light-point",
                    "menu-add-light-sun",
                    "menu-add-light-spot",
                    "menu-add-light-area",
                ]
                .iter()
                .enumerate()
                .map(|(i, key)| {
                    row(
                        self,
                        t(key),
                        "object.add_light",
                        &[("kind", PropertyValue::Enum(i as u32))],
                    )
                })
                .collect();
                let gears = [
                    "menu-add-gear",
                    "menu-add-gear-internal",
                    "menu-add-gear-rack",
                    "menu-add-gear-bevel",
                    "menu-add-gear-worm",
                ];
                // Gears and the parts they run with: the Manufacturing group.
                let mut manufacturing: Vec<_> = gears
                    .iter()
                    .enumerate()
                    .map(|(kind, key)| {
                        row(
                            self,
                            t(key),
                            "object.add_gear",
                            &[("kind", PropertyValue::Enum(kind as u32))],
                        )
                    })
                    .collect();
                for (key, id, pairs) in [
                    ("mech-bearing", "object.add_bearing", None),
                    ("mech-shaft", "object.add_shaft", None),
                    ("mech-pulley", "object.add_pulley", None),
                    ("mech-sprocket", "object.add_sprocket", None),
                    (
                        "mech-spring-compression",
                        "object.add_spring",
                        Some(("kind", PropertyValue::Enum(0))),
                    ),
                    (
                        "mech-spring-extension",
                        "object.add_spring",
                        Some(("kind", PropertyValue::Enum(1))),
                    ),
                    (
                        "mech-spring-torsion",
                        "object.add_spring",
                        Some(("kind", PropertyValue::Enum(2))),
                    ),
                    (
                        "mech-cap-screw",
                        "object.add_fastener",
                        Some(("kind", PropertyValue::Enum(0))),
                    ),
                    (
                        "mech-hex-bolt",
                        "object.add_fastener",
                        Some(("kind", PropertyValue::Enum(1))),
                    ),
                    (
                        "mech-nut",
                        "object.add_fastener",
                        Some(("kind", PropertyValue::Enum(2))),
                    ),
                    (
                        "mech-washer",
                        "object.add_fastener",
                        Some(("kind", PropertyValue::Enum(3))),
                    ),
                    (
                        "mech-sheet-folded",
                        "object.add_sheet_metal",
                        Some(("flat", PropertyValue::Bool(false))),
                    ),
                    (
                        "mech-sheet-flat",
                        "object.add_sheet_metal",
                        Some(("flat", PropertyValue::Bool(true))),
                    ),
                ] {
                    let pairs: Vec<(&'static str, PropertyValue)> = pairs.into_iter().collect();
                    manufacturing.push(row(self, t(key), id, &pairs));
                }
                let architecture: Vec<_> = kreate::shell::ADD_ARCHITECTURE
                    .iter()
                    .map(|(id, key)| row(self, t(key), id, &[]))
                    .collect();
                let other = vec![
                    row(self, t("menu-add-camera"), "object.add_camera", &[]),
                    row(self, t("menu-add-empty"), "object.add_empty", &[]),
                    row(self, t("menu-add-text"), "app.place_text", &[]),
                    row(self, t("menu-add-armature"), "object.add_armature", &[]),
                ];
                vec![
                    Group {
                        title: t("menu-add-mesh"),
                        entries: mesh,
                    },
                    Group {
                        title: t("menu-add-light"),
                        entries: lights,
                    },
                    Group {
                        title: t("menu-add-manufacturing"),
                        entries: manufacturing,
                    },
                    Group {
                        title: t("menu-add-architecture"),
                        entries: architecture,
                    },
                    Group {
                        title: String::new(),
                        entries: other,
                    },
                ]
            }
            "collection" => {
                let mut entries = vec![row(
                    self,
                    t("collection-scene"),
                    "object.move_to_collection",
                    &[("collection", PropertyValue::Text(String::new()))],
                )];
                let names: Vec<String> = self
                    .doc
                    .collections()
                    .map(|(_, c)| c.name.clone())
                    .collect();
                for name in names {
                    entries.push(row(
                        self,
                        name.clone(),
                        "object.move_to_collection",
                        &[("collection", PropertyValue::Text(name))],
                    ));
                }
                entries.push(row(
                    self,
                    t("collection-new"),
                    "object.move_to_collection",
                    &[
                        ("collection", PropertyValue::Text("Collection".into())),
                        ("new", PropertyValue::Bool(true)),
                    ],
                ));
                vec![Group {
                    title: t("menu-move-to-collection"),
                    entries,
                }]
            }
            // Blender's Shift+G: what to compare, for the element type in use.
            "similar" => {
                let entries = kreate_mesh::selection::SIMILAR
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, mode))| {
                        *mode
                            == match self.ui.select_mode {
                                1 => kreate_mesh::bmesh::SelectMode::Edge,
                                2 => kreate_mesh::bmesh::SelectMode::Face,
                                _ => kreate_mesh::bmesh::SelectMode::Vert,
                            }
                    })
                    .map(|(i, (name, _))| {
                        row(
                            self,
                            t(&format!("similar-{}", name.replace('_', "-"))),
                            "mesh.select_similar",
                            &[("kind", PropertyValue::Enum(i as u32))],
                        )
                    })
                    .collect();
                vec![Group {
                    title: t("menu-select-similar"),
                    entries,
                }]
            }
            // Blender's Ctrl+P: what to parent to.
            "parent" => vec![Group {
                title: t("menu-parent-to"),
                entries: vec![
                    row(self, t("parent-object"), "object.parent", &[]),
                    row(
                        self,
                        t("bind-automatic"),
                        "object.bind_armature",
                        &[("weights", PropertyValue::Enum(0))],
                    ),
                    row(
                        self,
                        t("bind-empty"),
                        "object.bind_armature",
                        &[("weights", PropertyValue::Enum(1))],
                    ),
                ],
            }],
            // Blender's Alt+P.
            "parent_clear" => vec![Group {
                title: t("parent-clear"),
                entries: vec![
                    row(
                        self,
                        t("parent-clear"),
                        "object.parent",
                        &[("clear", PropertyValue::Bool(true))],
                    ),
                    row(self, t("unbind-armature"), "object.unbind_armature", &[]),
                ],
            }],
            "extrude" => vec![Group {
                title: t("menu-extrude-kind"),
                entries: vec![
                    row(self, t("extrude-region"), "app.extrude", &[]),
                    row(
                        self,
                        t("extrude-along-normals"),
                        "app.extrude",
                        &[("along_normals", PropertyValue::Bool(true))],
                    ),
                ],
            }],
            "sketch_plane" => {
                let plane = |app: &App, key: &str, i: u32| {
                    row(
                        app,
                        t(key),
                        "sketch.new",
                        &[("plane", PropertyValue::Enum(i))],
                    )
                };
                vec![Group {
                    title: t("sketch-plane"),
                    entries: vec![
                        plane(self, "sketch-plane-xy", 0),
                        plane(self, "sketch-plane-xz", 1),
                        plane(self, "sketch-plane-yz", 2),
                        plane(self, "sketch-plane-cursor", 3),
                        row(self, t("sketch-plane-face"), "app.sketch_on_face", &[]),
                    ],
                }]
            }
            "apply" => {
                let apply = |app: &App, key: &str, l: bool, r: bool, s: bool| {
                    row(
                        app,
                        t(key),
                        "object.transform_apply",
                        &[
                            ("location", PropertyValue::Bool(l)),
                            ("rotation", PropertyValue::Bool(r)),
                            ("scale", PropertyValue::Bool(s)),
                        ],
                    )
                };
                vec![Group {
                    title: t("menu-apply"),
                    entries: vec![
                        apply(self, "apply-location", true, false, false),
                        apply(self, "apply-rotation", false, true, false),
                        apply(self, "apply-scale", false, false, true),
                        apply(self, "apply-all", true, true, true),
                        apply(self, "apply-rotation-scale", false, true, true),
                    ],
                }]
            }
            _ => return,
        };
        self.ui.context_menu.groups = groups;
        let Some((rx, ry, rw, rh)) = self.viewport_px else {
            return;
        };
        // At the pointer, but inside the viewport: the menu is drawn by the
        // viewport and only there, and a menu opened from a button above it —
        // the Sketch mode tab asking for a plane — would otherwise open off
        // the viewport and never be seen.
        let (x, y) = self
            .input
            .cursor
            .map(|(x, y)| (x as f32, y as f32))
            .unwrap_or((rx + rw * 0.5, ry + rh * 0.5));
        let x = x.clamp(rx + 8.0, (rx + rw - 8.0).max(rx + 8.0));
        let y = y.clamp(ry + 8.0, (ry + rh - 8.0).max(ry + 8.0));
        let scale = self.egui_ctx.pixels_per_point().max(0.01);
        self.ui.context_menu.at = Some(egui::pos2(x / scale, y / scale));
        self.redraw.set(RedrawReason::UI);
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
            ("mesh.shade", &[("smooth", PropertyValue::Bool(true))]),
            ("mesh.shade", &[("smooth", PropertyValue::Bool(false))]),
            ("mesh.normals", &[("action", PropertyValue::Enum(0))]),
            ("mesh.select", &[("action", PropertyValue::Enum(0))]),
            ("mesh.select", &[("action", PropertyValue::Enum(1))]),
            ("mesh.select", &[("action", PropertyValue::Enum(2))]),
            ("mesh.select", &[("action", PropertyValue::Enum(3))]),
            ("mesh.select", &[("action", PropertyValue::Enum(6))]),
            ("mesh.select", &[("action", PropertyValue::Enum(7))]),
            ("mesh.select", &[("action", PropertyValue::Enum(8))]),
            ("mesh.hide", &[("unselected", PropertyValue::Bool(false))]),
            ("mesh.hide", &[("unselected", PropertyValue::Bool(true))]),
            ("mesh.reveal", &[]),
            ("view.cursor_place", &[]),
            ("pose.ik_add", &[]),
            ("mesh.knife", &[]),
            ("uv.unwrap", &[]),
            ("object.select_all", &[("action", PropertyValue::Enum(0))]),
            ("object.select_all", &[("action", PropertyValue::Enum(1))]),
            ("object.select_all", &[("action", PropertyValue::Enum(3))]),
            ("object.add_light", &[]),
            ("object.add_camera", &[]),
            ("object.add_empty", &[]),
            ("object.add_curve", &[]),
            ("object.follow_path", &[]),
            ("object.add_armature", &[]),
            (
                "object.bind_armature",
                &[("weights", PropertyValue::Enum(0))],
            ),
            (
                "object.bind_armature",
                &[("weights", PropertyValue::Enum(1))],
            ),
            ("object.unbind_armature", &[]),
            ("pose.translate", &[]),
            ("pose.rotate", &[]),
            ("pose.scale", &[]),
            ("pose.select_all", &[("action", PropertyValue::Enum(0))]),
            ("pose.select_all", &[("action", PropertyValue::Enum(1))]),
            ("pose.select_all", &[("action", PropertyValue::Enum(3))]),
            ("pose.clear_transform", &[("what", PropertyValue::Enum(1))]),
            ("pose.clear_transform", &[("what", PropertyValue::Enum(2))]),
            ("pose.clear_transform", &[("what", PropertyValue::Enum(3))]),
            ("armature.translate", &[]),
            ("armature.rotate", &[]),
            ("armature.scale", &[]),
            ("armature.extrude", &[]),
            ("armature.delete", &[]),
            ("armature.bone_add", &[]),
            ("armature.select_all", &[("action", PropertyValue::Enum(0))]),
            ("armature.select_all", &[("action", PropertyValue::Enum(1))]),
            ("armature.select_all", &[("action", PropertyValue::Enum(3))]),
            ("anim.keyframe_insert", &[]),
            ("anim.keyframe_delete", &[]),
            // The application's own actions, which have keys too: a menu entry
            // that says "Ctrl+S" beside Save is how the key gets learnt.
            ("file.new", &[]),
            ("file.open", &[]),
            ("file.save", &[]),
            ("file.save_as", &[]),
            ("edit.undo", &[]),
            ("edit.redo", &[]),
            ("edit.copy", &[]),
            ("edit.paste", &[]),
            ("view.frame_all", &[]),
            ("view.walk", &[]),
            ("view.toggle_projection", &[]),
            ("view.toggle_sidebar", &[]),
            ("view.toggle_grid", &[]),
            ("view.toggle_ground_shadow", &[]),
            ("view.toggle_wireframe", &[]),
            ("app.search", &[]),
            ("edit.copy_script", &[]),
            ("file.run_script", &[]),
        ];
        // Keyed by the *variant* — the operator with the parameters that make
        // it this entry — so Select ▸ All and Select ▸ None, both `mesh.select`,
        // each keep their own shortcut. An entry with no parameters is keyed by
        // the bare id, which is the same string.
        self.ui.command_info = SHOWN
            .iter()
            .map(|(id, pairs)| {
                let mut params = PropertyMap::new();
                for (name, value) in *pairs {
                    params.set(name, value.clone());
                }
                (
                    kreate_ui::command_info::variant_id(id, &params),
                    self.command_info(id, &params),
                )
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
        if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Bones {
            // The operator's own pivot, for the reason the edit-mode branch
            // gives: a gesture measured about another point would turn the
            // bones out from under the pointer.
            return kreate_ops::ops::pose::pivot(&self.doc);
        }
        if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::BoneJoints {
            return kreate_ops::ops::armature::pivot(&self.doc);
        }
        if self.op_ctx.mode.edits_elements() {
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
    fn build_snap_world(&self, mode: kreate::snap::Mode) -> kreate::snap::SnapWorld {
        let mut world = kreate::snap::SnapWorld::default();
        if !mode.needs_geometry() {
            // Increment snapping needs no candidates, and gathering a scene's
            // worth of them to ignore would be the whole cost for none of the
            // benefit.
            return world;
        }

        let edited = self
            .op_ctx
            .mode
            .edits_elements()
            .then(|| kreate_ops::ops::mesh::target_mesh(&self.doc))
            .flatten();
        let moving = self.moving_objects();

        for (id, mesh) in &self.evaluated {
            let object = self.doc.object(*id);
            let Some(object) = object else { continue };
            if !self.doc.is_shown(*id) {
                continue;
            }
            let to_world = self.doc.world_transform(*id);

            let moving: Vec<u32> = if self.op_ctx.mode.edits_elements() {
                // In edit mode only the selected elements move, and only within
                // the mesh being edited. Every other object is a candidate whole.
                if object.mesh() == edited {
                    kreate_ops::ops::mesh::selection_vertices(mesh)
                } else {
                    Vec::new()
                }
            } else if moving.contains(id) {
                // In object mode a selected object moves entirely, and every
                // object under it with it, so none of them is a candidate:
                // a part snapped to its own child is snapped to itself.
                continue;
            } else {
                Vec::new()
            };

            world.add_mesh(mesh, &to_world, &moving);
        }
        world
    }

    /// What a drag is about to move, as snap candidates: the selected
    /// objects whole in object mode, the selected part of the edited mesh in
    /// edit mode — where a move's snap base is picked from.
    fn build_own_snap_world(&self) -> kreate::snap::SnapWorld {
        let mut world = kreate::snap::SnapWorld::default();
        let moving = self.moving_objects();
        let edited = self
            .op_ctx
            .mode
            .edits_elements()
            .then(|| kreate_ops::ops::mesh::target_mesh(&self.doc))
            .flatten();
        for (id, mesh) in &self.evaluated {
            let Some(object) = self.doc.object(*id) else {
                continue;
            };
            if !self.doc.is_shown(*id) {
                continue;
            }
            let to_world = self.doc.world_transform(*id);
            if self.op_ctx.mode.edits_elements() {
                if edited.is_some() && object.mesh() == edited {
                    let selected = kreate_ops::ops::mesh::selection_vertices(mesh);
                    world.add_mesh_only(mesh, &to_world, &selected);
                }
            } else if moving.contains(id) {
                world.add_mesh(mesh, &to_world, &[]);
            }
        }
        world
    }

    /// The objects a move in object mode carries: the selected ones and
    /// every object under them.
    fn moving_objects(&self) -> std::collections::HashSet<kreate_scene::document::ObjectId> {
        let selected = &self.doc.selection.objects;
        self.doc
            .objects()
            .map(|(id, _)| id)
            .filter(|id| {
                selected.contains(id) || selected.iter().any(|s| self.doc.is_ancestor(*s, *id))
            })
            .collect()
    }

    /// Takes the snap base under the pointer and goes on moving, with what
    /// the base may land on gathered if the drag had nothing yet.
    fn finish_base_pick(&mut self) {
        let picked = self
            .gesture
            .as_mut()
            .is_some_and(|(gesture, _)| gesture.finish_base_pick());
        if self.gesture.as_ref().is_some_and(|(g, _)| g.wants_world()) {
            let world = self.build_snap_world(kreate::snap::Mode::Element);
            if let Some((gesture, _)) = &mut self.gesture {
                gesture.set_world(world);
            }
        }
        if !picked {
            self.status("no point of the selection there; snapping by the nearest");
        }
        self.drive_gesture();
    }

    /// The points of what a drag is about to move, in the world: the
    /// selected vertices in edit mode, the selected objects' vertices (or
    /// origins) in object mode. A snap lands the one nearest its target.
    ///
    /// Taken at the start, because the drag moves them. Left empty past a
    /// size where looking through them every frame would be felt — the
    /// pivot lands instead.
    fn snap_sources(&self) -> Vec<DVec3> {
        const MAX: usize = 200_000;
        let mut out = Vec::new();
        let point =
            |to_world: &glam::DAffine3, p: glam::Vec3| to_world.transform_point3(p.as_dvec3());
        if self.op_ctx.mode.edits_elements() {
            let Some(edited) = kreate_ops::ops::mesh::target_mesh(&self.doc) else {
                return out;
            };
            for (id, mesh) in &self.evaluated {
                if self.doc.object(*id).and_then(|o| o.mesh()) != Some(edited) {
                    continue;
                }
                let to_world = self.doc.world_transform(*id);
                let selected = kreate_ops::ops::mesh::selection_vertices(mesh);
                if out.len() + selected.len() > MAX {
                    return Vec::new();
                }
                out.extend(
                    selected
                        .iter()
                        .map(|v| point(&to_world, mesh.positions()[*v as usize])),
                );
            }
        } else if self.op_ctx.mode.info().acts_on == kreate_ops::mode::ActsOn::Objects {
            for id in &self.moving_objects() {
                let to_world = self.doc.world_transform(*id);
                match self.evaluated.iter().find(|(e, _)| e == id) {
                    Some((_, mesh)) => {
                        if out.len() + mesh.positions().len() > MAX {
                            return Vec::new();
                        }
                        out.extend(mesh.positions().iter().map(|p| point(&to_world, *p)));
                    }
                    None => out.push(to_world.translation),
                }
            }
        }
        out
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
        // A loop cut the pointer chose needs no selection: it pivots on its
        // edge. Everything else pivots on what is selected.
        let hovered_edge = self
            .loop_cut_seed
            .filter(|_| kind == kreate::modal::Kind::LoopCut)
            .and_then(|(edge, _)| kreate_ops::ops::mesh::edge_in_world(&self.doc, edge))
            .map(|(a, b)| (a + b) * 0.5);
        let Some(pivot) = hovered_edge
            .or_else(|| self.pivot_override(kind))
            .or_else(|| self.selection_pivot())
        else {
            self.status("nothing selected");
            return;
        };

        let mode = self.op_ctx.mode;
        // An extrude's frame: the selected faces' normal as its local Z, which
        // the gesture is then held to.
        let along_normals = std::mem::take(&mut self.extrude_along_normals);
        let extrude_normal = if kind == kreate::modal::Kind::Extrude {
            match kreate_ops::ops::mesh::selected_faces_normal(&self.doc) {
                Some(n) => Some(n),
                // A closed solid's normals cancel; along its own normals each
                // face still has somewhere to go.
                None if along_normals && kreate_ops::ops::mesh::any_face_selected(&self.doc) => {
                    Some(glam::DVec3::Z)
                }
                None => {
                    self.status("select faces to extrude");
                    return;
                }
            }
        } else {
            None
        };
        let basis = match extrude_normal {
            Some(n) => {
                let n = n.as_vec3();
                let t = n.any_orthonormal_vector();
                [t, n.cross(t), n]
            }
            None if self.ui.tools.orientation == 2 => self.cursor_basis(),
            None => self.selection_basis(),
        };
        let mut gesture = kreate::modal::Gesture::begin(
            kind,
            cursor,
            pivot,
            &self.camera,
            (w, h),
            basis,
            kreate::modal::Snap {
                length: snap_step(self.doc.design.length_unit),
                ..self.snap
            },
            self.build_snap_world(self.snap.mode),
        );
        if matches!(
            kind,
            kreate::modal::Kind::Move
                | kreate::modal::Kind::ExtrudeFree
                | kreate::modal::Kind::Extrude
        ) {
            gesture.set_sources(self.snap_sources());
        }
        if extrude_normal.is_some() {
            gesture.set_object_basis(if self.ui.tools.orientation == 2 {
                self.cursor_basis()
            } else {
                self.selection_basis()
            });
            if along_normals {
                let has_normal = kreate_ops::ops::mesh::selected_faces_normal(&self.doc).is_some();
                gesture.extrude_along_normals(has_normal);
            } else {
                gesture.extrude_along_normal();
            }
        }
        // A loop cut slides along the edge it cuts across from.
        if kind == kreate::modal::Kind::LoopCut {
            let seed = match self.loop_cut_seed.take() {
                Some((edge, cuts)) => {
                    gesture.edge = Some(edge);
                    gesture.cuts = cuts;
                    kreate_ops::ops::mesh::edge_in_world(&self.doc, edge)
                }
                None => kreate_ops::ops::mesh::loop_cut_edge(&self.doc),
            };
            match seed {
                Some((a, b)) => gesture.set_slide_edge(a, b),
                None => {
                    self.status("select an edge to loop cut across");
                    return;
                }
            }
        }
        // An edge slide measures the pointer along the way the edges go.
        if kind == kreate::modal::Kind::EdgeSlide {
            match kreate_ops::ops::mesh::edge_slide_handle(&self.doc) {
                Some((a, b)) => gesture.set_slide_edge(a, b),
                None => {
                    self.status("select edges with faces beside them to slide");
                    return;
                }
            }
        }
        // Ctrl inverts snapping, and it may already be held when the gesture
        // starts. Read once here rather than waiting for the next modifier
        // event, which might never come.
        gesture.set_snap_inverted(self.input.ctrl_down);
        gesture.set_first_orientation(if matches!(self.ui.tools.orientation, 1 | 2) {
            kreate::modal::Orientation::Local
        } else {
            kreate::modal::Orientation::Global
        });
        // Proportional editing is meaningless in object mode: there is no
        // neighbouring geometry to fall off towards, only other objects, and
        // dragging one object part-way because another is near it is not
        // something anyone asks for.
        // Asked of the mode, because that is the claim: proportional editing
        // falls off towards neighbouring *geometry*, and only a mode that has
        // the elements on screen has any.
        if mode.edits_elements() && kind.transforms() {
            gesture.set_proportional(self.proportional);
            gesture.set_proportional_falloff(self.proportional_falloff);
        }
        let op_id = kind.op_id(mode);
        // Started with the identity parameters, so pressing the key alone moves
        // nothing. The first pointer motion is what does anything.
        let start = self.app_params().apply(op_id, &gesture.params());
        match self.history.begin_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            op_id,
            &start,
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
        let Some(weighing) = kreate_ops::ops::mesh::weigh_edit_selection_with(
            &self.doc,
            radius,
            self.proportional_falloff,
        ) else {
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
        let pivot = if self.ui.pivot == 3 {
            self.doc.cursor
        } else {
            weighing.pivot
        };
        let Some(place) = kreate_render::Placement::new(to_world, pivot) else {
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
        let weighing = kreate_ops::ops::mesh::weigh_edit_selection_with(
            &self.doc,
            radius,
            self.proportional_falloff,
        );
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
        self.ui.snap_target = self.gesture.as_ref().and_then(|(g, _)| g.snap_point());

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
        let fill = self.app_params();
        let Some((_, session)) = &mut self.gesture else {
            return;
        };
        let params = fill.apply(session.op_id(), &params);
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
        let fill = self.app_params();
        let Some((gesture, mut session)) = self.gesture.take() else {
            return;
        };
        self.ui.snap_target = None;
        // A previewed drag never wrote the document, so the operator has to run
        // now — once, with the final parameters. Which is exactly the shape
        // §1.6 wants anyway: the history entry, the adjust panel and the script
        // line all come from one `(op_id, params)` call, and the pointer's whole
        // journey leaves no trace beyond where it ended up.
        //
        // The preview is dropped rather than restored: this run rewrites the
        // positions, which makes a new handle, which `gpu_sync` repositions. One
        // extraction, at mouse-up, instead of one per frame.
        let final_params = fill.apply(session.op_id(), &gesture.params());
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
                &final_params,
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
        self.ui.snap_target = None;
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
    /// The application's additions to an operator's parameters, as of now.
    /// `at`, or the first place along +X from it that is not inside a
    /// visible object — where a new object can be seen.
    fn free_spot(&self, at: DVec3) -> DVec3 {
        spot_beside(at, &self.mesh_boxes())
    }

    /// Every shown mesh's world box, low and high corner.
    fn mesh_boxes(&self) -> Vec<(DVec3, DVec3)> {
        self.mesh_boxes_by_object()
            .into_iter()
            .map(|(_, b)| b)
            .collect()
    }

    /// The same, each with its object.
    fn mesh_boxes_by_object(&self) -> Vec<(kreate_scene::ObjectId, (DVec3, DVec3))> {
        self.evaluated
            .iter()
            .filter_map(|(id, mesh)| {
                let p = mesh.positions();
                let first = *p.first()?;
                let (lo, hi) = p
                    .iter()
                    .fold((first, first), |(lo, hi), q| (lo.min(*q), hi.max(*q)));
                let to = self.doc.world_transform(*id);
                let mut wlo = DVec3::splat(f64::MAX);
                let mut whi = DVec3::splat(f64::MIN);
                for i in 0..8 {
                    let c = glam::Vec3::new(
                        if i & 1 == 0 { lo.x } else { hi.x },
                        if i & 2 == 0 { lo.y } else { hi.y },
                        if i & 4 == 0 { lo.z } else { hi.z },
                    );
                    let w = to.transform_point3(c.as_dvec3());
                    wlo = wlo.min(w);
                    whi = whi.max(w);
                }
                Some((*id, (wlo, whi)))
            })
            .collect()
    }

    fn app_params(&self) -> AppParams {
        AppParams {
            key: self.autokey_frame(),
            pivot: self.ui.pivot,
            cursor: self.doc.cursor,
            plane: (self.ui.tools.orientation == 2).then_some(self.doc.cursor_rotation),
            auto_merge: self.ui.tools.auto_merge,
            frame: self.doc.playback.frame.round(),
        }
    }

    /// Where a rotation or scale turns about, on screen as in the operator:
    /// the header's pivot point, or `None` for the selection's own middle.
    fn pivot_override(&self, kind: kreate::modal::Kind) -> Option<DVec3> {
        if !matches!(
            kind,
            kreate::modal::Kind::Rotate | kreate::modal::Kind::Scale
        ) {
            return None;
        }
        match self.ui.pivot {
            3 => Some(self.doc.cursor),
            2 if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Objects => self
                .doc
                .selection
                .active
                .map(|a| self.doc.world_transform(a).translation),
            _ => None,
        }
    }

    /// The panel's conversation, written to its page in `chats/`: called
    /// whenever it changes. Begun on the first message, with the document
    /// that was open.
    pub(crate) fn save_chat(&mut self) {
        use kreate_ui::chat::ChatRole;
        let Some(dir) = kreate::chatlog::chat_dir() else {
            return;
        };
        let messages: Vec<(String, String)> = self
            .ui
            .chat_log
            .iter()
            .filter_map(|m| {
                let who = match m.role {
                    ChatRole::User => "you",
                    ChatRole::Claude => "assistant",
                    ChatRole::Tool => "tool",
                    ChatRole::Error => "error",
                    ChatRole::Note => "note",
                    ChatRole::Turn(_) => return None,
                };
                Some((who.to_string(), m.text.clone()))
            })
            .collect();
        if messages.is_empty() {
            return;
        }
        let assistant = self
            .chat
            .as_ref()
            .map(|c| c.provider().name().to_string())
            .unwrap_or_default();
        let session = self
            .chat
            .as_ref()
            .and_then(|c| c.session().map(str::to_string));
        let document = self.file.path().map(|p| p.display().to_string());
        let record = self
            .chat_record
            .get_or_insert_with(|| kreate::chatlog::Record {
                started: kreate::chatlog::now(),
                assistant: assistant.clone(),
                document,
                ..Default::default()
            });
        if record.messages == messages && record.session == session {
            return;
        }
        record.messages = messages;
        record.session = session;
        record.assistant = assistant;
        if let Err(e) = kreate::chatlog::save_in(&dir, record) {
            tracing::warn!("cannot keep the conversation: {e}");
        }
    }

    /// The last conversation, shown again at start and carried on with the
    /// same assistant.
    fn restore_chat(&mut self) {
        use kreate_ui::chat::{ChatMessage, ChatRole};
        let Some(record) =
            kreate::chatlog::chat_dir().and_then(|d| kreate::chatlog::load_last_in(&d))
        else {
            return;
        };
        let Some(chat) = self.chat.as_mut() else {
            return;
        };
        if chat.provider().name() != record.assistant {
            return;
        }
        // Carried on only where the assistant still has the session: one
        // from before Claude ran in its own folder is not there to resume,
        // and asking for it would fail the next message.
        if let Some(s) = &record.session {
            let there = chat.provider() != kreate::claude::Provider::Claude
                || kreate::claude::claude_has_session(s);
            if there {
                chat.resume(s.clone());
            }
        }
        self.ui.chat_log = record
            .messages
            .iter()
            .map(|(who, text)| ChatMessage {
                role: match who.as_str() {
                    "you" => ChatRole::User,
                    "assistant" => ChatRole::Claude,
                    "tool" => ChatRole::Tool,
                    "error" => ChatRole::Error,
                    _ => ChatRole::Note,
                },
                text: text.clone(),
            })
            .collect();
        self.ui.chat_log.push(ChatMessage {
            role: ChatRole::Note,
            text: self.ui.catalog.get("chat-restored"),
        });
        self.chat_record = Some(record);
    }

    /// The conversation as a Markdown file, where the user says.
    fn export_chat(&mut self) {
        self.save_chat();
        let Some(record) = self.chat_record.clone() else {
            let message = self.ui.catalog.get("chat-nothing-to-export");
            self.status(&message);
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Markdown", &["md"])
            .set_file_name(format!(
                "Kreate chat {}.md",
                record.started.replace(':', "-")
            ))
            .save_file()
        else {
            return;
        };
        match std::fs::write(&path, kreate::chatlog::markdown(&record)) {
            Ok(()) => self.status(&path.display().to_string()),
            Err(e) => self.status(&e.to_string()),
        }
    }

    /// A sketch on the flat face under the pointer: its plane, the face's
    /// outline in it. Said, and nothing made, off every part or on a
    /// curved face.
    fn sketch_on_face_under_cursor(&mut self) {
        self.sketch_face_pick = false;
        let Some(ndc) = self.cursor_ndc() else { return };
        let aspect = self.viewport_aspect();
        self.evaluate_scene();
        let hit = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect);
        let Some(hit) = hit else {
            self.sketch_waiting = false;
            let message = self.ui.catalog.get("sketch-no-face");
            self.status(&message);
            return;
        };
        let name = self
            .doc
            .object(hit.object)
            .map(|o| o.name.clone())
            .unwrap_or_default();
        let mut params = PropertyMap::new();
        params.set("plane", PropertyValue::Enum(4));
        params.set("object", PropertyValue::Text(name));
        params.set("at", PropertyValue::Vec3(hit.point.to_array()));
        self.run_op("sketch.new", params);
        if kreate_ops::ops::sketch_draw::active_sketch(&self.doc).is_none() {
            self.sketch_waiting = false;
        }
    }

    /// Puts text where the pointer is: on the surface under it, lying on that
    /// face and reading along it level; past every surface, on the floor;
    /// and with the floor out of sight, on the plane through the 3D cursor
    /// facing the view. One `object.add_text`, its place and turn given, so
    /// it is one undo step and replays where it went.
    fn place_text_under_cursor(&mut self) {
        self.text_place = false;
        let Some(ndc) = self.cursor_ndc() else { return };
        let aspect = self.viewport_aspect();
        self.evaluate_scene();
        let eye = self.camera.eye();
        let dir = self.camera.ray_direction(ndc, aspect).as_dvec3();
        let hit = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect);
        let (at, normal) = match hit {
            Some(hit) => {
                let normal = self
                    .surface_normal(hit.object, hit.face)
                    .unwrap_or(DVec3::Z);
                // The face's front: text on the side the view is on.
                let normal = if normal.dot(dir) > 0.0 {
                    -normal
                } else {
                    normal
                };
                (hit.point, normal)
            }
            None if dir.z.abs() > 1e-9 && -eye.z / dir.z > 0.0 => {
                (eye + dir * (-eye.z / dir.z), DVec3::Z)
            }
            None => {
                let normal = self.camera.forward();
                let denom = dir.dot(normal);
                if denom.abs() < 1e-9 {
                    return;
                }
                let at = eye + dir * ((self.doc.cursor - eye).dot(normal) / denom);
                (at, -normal)
            }
        };
        let turn = work_plane_turn(normal);
        let mut params = PropertyMap::new();
        params.set("location", PropertyValue::Vec3(at.to_array()));
        params.set(
            "rotation",
            PropertyValue::Vec3(kreate_scene::animation::to_euler(turn).to_array()),
        );
        self.status("");
        self.run_op("object.add_text", params);
    }

    /// The outward normal, in the world, of face `face` of object `id`'s
    /// evaluated mesh — the face a pick names.
    fn surface_normal(&self, id: ObjectId, face: u32) -> Option<DVec3> {
        let mesh = &self.evaluated.iter().find(|(o, _)| *o == id)?.1;
        let corners = mesh.faces().nth(face as usize)?;
        let p = mesh.positions();
        let mut n = DVec3::ZERO;
        for i in 0..corners.len() {
            let a = p.get(corners[i] as usize)?.as_dvec3();
            let b = p.get(corners[(i + 1) % corners.len()] as usize)?.as_dvec3();
            n += a.cross(b);
        }
        let world = self.doc.world_transform(id);
        let n = world.matrix3.inverse().transpose() * n;
        (n.length() > 1e-12).then(|| n.normalize())
    }

    /// Shift+Right-click: the 3D cursor onto whatever is under the pointer, or
    /// onto the plane through where it is now, facing the view.
    fn place_cursor(&mut self) {
        let Some(ndc) = self.cursor_ndc() else { return };
        let aspect = self.viewport_aspect();
        self.evaluate_scene();
        let hit = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect);
        let at = match hit {
            Some(hit) => hit.point,
            None => {
                let eye = self.camera.eye();
                let dir = self.camera.ray_direction(ndc, aspect).as_dvec3();
                let normal = self.camera.forward();
                let denom = dir.dot(normal);
                if denom.abs() < 1e-9 {
                    return;
                }
                eye + dir * ((self.doc.cursor - eye).dot(normal) / denom)
            }
        };
        self.set_cursor(at);
    }

    fn set_cursor(&mut self, at: DVec3) {
        let mut params = PropertyMap::new();
        params.set("location", PropertyValue::Vec3(at.to_array()));
        self.run_op("scene.cursor_set", params);
    }

    /// The work plane's axes: the 3D cursor's X, Y and its normal Z.
    fn cursor_basis(&self) -> [Vec3; 3] {
        let q = self.doc.cursor_rotation.as_quat();
        [q * Vec3::X, q * Vec3::Y, q * Vec3::Z]
    }

    /// Puts the cursor somewhere and turns it: the work plane.
    fn place_cursor_turned(&mut self, at: DVec3, turn: kreate_math::DQuat) {
        let mut params = PropertyMap::new();
        params.set("location", PropertyValue::Vec3(at.to_array()));
        params.set("orient", PropertyValue::Bool(true));
        params.set(
            "rotation",
            PropertyValue::Vec3(kreate_scene::animation::to_euler(turn).to_array()),
        );
        self.run_op("scene.cursor_set", params);
    }

    /// Shift+S > Cursor to Selected. With faces selected in edit mode the
    /// cursor also turns to lie on them (Z their normal, X level where it
    /// can be), which is how a work plane is put on a face.
    fn cursor_to_selected(&mut self) {
        let Some(at) = self.selection_pivot() else {
            self.status("nothing selected");
            return;
        };
        let normal = self
            .op_ctx
            .mode
            .edits_elements()
            .then(|| kreate_ops::ops::mesh::selected_faces_normal(&self.doc))
            .flatten();
        match normal {
            Some(n) => {
                self.place_cursor_turned(at, work_plane_turn(n));
                self.status("cursor on the faces: the Cursor orientation works in their plane");
            }
            None => self.set_cursor(at),
        }
    }

    /// Shift+S ▸ Selection to Cursor: the selection moved so its middle is on
    /// the cursor, by the mode's own move — one step, and the move a script
    /// would write.
    fn selection_to_cursor(&mut self) {
        let Some(from) = self.selection_pivot() else {
            self.status("nothing selected");
            return;
        };
        let op = self.op_ctx.mode.info().transform[0];
        let delta = self.doc.cursor - from;
        let mut params = PropertyMap::new();
        params.set("translation", PropertyValue::Vec3(delta.to_array()));
        self.run_op(op, params);
    }

    /// Where the selection is and how big: the selected elements in edit mode,
    /// the selected objects' drawn geometry otherwise.
    fn selection_sphere(&self) -> Option<(DVec3, f64)> {
        let mut points: Vec<DVec3> = Vec::new();
        if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements {
            let weighing = kreate_ops::ops::mesh::weigh_edit_selection(&self.doc, 0.0)?;
            for part in &weighing.parts {
                let Some(mesh) = self.doc.mesh(part.target.mesh) else {
                    continue;
                };
                for (i, w) in part.indices.iter().zip(&part.weights) {
                    if *w >= 1.0 {
                        points.push(
                            part.target
                                .to_world
                                .transform_point3(mesh.positions()[*i as usize].as_dvec3()),
                        );
                    }
                }
            }
        } else {
            for id in &self.doc.selection.objects {
                let world = self.doc.world_transform(*id);
                match self.evaluated.iter().find(|(o, _)| o == id) {
                    Some((_, mesh)) => points.extend(
                        mesh.positions()
                            .iter()
                            .map(|p| world.transform_point3(p.as_dvec3())),
                    ),
                    None => points.push(world.translation),
                }
            }
        }
        if points.is_empty() {
            return None;
        }
        let lo = points.iter().fold(DVec3::splat(f64::MAX), |a, p| a.min(*p));
        let hi = points.iter().fold(DVec3::splat(f64::MIN), |a, p| a.max(*p));
        let centre = (lo + hi) * 0.5;
        // A lone point still gets a view a little way back from it.
        Some((centre, ((hi - lo).length() * 0.5).max(0.5)))
    }

    /// Numpad `.`: the view glides to frame the selection, or everything when
    /// nothing is selected.
    fn frame_selected(&mut self) {
        let Some((centre, radius)) = self.selection_sphere() else {
            self.frame_all();
            return;
        };
        let aspect = self
            .gfx
            .as_ref()
            .map(|g| g.viewport.aspect())
            .unwrap_or(16.0 / 9.0);
        let mut destination = self.camera;
        destination.frame_sphere(centre, radius, aspect);
        self.motion
            .animate_to(self.camera, destination, kreate_render::DEFAULT_TRANSITION);
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// `/`: only the selected objects, framed — and again, everything back
    /// where the view was.
    fn toggle_local_view(&mut self) {
        if let Some((_, before)) = self.local_view.take() {
            self.motion
                .animate_to(self.camera, before, kreate_render::DEFAULT_TRANSITION);
            self.status("");
            self.redraw.set(RedrawReason::SCENE_CHANGED);
            return;
        }
        let keep: Vec<_> = self.doc.selection.objects.clone();
        if keep.is_empty() {
            self.status("select something for local view");
            return;
        }
        let aside: Vec<_> = self
            .doc
            .objects()
            .map(|(id, _)| id)
            .filter(|id| !keep.contains(id))
            .collect();
        self.local_view = Some((aside, self.camera));
        self.evaluate_scene();
        self.frame_selected();
        self.status("Local View — / to leave");
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Numpad 0: the view through the scene camera — the active object when
    /// it is one, the first camera otherwise — and again, back.
    fn toggle_camera_view(&mut self) {
        // Already looking through one camera with another one active: go
        // through that one, as Blender's Ctrl+Numpad 0 does, rather than out.
        let other = self.doc.selection.active.filter(|a| {
            Some(*a) != self.camera_view_object
                && self.doc.object(*a).is_some_and(|o| {
                    matches!(o.data, kreate_scene::document::ObjectData::Camera(_))
                })
        });
        if let (Some(_), Some(object)) = (self.camera_view, other) {
            self.camera_view_object = Some(object);
            let destination = self.view_through(object);
            self.camera_view_pose = Some(destination);
            self.motion
                .animate_to(self.camera, destination, kreate_render::DEFAULT_TRANSITION);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
            return;
        }
        if let Some(before) = self.camera_view.take() {
            self.camera_view_object = None;
            self.camera_view_pose = None;
            self.motion
                .animate_to(self.camera, before, kreate_render::DEFAULT_TRANSITION);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
            return;
        }
        let Some(object) = self.scene_camera(None) else {
            self.status("the scene has no camera");
            return;
        };
        self.camera_view_object = Some(object);
        let destination = self.view_through(object);
        self.camera_view = Some(self.camera);
        self.camera_view_pose = Some(destination);
        self.motion
            .animate_to(self.camera, destination, kreate_render::DEFAULT_TRANSITION);
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// Keeps the camera view on the camera: as it plays an animation, is
    /// moved, or has its lens changed. Blender's rule for the other half: a
    /// view the user has moved away is no longer the camera's, and the camera
    /// view is left where they took it rather than pulled back.
    fn follow_camera_view(&mut self) {
        // A capture or a render places the view itself, and puts it back.
        if self.capturing || self.motion.is_animating() {
            return;
        }
        let Some(object) = self.camera_view_object else {
            return;
        };
        let leave = |app: &mut Self| {
            app.camera_view = None;
            app.camera_view_object = None;
            app.camera_view_pose = None;
        };
        if self.doc.object(object).is_none() {
            return leave(self);
        }
        let Some(pose) = self.camera_view_pose else {
            self.camera_view_pose = Some(self.camera);
            return;
        };
        if !same_view(&self.camera, &pose) {
            return leave(self);
        }
        // A marker cuts to another camera: the view cuts with it, as the
        // render will.
        let object = match self.doc.marker_camera(self.doc.playback.frame) {
            Some(cut) if cut != object => {
                self.camera_view_object = Some(cut);
                cut
            }
            _ => object,
        };
        let now = self.view_through(object);
        if !same_view(&now, &pose) {
            self.camera = now;
            self.motion.target = now;
            self.camera_view_pose = Some(now);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
    }

    /// The camera object to look through: the one called `name`; or else
    /// the one the markers cut to at this frame, or the scene camera; or
    /// else the active object when it is a camera, or else the first camera.
    fn scene_camera(&self, name: Option<&str>) -> Option<kreate_scene::document::ObjectId> {
        let is_camera = |id: kreate_scene::document::ObjectId| {
            self.doc
                .object(id)
                .is_some_and(|o| matches!(o.data, kreate_scene::document::ObjectData::Camera(_)))
        };
        if let Some(name) = name {
            return self
                .doc
                .objects()
                .find(|(id, o)| o.name == name && is_camera(*id))
                .map(|(id, _)| id);
        }
        if let Some(chosen) = self.doc.camera_at(self.doc.playback.frame) {
            return Some(chosen);
        }
        self.doc
            .selection
            .active
            .filter(|a| is_camera(*a))
            .or_else(|| {
                self.doc
                    .objects()
                    .map(|(id, _)| id)
                    .find(|id| is_camera(*id))
            })
    }

    /// The viewport camera that sees what a scene camera sees.
    fn view_through(&self, object: kreate_scene::document::ObjectId) -> kreate_render::Camera {
        let aspect = self
            .gfx
            .as_ref()
            .map(|g| g.viewport.aspect())
            .unwrap_or(16.0 / 9.0);
        let (_, rotation, eye) = self
            .doc
            .world_transform(object)
            .to_scale_rotation_translation();
        let lens = match self.doc.object(object).map(|o| &o.data) {
            Some(kreate_scene::document::ObjectData::Camera(c)) => {
                self.doc.camera(*c).map(|c| c.fov_y(aspect))
            }
            _ => None,
        };
        let mut destination = self.camera;
        destination.projection = kreate_render::Projection::Perspective;
        destination.rotation = rotation.normalize();
        destination.distance = self.camera.distance.max(1.0);
        destination.pivot = eye + destination.forward() * destination.distance;
        if let Some(fov) = lens.filter(|f| f.is_finite() && *f > 0.0) {
            destination.fov_y = fov;
        }
        destination
    }

    /// `C`: turns the selection brush on, or off again.
    fn toggle_circle_select(&mut self) {
        if self.circle.is_some() {
            self.end_circle_select();
            return;
        }
        if !matches!(
            self.op_ctx.mode.acts_on(),
            kreate_ops::mode::ActsOn::Objects | kreate_ops::mode::ActsOn::MeshElements
        ) {
            return;
        }
        let radius = self.circle_radius;
        self.circle = Some(CircleSelect {
            radius,
            stroke: None,
        });
        self.show_circle();
        self.status("Circle Select — drag to select, Shift+drag to deselect, wheel for size, Esc or right-click to stop");
    }

    fn show_circle(&mut self) {
        let ppp = self.egui_ctx.pixels_per_point().max(0.01);
        self.ui.circle_select = self.circle.as_ref().map(|c| c.radius / ppp);
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    fn end_circle_select(&mut self) {
        self.finish_circle_stroke();
        if let Some(c) = self.circle.take() {
            self.circle_radius = c.radius;
        }
        self.ui.circle_select = None;
        self.status("");
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// A press with the brush on: a stroke begins.
    fn begin_circle_stroke(&mut self) {
        let before: Vec<u32> = self
            .doc
            .selection
            .objects
            .iter()
            .map(|id| shell::object_key(*id))
            .collect();
        let deselect = self.input.shift_down || self.input.ctrl_down;
        // What an earlier drag left in the reduction is not this stroke's.
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.region.clear();
        }
        if let Some(c) = &mut self.circle {
            c.stroke = Some(CircleStroke {
                deselect,
                elements: Default::default(),
                session: None,
                before,
            });
        }
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// Adds what the brush has covered since the last frame to the stroke,
    /// and applies it — live, so the selection grows under the brush.
    fn drive_circle_stroke(&mut self) {
        let Some(mut resolved) = self
            .gfx
            .as_ref()
            .and_then(|g| g.region.resolved())
            .map(|e| e.to_vec())
        else {
            return;
        };
        let edits = self.op_ctx.mode.edits_elements();
        if self.xray {
            if let (Some((x, y)), Some(radius)) = (
                self.cursor_viewport_px(),
                self.circle.as_ref().map(|c| c.radius),
            ) {
                let brush = kreate_render::SelectShape::Circle {
                    centre: (x as f32, y as f32),
                    radius,
                };
                resolved.extend(if edits {
                    self.xray_region_elements(&brush)
                } else {
                    self.xray_region_objects(&brush)
                });
            }
        }
        let mode = match self.pick_domain() {
            kreate_render::PickDomain::Vert => 0u32,
            kreate_render::PickDomain::Edge => 1,
            _ => 2,
        };
        let Some(stroke) = self.circle.as_mut().and_then(|c| c.stroke.as_mut()) else {
            return;
        };
        let before = stroke.elements.len();
        stroke.elements.extend(resolved);
        if stroke.elements.len() == before {
            return;
        }
        if !edits {
            // Objects: the selection the stroke began with, plus or minus
            // everything the brush has passed over.
            let mut objects: Vec<u32> = if stroke.deselect {
                stroke
                    .before
                    .iter()
                    .copied()
                    .filter(|k| !stroke.elements.contains(k))
                    .collect()
            } else {
                let mut all = stroke.before.clone();
                all.extend(
                    stroke
                        .elements
                        .iter()
                        .copied()
                        .filter(|k| !stroke.before.contains(k)),
                );
                all
            };
            objects.dedup();
            let active = objects.last().copied();
            self.ui
                .queue
                .push(kreate_ui::Command::SetSelection { objects, active });
            self.redraw.set(RedrawReason::SCENE_CHANGED);
            return;
        }
        let mut params = PropertyMap::new();
        params.set(
            "elements",
            PropertyValue::Ints(stroke.elements.iter().copied().collect()),
        );
        params.set("mode", PropertyValue::Enum(mode));
        params.set("extend", PropertyValue::Bool(true));
        params.set("deselect", PropertyValue::Bool(stroke.deselect));
        match stroke.session.as_mut() {
            Some(session) => {
                if let Err(e) = self.history.update_modal(
                    &self.registry,
                    &mut self.doc,
                    Some(&mut self.edit),
                    &self.op_ctx,
                    session,
                    &params,
                ) {
                    tracing::warn!(error = %e, "circle select failed");
                }
            }
            None => {
                // Begun on the first stroke that changes anything, so a stroke
                // over what was already selected leaves no step behind.
                if let Ok(Some(session)) = self.history.begin_modal(
                    &self.registry,
                    &mut self.doc,
                    Some(&mut self.edit),
                    &self.op_ctx,
                    "mesh.select_set",
                    &params,
                ) {
                    stroke.session = Some(session);
                }
            }
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// The button comes up: the stroke is one step to undo.
    fn finish_circle_stroke(&mut self) {
        let Some(stroke) = self.circle.as_mut().and_then(|c| c.stroke.take()) else {
            return;
        };
        if let Some(session) = stroke.session {
            self.history.commit_modal(session);
            if let Some(entry) = self.history.last() {
                println!("{}", entry.to_script_preview(kreate::shell::SCRIPT_PREVIEW));
            }
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// `K`: from here until Enter, each click adds a point to the cut.
    /// Runs every Mesh Check on the active mesh, as it is in the document
    /// (what selecting will select), in the world, and keeps the counts for
    /// the Object tab.
    fn run_mesh_check(&mut self, thickness: f64) {
        let Some(id) = self.doc.selection.active else {
            return;
        };
        let Some(mesh) = self
            .doc
            .object(id)
            .and_then(|o| o.mesh())
            .and_then(|m| self.doc.mesh(m))
        else {
            return;
        };
        let to_world = self.doc.world_transform(id);
        let r = kreate_mesh::check::check(mesh, |p| to_world.transform_point3(p), thickness, true);
        let counts = [
            r.open_edges.len(),
            r.non_manifold_edges.len(),
            r.flipped_edges.len(),
            r.degenerate_faces.len(),
            r.loose_verts.len(),
            r.intersecting_faces.len(),
            r.thin_faces.len(),
        ];
        self.ui.mesh_check = Some((id, counts));
        let total: usize = counts.iter().sum();
        self.status(&if total == 0 {
            "mesh check: no problems".to_string()
        } else {
            format!("mesh check: {total} problem elements")
        });
        self.redraw.set(RedrawReason::UI);
    }

    /// The bill of materials: each part, how many there are of it (objects
    /// sharing a mesh are one part used more than once), its size, volume and
    /// mass by what each is made of — as CSV, in the scene's unit.
    fn parts_list(&self) -> String {
        let unit = self.doc.design.length_unit;
        let symbol = unit.symbol();
        /// One part of the list.
        struct Row {
            name: String,
            count: usize,
            size: [f64; 3],
            volume: f64,
            faces: usize,
            material: String,
            density: f64,
        }
        let mut rows: Vec<Row> = Vec::new();
        let mut by_mesh: Vec<(usize, usize)> = Vec::new();
        for (name, mesh, world) in self.printable_parts() {
            let key = Arc::as_ptr(&mesh) as usize;
            if let Some((_, row)) = by_mesh.iter().find(|(k, _)| *k == key) {
                rows[*row].count += 1;
                continue;
            }
            let points: Vec<DVec3> = mesh
                .positions()
                .iter()
                .map(|p| world.transform_point3(p.as_dvec3()))
                .collect();
            let (lo, hi) = points.iter().fold(
                (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
                |(lo, hi), p| (lo.min(*p), hi.max(*p)),
            );
            let size = if points.is_empty() {
                DVec3::ZERO
            } else {
                hi - lo
            };
            let mass = kreate_mesh::measure::mass_properties(&mesh, |p| world.transform_point3(p));
            // What it is made of: its first material, and that material's
            // density or the scene's.
            let object = self
                .doc
                .objects()
                .find(|(_, o)| o.name == name)
                .map(|(id, _)| id);
            let material = object
                .and_then(|id| self.doc.object(id)?.mesh())
                .and_then(|m| self.doc.mesh_slots(m).iter().flatten().next().copied())
                .and_then(|m| self.doc.material(m))
                .map(|m| m.name.clone())
                .unwrap_or_default();
            let density = object.map_or(self.doc.design.density, |id| self.doc.density_of(id));
            by_mesh.push((key, rows.len()));
            rows.push(Row {
                name,
                count: 1,
                size: size.to_array(),
                volume: mass.volume.abs(),
                faces: mesh.num_faces() as usize,
                material,
                density,
            });
        }
        let per = unit.metres();
        let mut out = format!(
            "Part,Quantity,Material,Size X ({symbol}),Size Y ({symbol}),Size Z ({symbol}),Volume ({symbol}3),Mass each (kg),Faces
"
        );
        let quote = |text: String| {
            if text.contains([',', '"']) {
                format!("\"{}\"", text.replace('"', "\"\""))
            } else {
                text
            }
        };
        for Row {
            name,
            count,
            size,
            volume,
            faces,
            material,
            density,
        } in rows
        {
            let name = quote(name);
            let material = quote(material);
            out.push_str(&format!(
                "{name},{count},{material},{:.4},{:.4},{:.4},{:.4},{:.6},{faces}
",
                size[0] / per,
                size[1] / per,
                size[2] / per,
                volume / per.powi(3),
                volume * density,
            ));
        }
        out
    }

    /// File > Export Drawing: the shown parts seen square from one side, as
    /// a DXF or an SVG drawing in the scene's unit, hidden lines dashed.
    fn export_drawing(&mut self, view: u32) {
        let view = kreate_io::drawing::View::ALL[(view as usize).min(2)];
        let name = match view {
            kreate_io::drawing::View::Top => "top",
            kreate_io::drawing::View::Front => "front",
            kreate_io::drawing::View::Right => "right",
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("DXF drawing", &["dxf"])
            .add_filter("SVG drawing", &["svg"])
            .set_title("Export Drawing")
            .set_file_name(format!("{name}.dxf"))
            .save_file()
        else {
            return;
        };
        let _ = self.write_drawing(&path, view);
    }

    fn write_drawing(
        &mut self,
        path: &std::path::Path,
        view: kreate_io::drawing::View,
    ) -> Result<(), String> {
        let parts = self.printable_parts();
        let refs: Vec<(&MeshData, glam::DAffine3)> = parts
            .iter()
            .map(|(_, mesh, world)| (&**mesh, *world))
            .collect();
        let lines = kreate_io::drawing::project_drawing(&refs, view, 30f64.to_radians());
        let unit = self.doc.design.length_unit;
        // The overall width and height, written as the scene writes lengths.
        let dims = kreate_io::drawing::overall_dimensions(&lines, kreate_reflect::format_length);
        let svg = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("svg"));
        let text = if svg {
            let suffix = match unit.symbol() {
                "m" | "ft" => "",
                s => s,
            };
            kreate_io::drawing::write_svg_with(&lines, &dims, unit.metres(), suffix)
        } else {
            kreate_io::drawing::write_dxf_with(&lines, &dims, unit.metres())
        };
        match std::fs::write(path, text) {
            Ok(()) => {
                self.status(&format!("drawing written: {} lines", lines.len()));
                Ok(())
            }
            Err(e) => {
                let message = format!("cannot write the drawing: {e}");
                self.status(&message);
                Err(message)
            }
        }
    }

    /// Every shown mesh object as it stands, for a print: its evaluated mesh
    /// (modifiers applied, not cut by a section view) and where it is.
    fn printable_parts(&self) -> Vec<(String, Arc<MeshData>, glam::DAffine3)> {
        self.evaluated
            .iter()
            .filter(|(id, _)| self.doc.is_shown(*id))
            .filter_map(|(id, mesh)| {
                let object = self.doc.object(*id)?;
                object.mesh()?;
                let whole = self
                    .sectioned
                    .iter()
                    .find(|(_, _, out)| Arc::ptr_eq(out, mesh))
                    .map(|(input, _, _)| input.clone())
                    .unwrap_or_else(|| mesh.clone());
                Some((object.name.clone(), whole, self.doc.world_transform(*id)))
            })
            .collect()
    }

    /// The section view: every evaluated mesh but the ones being edited cut
    /// by the plane and capped, in its own space. Kept per input and plane,
    /// so an unchanged scene is not cut again every frame.
    fn cut_sections(&mut self, editing: &[ObjectId]) {
        let Some(section) = self.doc.design.section else {
            self.sectioned.clear();
            return;
        };
        let (point, normal) = section.plane(self.doc.cursor, self.doc.cursor_rotation);
        let mut kept = Vec::new();
        for (id, mesh) in self.evaluated.iter_mut() {
            if editing.contains(id) {
                continue;
            }
            let world = self.doc.world_transform(*id);
            let local_point = world.inverse().transform_point3(point);
            let local_normal = (world.matrix3.transpose() * normal).normalize_or_zero();
            let key = [
                local_point.x,
                local_point.y,
                local_point.z,
                local_normal.x,
                local_normal.y,
                local_normal.z,
                f64::from(u8::from(section.cap)),
            ];
            let out = self
                .sectioned
                .iter()
                .find(|(input, k, _)| Arc::ptr_eq(input, mesh) && *k == key)
                .map(|(_, _, out)| out.clone())
                .unwrap_or_else(|| {
                    Arc::new(kreate_mesh::section::section(
                        mesh,
                        local_point.as_vec3(),
                        local_normal.as_vec3(),
                        section.cap,
                    ))
                });
            kept.push((mesh.clone(), key, out.clone()));
            *mesh = out;
        }
        self.sectioned = kept;
    }

    /// The active mesh object's volume, surface and middle, for the Object
    /// tab: as evaluated (modifiers included) and in the world, so it is what
    /// is on screen. Not during a sculpt stroke, which changes the mesh every
    /// frame; the stroke's end measures it again.
    fn publish_mass(&mut self) {
        if self.sculpt_stroke.is_some() {
            return;
        }
        let found = self.doc.selection.active.and_then(|id| {
            self.evaluated
                .iter()
                .find(|(o, _)| *o == id)
                .map(|(_, m)| (id, m.clone()))
        });
        // A section view cuts what is drawn, not the part: measure the whole.
        let found = found.map(|(id, m)| {
            let whole = self
                .sectioned
                .iter()
                .find(|(_, _, out)| Arc::ptr_eq(out, &m))
                .map(|(input, _, _)| input.clone());
            (id, whole.unwrap_or(m))
        });
        let Some((id, mesh)) = found else {
            self.mass_cache = None;
            self.ui.mass = None;
            return;
        };
        let to_world = self.doc.world_transform(id);
        let key = (Arc::as_ptr(&mesh) as usize, to_world.to_cols_array());
        let props = match &self.mass_cache {
            Some((ptr, cols, props)) if (*ptr, *cols) == key => *props,
            _ => {
                let props =
                    kreate_mesh::measure::mass_properties(&mesh, |p| to_world.transform_point3(p));
                self.mass_cache = Some((key.0, key.1, props));
                props
            }
        };
        self.ui.mass = (mesh.num_faces() > 0).then_some(props);
    }

    /// The face under the pointer: whose it is, its middle and its outward
    /// normal, in the world, as it is drawn.
    fn face_under_pointer(&mut self) -> Option<(ObjectId, DVec3, DVec3)> {
        let ndc = self.cursor_ndc()?;
        let aspect = self.viewport_aspect();
        let hit = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect)?;
        let mesh = self
            .evaluated
            .iter()
            .find(|(id, _)| *id == hit.object)?
            .1
            .clone();
        let world = self.doc.world_transform(hit.object);
        let ring = mesh.face_verts(hit.face);
        let points: Vec<DVec3> = ring
            .iter()
            .map(|v| world.transform_point3(mesh.positions()[*v as usize].as_dvec3()))
            .collect();
        let mut normal = DVec3::ZERO;
        for i in 0..points.len() {
            normal += points[i].cross(points[(i + 1) % points.len()]);
        }
        let middle = points.iter().sum::<DVec3>() / points.len().max(1) as f64;
        Some((hit.object, middle, normal.normalize_or_zero()))
    }

    /// A click with Align Faces: the first picks the moving part's face, the
    /// second the face it goes against — and the part goes.
    fn mate_click(&mut self) {
        let Some(picked) = self.face_under_pointer() else {
            self.status("Align Faces: click on a face");
            return;
        };
        match self.mate {
            Some(None) => {
                self.mate = Some(Some(picked));
                self.status("Align Faces: now the face it goes against");
            }
            Some(Some((moving, from, from_normal))) => {
                if picked.0 == moving {
                    self.status("Align Faces: pick a face of another part");
                    return;
                }
                let Some(uid) = self.doc.object_uid(moving) else {
                    return;
                };
                let mut p = PropertyMap::new();
                p.set("uid", PropertyValue::Text(uid.raw().to_string()));
                p.set("from", PropertyValue::Vec3(from.to_array()));
                p.set("from_normal", PropertyValue::Vec3(from_normal.to_array()));
                p.set("to", PropertyValue::Vec3(picked.1.to_array()));
                p.set("to_normal", PropertyValue::Vec3(picked.2.to_array()));
                if let Some(target) = self.doc.object_uid(picked.0) {
                    p.set("target", PropertyValue::Text(target.raw().to_string()));
                }
                self.run_op("object.mate", p);
                self.mate = None;
            }
            None => {}
        }
    }

    /// View ▸ Measure: click two points for the distance between them, or
    /// hold Ctrl on the second for the corner of an angle and click a third.
    /// Points catch on vertices, edges and faces of anything shown; each
    /// finished measurement stays on the scene as a dimension.
    fn begin_measure(&mut self) {
        if self.gesture.is_some() || self.measure.is_some() {
            return;
        }
        self.end_loop_cut_hover();
        let mut world = kreate::snap::SnapWorld::default();
        for (id, mesh) in &self.evaluated {
            if self.doc.is_shown(*id) {
                world.add_mesh(mesh, &self.doc.world_transform(*id), &[]);
            }
        }
        self.measure = Some(MeasureTool {
            world,
            points: Vec::new(),
            hover: None,
            angle: false,
        });
        self.status("Measure: click two points (Ctrl on the second for an angle), Esc to stop");
        self.update_measure();
    }

    /// Where the next point would go: what the pointer is near, or failing
    /// that the point under it on the plane facing the view through the last
    /// point taken (the 3D cursor for the first).
    fn update_measure(&mut self) {
        let Some(mut tool) = self.measure.take() else {
            return;
        };
        let size = self.gfx.as_ref().map(|g| g.viewport.render_size());
        let caught = match (self.cursor_viewport_px(), size) {
            (Some((x, y)), Some((w, h))) => kreate::snap::resolve(
                &tool.world,
                kreate::snap::Mode::Element,
                (x as f32, y as f32),
                &self.camera,
                (w as f32, h as f32),
            )
            .map(|hit| hit.point),
            _ => None,
        };
        tool.hover = caught.or_else(|| {
            let ndc = self.cursor_ndc()?;
            let eye = self.camera.eye();
            let dir = self
                .camera
                .ray_direction(ndc, self.viewport_aspect())
                .as_dvec3();
            let n = self.camera.forward();
            let through = tool.points.last().copied().unwrap_or(self.doc.cursor);
            let denom = dir.dot(n);
            (denom.abs() > 1e-9).then(|| eye + dir * ((through - eye).dot(n) / denom))
        });
        let mut preview = tool.points.clone();
        preview.extend(tool.hover);
        self.ui.measure_preview = preview;
        self.measure = Some(tool);
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// A click with the Measure tool: a point taken, and the measurement left
    /// on the scene once it has its two points — or three, for an angle.
    fn measure_click(&mut self) {
        let ctrl = self.input.ctrl_down;
        let Some(tool) = &mut self.measure else {
            return;
        };
        let Some(at) = tool.hover else { return };
        tool.points.push(at);
        if tool.points.len() == 2 && ctrl {
            tool.angle = true;
        }
        let done = match tool.points.len() {
            2 => !tool.angle,
            3 => true,
            _ => false,
        };
        if done {
            let points: Vec<f64> = tool.points.iter().flat_map(|p| p.to_array()).collect();
            tool.points.clear();
            tool.angle = false;
            self.run_op_with("measure.add", &[("points", PropertyValue::Floats(points))]);
        }
        self.update_measure();
    }

    fn end_measure(&mut self, status: &str) {
        self.measure = None;
        self.ui.measure_preview.clear();
        if !status.is_empty() {
            self.status(status);
        }
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    // --- sketching -----------------------------------------------------------

    /// The active sketch and where it is in the world.
    fn sketch_target(&self) -> Option<(kreate_scene::document::ObjectId, glam::DAffine3)> {
        let id = kreate_ops::ops::sketch_draw::active_sketch(&self.doc)?;
        Some((id, self.doc.world_transform(id)))
    }

    /// The ray under the pointer: from the eye in perspective, from the view's
    /// plane straight on in orthographic.
    fn cursor_ray(&self) -> Option<(DVec3, DVec3)> {
        let ndc = self.cursor_ndc()?;
        let aspect = self.viewport_aspect();
        match self.camera.projection {
            kreate_render::Projection::Perspective => {
                let d = self.camera.ray_direction(ndc, aspect);
                Some((self.camera.eye(), d.as_dvec3().normalize()))
            }
            kreate_render::Projection::Orthographic => {
                let h = self.camera.ortho_height() * 0.5;
                let origin = self.camera.eye()
                    + self.camera.right() * (ndc.x as f64 * h * aspect)
                    + self.camera.up() * (ndc.y as f64 * h);
                Some((origin, self.camera.forward()))
            }
        }
    }

    /// What of the active curve is under the pointer: its spline, point,
    /// and which of the point's three places — the point itself or a
    /// handle — within `radius` on screen.
    fn curve_pick(&self, radius: f64) -> Option<(u32, u32, CurvePart)> {
        if self.op_ctx.mode != OpMode::RESTING {
            return None;
        }
        let id = kreate_ops::ops::curve::active_curve(&self.doc)?;
        let curve = self.doc.object(id)?.curve.as_ref()?;
        let world = self.doc.world_transform(id);
        let projector = self.ui.projector?;
        let (x, y) = self.input.cursor?;
        let ppp = self.egui_ctx.pixels_per_point() as f64;
        let at = glam::DVec2::new(x / ppp, y / ppp);
        let screen = |p: DVec3| {
            projector
                .project(world.transform_point3(p))
                .map(|q| glam::DVec2::new(q.x as f64, q.y as f64))
        };
        let mut best: Option<((u32, u32, CurvePart), f64)> = None;
        for (s, spline) in curve.splines.iter().enumerate() {
            for (i, p) in spline.points.iter().enumerate() {
                let mut places = vec![(CurvePart::Point, p.position)];
                if spline.kind == kreate_scene::curve::SplineKind::Bezier {
                    places.push((CurvePart::Left, p.left));
                    places.push((CurvePart::Right, p.right));
                }
                for (part, place) in places {
                    let Some(sp) = screen(place) else { continue };
                    let d = sp.distance(at);
                    // A handle on its point: the point wins, as the square is
                    // drawn over the ring.
                    let d = if part == CurvePart::Point { d - 0.5 } else { d };
                    if d <= radius && best.is_none_or(|b| d < b.1) {
                        best = Some(((s as u32, i as u32, part), d));
                    }
                }
            }
        }
        best.map(|b| b.0)
    }

    /// The curve's points and handles: left takes hold of one and drags it
    /// across the view; right or Esc puts it back.
    fn curve_mouse(&mut self, button: MouseButton, pressed: bool) -> bool {
        match (button, pressed) {
            (MouseButton::Left, true) if self.press_belongs_to_viewport() => {
                let Some((spline, index, part)) = self.curve_pick(8.0) else {
                    return false;
                };
                let Some(id) = kreate_ops::ops::curve::active_curve(&self.doc) else {
                    return false;
                };
                let Some(point) = self
                    .doc
                    .object(id)
                    .and_then(|o| o.curve.as_ref())
                    .and_then(|c| c.splines.get(spline as usize))
                    .and_then(|s| s.points.get(index as usize))
                    .copied()
                else {
                    return false;
                };
                let world = self.doc.world_transform(id);
                let grabbed = world.transform_point3(match part {
                    CurvePart::Point => point.position,
                    CurvePart::Left => point.left,
                    CurvePart::Right => point.right,
                });
                // Where the pointer's ray meets the plane facing the view
                // through what was grabbed: the drag is measured from there.
                let Some(from) = self.ray_on_plane(grabbed) else {
                    return false;
                };
                self.curve_drag = Some(CurveDrag {
                    object: id,
                    spline,
                    index,
                    part,
                    point,
                    plane: grabbed,
                    from,
                    session: None,
                });
                true
            }
            (MouseButton::Left, false) => {
                let Some(drag) = self.curve_drag.take() else {
                    return false;
                };
                if let Some(session) = drag.session {
                    self.history.commit_modal(session);
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
                true
            }
            (MouseButton::Right, true) => self.cancel_curve_drag(),
            _ => false,
        }
    }

    /// Where the pointer's ray meets the plane facing the view through `at`.
    fn ray_on_plane(&self, at: DVec3) -> Option<DVec3> {
        let (o, d) = self.cursor_ray()?;
        let n = self.camera.forward();
        let denom = d.dot(n);
        if denom.abs() < 1e-12 {
            return None;
        }
        let t = (at - o).dot(n) / denom;
        Some(o + d * t)
    }

    /// Moves what is dragged under the pointer, live, as one step.
    fn drive_curve_drag(&mut self) {
        let Some(drag) = self.curve_drag.as_ref() else {
            return;
        };
        let (object, spline, index, part, point, plane, from, begun) = (
            drag.object,
            drag.spline,
            drag.index,
            drag.part,
            drag.point,
            drag.plane,
            drag.from,
            drag.session.is_some(),
        );
        let Some(now) = self.ray_on_plane(plane) else {
            return;
        };
        let to_local = self.doc.world_transform(object).inverse();
        let delta = to_local.transform_vector3(now - from);
        if !begun && delta.length() < 1e-9 {
            return;
        }
        let mut p = point;
        match part {
            CurvePart::Point => p.translate(delta),
            CurvePart::Left => p.left += delta,
            CurvePart::Right => p.right += delta,
        }
        let params = kreate_reflect::OpParams::to_map(&kreate_ops::ops::curve::point_params(
            spline, index, &p,
        ));
        let Some(stored) = self.curve_drag.as_mut() else {
            return;
        };
        match stored.session.as_mut() {
            Some(session) => {
                if let Err(e) = self.history.update_modal(
                    &self.registry,
                    &mut self.doc,
                    None,
                    &self.op_ctx,
                    session,
                    &params,
                ) {
                    tracing::warn!(error = %e, "curve drag failed");
                }
            }
            None => match self.history.begin_modal(
                &self.registry,
                &mut self.doc,
                None,
                &self.op_ctx,
                "curve.point_set",
                &params,
            ) {
                Ok(Some(session)) => stored.session = Some(session),
                Ok(None) => {}
                Err(e) => tracing::warn!(error = %e, "curve drag failed"),
            },
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Puts a dragged point back where it was taken.
    fn cancel_curve_drag(&mut self) -> bool {
        let Some(drag) = self.curve_drag.take() else {
            return false;
        };
        if let Some(session) = drag.session {
            History::cancel_modal(session, &mut self.doc);
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
        true
    }

    /// The pointer on the active sketch's plane, in the sketch's own space.
    fn sketch_plane_point(&self) -> Option<glam::DVec2> {
        let (_, world) = self.sketch_target()?;
        let (o, d) = self.cursor_ray()?;
        let back = world.inverse();
        let (o, d) = (back.transform_point3(o), back.transform_vector3(d));
        if d.z.abs() < 1e-12 {
            return None;
        }
        let t = -o.z / d.z;
        let p = o + d * t;
        Some(glam::DVec2::new(p.x, p.y))
    }

    /// The pointer in interface points, as the projector puts things.
    fn sketch_cursor(&self) -> Option<glam::DVec2> {
        let (x, y) = self.input.cursor?;
        let ppp = self.egui_ctx.pixels_per_point() as f64;
        Some(glam::DVec2::new(x / ppp, y / ppp))
    }

    /// The active sketch as the tools see it.
    fn sketch_geometry(&self) -> Option<kreate::sketching::Geometry> {
        let (id, world) = self.sketch_target()?;
        let projector = self.ui.projector?;
        let object = self.doc.object(id)?;
        let points = kreate_ops::ops::sketch_draw::points(&self.doc, id);
        Some(kreate::sketching::Geometry::new(
            points,
            object.sketch_lines.clone(),
            &object.sketch,
            |p| {
                projector
                    .project(world.transform_point3(DVec3::new(p.x, p.y, 0.0)))
                    .map(|q| glam::DVec2::new(q.x as f64, q.y as f64))
            },
        ))
    }

    /// The pointer on the plane, caught by a point of the sketch near it.
    fn sketch_hover(&mut self) {
        let Some(on_plane) = self.sketch_plane_point() else {
            self.sketching.hover = None;
            self.sketching.trim = None;
            return;
        };
        if self.sketching.tool == kreate::sketching::Tool::Trim {
            let trim = self
                .sketch_cursor()
                .zip(self.sketch_geometry())
                .and_then(|(at, g)| g.line_near(at, 8.0))
                .map(|line| (line, on_plane));
            if trim != self.sketching.trim {
                self.sketching.trim = trim;
                self.redraw.set(RedrawReason::UI);
            }
        }
        let pointer = self.sketch_cursor();
        if let (Some(band), Some(at)) = (self.sketching.band.as_mut(), pointer) {
            band[1] = at;
        }
        let caught = self
            .sketch_cursor()
            .zip(self.sketch_snap_geometry())
            .and_then(|(at, g)| Some((g.point_near(at, 10.0)?, g)));
        // A point of the sketch first, always: drawing from it joins it.
        // Then, as always, the other sketches' points and lines, as Fusion
        // catches them. Then, with snapping on — the magnet, or Ctrl the
        // other way — what the snap mode names.
        let snapping = self.snap.enabled != self.input.ctrl_down;
        let (hover, snapped) = match caught {
            Some((v, g)) => ((g.points[v as usize], v + 1), false),
            None => match self.sketch_other_snap() {
                Some(p) => ((p, 0), true),
                None => match snapping.then(|| self.sketch_snap(on_plane)).flatten() {
                    Some(p) => ((p, 0), true),
                    None => ((on_plane, 0), false),
                },
            },
        };
        // Drawing an outline, back at where it began: caught there, before
        // anything else. The first point is not in the sketch until the
        // outline is, so nothing above can find it — and an outline begun
        // in the open could not be closed.
        let first = self.sketching.clicks.first().copied();
        let closing = matches!(
            self.sketching.tool,
            kreate::sketching::Tool::Line | kreate::sketching::Tool::Spline
        ) && self.sketching.clicks.len() >= 3
            && first.is_some_and(|(p, _)| {
                self.sketch_screen(p)
                    .zip(self.sketch_cursor())
                    .is_some_and(|(s, at)| s.distance(at) <= 10.0)
            });
        let (hover, snapped) = match (closing, first) {
            (true, Some(first)) => (first, true),
            _ => (hover, snapped),
        };
        self.sketching.closing = closing;
        self.sketching.hover = Some(hover);
        self.sketching.snapped = snapped;
        // A drag follows: where the pointer is, snapped as drawing is.
        if self.sketching.drag.is_some() {
            self.drive_sketch_drag(hover.0);
        }
        self.redraw.set(RedrawReason::UI);
    }

    /// The other sketches' points and lines near the pointer, laid onto this
    /// sketch's plane: a point within reach first, then the nearest place
    /// on a line. Whatever the snap toggle says — they are drawing too, and
    /// what a new line most often starts from.
    fn sketch_other_snap(&self) -> Option<glam::DVec2> {
        const REACH: f64 = 10.0;
        let (id, world) = self.sketch_target()?;
        let at = self.sketch_cursor()?;
        let projector = self.ui.projector?;
        let screen = |p: DVec3| {
            projector
                .project(p)
                .map(|q| glam::DVec2::new(q.x as f64, q.y as f64))
        };
        let to_plane = |p: DVec3| {
            let local = world.inverse().transform_point3(p);
            glam::DVec2::new(local.x, local.y)
        };
        let points = kreate::sketching::other_points(&self.doc, Some(id));
        let near = points
            .iter()
            .filter_map(|p| Some((*p, screen(*p)?.distance(at))))
            .filter(|(_, d)| *d <= REACH)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((p, _)) = near {
            return Some(to_plane(p));
        }
        let mut best: Option<(DVec3, f64)> = None;
        for [a, b] in kreate::sketching::other_lines(&self.doc, Some(id)) {
            let (Some(sa), Some(sb)) = (screen(a), screen(b)) else {
                continue;
            };
            let d = sb - sa;
            let t = if d.length_squared() > 1e-12 {
                ((at - sa).dot(d) / d.length_squared()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let gap = (sa + d * t).distance(at);
            if gap <= REACH && best.is_none_or(|(_, g)| gap < g) {
                best = Some((a.lerp(b, t), gap));
            }
        }
        best.map(|(p, _)| to_plane(p))
    }

    /// Where drawing snaps to near `on_plane`, by the snap mode: the grid
    /// for increment; otherwise the sketch's own middles and crossings, the
    /// rest of the scene — other sketches, meshes, bodies — laid onto the
    /// sketch's plane, and, for edges, a point along a line.
    fn sketch_snap(&mut self, on_plane: glam::DVec2) -> Option<glam::DVec2> {
        use kreate::snap::Mode;
        let mode = self.snap.mode;
        if !mode.needs_geometry() {
            let step = self.snap.length;
            if step <= 0.0 {
                return None;
            }
            return Some((on_plane / step).round() * step);
        }
        let (id, world) = self.sketch_target()?;
        let ppp = self.egui_ctx.pixels_per_point() as f64;
        let at = self.sketch_cursor()?;
        let g = self.sketch_snap_geometry()?;
        let projector = self.ui.projector?;
        let screen = |p: glam::DVec2| {
            projector
                .project(world.transform_point3(DVec3::new(p.x, p.y, 0.0)))
                .map(|q| glam::DVec2::new(q.x as f64, q.y as f64))
        };
        const REACH: f64 = 10.0;
        let mut best: Option<(glam::DVec2, f64)> = None;
        let offer = |p: glam::DVec2, d: f64, best: &mut Option<(glam::DVec2, f64)>| {
            if d <= REACH && best.is_none_or(|b| d < b.1) {
                *best = Some((p, d));
            }
        };
        // The sketch's own: a line's middle, two lines' crossing.
        if matches!(mode, Mode::Element | Mode::Vertex | Mode::EdgeMidpoint) {
            for p in g.snap_points() {
                if let Some(s) = screen(p) {
                    offer(p, s.distance(at), &mut best);
                }
            }
        }
        // The rest of the scene, laid onto the plane.
        let size = self.gfx.as_ref().map(|g| g.viewport.render_size());
        if let (Some((x, y)), Some((w, h))) = (self.cursor_viewport_px(), size) {
            let scene = self.sketch_snap_world(id, mode);
            if let Some(hit) = kreate::snap::resolve(
                &scene,
                mode,
                (x as f32, y as f32),
                &self.camera,
                (w as f32, h as f32),
            ) {
                let local = world.inverse().transform_point3(hit.point);
                offer(
                    glam::DVec2::new(local.x, local.y),
                    hit.distance_px as f64 / ppp,
                    &mut best,
                );
            }
        }
        if let Some((p, _)) = best {
            return Some(p);
        }
        // Last, a point along one of the sketch's lines.
        if matches!(mode, Mode::Element | Mode::Edge) {
            return g.on_line(at, on_plane, REACH);
        }
        None
    }

    /// The scene's snap candidates for drawing in sketch `id`: everything
    /// shown but the sketch itself, with the other sketches' lines. Kept
    /// while the scene and the mode stay as they are.
    fn sketch_snap_world(
        &mut self,
        id: ObjectId,
        mode: kreate::snap::Mode,
    ) -> Arc<kreate::snap::SnapWorld> {
        let version = self.doc.version();
        if let Some((v, m, world)) = &self.sketch_snap_cache {
            // A drag changes the sketch every move, and nothing else: the
            // scene it snaps to stays as it was gathered.
            if (*v == version || self.sketching.drag.is_some()) && *m == mode {
                return world.clone();
            }
        }
        let mut world = self.build_snap_world(mode);
        for [a, b] in kreate::sketching::other_lines(&self.doc, Some(id)) {
            world.verts.push(a);
            world.verts.push(b);
            world.edges.push([a, b]);
        }
        let world = Arc::new(world);
        self.sketch_snap_cache = Some((version, mode, world.clone()));
        world
    }

    /// A point of the active sketch's plane on screen, in interface points.
    fn sketch_screen(&self, p: glam::DVec2) -> Option<glam::DVec2> {
        let (_, world) = self.sketch_target()?;
        let projector = self.ui.projector?;
        projector
            .project(world.transform_point3(DVec3::new(p.x, p.y, 0.0)))
            .map(|q| glam::DVec2::new(q.x as f64, q.y as f64))
    }

    /// The sketch as what is dragged can snap to: all of it but what moves.
    fn sketch_snap_geometry(&self) -> Option<kreate::sketching::Geometry> {
        let g = self.sketch_geometry()?;
        Some(match &self.sketching.drag {
            Some(drag) => g.without(&drag.points),
            None => g,
        })
    }

    /// Moves what is dragged so its anchor is at `to`, live: one modal step,
    /// begun on the first move and redone from where it began on each one
    /// after.
    fn drive_sketch_drag(&mut self, to: glam::DVec2) {
        let Some(drag) = self.sketching.drag.clone() else {
            return;
        };
        let offset = to - drag.anchor;
        // A click is not a drag: nothing moves until the pointer does.
        if self.sketch_drag_session.is_none() && offset.length() < 1e-9 {
            return;
        }
        let mut params = PropertyMap::new();
        params.set("points", PropertyValue::Ints(drag.points.clone()));
        params.set("offset", PropertyValue::Vec3([offset.x, offset.y, 0.0]));
        match self.sketch_drag_session.as_mut() {
            Some(session) => {
                if let Err(e) = self.history.update_modal(
                    &self.registry,
                    &mut self.doc,
                    None,
                    &self.op_ctx,
                    session,
                    &params,
                ) {
                    tracing::warn!(error = %e, "sketch drag failed");
                }
            }
            None => match self.history.begin_modal(
                &self.registry,
                &mut self.doc,
                None,
                &self.op_ctx,
                "sketch.move",
                &params,
            ) {
                Ok(Some(session)) => self.sketch_drag_session = Some(session),
                Ok(None) => {}
                Err(e) => tracing::warn!(error = %e, "sketch drag failed"),
            },
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Puts a drag back where it began: Esc or the right button.
    fn cancel_sketch_drag(&mut self) -> bool {
        if self.sketching.drag.take().is_none() {
            return false;
        }
        if let Some(session) = self.sketch_drag_session.take() {
            History::cancel_modal(session, &mut self.doc);
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
        true
    }

    fn sketch_tool_done(&mut self) {
        self.sketching.reset();
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// A line tool's points so far, run as one stroke.
    fn commit_sketch_line(&mut self, closed: bool) {
        let clicks = std::mem::take(&mut self.sketching.clicks);
        if clicks.len() >= 2 {
            self.run_op_with(
                "sketch.line",
                &[
                    (
                        "points",
                        PropertyValue::Floats(
                            clicks.iter().flat_map(|(p, _)| [p.x, p.y]).collect(),
                        ),
                    ),
                    (
                        "joins",
                        PropertyValue::Ints(clicks.iter().map(|(_, j)| *j).collect()),
                    ),
                    ("closed", PropertyValue::Bool(closed)),
                ],
            );
        }
        self.sketch_tool_done();
    }

    /// A spline tool's points so far, run as one stroke.
    fn commit_sketch_spline(&mut self, closed: bool) {
        let clicks = std::mem::take(&mut self.sketching.clicks);
        if clicks.len() >= 2 {
            self.run_op_with(
                "sketch.spline",
                &[
                    (
                        "points",
                        PropertyValue::Floats(
                            clicks.iter().flat_map(|(p, _)| [p.x, p.y]).collect(),
                        ),
                    ),
                    (
                        "joins",
                        PropertyValue::Ints(clicks.iter().map(|(_, j)| *j).collect()),
                    ),
                    ("closed", PropertyValue::Bool(closed)),
                ],
            );
        }
        self.sketch_tool_done();
    }

    /// The sketch mouse: returns whether the press or release was the tools'.
    fn sketch_mouse(&mut self, button: MouseButton, pressed: bool) -> bool {
        use kreate::sketching::{Pick, Tool};
        let tool = self.sketching.tool;
        match (button, pressed) {
            (MouseButton::Right, true) => {
                if self.cancel_sketch_drag() {
                    return true;
                }
                if tool == Tool::Line && self.sketching.clicks.len() >= 2 {
                    self.commit_sketch_line(false);
                } else if tool == Tool::Spline && self.sketching.clicks.len() >= 2 {
                    self.commit_sketch_spline(false);
                } else if !self.sketching.clicks.is_empty() || !self.sketching.picks.is_empty() {
                    self.sketch_tool_done();
                } else {
                    self.sketching.take(Tool::Select);
                }
                true
            }
            (MouseButton::Left, false) if self.sketching.band.is_some() => {
                // The end of a drag-box: what is inside is selected.
                if let (Some([a, b]), Some(g)) =
                    (self.sketching.band.take(), self.sketch_geometry())
                {
                    let (points, lines) = kreate::sketching::in_band(&g, a, b);
                    for v in points {
                        if !self.sketching.points.contains(&v) {
                            self.sketching.points.push(v);
                        }
                    }
                    for l in lines {
                        if !self.sketching.lines.contains(&l) {
                            self.sketching.lines.push(l);
                        }
                    }
                }
                self.redraw.set(RedrawReason::UI);
                true
            }
            (MouseButton::Left, false) => {
                // The end of a drag: what it moved is one step.
                if let Some(drag) = self.sketching.drag.take() {
                    if let Some(session) = self.sketch_drag_session.take() {
                        self.history.commit_modal(session);
                        // One point dropped on another of the sketch: joined.
                        let (hover, snapped) = (self.sketching.hover, self.sketching.snapped);
                        if let (Some(v), Some((_, join)), false) = (drag.single, hover, snapped) {
                            if join > 0 && join - 1 != v {
                                let kind = kreate_scene::sketch::KINDS
                                    .iter()
                                    .position(|k| *k == "coincident")
                                    .unwrap_or(0) as u32;
                                self.run_op_with(
                                    "sketch.constrain",
                                    &[
                                        ("kind", PropertyValue::Enum(kind)),
                                        ("points", PropertyValue::Ints(vec![v, join - 1])),
                                    ],
                                );
                            }
                        }
                    }
                    self.redraw.set(RedrawReason::SCENE_CHANGED);
                    return true;
                }
                false
            }
            (MouseButton::Left, true) if self.press_belongs_to_viewport() => {
                self.sketch_hover();
                let Some((p, join)) = self.sketching.hover else {
                    return true;
                };
                let pick = self
                    .sketch_cursor()
                    .zip(self.sketch_geometry())
                    .and_then(|(at, g)| g.pick(at));
                let v3 = |q: glam::DVec2| PropertyValue::Vec3([q.x, q.y, 0.0]);
                match tool {
                    Tool::Line => {
                        // On the outline's first point: closed.
                        if self.sketching.closing {
                            self.commit_sketch_line(true);
                            return true;
                        }
                        let first = self.sketching.clicks.first().map(|c| c.1);
                        self.sketching.clicks.push((p, join));
                        self.sketching.typed.clear();
                        // Back on the first point: the outline is closed.
                        if self.sketching.clicks.len() >= 3 && join > 0 && first == Some(join) {
                            self.sketching.clicks.pop();
                            self.commit_sketch_line(true);
                        }
                    }
                    Tool::Spline => {
                        if self.sketching.closing {
                            self.commit_sketch_spline(true);
                            return true;
                        }
                        self.sketching.clicks.push((p, join));
                        self.sketching.typed.clear();
                    }
                    // Mirror: the axis is the line clicked; what is selected
                    // goes across it.
                    Tool::Mirror => {
                        if let Some(Pick::Line(l)) = pick {
                            let g = self.sketch_geometry().unwrap_or_default();
                            if let Some(ends) = g.lines.get(l as usize).copied() {
                                let (points, lines) =
                                    (self.sketching.points.clone(), self.sketching.lines.clone());
                                self.run_op_with(
                                    "sketch.mirror",
                                    &[
                                        ("points", PropertyValue::Ints(points)),
                                        ("lines", PropertyValue::Ints(lines)),
                                        ("axis", PropertyValue::Ints(ends.to_vec())),
                                    ],
                                );
                                self.sketch_tool_done();
                            }
                        }
                    }
                    // Offset: the selected lines, as far as the click is
                    // from the first of them, to its side — or as typed.
                    Tool::Offset => {
                        let lines = self.sketching.lines.clone();
                        let g = self.sketch_geometry().unwrap_or_default();
                        if let Some(&[a, b]) = lines.first().and_then(|l| g.lines.get(*l as usize))
                        {
                            let (pa, pb) = (g.points[a as usize], g.points[b as usize]);
                            let n = (pb - pa).perp().normalize_or_zero();
                            let side = (p - pa).dot(n);
                            let typed = kreate::sketching::typed_lengths(
                                &self.sketching.typed,
                                self.doc.design.length_unit.metres(),
                            );
                            let d = typed
                                .first()
                                .map(|t| t.abs() * side.signum())
                                .unwrap_or(side);
                            if d.abs() > 1e-9 {
                                self.run_op_with(
                                    "sketch.offset",
                                    &[
                                        ("lines", PropertyValue::Ints(lines)),
                                        ("distance", PropertyValue::Float(d)),
                                    ],
                                );
                            }
                            self.sketch_tool_done();
                        }
                    }
                    Tool::Fillet => {
                        if let Some(Pick::Point(v)) = pick {
                            let typed = kreate::sketching::typed_lengths(
                                &self.sketching.typed,
                                self.doc.design.length_unit.metres(),
                            );
                            let mut args = vec![("point", PropertyValue::Int(v as i64))];
                            if let Some(r) = typed.first() {
                                args.push(("radius", PropertyValue::Float(r.abs())));
                            }
                            self.run_op_with("sketch.fillet", &args);
                            self.sketching.typed.clear();
                        }
                    }
                    Tool::Rectangle | Tool::Circle | Tool::Arc | Tool::Polygon | Tool::Slot => {
                        self.sketching.clicks.push((p, join));
                        let c = self.sketching.clicks.clone();
                        let unit = self.doc.design.length_unit.metres();
                        let typed = kreate::sketching::typed_lengths(&self.sketching.typed, unit);
                        match (tool, c.len()) {
                            (Tool::Polygon, 2) => {
                                let r = typed
                                    .first()
                                    .map(|d| d.abs())
                                    .unwrap_or(c[0].0.distance(c[1].0));
                                let rotation = (c[1].0 - c[0].0).y.atan2((c[1].0 - c[0].0).x);
                                if r > 1e-9 {
                                    self.run_op_with(
                                        "sketch.polygon",
                                        &[
                                            ("centre", v3(c[0].0)),
                                            ("radius", PropertyValue::Float(r)),
                                            ("rotation", PropertyValue::Float(rotation)),
                                            ("join", PropertyValue::Int(c[0].1 as i64)),
                                        ],
                                    );
                                }
                                self.sketch_tool_done();
                            }
                            (Tool::Slot, 3) => {
                                let d = c[1].0 - c[0].0;
                                let side = glam::DVec2::new(-d.y, d.x).normalize_or_zero();
                                let w = typed
                                    .first()
                                    .map(|w| w.abs())
                                    .unwrap_or(((c[2].0 - c[0].0).dot(side)).abs() * 2.0);
                                if w > 1e-9 && d.length() > 1e-9 {
                                    self.run_op_with(
                                        "sketch.slot",
                                        &[
                                            ("a", v3(c[0].0)),
                                            ("b", v3(c[1].0)),
                                            ("width", PropertyValue::Float(w)),
                                            ("joins", PropertyValue::Ints(vec![c[0].1, c[1].1])),
                                        ],
                                    );
                                }
                                self.sketch_tool_done();
                            }
                            (Tool::Rectangle, 2) => {
                                self.run_op_with(
                                    "sketch.rectangle",
                                    &[
                                        ("a", v3(c[0].0)),
                                        ("b", v3(c[1].0)),
                                        ("join", PropertyValue::Int(c[0].1 as i64)),
                                    ],
                                );
                                self.sketch_tool_done();
                            }
                            (Tool::Circle, 2) => {
                                let r = c[0].0.distance(c[1].0);
                                if r > 1e-9 {
                                    self.run_op_with(
                                        "sketch.circle",
                                        &[
                                            ("centre", v3(c[0].0)),
                                            ("radius", PropertyValue::Float(r)),
                                            ("join", PropertyValue::Int(c[0].1 as i64)),
                                        ],
                                    );
                                }
                                self.sketch_tool_done();
                            }
                            // Start, end, then a point on the way.
                            (Tool::Arc, 3) => {
                                self.run_op_with(
                                    "sketch.arc",
                                    &[
                                        ("start", v3(c[0].0)),
                                        ("end", v3(c[1].0)),
                                        ("through", v3(c[2].0)),
                                        ("joins", PropertyValue::Ints(vec![c[0].1, c[1].1])),
                                    ],
                                );
                                self.sketch_tool_done();
                            }
                            _ => {}
                        }
                    }
                    Tool::Point => {
                        self.run_op_with("sketch.point", &[("at", v3(p))]);
                        self.sketch_tool_done();
                    }
                    // Text goes where it is clicked; its words and font are
                    // then the last operation's to change.
                    Tool::Text => {
                        self.run_op_with("sketch.text", &[("at", v3(p))]);
                        self.sketch_tool_done();
                    }
                    // Trim stays in hand: one click a piece, as many as there are.
                    Tool::Trim => {
                        if let Some((line, on)) = self.sketching.trim.take() {
                            self.run_op_with(
                                "sketch.trim",
                                &[("line", PropertyValue::Int(line as i64)), ("at", v3(on))],
                            );
                            self.redraw.set(RedrawReason::SCENE_CHANGED);
                            self.sketch_hover();
                        }
                    }
                    Tool::Constrain(_) | Tool::Dimension => {
                        let kind = match tool {
                            Tool::Constrain(k) => k,
                            _ => "",
                        };
                        let Some(pick) = pick else {
                            return true;
                        };
                        self.sketching.picks.push(pick);
                        let g = self.sketch_geometry().unwrap_or_default();
                        let result = if tool == Tool::Dimension {
                            kreate::sketching::dimension_for(&self.sketching.picks, &g)
                        } else {
                            kreate::sketching::constraint_for(kind, &self.sketching.picks, &g)
                        };
                        match result {
                            Ok(Some((tag, points))) => {
                                let kind = kreate_scene::sketch::KINDS
                                    .iter()
                                    .position(|k| *k == tag)
                                    .unwrap_or(0) as u32;
                                self.run_op_with(
                                    "sketch.constrain",
                                    &[
                                        ("kind", PropertyValue::Enum(kind)),
                                        ("points", PropertyValue::Ints(points)),
                                    ],
                                );
                                // A dimension placed is a number to type.
                                if tool == Tool::Dimension {
                                    if let Some((id, _)) = self.sketch_target() {
                                        let n = self.doc.object(id).map_or(0, |o| o.sketch.len());
                                        self.ui.sketch_dim_edit =
                                            n.checked_sub(1).map(|i| (i, String::new()));
                                    }
                                }
                                self.sketch_tool_done();
                            }
                            Ok(None) => {}
                            Err(_) => {
                                self.status(&self.ui.catalog.get("sketch-cannot-constrain"));
                                self.sketch_tool_done();
                            }
                        }
                    }
                    // The pattern's centre, when the dialog asks for it.
                    Tool::Pattern
                        if self.ui.pattern_dialog.circular
                            && self.ui.pattern_dialog.picking_centre =>
                    {
                        self.ui.pattern_dialog.centre = [p.x, p.y];
                        self.ui.pattern_dialog.picking_centre = false;
                    }
                    Tool::Select | Tool::Pattern => {
                        let extend = self.input.shift_down;
                        let g = self.sketch_geometry().unwrap_or_default();
                        let lines_of = |pick: Pick| match pick {
                            Pick::Line(l) => vec![l],
                            Pick::Round { centre, on } => g.lines_of_round(centre, on),
                            Pick::Point(_) => Vec::new(),
                        };
                        // Taking hold of what is already selected drags all
                        // of it; anything else is selected first (Shift adds).
                        let held = match pick {
                            Some(Pick::Point(v)) => self.sketching.points.contains(&v),
                            Some(other) => {
                                let ls = lines_of(other);
                                !ls.is_empty()
                                    && ls.iter().all(|l| self.sketching.lines.contains(l))
                            }
                            None => false,
                        };
                        if !extend && !held {
                            self.sketching.points.clear();
                            self.sketching.lines.clear();
                        }
                        match pick {
                            Some(Pick::Point(v)) if !self.sketching.points.contains(&v) => {
                                self.sketching.points.push(v);
                            }
                            Some(other @ (Pick::Line(_) | Pick::Round { .. })) => {
                                for l in lines_of(other) {
                                    if !self.sketching.lines.contains(&l) {
                                        self.sketching.lines.push(l);
                                    }
                                }
                            }
                            _ => {}
                        }
                        // On nothing: a drag-box, from here.
                        if pick.is_none() {
                            if let Some(at) = self.sketch_cursor() {
                                self.sketching.band = Some([at, at]);
                            }
                        }
                        // Only the select tool drags; the pattern selects.
                        let pick = pick.filter(|_| tool == Tool::Select);
                        if let Some(pick) = pick {
                            // Everything selected moves: its points and its
                            // lines' ends, a circle's centre with its ring.
                            let mut points = self.sketching.points.clone();
                            for l in &self.sketching.lines {
                                if let Some(ends) = g.lines.get(*l as usize) {
                                    points.extend_from_slice(ends);
                                }
                            }
                            if matches!(pick, Pick::Round { .. }) {
                                points.extend(g.points_of(pick));
                            }
                            for (c, ring, _) in &g.rounds {
                                if ring.iter().all(|v| points.contains(v)) {
                                    points.push(*c);
                                }
                            }
                            points.sort_unstable();
                            points.dedup();
                            let raw = self.sketch_plane_point().unwrap_or(p);
                            let (anchor, single) = match pick {
                                Pick::Point(v) => (
                                    g.points.get(v as usize).copied().unwrap_or(raw),
                                    (points.len() == 1).then_some(v),
                                ),
                                _ => (raw, None),
                            };
                            self.sketching.drag = Some(kreate::sketching::SketchDrag {
                                points,
                                anchor,
                                single,
                            });
                        }
                    }
                    Tool::Extrude | Tool::Revolve => {
                        if let (Tool::Revolve, Some(Pick::Line(l))) = (tool, pick) {
                            if let Some(ends) = self
                                .sketch_target()
                                .and_then(|(id, _)| self.doc.object(id))
                                .and_then(|o| o.sketch_lines.get(l as usize).copied())
                            {
                                self.set_revolve_axis(ends);
                                return true;
                            }
                        }
                        // A region: picked, or put back if it already was.
                        if let Some((id, _)) = self.sketch_target() {
                            let regions = match (tool, self.sketching.axis) {
                                (Tool::Revolve, Some(axis)) => {
                                    kreate_scene::feature::shape_regions(
                                        &self.doc,
                                        id,
                                        &kreate::sketching::revolve_shape(axis),
                                    )
                                }
                                _ => kreate_scene::feature::sketch_regions(&self.doc, id),
                            };
                            let at = |q: glam::DVec2| {
                                kreate_mesh::regions::region_at(&regions, q.as_vec2())
                            };
                            if let Some(r) = at(p) {
                                let before = self.sketching.seeds.len();
                                self.sketching.seeds.retain(|s| at(*s) != Some(r));
                                if self.sketching.seeds.len() == before {
                                    self.sketching.seeds.push(p);
                                }
                            }
                        }
                    }
                }
                self.redraw.set(RedrawReason::UI);
                true
            }
            _ => false,
        }
    }

    /// The sketch keys: typing a length, Enter, Backspace and Escape.
    /// Returns whether the key was the tools'.
    fn sketch_key(&mut self, code: KeyCode) -> bool {
        use kreate::sketching::Tool;
        let tool = self.sketching.tool;
        let drawing = tool.draws() && !self.sketching.clicks.is_empty();
        let typed = match code {
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
            KeyCode::Comma | KeyCode::Space => Some(','),
            _ => None,
        };
        if let (true, Some(ch)) = (drawing, typed) {
            self.sketching.typed.push(ch);
            self.redraw.set(RedrawReason::UI);
            return true;
        }
        let unit = self.doc.design.length_unit.metres();
        match code {
            KeyCode::Backspace if drawing => {
                if self.sketching.typed.pop().is_none() {
                    self.sketching.clicks.pop();
                }
                self.redraw.set(RedrawReason::UI);
                true
            }
            KeyCode::Enter | KeyCode::NumpadEnter if drawing => {
                let lengths = kreate::sketching::typed_lengths(&self.sketching.typed, unit);
                let (last, toward) = (
                    self.sketching
                        .clicks
                        .last()
                        .map(|c| c.0)
                        .unwrap_or_default(),
                    self.sketching.hover.map(|h| h.0).unwrap_or_default(),
                );
                let v3 = |q: glam::DVec2| PropertyValue::Vec3([q.x, q.y, 0.0]);
                match (tool, lengths.as_slice()) {
                    (Tool::Line, [length, ..]) => {
                        let next = kreate::sketching::along(last, toward, *length);
                        self.sketching.clicks.push((next, 0));
                        self.sketching.typed.clear();
                    }
                    (Tool::Line, []) => self.commit_sketch_line(false),
                    (Tool::Spline, []) => self.commit_sketch_spline(false),
                    (Tool::Rectangle, [w, h, ..]) => {
                        let first = self.sketching.clicks[0];
                        let far = kreate::sketching::corner(first.0, toward, *w, *h);
                        self.run_op_with(
                            "sketch.rectangle",
                            &[
                                ("a", v3(first.0)),
                                ("b", v3(far)),
                                ("join", PropertyValue::Int(first.1 as i64)),
                            ],
                        );
                        self.sketch_tool_done();
                    }
                    // A circle's number is its diameter, as Fusion's is.
                    (Tool::Circle, [d, ..]) => {
                        let first = self.sketching.clicks[0];
                        self.run_op_with(
                            "sketch.circle",
                            &[
                                ("centre", v3(first.0)),
                                ("radius", PropertyValue::Float(d.abs() * 0.5)),
                                ("join", PropertyValue::Int(first.1 as i64)),
                            ],
                        );
                        self.sketch_tool_done();
                    }
                    _ => {}
                }
                self.redraw.set(RedrawReason::UI);
                true
            }
            KeyCode::Escape => {
                if self.sketching.band.take().is_some() {
                    // The box let go, nothing else.
                } else if self.cancel_sketch_drag() {
                    // Put back, nothing else.
                } else if tool == Tool::Line && self.sketching.clicks.len() >= 2 {
                    self.commit_sketch_line(false);
                } else if tool == Tool::Spline && self.sketching.clicks.len() >= 2 {
                    self.commit_sketch_spline(false);
                } else if !self.sketching.clicks.is_empty()
                    || !self.sketching.picks.is_empty()
                    || !self.sketching.seeds.is_empty()
                {
                    self.sketch_tool_done();
                } else if tool != Tool::Select {
                    self.sketching.take(Tool::Select);
                } else {
                    self.sketching.points.clear();
                    self.sketching.lines.clear();
                }
                self.redraw.set(RedrawReason::UI);
                true
            }
            _ => false,
        }
    }

    /// Turns the camera to look straight at the sketch's plane, once per
    /// sketch entered; and keeps the overlay's picture of the sketch.
    fn update_sketch_view(&mut self) {
        use kreate::sketching::Tool;
        // Every sketch's lines, thin, whatever the mode.
        let active = (self.op_ctx.mode == OpMode::Sketch)
            .then(|| self.sketch_target().map(|t| t.0))
            .flatten();
        self.ui.sketch_lines = kreate::sketching::other_lines_marked(&self.doc, active);
        self.ui.length_unit = self.doc.design.length_unit.metres();
        self.ui.features = kreate::sketching::feature_rows(&self.doc, &self.ui.catalog);

        // Waiting for a plane: entered once the new sketch is active, given
        // up when the menu closed without one.
        if self.sketch_waiting {
            if kreate_ops::ops::sketch_draw::active_sketch(&self.doc).is_some() {
                self.sketch_waiting = false;
                self.set_mode(OpMode::Sketch);
            } else if self.ui.context_menu.groups.is_empty() && !self.sketch_face_pick {
                self.sketch_waiting = false;
            }
        }
        if self.op_ctx.mode != OpMode::Sketch {
            self.ui.sketch_view = None;
            self.sketch_faced = None;
            if self.sketching.tool != Tool::Select || !self.sketching.clicks.is_empty() {
                self.sketching = Default::default();
            }
            return;
        }
        let Some((id, world)) = self.sketch_target() else {
            self.ui.sketch_view = None;
            return;
        };
        let uid = self.doc.object(id).map(|o| o.uid);
        if self.sketch_faced != uid {
            self.sketch_faced = uid;
            self.sketching = Default::default();
            // Face the plane: the camera turned as the sketch is, looking
            // down its Z, orthographic, about its origin.
            let (_, rotation, translation) = world.to_scale_rotation_translation();
            let mut camera = self.camera;
            camera.rotation = rotation;
            camera.pivot = translation;
            camera.projection = kreate_render::Projection::Orthographic;
            self.camera = camera;
            self.motion = kreate_render::CameraMotion::new(camera);
            self.redraw.set(RedrawReason::SCENE_CHANGED);
        }
        // Pinned, worked out again only when the sketch changed.
        let points = kreate_ops::ops::sketch_draw::points(&self.doc, id);
        let constraints = self
            .doc
            .object(id)
            .map(|o| o.sketch.clone())
            .unwrap_or_default();
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            format!("{constraints:?}").hash(&mut h);
            for p in &points {
                p.x.to_bits().hash(&mut h);
                p.y.to_bits().hash(&mut h);
            }
            h.finish()
        };
        if self.sketch_pinned.0 != key {
            self.sketch_pinned = (key, kreate::sketching::pinned(&points, &constraints));
        }
        self.ui.sketch_view = Some(kreate::sketching::view(
            &self.doc,
            id,
            &self.sketching,
            &self.ui.catalog,
            &self.sketch_pinned.1,
        ));
        self.preview_sketch_pattern(id, world);
        self.choose_solid_operation(id);
    }

    /// The pattern's copies, as they would be made, into the view's preview;
    /// a circular one's centre marked.
    fn preview_sketch_pattern(&mut self, id: ObjectId, world: glam::DAffine3) {
        if self.sketching.tool != kreate::sketching::Tool::Pattern {
            return;
        }
        let Some(object) = self.doc.object(id) else {
            return;
        };
        let pts = kreate_ops::ops::sketch_draw::points(&self.doc, id);
        let (_, with) = kreate_ops::ops::sketch_draw::pattern_source(
            &object.sketch_lines,
            &object.sketch,
            &self.sketching.points,
            &self.sketching.lines,
            pts.len(),
        );
        let dialog = &self.ui.pattern_dialog;
        let params = dialog.params(self.doc.design.length_unit.metres());
        let moves = kreate_ops::ops::sketch_draw::pattern_moves(&params);
        let to_world = |p: glam::DVec2| world.transform_point3(DVec3::new(p.x, p.y, 0.0));
        let mut preview = Vec::new();
        for m in moves.iter().take(2000) {
            for l in &with {
                let [a, b] = object.sketch_lines[*l as usize];
                let (Some(pa), Some(pb)) = (pts.get(a as usize), pts.get(b as usize)) else {
                    continue;
                };
                preview.push(vec![
                    to_world(m.transform_point2(*pa)),
                    to_world(m.transform_point2(*pb)),
                ]);
            }
        }
        let centre = dialog
            .circular
            .then(|| to_world(glam::DVec2::from(dialog.centre)));
        if let Some(view) = self.ui.sketch_view.as_mut() {
            view.preview.extend(preview);
            if centre.is_some() {
                view.snap = centre;
            }
        }
    }

    /// Fusion's choice for an extrude, while its dialog is open and nothing
    /// was picked by hand: going into an object, a cut from it; coming out
    /// of one, a join to it; in the open, a new body. Looked at in the
    /// middle of what the extrude would fill, and just behind the sketch.
    fn choose_solid_operation(&mut self, sketch: ObjectId) {
        let seeds = self.sketching.seeds.clone();
        if seeds.is_empty() {
            // Closed: the next extrude chooses afresh.
            self.ui.solid_dialog.chosen = false;
            self.solid_auto = None;
            return;
        }
        if self.sketching.tool != kreate::sketching::Tool::Extrude || self.ui.solid_dialog.chosen {
            return;
        }
        let d = self.ui.solid_dialog.distance * self.ui.length_unit;
        let symmetric = self.ui.solid_dialog.symmetric;
        let key = (
            self.doc.version(),
            seeds
                .iter()
                .map(|s| [s.x.to_bits(), s.y.to_bits()])
                .collect::<Vec<_>>(),
            d.to_bits(),
            symmetric,
        );
        if self.solid_auto.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let world = self.doc.world_transform(sketch);
        let normal = world.transform_vector3(DVec3::Z).normalize_or_zero();
        let along = if symmetric { d.abs() * 0.25 } else { d * 0.5 };
        let behind = -d.signum() * (d.abs() * 0.02).max(1e-4);
        let at = |s: glam::DVec2| world.transform_point3(DVec3::new(s.x, s.y, 0.0));
        let mut cut: Option<String> = None;
        let mut join: Option<String> = None;
        for (id, mesh) in &self.evaluated {
            let Some(o) = self.doc.object(*id) else {
                continue;
            };
            if *id == sketch
                || !self.doc.is_shown(*id)
                || !o.sketch.is_empty()
                || !o.sketch_lines.is_empty()
            {
                continue;
            }
            let to_local = self.doc.world_transform(*id).inverse();
            let inside =
                |p: DVec3| kreate_mesh::measure::contains(mesh, to_local.transform_point3(p));
            if cut.is_none() && seeds.iter().any(|s| inside(at(*s) + normal * along)) {
                cut = Some(o.name.clone());
            }
            if join.is_none()
                && !symmetric
                && seeds.iter().any(|s| inside(at(*s) + normal * behind))
            {
                join = Some(o.name.clone());
            }
        }
        let choice = match (cut, join) {
            (Some(name), _) => (2, name),
            (None, Some(name)) => (1, name),
            (None, None) => (0, String::new()),
        };
        self.ui.solid_dialog.operation = choice.0;
        self.ui.solid_dialog.target = choice.1.clone();
        self.solid_auto = Some((key, choice));
    }

    fn begin_knife(&mut self) {
        if self.gesture.is_some() || self.knife.is_some() {
            return;
        }
        self.end_loop_cut_hover();
        self.knife = Some((Vec::new(), None));
        self.update_knife();
    }

    /// Where the next knife point would go: on the active mesh's surface under
    /// the pointer, pulled onto a vertex or an edge of that face when the
    /// pointer is close to one on screen — a cut that starts and ends on edges
    /// is what splits faces.
    fn update_knife(&mut self) {
        let Some((points, _)) = self.knife.take() else {
            return;
        };
        if self.op_ctx.mode.acts_on() != kreate_ops::mode::ActsOn::MeshElements {
            self.end_knife("");
            return;
        }
        let mut hover = None;
        let object = kreate_ops::ops::mesh::target_object(&self.doc);
        let mesh_id = kreate_ops::ops::mesh::target_mesh(&self.doc);
        if let (Some(ndc), Some(object), Some(mesh_id), Some(projector), Some((x, y))) = (
            self.cursor_ndc(),
            object,
            mesh_id,
            self.ui.projector,
            self.input.cursor,
        ) {
            let aspect = self.viewport_aspect();
            let hit = self
                .pick
                .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect);
            let stored = self.doc.mesh_arc(mesh_id).cloned();
            let as_drawn = self
                .evaluated
                .iter()
                .find(|(id, _)| *id == object)
                .map(|(_, m)| m.clone());
            if let (Some(hit), Some(mesh), Some(drawn)) = (hit, stored, as_drawn) {
                if hit.object == object && Arc::ptr_eq(&mesh, &drawn) && hit.face < mesh.num_faces()
                {
                    let ppp = self.egui_ctx.pixels_per_point();
                    let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
                    let world = self.doc.world_transform(object);
                    let place =
                        |v: u32| world.transform_point3(mesh.positions()[v as usize].as_dvec3());
                    let ring = mesh.face_verts(hit.face).to_vec();
                    let mut best: Option<(f32, DVec3)> = None;
                    // A vertex within ten points wins outright.
                    for v in &ring {
                        let w = place(*v);
                        if let Some(s) = projector.project(w) {
                            let d = s.distance(at);
                            if d < 10.0 && best.is_none_or(|(bd, _)| d < bd) {
                                best = Some((d, w));
                            }
                        }
                    }
                    // Otherwise the nearest point of an edge within eight.
                    if best.is_none() {
                        for i in 0..ring.len() {
                            let (wa, wb) = (place(ring[i]), place(ring[(i + 1) % ring.len()]));
                            let (Some(sa), Some(sb)) =
                                (projector.project(wa), projector.project(wb))
                            else {
                                continue;
                            };
                            let ab = sb - sa;
                            let t = if ab.length_sq() > 0.0 {
                                ((at - sa).dot(ab) / ab.length_sq()).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            let d = (sa + ab * t).distance(at);
                            if d < 8.0 && best.is_none_or(|(bd, _)| d < bd) {
                                best = Some((d, wa.lerp(wb, t as f64)));
                            }
                        }
                    }
                    hover = Some(best.map(|(_, w)| w).unwrap_or(hit.point));
                }
            }
        }
        let mut preview = points.clone();
        preview.extend(hover);
        self.ui.knife_preview = preview;
        self.status(&format!(
            "Knife  {} points — click on edges to cut, Enter to finish, Backspace to undo a point, Esc to cancel",
            points.len()
        ));
        self.knife = Some((points, hover));
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    fn end_knife(&mut self, status: &str) {
        self.knife = None;
        self.ui.knife_preview.clear();
        if !status.is_empty() {
            self.status(status);
        }
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// Enter: the path, in the mesh's own space, to the operator.
    fn commit_knife(&mut self) {
        let Some((points, _)) = self.knife.clone() else {
            return;
        };
        self.end_knife("");
        if points.len() < 2 {
            self.status("Knife cancelled: click at least two points");
            return;
        }
        let Some(object) = kreate_ops::ops::mesh::target_object(&self.doc) else {
            return;
        };
        let to_local = self.doc.world_transform(object).inverse();
        if !to_local.is_finite() {
            return;
        }
        let flat: Vec<f64> = points
            .iter()
            .flat_map(|p| to_local.transform_point3(*p).to_array())
            .collect();
        let view = to_local
            .transform_vector3(self.camera.forward())
            .normalize_or_zero();
        let mut params = PropertyMap::new();
        params.set("points", PropertyValue::Floats(flat));
        params.set("view", PropertyValue::Vec3(view.to_array()));
        self.run_op("mesh.knife", params);
    }

    /// `Ctrl+R`: from here until a click, the edge under the pointer shows the
    /// loops that would be cut across it. Blender's way — no need to select
    /// the edge first.
    fn begin_loop_cut_hover(&mut self) {
        if self.gesture.is_some() || self.loop_cut_hover.is_some() {
            return;
        }
        self.loop_cut_hover = Some((None, 1));
        self.update_loop_cut_hover();
    }

    /// Finds the edge under the pointer and redraws the preview.
    ///
    /// The face under the pointer is found the way auto-depth finds it, by a
    /// ray, so what is behind the model is not offered; the edge is whichever
    /// of that face's edges is nearest the pointer on screen. On the active
    /// mesh only, and only while it is drawn as it is stored — a modifier
    /// stack numbers its faces differently.
    /// The edge of the active mesh nearest the pointer on the face under it:
    /// a ray finds the face, so what is behind the model is not offered, and
    /// the edge is whichever of that face's is nearest on screen. Only while the
    /// mesh is drawn as it is stored — a modifier stack numbers its faces
    /// otherwise.
    fn edge_under_cursor(&mut self) -> Option<u32> {
        let object = kreate_ops::ops::mesh::target_object(&self.doc)?;
        let mesh_id = kreate_ops::ops::mesh::target_mesh(&self.doc)?;
        let (ndc, projector, (x, y)) = (self.cursor_ndc()?, self.ui.projector?, self.input.cursor?);
        let aspect = self.viewport_aspect();
        let hit = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect)?;
        let mesh = self.doc.mesh_arc(mesh_id).cloned()?;
        let drawn = self
            .evaluated
            .iter()
            .find(|(id, _)| *id == object)
            .map(|(_, m)| m.clone())?;
        if hit.object != object || !Arc::ptr_eq(&mesh, &drawn) || hit.face >= mesh.num_faces() {
            return None;
        }
        let ppp = self.egui_ctx.pixels_per_point();
        let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
        let world = self.doc.world_transform(object);
        let corner_edges = mesh.corner_edges();
        let screen = |v: u32| {
            projector.project(world.transform_point3(mesh.positions()[v as usize].as_dvec3()))
        };
        let mut best: Option<(f32, u32)> = None;
        for c in mesh.face_corner_range(hit.face) {
            let e = corner_edges[c];
            let [a, b] = mesh.edges()[e as usize];
            if let (Some(pa), Some(pb)) = (screen(a), screen(b)) {
                let d = segment_distance(at, pa, pb);
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, e));
                }
            }
        }
        best.map(|(_, e)| e)
    }

    /// Alt+click: the loop through the edge under the pointer — Ctrl for a
    /// ring, Shift to add; in face mode, a loop of faces.
    fn loop_pick(&mut self) {
        let Some(edge) = self.edge_under_cursor() else {
            return;
        };
        let mut params = PropertyMap::new();
        params.set("edge", PropertyValue::Int(edge as i64));
        params.set("ring", PropertyValue::Bool(self.input.ctrl_down));
        params.set("extend", PropertyValue::Bool(self.input.shift_down));
        params.set(
            "faces",
            PropertyValue::Bool(self.select_mode == 2 && !self.input.ctrl_down),
        );
        self.run_op("mesh.select_loop_pick", params);
    }

    /// L: the piece under the pointer, added — or with `deselect`, taken out.
    fn linked_pick(&mut self, deselect: bool) {
        let Some(hit) = self.hover else {
            self.status("point at the mesh to select what is linked to it");
            return;
        };
        let mode = match self.pick_domain() {
            kreate_render::PickDomain::Vert => 0u32,
            kreate_render::PickDomain::Edge => 1,
            _ => 2,
        };
        let mut params = PropertyMap::new();
        params.set("element", PropertyValue::Int(hit.element as i64));
        params.set("mode", PropertyValue::Enum(mode));
        params.set("deselect", PropertyValue::Bool(deselect));
        self.run_op("mesh.select_linked_pick", params);
    }

    fn update_loop_cut_hover(&mut self) {
        let Some((_, cuts)) = self.loop_cut_hover else {
            return;
        };
        if self.op_ctx.mode.acts_on() != kreate_ops::mode::ActsOn::MeshElements {
            self.end_loop_cut_hover();
            return;
        }
        let mut found = None;
        let object = kreate_ops::ops::mesh::target_object(&self.doc);
        let mesh_id = kreate_ops::ops::mesh::target_mesh(&self.doc);
        if let (Some(ndc), Some(object), Some(mesh_id), Some(projector), Some((x, y))) = (
            self.cursor_ndc(),
            object,
            mesh_id,
            self.ui.projector,
            self.input.cursor,
        ) {
            let aspect = self.viewport_aspect();
            let hit = self
                .pick
                .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect);
            let stored = self.doc.mesh_arc(mesh_id).cloned();
            let as_drawn = self
                .evaluated
                .iter()
                .find(|(id, _)| *id == object)
                .map(|(_, m)| m.clone());
            if let (Some(hit), Some(mesh), Some(drawn)) = (hit, stored, as_drawn) {
                if hit.object == object && Arc::ptr_eq(&mesh, &drawn) && hit.face < mesh.num_faces()
                {
                    let ppp = self.egui_ctx.pixels_per_point();
                    let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
                    let world = self.doc.world_transform(object);
                    let corner_edges = mesh.corner_edges();
                    let screen = |v: u32| {
                        projector.project(
                            world.transform_point3(mesh.positions()[v as usize].as_dvec3()),
                        )
                    };
                    let mut best = f32::INFINITY;
                    for c in mesh.face_corner_range(hit.face) {
                        let e = corner_edges[c];
                        let [a, b] = mesh.edges()[e as usize];
                        if let (Some(pa), Some(pb)) = (screen(a), screen(b)) {
                            let d = segment_distance(at, pa, pb);
                            if d < best {
                                best = d;
                                found = Some(e);
                            }
                        }
                    }
                    let lines = found
                        .map(|e| kreate_mesh::edit::loopcut::preview_lines(&mesh, e, cuts))
                        .unwrap_or_default();
                    self.ui.loop_cut_preview = lines
                        .iter()
                        .map(|line| {
                            line.iter()
                                .map(|p| world.transform_point3(p.as_dvec3()))
                                .collect()
                        })
                        .collect();
                }
            }
        }
        if found.is_none() {
            self.ui.loop_cut_preview.clear();
        }
        self.loop_cut_hover = Some((found, cuts));
        self.status(&format!(
            "Loop Cut  cuts {cuts}  — point at an edge, click to cut, wheel for more loops, Esc to cancel"
        ));
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    fn end_loop_cut_hover(&mut self) {
        self.loop_cut_hover = None;
        self.ui.loop_cut_preview.clear();
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// The click: cut across the edge under the pointer, then slide it.
    fn commit_loop_cut_hover(&mut self) {
        let Some((edge, cuts)) = self.loop_cut_hover else {
            return;
        };
        self.end_loop_cut_hover();
        let Some(edge) = edge else {
            self.status("Loop Cut cancelled: no edge under the pointer");
            return;
        };
        self.loop_cut_seed = Some((edge, cuts));
        self.begin_gesture(kreate::modal::Kind::LoopCut);
        self.loop_cut_seed = None;
    }

    /// Adds or takes away loops from a running loop cut: the wheel, as in
    /// Blender. `false` when no loop cut is running.
    fn adjust_loop_cuts(&mut self, notches: f32) -> bool {
        if let Some((_, cuts)) = &mut self.loop_cut_hover {
            let step = notches.signum() as i32;
            *cuts = (*cuts as i32 + step).clamp(1, 64) as u32;
            self.update_loop_cut_hover();
            return true;
        }
        let Some((gesture, _)) = &mut self.gesture else {
            return false;
        };
        if gesture.kind != kreate::modal::Kind::LoopCut {
            return false;
        }
        let step = notches.signum() as i32;
        if step != 0 {
            gesture.add_cuts(step);
            self.drive_gesture();
        }
        true
    }

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
            // `G` again in a mesh's move, Blender's `G G`: the move undone
            // and the selected edges slid instead.
            KeyCode::KeyG
                if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements
                    && self
                        .gesture
                        .as_ref()
                        .is_some_and(|(g, _)| g.kind == kreate::modal::Kind::Move) =>
            {
                self.cancel_gesture();
                self.begin_gesture(kreate::modal::Kind::EdgeSlide);
                true
            }
            KeyCode::Enter | KeyCode::NumpadEnter => {
                if self.gesture.as_ref().is_some_and(|(g, _)| g.picking_base()) {
                    self.finish_base_pick();
                } else {
                    self.confirm_gesture();
                }
                true
            }
            KeyCode::Escape => {
                self.cancel_gesture();
                true
            }
            // A move's snap base, Blender's `B`: the drag holds still, the
            // pointer picks a point of what moves, and from then on that
            // point is what lands on the target. Pressed again while picking,
            // it takes the point under the pointer.
            // B in a loop cut: the cut snaps to the nearer end of its edge,
            // and a number typed after is a distance from that end.
            KeyCode::KeyB
                if self
                    .gesture
                    .as_ref()
                    .is_some_and(|(g, _)| g.kind == kreate::modal::Kind::LoopCut) =>
            {
                if let Some((gesture, _)) = &mut self.gesture {
                    gesture.anchor_slide();
                }
                self.drive_gesture();
                true
            }
            KeyCode::KeyB if self.gesture.as_ref().is_some_and(|(g, _)| g.takes_base()) => {
                if self.gesture.as_ref().is_some_and(|(g, _)| g.picking_base()) {
                    self.finish_base_pick();
                } else {
                    let own = self.build_own_snap_world();
                    if let Some((gesture, _)) = &mut self.gesture {
                        gesture.begin_base_pick(own);
                    }
                    self.drive_gesture();
                }
                true
            }
            // Snapping for the rest of any other drag. The header says
            // `snap` while it is on, so the toggle is readable.
            KeyCode::KeyB => {
                if let Some((gesture, _)) = &mut self.gesture {
                    gesture.toggle_snap();
                }
                // A drag that began with increment snapping gathered nothing
                // to catch on. What is not moving is still where it was.
                if self.gesture.as_ref().is_some_and(|(g, _)| g.wants_world()) {
                    let world = self.build_snap_world(kreate::snap::Mode::Element);
                    if let Some((gesture, _)) = &mut self.gesture {
                        gesture.set_world(world);
                    }
                }
                self.drive_gesture();
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

    /// Starts the still view's mean again when this frame's reasons include a
    /// change its own signature cannot see — a mesh rewritten in place, a
    /// light, a setting.
    fn restart_mean_if_changed(&mut self) {
        let changed = RedrawReason::SCENE_CHANGED
            | RedrawReason::RESIZE
            | RedrawReason::SETTINGS
            | RedrawReason::PLAYING;
        if self.redraw.0 & changed != 0 {
            if let Some(gfx) = self.gfx.as_mut() {
                gfx.viewport.restart_accumulation();
            }
        }
    }

    // --- the connector -----------------------------------------------------
    //
    // What `kreate-mcp` asks for, served on the main thread, where the
    // document is. See `kreate::bridge`.

    // --- rendering ---------------------------------------------------------

    /// Applies the commands the panels queued this frame.
    fn drain_ui_commands(&mut self) {
        for command in self.ui.queue.drain() {
            match command {
                // A boolean from a menu or the ribbon is chosen in the
                // viewport, as Fusion's Combine is; a script's runs as given.
                Command::RunOperator { op_id, params } if op_id == "object.boolean" => {
                    let operation = params.get("operation").and_then(|v| match v {
                        PropertyValue::Enum(n) => Some(*n),
                        PropertyValue::Int(n) => Some(*n as u32),
                        _ => None,
                    });
                    self.begin_boolean_pick(operation.unwrap_or(0));
                }
                Command::RunOperator { op_id, params } => self.run_op(&op_id, params),
                Command::PickObjectField { op_id, params } => {
                    self.ui.object_pick = Some(kreate::shell::ObjectPick::Field { op_id, params });
                    let message = self.ui.catalog.get("pick-field-hint");
                    self.status(&message);
                }
                Command::ObjectPickConfirm => self.finish_object_pick(),
                Command::ObjectPickCancel => {
                    self.ui.object_pick = None;
                    self.redraw.set(RedrawReason::UI);
                }
                // The same path a key press takes, so a menu entry that says
                // "Move" starts the drag the G key starts. See `dispatch`.
                Command::Dispatch { op_id, params } => {
                    self.dispatch(&Action { op_id, params });
                }
                // Through the history with the edit sessions, exactly as a
                // key press would go: one undo step per line, and an
                // edit-mode operator works on the open session.
                Command::RunScript { source } => self.run_script(&source),
                Command::ChatSend { text } => self.chat_send(text),
                Command::ChatAttach => self.pick_chat_attachments(),
                Command::ChatDetach { index } => {
                    if index < self.chat_attachments.len() {
                        self.chat_attachments.remove(index);
                        self.publish_chat_attachments();
                    }
                }
                Command::ChatStop => {
                    if let Some(chat) = &mut self.chat {
                        chat.stop();
                    }
                }
                Command::ChatProvider { index } => {
                    let Some(provider) = kreate::claude::Provider::ALL.get(index).copied() else {
                        continue;
                    };
                    if let Some(chat) = &mut self.chat {
                        chat.set_provider(provider);
                    }
                    self.ui.chat_provider = index;
                    self.ui.chat_models.clear();
                    self.ui.chat_log.clear();
                    self.ui.chat_account = kreate_ui::chat::ChatAccount::Checking;
                    self.ui.chat_busy = false;
                }
                Command::ChatModel { index } => {
                    if let Some(chat) = &mut self.chat {
                        chat.set_model(index);
                        self.ui.chat_model = chat.model().unwrap_or(0);
                    }
                }
                Command::ChatSignIn => {
                    if let Some(chat) = &mut self.chat {
                        match chat.sign_in() {
                            Ok(()) => {
                                self.ui.chat_account =
                                    kreate_ui::chat::ChatAccount::SigningIn { url: None }
                            }
                            Err(e) => {
                                self.ui.chat_account = kreate_ui::chat::ChatAccount::Unavailable(e)
                            }
                        }
                    }
                }
                Command::ChatSignInCode { code } => {
                    if let Some(chat) = &mut self.chat {
                        chat.sign_in_code(&code);
                    }
                }
                Command::ChatSignOut => {
                    if let Some(chat) = &mut self.chat {
                        match chat.sign_out() {
                            Ok(()) => {
                                self.ui.chat_log.clear();
                                self.ui.chat_busy = false;
                                self.ui.chat_account = kreate_ui::chat::ChatAccount::Checking;
                            }
                            Err(e) => {
                                self.ui.chat_account = kreate_ui::chat::ChatAccount::Unavailable(e)
                            }
                        }
                    }
                }
                Command::ChatSignInCancel => {
                    if let Some(chat) = &mut self.chat {
                        chat.cancel_sign_in();
                    }
                }
                Command::OpenContact => self.ui.contact.open(),
                Command::OpenCollab => self.open_collab(),
                Command::CollabHost { name, port, key } => self.collab_host(name, port, key),
                Command::CollabJoin { name, address, key } => self.collab_join(name, address, key),
                Command::CollabLeave => self.collab_leave(),
                Command::OpenAbout => self.ui.about_open = true,
                Command::Donate => {
                    kreate::desktop::open_url(kreate::desktop::SPONSORS_URL);
                    let message = self.ui.catalog.get("donate-opened");
                    self.status(&message);
                }
                Command::ContactSend {
                    kind,
                    name,
                    email,
                    message,
                    include_details,
                } => {
                    let details = include_details
                        .then(|| (kreate::version::describe(), kreate::version::platform()));
                    match &self.outbox {
                        Some(outbox) => outbox.send(&kreate::contact::Message {
                            kind,
                            name,
                            email,
                            message,
                            details,
                        }),
                        None => {
                            self.ui.contact.status = kreate_ui::contact::ContactStatus::Failed(
                                "messages cannot be sent from here".into(),
                            )
                        }
                    }
                }
                Command::ChatNew => {
                    if let Some(chat) = &mut self.chat {
                        chat.reset();
                    }
                    self.ui.chat_log.clear();
                    self.chat_turns.clear();
                    // A new page for what comes next; nothing to bring back
                    // after a restart.
                    self.chat_record = None;
                    if let Some(dir) = kreate::chatlog::chat_dir() {
                        kreate::chatlog::forget_last_in(&dir);
                    }
                }
                Command::ChatHistory => {
                    if let Some(dir) = kreate::chatlog::chat_dir() {
                        kreate::logs::reveal(&dir);
                    }
                }
                Command::ChatExport => self.export_chat(),
                Command::OpenLogFolder => {
                    if let Some(dir) = kreate::logs::log_dir() {
                        kreate::logs::reveal(&dir);
                    }
                }
                Command::ChatRevertTurn { turn } => self.revert_chat_turn(turn),
                Command::ConnectApp { index } => {
                    let said = match (
                        kreate::connect::App::ALL.get(index),
                        kreate::claude::find_connector(),
                    ) {
                        (Some(app), Some(connector)) => app.connect(&connector),
                        _ => Err("kreate-mcp is not beside Kreate".to_string()),
                    };
                    let (level, text) = match said {
                        Ok(text) => (rfd::MessageLevel::Info, text),
                        Err(text) => (rfd::MessageLevel::Warning, text),
                    };
                    self.status(&text);
                    rfd::MessageDialog::new()
                        .set_level(level)
                        .set_title(self.ui.catalog.get("connect-menu"))
                        .set_description(text)
                        .set_buttons(rfd::MessageButtons::Ok)
                        .show();
                }
                // The outliner's key resolved against the *current* document,
                // then handed to the operator as the identity it wants — so
                // the eye is one undo step like any other edit.
                Command::ParentObject { child, parent } => {
                    let uid = |key| {
                        shell::object_from_key(&self.doc, key)
                            .and_then(|id| self.doc.object_uid(id))
                    };
                    if let (Some(c), Some(p)) = (uid(child), uid(parent)) {
                        let mut params = PropertyMap::new();
                        params.set("child", PropertyValue::Text(c.raw().to_string()));
                        params.set("parent", PropertyValue::Text(p.raw().to_string()));
                        self.run_op("object.parent_one", params);
                    }
                }
                Command::MoveObjectToCollection { object, collection } => {
                    let uid = shell::object_from_key(&self.doc, object)
                        .and_then(|id| self.doc.object_uid(id));
                    if let Some(uid) = uid {
                        let mut params = PropertyMap::new();
                        params.set("uid", PropertyValue::Text(uid.raw().to_string()));
                        params.set("collection", PropertyValue::Text(collection));
                        self.run_op("object.move_to_collection", params);
                    }
                }
                Command::OpenPath(path) => {
                    if self.confirm_discarding_changes("unsaved-what-open") {
                        self.open_from(&path);
                    }
                }
                Command::SetVisible { object, visible } => {
                    let uid = shell::object_from_key(&self.doc, object)
                        .and_then(|id| self.doc.object_uid(id));
                    if let Some(uid) = uid {
                        let mut params = PropertyMap::new();
                        params.set("uid", PropertyValue::Text(uid.raw().to_string()));
                        params.set("visible", PropertyValue::Bool(visible));
                        self.run_op("object.set_visible", params);
                    }
                }
                Command::ToggleEditMode => self.toggle_edit_mode(),
                // An unknown tag is ignored, which is what a script or a keymap
                // written against a later version should get.
                Command::SetMode(tag) => {
                    if let Some(mode) = OpMode::from_tag(&tag) {
                        self.set_mode(mode);
                    }
                }
                Command::EditSketch(name) => {
                    let found = self
                        .doc
                        .objects()
                        .find(|(_, o)| o.name == name)
                        .map(|(id, _)| id);
                    if let Some(id) = found {
                        if self.op_ctx.mode != OpMode::RESTING {
                            self.set_mode(OpMode::RESTING);
                        }
                        self.doc.selection.clear();
                        self.doc.selection.add(id);
                        self.doc.selection.active = Some(id);
                        self.set_mode(OpMode::Sketch);
                    }
                }
                Command::SetSelectMode(mode) => self.set_select_mode(mode),
                Command::SetSculptBrush(brush) => {
                    self.ui.sculpt.brush =
                        brush.min(kreate_mesh::sculpt::Brush::ALL.len() as u32 - 1);
                    self.redraw.set(RedrawReason::UI);
                }
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
                Command::AdjustLast { mut params } => {
                    // A panel adjusting the last operation sends the values it
                    // shows, and hidden ones — an auto-key's frame — are not
                    // shown. Carried over from the step being adjusted, or a
                    // drag of a sidebar field would drop the key its first
                    // frame put down.
                    if let Some(last) = self.history.last() {
                        if let Some(info) = self.registry.get(&last.op_id) {
                            for desc in info.schema.iter().filter(|d| d.hidden) {
                                if params.get(desc.name).is_none() {
                                    if let Some(v) = last.params.get(desc.name) {
                                        params.set(desc.name, v.clone());
                                    }
                                }
                            }
                        }
                    }
                    match self.history.adjust_last(
                        &self.registry,
                        &mut self.doc,
                        &self.op_ctx,
                        &params,
                    ) {
                        Ok(_) => self.redraw.set(RedrawReason::SCENE_CHANGED),
                        // Said where it is seen: a value the panel takes back
                        // without a word reads as one that was ignored.
                        Err(e) => self.status(&e.to_string()),
                    }
                }
                Command::SetShading(mode) => {
                    if let Some(gfx) = self.gfx.as_mut() {
                        gfx.viewport.shading = offered(mode);
                        self.redraw.set(RedrawReason::SETTINGS);
                    }
                }
                Command::Undo => self.undo(),
                Command::Redo => self.redo(),
                Command::UndoTo(target) => {
                    while self.history.cursor() > target && self.history.undo(&mut self.doc) {}
                    while self.history.cursor() < target {
                        match self
                            .history
                            .redo(&self.registry, &mut self.doc, &self.op_ctx)
                        {
                            Ok(true) => {}
                            _ => break,
                        }
                    }
                    self.collab_after_undo();
                    self.redraw.set(RedrawReason::SCENE_CHANGED);
                }
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
                Command::SelectBone {
                    object,
                    bone,
                    extend,
                } => {
                    // The pick of whatever mode this is, on the armature being
                    // posed or edited; in object mode the armature is simply
                    // selected, which is what clicking its bone there does.
                    let Some(id) = shell::object_from_key(&self.doc, object) else {
                        continue;
                    };
                    let mut params = PropertyMap::new();
                    params.set("bone", PropertyValue::Text(bone));
                    params.set("extend", PropertyValue::Bool(extend));
                    match self.op_ctx.mode.acts_on() {
                        kreate_ops::mode::ActsOn::Bones
                            if kreate_ops::ops::pose::posed_object(&self.doc) == Some(id) =>
                        {
                            self.run_op("pose.select_pick", params)
                        }
                        kreate_ops::mode::ActsOn::BoneJoints
                            if kreate_ops::ops::armature::edited_object(&self.doc) == Some(id) =>
                        {
                            params.set("part", PropertyValue::Enum(0));
                            self.run_op("armature.select_pick", params)
                        }
                        _ => {
                            self.doc.selection.clear();
                            self.doc.selection.add(id);
                            self.redraw.set(RedrawReason::SCENE_CHANGED);
                        }
                    }
                }
                Command::FrameSelection => self.frame_selected(),
                Command::NewDocument => self.new_document(),
                Command::OpenDocument => self.open_document(),
                Command::SaveDocument => self.save_document(false),
                Command::SaveDocumentAs => self.save_document(true),
                Command::ExportScene => self.export_scene(),
                Command::AssetSource { index } => {
                    self.ui.asset_source = index;
                    self.reset_asset_list();
                }
                Command::AssetKind { index } => {
                    self.ui.asset_kind = index;
                    self.reset_asset_list();
                }
                Command::AssetResolution { index } => self.ui.asset_resolution = index,
                Command::AssetRefresh => self.reset_asset_list(),
                Command::AssetSearch { text } => {
                    self.ui.asset_query.text = text;
                    self.reset_asset_list();
                }
                Command::AssetAnimatedOnly { on } => {
                    self.ui.asset_query.animated = on;
                    self.reset_asset_list();
                }
                Command::AssetGenerate { text } => {
                    let request = match text {
                        Some(words) => Some(kreate::generate::Request::Text(words)),
                        None => rfd::FileDialog::new()
                            .add_filter("Picture", &["png", "jpg", "jpeg", "webp"])
                            .set_title("A picture of one object")
                            .pick_file()
                            .map(kreate::generate::Request::Image),
                    };
                    if let (Some(request), Some(library)) = (request, &self.assets) {
                        library.generate(request);
                        self.ui.asset_generating = true;
                        self.ui.asset_error = None;
                        self.ui.asset_busy = Some(self.ui.catalog.get("asset-gen-text"));
                    }
                }
                Command::AssetHfToken { token } => {
                    if let Err(why) = kreate::generate::save_token(&token) {
                        self.ui.asset_error = Some(why);
                    }
                }
                Command::AssetKey { key } => match if kreate::assets::Source::ALL
                    .get(self.ui.asset_source)
                    == Some(&kreate::assets::Source::Sketchfab)
                {
                    kreate::sketchfab::save_token(&key)
                } else {
                    kreate::assets::save_polypizza_key(&key)
                } {
                    Ok(()) => {
                        self.ui.asset_key_held = !key.trim().is_empty();
                        self.reset_asset_list();
                    }
                    Err(why) => self.ui.asset_error = Some(why),
                },
                Command::AssetLink { index } => {
                    if let Some(path) = self
                        .ui
                        .asset_entries
                        .get(index)
                        .and_then(|e| e.path.clone())
                    {
                        let mut params = PropertyMap::new();
                        params.set("path", PropertyValue::Text(path.display().to_string()));
                        self.run_op("object.link_library", params);
                    }
                }
                Command::AssetSelect { index } => {
                    self.ui.asset_selected = self.ui.asset_entries.get(index).map(|e| e.id.clone());
                }
                Command::AssetAdd { index } => {
                    let entry = self.ui.asset_entries.get(index).cloned();
                    // Already on its way: the first request adds it.
                    let fresh = entry
                        .as_ref()
                        .is_some_and(|e| self.ui.asset_fetching.insert(e.id.clone()));
                    if let (Some(entry), Some(library), true) = (entry, &self.assets, fresh) {
                        self.ui.asset_selected = Some(entry.id.clone());
                        let size = kreate::assets::RESOLUTIONS
                            .get(self.ui.asset_resolution)
                            .copied()
                            .unwrap_or("1k");
                        library.fetch(entry.clone(), size);
                        self.ui.asset_error = None;
                        self.ui.asset_busy = Some(self.ui.catalog.format(
                            "asset-fetching",
                            kreate_i18n::TextKind::Ui,
                            &[("name", entry.name.as_str())],
                        ));
                    }
                }
                Command::AssetAddFolder => {
                    if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                        if !self.ui.asset_folders.contains(&folder) {
                            self.ui.asset_folders.push(folder);
                            let _ = kreate::assets::save_folders(&self.ui.asset_folders);
                        }
                        self.reset_asset_list();
                    }
                }
                Command::AssetRemoveFolder { index } => {
                    if index < self.ui.asset_folders.len() {
                        self.ui.asset_folders.remove(index);
                        let _ = kreate::assets::save_folders(&self.ui.asset_folders);
                        self.reset_asset_list();
                    }
                }
                Command::ImportScene => {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Scene", kreate_io::scene_file::SCENE_FORMATS)
                        .add_filter("Kreate document", &["kreate"])
                        .set_title("Import")
                        .pick_file()
                    {
                        self.import_into(&path);
                    }
                }
                Command::ImportFormat { index } => {
                    if let Some(format) = kreate::formats::IMPORT.get(index) {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(format.name, format.extensions)
                            .set_title(format!("Import {}", format.name))
                            .pick_file()
                        {
                            self.import_into(&path);
                        }
                    }
                }
                Command::ExportFormat { index } => {
                    if let Some(format) = kreate::formats::EXPORT.get(index) {
                        let stem = self
                            .file
                            .path()
                            .and_then(|p| p.file_stem())
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "untitled".into());
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(format.name, format.extensions)
                            .set_title(format!("Export {}", format.name))
                            .set_file_name(format!("{stem}.{}", format.extensions[0]))
                            .save_file()
                        {
                            let _ = self.export_to(&path);
                        }
                    }
                }
                Command::PickWorldImage => self.pick_world_image(),
                Command::PickMaterialImage { map } => self.pick_material_image(map),
                Command::ClearMaterialImage { map } => self.clear_material_image(map),
                Command::PickNodeImage { slot, node } => self.pick_node_image(slot, node),
                Command::NewMaterial => {
                    let params = self.material_slot_param();
                    self.run_op("material.new", params)
                }
                Command::ClearWorldImage => self.clear_world_image(),
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

    /// Closes Kreate, asking first about unsaved work as a new document
    /// does. A clean close clears this session's autosaves: they are for a
    /// crash, and after one the user has had their say they would only pile
    /// up as recoveries of work that was deliberately left.
    fn quit(&mut self, event_loop: &ActiveEventLoop) {
        // Clicks on the close button while the question was up arrive after
        // its answer: once leaving is decided they leave, and while it is
        // being asked they wait. Each used to ask again, one dialog after
        // another — in a shared session, where the others' edits keep the
        // document changed, every time.
        if self.quitting {
            event_loop.exit();
            return;
        }
        if self.asking || !self.confirm_discarding_changes("unsaved-what-quit") {
            return;
        }
        self.quitting = true;
        // Out of a shared session first, so the others see the leaving now
        // rather than when the connection times out.
        self.collab_leave_quietly();
        self.file.end_cleanly();
        event_loop.exit();
    }

    /// Replaces the document with an empty one.
    fn new_document(&mut self) {
        if !self.confirm_discarding_changes("unsaved-what-new") {
            return;
        }
        // The view a fresh start shows: Kreate's own camera, out of any
        // scene camera and any glide in progress, and then framed as
        // `adopt` frames. Keeping the last document's angle, lens or
        // orthographic view made a new document look like an old one.
        self.camera = kreate::startup::starting_camera();
        self.motion = kreate_render::CameraMotion::new(kreate::startup::starting_camera());
        self.camera_view = None;
        self.camera_view_object = None;
        // The grid too, which a fresh start always shows.
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.viewport.show_grid = true;
        }
        self.adopt(load_document(None));
        self.file = DocumentFile::new();
        self.status("new document");
    }

    fn open_document(&mut self) {
        if !self.confirm_discarding_changes("unsaved-what-open") {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Kreate document", &["kreate"])
            .add_filter("Scene", kreate_io::scene_file::SCENE_FORMATS)
            .set_title("Open")
            .pick_file()
        else {
            return;
        };
        self.open_from(&path);
    }

    fn load_from(&mut self, path: &std::path::Path) {
        // A scene file rather than a document: read through the IR and built
        // into a fresh document, placement and lights and all. It keeps no
        // path, because saving back over a glTF would be saving a document as
        // something that cannot hold one.
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("kreate"))
        {
            match kreate_io::scene_file::read_scene_file(path) {
                Ok((document, notes)) => {
                    self.adopt(document);
                    self.file = DocumentFile::new();
                    let lost = notes.len();
                    for note in &notes {
                        tracing::info!(?note, "import note");
                    }
                    self.status(&format!(
                        "imported {}{}",
                        path.display(),
                        if lost > 0 {
                            format!(" ({lost} notes)")
                        } else {
                            String::new()
                        }
                    ));
                }
                Err(e) => self.status(&format!("cannot import {}: {e}", path.display())),
            }
            return;
        }
        match DocumentFile::open(path) {
            Ok((document, file)) => {
                self.adopt(document);
                // Where it was looked at from when it was saved — over the
                // framing `adopt` gave it, which is for a file that kept none.
                if let Some(view) = file.view() {
                    self.camera = view;
                    self.motion = kreate_render::CameraMotion::new(view);
                    self.camera_view = None;
                    self.camera_view_object = None;
                }
                self.file = file;
                self.remember_recent(path);
                self.status(&format!("opened {}", path.display()));
            }
            // Reported, never fatal. A file that will not open is the most
            // ordinary thing a user can hand an application, and losing the
            // session they already have over it would be indefensible.
            Err(e) => self.status(&format!("cannot open {}: {e}", path.display())),
        }
    }

    /// Puts a document at the top of File ▸ Open Recent, and keeps the list.
    fn remember_recent(&mut self, path: &std::path::Path) {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        kreate::prefs::note_recent(&mut self.ui.recent_files, &path);
        if let Some(file) = kreate::prefs::recent_path() {
            if let Err(e) = kreate::prefs::save_recent(&file, &self.ui.recent_files) {
                tracing::warn!(error = %e, "could not keep the recent files");
            }
        }
    }

    fn save_document(&mut self, ask: bool) {
        let existing = self.file.path().map(|p| p.to_path_buf());
        let path =
            match (&existing, ask) {
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
                        // Never saved: the user chooses where it lives, starting
                        // from where they last kept one rather than wherever the
                        // system's dialog last was.
                        dialog = dialog.set_file_name("untitled.kreate").set_title(
                            self.ui
                                .catalog
                                .format("save-where", kreate_i18n::TextKind::Ui, &[]),
                        );
                        if let Some(folder) = self.first_save_folder() {
                            dialog = dialog.set_directory(folder);
                        }
                    }
                    match dialog.save_file() {
                        Some(path) => path,
                        None => return,
                    }
                }
            };
        let _ = self.save_to(path);
    }

    /// Where a document never saved is offered to go: the folder of the
    /// document most recently opened or saved, or the user's Documents.
    fn first_save_folder(&self) -> Option<std::path::PathBuf> {
        self.ui
            .recent_files
            .iter()
            .filter_map(|p| p.parent())
            .find(|d| d.is_dir())
            .map(std::path::Path::to_path_buf)
            .or_else(|| {
                let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
                let documents = std::path::Path::new(&home).join("Documents");
                documents.is_dir().then_some(documents)
            })
    }

    /// Saves to `path`, which becomes the document's file. Says what happened.
    fn save_to(&mut self, path: std::path::PathBuf) -> Result<String, String> {
        // The dialog may hand back a name with no extension, and a document
        // called `model` is one the shell will not associate with anything.
        let path = if path.extension().is_none() {
            path.with_extension("kreate")
        } else {
            path
        };
        self.file.set_view(&self.camera);
        match self.file.save(&path, &self.doc) {
            Ok(()) => {
                self.remember_recent(&path);
                let message = format!("saved {}", path.display());
                self.status(&message);
                Ok(message)
            }
            Err(e) => {
                let message = format!("cannot save {}: {e}", path.display());
                self.status(&message);
                Err(message)
            }
        }
    }

    /// Writes the scene to an interchange format.
    ///
    /// The other half of `load_from`, and until now it did not exist in the
    /// application at all: a glTF scene could be opened and there was no way to
    /// write one back. §7.7's gate says a pillar is finished when the feature
    /// can be reached from the interface, and a round trip that only the
    /// converter can perform is a round trip the user does not have.
    fn export_scene(&mut self) {
        let mut dialog = rfd::FileDialog::new();
        for format in kreate::formats::EXPORT {
            dialog = dialog.add_filter(format.name, format.extensions);
        }
        let Some(path) = dialog
            .set_title("Export")
            .set_file_name("untitled.glb")
            .save_file()
        else {
            return;
        };
        let _ = self.export_to(&path);
    }

    /// Writes the scene to `path`, in the format its extension names. Says
    /// what happened.
    fn export_to(&mut self, path: &std::path::Path) -> Result<String, String> {
        let (scene, mut notes) = kreate_io::scene_from_document(&self.doc);
        let lower = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_else(|| "glb".to_string());
        let written = match lower.as_str() {
            // For printing: every shown part as it stands — modifiers applied,
            // placed in the world — in the scene's unit.
            // For CAD: each part a faceted solid, in the scene's unit.
            "step" | "stp" => {
                let parts = self.printable_parts();
                if parts.is_empty() {
                    Err("there is no geometry to write".to_string())
                } else {
                    let refs: Vec<(String, &MeshData, glam::DAffine3)> = parts
                        .iter()
                        .map(|(name, mesh, world)| (name.clone(), &**mesh, *world))
                        .collect();
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    Ok(kreate_io::step::write_step(
                        &name,
                        &refs,
                        self.doc.design.length_unit.metres(),
                    )
                    .into_bytes())
                }
            }
            "csv" => Ok(self.parts_list().into_bytes()),
            "3mf" => {
                let parts = self.printable_parts();
                if parts.is_empty() {
                    Err("there is no geometry to write".to_string())
                } else {
                    let refs: Vec<(String, &MeshData, glam::DAffine3)> = parts
                        .iter()
                        .map(|(name, mesh, world)| (name.clone(), &**mesh, *world))
                        .collect();
                    Ok(kreate_io::threemf::write_3mf(
                        &refs,
                        self.doc.design.length_unit.metres(),
                    ))
                }
            }
            "usda" => {
                // The pictures beside it, where it names them.
                let package = kreate_io::usd::write_usd(&scene);
                notes.extend(package.notes);
                match write_beside(path, &package.files) {
                    Ok(()) => Ok(package.text.into_bytes()),
                    Err(e) => Err(e),
                }
            }
            "usdc" | "usd" => {
                let package = kreate_io::usd::write_usd_binary(&scene);
                notes.extend(package.notes);
                match write_beside(path, &package.files) {
                    Ok(()) => Ok(package.text),
                    Err(e) => Err(e),
                }
            }
            "usdz" => {
                let (bytes, more) = kreate_io::usd::write_usdz(&scene);
                notes.extend(more);
                Ok(bytes)
            }
            "urdf" => {
                let name = path
                    .file_stem()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "robot".into());
                let robot = kreate_io::urdf::write_urdf(&self.doc, &name);
                notes.extend(robot.notes);
                match write_beside(path, &robot.files) {
                    Ok(()) => Ok(robot.urdf.into_bytes()),
                    Err(e) => Err(e),
                }
            }
            "obj" | "stl" => {
                // Neither format has a node, a light or a camera, so what
                // travels is the geometry at the origin. Said out loud: a user
                // who exports a lit scene to STL and finds one object in it
                // should find out here rather than in the next program.
                if !scene.lights.is_empty() || scene.nodes.len() > scene.meshes.len() {
                    notes.push(kreate_io::LossyNote::Unsupported {
                        what: format!("{lower} holds no hierarchy, lights or cameras"),
                    });
                }
                let meshes = scene.meshes.clone();
                if lower == "obj" {
                    Ok(kreate_io::write_obj(&meshes, ObjOptions::default()).into_bytes())
                } else if let Some((_, mesh)) = meshes.first() {
                    if meshes.len() > 1 {
                        notes.push(kreate_io::LossyNote::Unsupported {
                            what: format!(
                                "STL holds one unnamed solid; {} meshes were not written",
                                meshes.len() - 1
                            ),
                        });
                    }
                    let (bytes, more) =
                        kreate_io::stl::write_stl(mesh, &kreate_io::stl::StlOptions::default());
                    notes.extend(more);
                    Ok(bytes)
                } else {
                    Err("there is no geometry to write".to_string())
                }
            }
            _ => {
                let options = kreate_io::GltfExportOptions {
                    // A `.glb` full of JSON is a file no viewer opens, so the
                    // extension decides rather than a setting.
                    binary: !lower.eq_ignore_ascii_case("gltf"),
                    ..kreate_io::GltfExportOptions::default()
                };
                let (bytes, more) = kreate_io::write_gltf_scene(&scene, &options);
                notes.extend(more);
                Ok(bytes)
            }
        };
        let bytes = match written {
            Ok(bytes) => bytes,
            Err(e) => {
                let message = format!("cannot export: {e}");
                self.status(&message);
                return Err(message);
            }
        };
        for note in &notes {
            tracing::info!(?note, "export note");
        }
        match std::fs::write(path, bytes) {
            // The path is deliberately not adopted. An export is not a save,
            // and a document whose path is a `.glb` would quietly lose
            // everything the format cannot hold on the next Ctrl+S.
            Ok(()) => {
                let message = format!(
                    "exported {}{}",
                    path.display(),
                    if notes.is_empty() {
                        String::new()
                    } else {
                        format!(" ({} notes)", notes.len())
                    }
                );
                self.status(&message);
                Ok(message)
            }
            Err(e) => {
                let message = format!("cannot write {}: {e}", path.display());
                self.status(&message);
                Err(message)
            }
        }
    }

    /// Loads an image and makes it the world's background.
    ///
    /// The file read is here and the document edit is an operator, and the
    /// split is deliberate: `world.set_image` can be replayed, scripted and
    /// undone, and a version of it that opened a file could be none of those.
    ///
    /// The datablock is added outside the history, so undoing the link leaves
    /// the image in the document with nothing pointing at it. That is what
    /// Blender does with an unused image and it is the right answer — the
    /// alternative is an undo that deletes a datablock a material may have
    /// started using in the meantime.
    fn pick_world_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Environment", &["hdr", "exr", "png", "jpg", "jpeg"])
            .set_title("Environment")
            .pick_file()
        else {
            return;
        };
        self.set_world_image_from(&path);
    }

    /// The world's background from an image file.
    fn set_world_image_from(&mut self, path: &std::path::Path) {
        // Raw, and it is not a guess: an environment is an emission and §2.1's
        // working space has no upper bound, so a float file is already what a
        // shader should sample. An eight-bit one is almost always sRGB and the
        // loader only consults this for those.
        let image = match kreate_io::load_image_file(path, kreate_scene::ColorSpace::Srgb) {
            Ok(image) => image,
            Err(e) => return self.status(&format!("cannot read {}: {e}", path.display())),
        };
        let name = image.name.clone();
        let id = self.doc.add_image(image);
        let Some(uid) = self.doc.image_uid(id) else {
            return self.status("the image has no identity");
        };
        let mut params = PropertyMap::new();
        params.set(
            "uid",
            kreate_reflect::PropertyValue::Text(uid.raw().to_string()),
        );
        self.run_op("world.set_image", params);
        self.status(&format!("environment: {name}"));
    }

    /// A file dropped on the window, as Blender takes one: a document opens,
    /// a scene is added to this one — through the paste operator, so it is one
    /// step to undo and replays from its script line — and an HDR becomes the
    /// world's light.
    fn open_dropped(&mut self, path: &std::path::Path) {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "kreate" => {
                if self.confirm_discarding_changes("unsaved-what-open") {
                    self.load_from(path);
                }
            }
            e if kreate_io::scene_file::SCENE_FORMATS.contains(&e) => self.import_into(path),
            "hdr" | "exr" => self.set_world_image_from(path),
            _ => self.status(&format!("cannot open {} here", path.display())),
        }
    }

    /// Forgets what the asset browser shows, so the next frame lists afresh.
    fn reset_asset_list(&mut self) {
        self.ui.asset_entries.clear();
        self.ui.asset_error = None;
        self.ui.asset_busy = None;
        self.ui.asset_list_wanted = true;
    }

    /// Adds a scene file's contents to the document, beside what is there.
    ///
    /// Read on a thread of its own and added when it is ready, so a large
    /// model does not stop the interface: [`App::serve_imports`] runs
    /// `object.import` for it, one undo step whose script line names the file.
    fn import_into(&mut self, path: &std::path::Path) {
        self.read_in_background(path, false);
    }

    /// File ▸ Open and the recent files: a document is read where it is, a
    /// scene file on a thread with a progress card, like an import — a CAD
    /// file can take minutes, and the window must not freeze meanwhile.
    fn open_from(&mut self, path: &std::path::Path) {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("kreate"))
        {
            self.load_from(path);
        } else {
            self.read_in_background(path, true);
        }
    }

    fn read_in_background(&mut self, path: &std::path::Path, opening: bool) {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.status(&format!("reading {name}…"));
        let progress = std::sync::Arc::new(kreate_io::scene_file::ReadProgress::default());
        self.next_import += 1;
        self.importing.push(Importing {
            id: self.next_import,
            path: path.to_path_buf(),
            started: Instant::now(),
            opening,
            progress: progress.clone(),
        });
        self.redraw.set(RedrawReason::UI);
        let (tx, wake) = (self.imports.0.clone(), self.import_wake.clone());
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let result = match kreate_io::scene_file::read_scene_file_with(&path, &progress) {
                Ok((document, _)) => {
                    let bounds = document.world_bounds();
                    kreate_io::scene_file::prepare(&path, document);
                    Ok((path, bounds, opening))
                }
                Err(why) => Err((
                    path.clone(),
                    format!("cannot import {}: {why}", path.display()),
                )),
            };
            let _ = tx.send(result);
            if let Some(wake) = wake {
                wake();
            }
        });
    }

    /// Adds what the import threads have read.
    ///
    /// And asks for a frame. A thread finishing wakes the loop, not the
    /// window: without the request the model was in the document and not on
    /// the screen until the mouse next moved.
    fn serve_imports(&mut self) {
        let ready: Vec<ImportReady> = self.imports.1.try_iter().collect();
        if ready.is_empty() {
            return;
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
        for result in ready {
            let path = match &result {
                Ok((path, _, _)) | Err((path, _)) => path.clone(),
            };
            if let Some(at) = self.importing.iter().position(|i| i.path == path) {
                self.importing.remove(at);
            }
            self.publish_imports();
            match result {
                Ok((path, _, true)) => match kreate_io::scene_file::take(&path) {
                    Ok(document) => {
                        self.adopt((*document).clone());
                        self.file = DocumentFile::new();
                        self.status(&format!("opened {}", path.display()));
                    }
                    Err(why) => self.status(&format!("cannot open {}: {why}", path.display())),
                },
                Ok((path, bounds, false)) => {
                    let mut wish = self.import_wishes.remove(&path).unwrap_or_default();
                    let mut params = PropertyMap::new();
                    params.set("path", PropertyValue::Text(path.display().to_string()));
                    if !bounds.is_empty() {
                        // Scaled about the origin first, as the operator does.
                        let (lo, hi) = (bounds.min.as_dvec3(), bounds.max.as_dvec3());
                        let factor = match wish.height {
                            Some(h) if hi.z - lo.z > 1e-9 => {
                                params.set("height", PropertyValue::Float(h));
                                h / (hi.z - lo.z)
                            }
                            _ => 1.0,
                        };
                        let (lo, hi) = (lo * factor, hi * factor);
                        let offset = match wish.location {
                            // Its base's middle on the spot asked for.
                            Some(at) => {
                                at - DVec3::new((lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0, lo.z)
                            }
                            None => clear_offset((lo, hi), &self.mesh_boxes()),
                        };
                        params.set("offset", PropertyValue::Vec3(offset.to_array()));
                    }
                    let before = self.history.cursor();
                    self.run_op("object.import", params);
                    if wish.reply.is_some() {
                        let answer = if self.history.cursor() > before {
                            self.evaluate_scene();
                            Ok(self.arrived_json())
                        } else {
                            Err(format!("{} could not be added", path.display()))
                        };
                        wish.answer(answer);
                    } else if self.history.cursor() > before {
                        // The view goes to what arrived: a CAD part is a few
                        // centimetres across and lands as a dot on the grid,
                        // a building out past the far edge of the view.
                        self.evaluate_scene();
                        self.frame_selected();
                    }
                    self.status(&format!("added {}", path.display()));
                }
                Err((path, why)) => {
                    if let Some(mut wish) = self.import_wishes.remove(&path) {
                        wish.answer(Err(why.clone()));
                    }
                    if why.ends_with(kreate_io::scene_file::CANCELLED) {
                        self.status(&format!("stopped reading {}", path.display()));
                    } else {
                        self.status(&why);
                    }
                }
            }
        }
    }

    /// What an import just added — it is the selection — by name, with
    /// where each stands and how big it is.
    fn arrived_json(&self) -> serde_json::Value {
        let boxes: std::collections::HashMap<kreate_scene::ObjectId, (DVec3, DVec3)> =
            self.mesh_boxes_by_object().into_iter().collect();
        let objects: Vec<serde_json::Value> = self
            .doc
            .selection
            .objects
            .iter()
            .filter_map(|id| {
                let o = self.doc.object(*id)?;
                let mut item = serde_json::json!({ "name": o.name });
                if let Some((lo, hi)) = boxes.get(id) {
                    let round = |v: DVec3| (v * 1000.0).round() / 1000.0;
                    item["min"] = serde_json::json!(round(*lo).to_array());
                    item["size"] = serde_json::json!(round(*hi - *lo).to_array());
                }
                Some(item)
            })
            .collect();
        serde_json::json!({ "added": objects })
    }

    /// Loads an image and makes it the active material's base colour texture.
    ///
    /// The same division as `pick_world_image`, and the same consequence for
    /// undo: the datablock is added outside the history, so undoing the link
    /// leaves the image in the document with nothing pointing at it.
    fn pick_material_image(&mut self, map: u32) {
        let Some(slot) = kreate_scene::MaterialMap::ALL.get(map as usize).copied() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "hdr", "exr"])
            .set_title(match slot {
                kreate_scene::MaterialMap::BaseColor => "Base Colour",
                kreate_scene::MaterialMap::MetallicRoughness => "Metallic / Roughness",
                kreate_scene::MaterialMap::Normal => "Normal Map",
                kreate_scene::MaterialMap::Emission => "Emission Map",
                kreate_scene::MaterialMap::Occlusion => "Occlusion Map",
            })
            .pick_file()
        else {
            return;
        };
        // The slot declares the space, and not as a guess: a base colour is a
        // reflectance an artist chose by looking at it, which is what an
        // eight-bit file in that slot always holds, and a roughness or normal
        // map is a measurement stored as plain numbers. The declaration also
        // picks the atlas the image is drawn from. A float file ignores it,
        // because it has no transfer function to undo.
        let image = match kreate_io::load_image_file(&path, slot.color_space()) {
            Ok(image) => image,
            Err(e) => return self.status(&format!("cannot read {}: {e}", path.display())),
        };
        let name = image.name.clone();
        let id = self.doc.add_image(image);
        let Some(uid) = self.doc.image_uid(id) else {
            return self.status("the image has no identity");
        };
        let mut params = self.material_slot_param();
        params.set("map", kreate_reflect::PropertyValue::Enum(map));
        params.set(
            "uid",
            kreate_reflect::PropertyValue::Text(uid.raw().to_string()),
        );
        self.run_op("material.set_image", params);
        self.status(&format!("{}: {name}", slot.tag()));
    }

    /// Runs the drivers if the document changed since they last ran.
    fn run_drivers(&mut self) {
        if self.doc.version() == self.driven_at {
            return;
        }
        if self.doc.drive() > 0 {
            self.redraw.set(RedrawReason::SCENE_CHANGED);
        }
        // A rigged robot's joints, where its bones now stand.
        if self.doc.sync_robot_rigs() > 0 {
            self.redraw.set(RedrawReason::SCENE_CHANGED);
        }
        // What the parameters hold, where they have it, after whatever moved.
        if self.doc.apply_parameters() > 0 {
            self.redraw.set(RedrawReason::SCENE_CHANGED);
        }
        if self.solve_active_sketch() {
            self.redraw.set(RedrawReason::SCENE_CHANGED);
        }
        // After the run, so the run's own writes are not a change to answer.
        self.driven_at = self.doc.version();
    }

    /// After any edit, the active sketch's constraints hold again: a point
    /// dragged by hand settles where they let it, as a CAD sketcher's does.
    /// Like the drivers, not a step of its own — the edit's undo puts back a
    /// sketch that was solved, and the solve is the same every time.
    fn solve_active_sketch(&mut self) -> bool {
        let Some(id) = self.doc.selection.active else {
            return false;
        };
        let Some(object) = self.doc.object(id) else {
            return false;
        };
        if object.sketch.is_empty() {
            return false;
        }
        let Some(mesh_id) = object.mesh() else {
            return false;
        };
        let constraints = object.sketch.clone();
        let Some(mesh) = self.doc.mesh(mesh_id) else {
            return false;
        };
        let before: Vec<glam::DVec2> = mesh
            .positions()
            .iter()
            .map(|p| glam::DVec2::new(p.x as f64, p.y as f64))
            .collect();
        let mut points = before.clone();
        kreate_scene::sketch::solve(&mut points, &constraints);
        if points
            .iter()
            .zip(&before)
            .all(|(a, b)| a.distance(*b) < 1e-7)
        {
            return false;
        }
        let Some(mesh) = self.doc.mesh_mut(mesh_id) else {
            return false;
        };
        for (p, q) in mesh.positions_mut().iter_mut().zip(&points) {
            p.x = q.x as f32;
            p.y = q.y as f32;
        }
        true
    }

    /// Moves the clock to `frame` and plays every animation there.
    fn set_frame(&mut self, frame: f64) {
        if !frame.is_finite() {
            return;
        }
        self.doc.animate(frame);
        // What follows a curve moves with the clock whether or not anything
        // is keyed: put where this frame has it, not left to the drivers'
        // next run.
        self.doc.follow_paths();
        if let Some(time) = kreate_math::SceneTime::new(frame, self.doc.playback.fps) {
            // The modifier stack's time too, so anything that reads the clock
            // is evaluated at the frame on screen.
            self.time = time;
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Steps the clock by the wall time since the last step, while playing.
    ///
    /// By elapsed time rather than a frame per redraw, so the animation plays
    /// at its own rate whatever the display does — a slow frame is a skipped
    /// frame, not a slower film.
    fn advance_playback(&mut self) {
        let Some(last) = self.playing else { return };
        let now = std::time::Instant::now();
        let dt = now.duration_since(last).as_secs_f64().min(0.25);
        self.playing = Some(now);
        let next = self.doc.playback.advanced(dt);
        self.set_frame(next);
        self.redraw.set(RedrawReason::PLAYING);
    }

    /// Hands the clock to the timeline panel.
    fn sync_timeline(&mut self) {
        // Every action the dope sheet shows: the selection's, its materials'
        // and its lamps', so a blinking material's keys are somewhere to aim.
        let mut keys: Vec<f64> = kreate_ops::ops::anim::shown_actions(&self.doc)
            .into_iter()
            .filter_map(|a| self.doc.action(a))
            .flat_map(|a| a.key_frames())
            .collect();
        keys.sort_by(f64::total_cmp);
        keys.dedup();
        let p = self.doc.playback;
        self.ui.timeline = kreate_ui::timeline::TimelineView {
            frame: p.frame,
            start: p.start,
            end: p.end,
            playing: self.playing.is_some(),
            autokey: self.autokey,
            keys,
            markers: self
                .doc
                .markers
                .iter()
                .map(|m| kreate_ui::timeline::TimelineMarker {
                    frame: m.frame,
                    name: m.name.clone(),
                    camera: m.camera.is_some_and(|c| self.doc.is_camera(c)),
                })
                .collect(),
            active_camera: self
                .doc
                .selection
                .active
                .is_some_and(|a| self.doc.is_camera(a)),
        };
    }

    /// Opens an image file into an Image Texture node.
    ///
    /// The space is the graph's to say — see `ShaderGraph::image_space` — so
    /// a file picked for a node that feeds a normal map arrives as numbers.
    fn pick_node_image(&mut self, slot: u32, node: u32) {
        let space = self
            .doc
            .selection
            .active
            .and_then(|object| self.doc.material_at(object, slot as usize))
            .and_then(|m| m.graph.as_ref())
            .map(|g| g.image_space(node as usize))
            .unwrap_or(kreate_scene::ColorSpace::Srgb);
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "hdr", "exr"])
            .set_title("Image Texture")
            .pick_file()
        else {
            return;
        };
        let image = match kreate_io::load_image_file(&path, space) {
            Ok(image) => image,
            Err(e) => return self.status(&format!("cannot read {}: {e}", path.display())),
        };
        let name = image.name.clone();
        let id = self.doc.add_image(image);
        let Some(uid) = self.doc.image_uid(id) else {
            return self.status("the image has no identity");
        };
        let mut params = PropertyMap::new();
        params.set("slot", kreate_reflect::PropertyValue::Int(slot as i64));
        params.set("node", kreate_reflect::PropertyValue::Int(node as i64));
        params.set(
            "uid",
            kreate_reflect::PropertyValue::Text(uid.raw().to_string()),
        );
        self.run_op("node.set_image", params);
        self.status(&format!("image texture: {name}"));
    }

    /// Takes one of the active material's textures off.
    fn clear_material_image(&mut self, map: u32) {
        let mut params = self.material_slot_param();
        params.set("map", kreate_reflect::PropertyValue::Enum(map));
        params.set("uid", kreate_reflect::PropertyValue::Text(String::new()));
        self.run_op("material.set_image", params);
    }

    /// Parameters naming the slot the Material tab is showing.
    ///
    /// Every material operator the panel's buttons run takes one, because the
    /// slot is interface state and the operators are not allowed to read it —
    /// see `SetMaterialParams::slot`. One place, so a button added later
    /// cannot forget and quietly edit slot zero instead.
    fn material_slot_param(&self) -> PropertyMap {
        let mut params = PropertyMap::new();
        params.set(
            "slot",
            kreate_reflect::PropertyValue::Int(self.ui.material_slot as i64),
        );
        params
    }

    /// Puts the world's background back to a colour.
    fn clear_world_image(&mut self) {
        let mut params = PropertyMap::new();
        params.set("uid", kreate_reflect::PropertyValue::Text(String::new()));
        self.run_op("world.set_image", params);
    }

    /// Asks before throwing away unsaved work: save it, discard it, or stay.
    /// `true` to proceed. `what` names the step, as a catalog key.
    ///
    /// A modal dialog, and deliberately: this is the one moment where carrying
    /// on regardless destroys something the user cannot get back, and a
    /// non-blocking notification they can miss is not consent.
    fn confirm_discarding_changes(&mut self, what: &str) -> bool {
        if !self.file.is_modified(&self.doc) {
            return true;
        }
        // A scene with no objects in it has nothing to lose: closing it, or
        // opening another over it, goes ahead without asking.
        if self.doc.object_count() == 0 {
            return true;
        }
        if self.asking {
            return false;
        }
        let t = |key: &str, args: &[(&str, &str)]| {
            self.ui.catalog.format(key, kreate_i18n::TextKind::Ui, args)
        };
        let name = self
            .file
            .path()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| t("unsaved-untitled", &[]));
        let (save, discard, cancel) = (
            t("unsaved-save", &[]),
            t("unsaved-discard", &[]),
            t("unsaved-cancel", &[]),
        );
        let what = t(what, &[]);
        let mut description = t(
            "unsaved-body",
            &[("name", name.as_str()), ("what", what.as_str())],
        );
        // Windows draws its own Yes, No and Cancel unless the program says
        // it may use newer controls, which Kreate does not; the line says
        // which is which there.
        let buttons = if cfg!(windows) {
            description.push_str("\n\n");
            description.push_str(&t("unsaved-windows", &[]));
            rfd::MessageButtons::YesNoCancel
        } else {
            rfd::MessageButtons::YesNoCancelCustom(save.clone(), discard.clone(), cancel)
        };
        let mut dialog = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title(t("unsaved-title", &[]))
            .set_description(description)
            .set_buttons(buttons);
        // Owned by the window, which it then holds still: without a parent
        // the window stayed live under it, and a second close opened a
        // second question.
        if let Some(window) = &self.window {
            dialog = dialog.set_parent(window.as_ref());
        }
        self.asking = true;
        let answer = dialog.show();
        self.asking = false;
        match answer {
            rfd::MessageDialogResult::Yes => self.save_then_go_on(),
            rfd::MessageDialogResult::Custom(label) if label == save => self.save_then_go_on(),
            rfd::MessageDialogResult::No => true,
            rfd::MessageDialogResult::Custom(label) if label == discard => true,
            _ => false,
        }
    }

    /// Saves before leaving the document — asking where, for one never
    /// saved — and says whether it was: a save dialog closed without
    /// choosing, or a save that failed, leaves the user where they were.
    fn save_then_go_on(&mut self) -> bool {
        self.save_document(false);
        !self.file.is_modified(&self.doc)
    }

    /// Replaces the document and everything derived from it.
    ///
    /// The caches are not optional to clear. `pick`, `modifiers` and `evaluated`
    /// are all keyed on the document that produced them, and a new document with
    /// the old caches shows the previous file's geometry — which reads as the
    /// open having silently failed.
    fn adopt(&mut self, document: Document) {
        self.doc = document;
        // Local view named objects of the scene that went; in the new one the
        // same numbers are other objects, or none.
        self.local_view = None;
        // In a shared session the new scene is shared too, and what is added
        // to it is numbered in this participant's own range.
        if let Some(site) = self.session.as_ref().and_then(|s| s.site()) {
            self.doc.set_uid_site(site);
        }
        self.collab_may_enter(OpMode::RESTING);
        self.history = History::new();
        self.edit.clear();
        self.op_ctx.mode = OpMode::RESTING;

        self.pick = kreate::pick::PickCache::default();
        self.modifiers = kreate_scene::ModifierCache::new();
        self.evaluated.clear();
        self.hover = None;
        self.drag_select = None;
        self.ui.adjust_params = None;
        // Stopped, at the frame the document was saved at — a new one's
        // first. The last document's playback ran on into the next, and the
        // clock the modifiers read stayed at its frame.
        self.playing = None;
        let frame = self.doc.playback.frame;
        self.set_frame(frame);
        // The whole scene in view, lamp and camera included, the way Blender's
        // default file opens. Snapped rather than animated: there is no
        // previous view to keep the user's bearings from, and a quarter-second
        // glide from the last document's camera would be a glide from nowhere.
        self.frame_all_with(false);
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
        // The view goes with an autosave too, so a recovered document opens
        // where it was being looked at.
        self.file.set_view(&self.camera);
        self.file.maybe_autosave(&self.doc);
        let title = self.file.title(&self.doc);
        if title != self.shown_title {
            self.ui.document_title = title.trim_end_matches(" — Kreate").to_string();
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
                    .map(|e| e.to_script_preview(kreate::shell::SCRIPT_PREVIEW))
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
            self.collab_after_undo();
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
                self.collab_after_undo();
                println!("  redo");
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            Ok(false) => println!("  nothing to redo"),
            Err(e) => println!("  redo failed: {e}"),
        }
    }

    fn frame_all(&mut self) {
        self.frame_all_with(true);
    }

    /// Frames the whole scene, gliding to it or jumping.
    fn frame_all_with(&mut self, animate: bool) {
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
        // Animated, never teleported, when there is a view to come from. A
        // camera that jumps costs the user their sense of where they are, and
        // they then have to orbit around to rebuild it — which is a worse
        // outcome than a quarter-second wait. A document that has just been
        // opened has no such view, and jumps.
        if animate {
            self.motion
                .animate_to(self.camera, destination, kreate_render::DEFAULT_TRANSITION);
        } else {
            self.camera = destination;
        }
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
    /// What an evaluation depends on besides the document.
    fn eval_ctx(&self) -> kreate_scene::modifier::EvalCtx {
        kreate_scene::modifier::EvalCtx::at(self.time)
    }

    fn evaluate_scene(&mut self) {
        self.evaluated = self.modifiers.evaluate_scene(&self.doc, &self.eval_ctx());
        // A boolean too big to finish leaves its input as it was; said once.
        if kreate_scene::modifier::take_boolean_gave_up() {
            let message = self.ui.catalog.get("boolean-too-complex");
            self.status(&message);
        }
        // What `H` hid is hidden in edit mode only: every other object, and
        // every object outside edit mode, is drawn whole. A copy without the
        // hide layers, kept per input so an unchanged mesh is not copied again.
        let editing: Vec<_> =
            if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements {
                kreate_ops::ops::mesh::edit_targets(&self.doc)
                    .into_iter()
                    .map(|t| t.object)
                    .collect()
            } else {
                Vec::new()
            };
        let mut unhidden = Vec::new();
        for (id, mesh) in self.evaluated.iter_mut() {
            if editing.contains(id) || !kreate_mesh::selection::any_hidden(mesh) {
                continue;
            }
            let shown = self
                .unhidden
                .iter()
                .find(|(input, _)| Arc::ptr_eq(input, mesh))
                .map(|(_, out)| out.clone())
                .unwrap_or_else(|| Arc::new(kreate_mesh::selection::without_hidden(mesh)));
            unhidden.push((mesh.clone(), shown.clone()));
            *mesh = shown;
        }
        self.unhidden = unhidden;
        self.cut_sections(&editing);
        // Local view: only what it was entered with is drawn or picked.
        if let Some((aside, _)) = &self.local_view {
            self.evaluated.retain(|(id, _)| !aside.contains(id));
        }
        // In weight paint mode the painted mesh is shown by its weights.
        if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Weights {
            let active = self.doc.selection.active;
            let group = active
                .and_then(|id| self.doc.object(id))
                .and_then(|o| o.active_group.clone());
            if let Some((_, mesh)) = self
                .evaluated
                .iter_mut()
                .find(|(id, _)| Some(*id) == active)
            {
                let reuse = self
                    .weight_display
                    .as_ref()
                    .filter(|(input, g, _)| Arc::ptr_eq(input, mesh) && *g == group)
                    .map(|(_, _, out)| out.clone());
                let shown = reuse.unwrap_or_else(|| {
                    let out = Arc::new(kreate_ops::ops::paint::weight_display(
                        mesh,
                        group.as_deref(),
                    ));
                    self.weight_display = Some((mesh.clone(), group.clone(), out.clone()));
                    out
                });
                *mesh = shown;
            }
        } else {
            self.weight_display = None;
        }
        // In sculpt mode the mask and face sets are shown on the surface.
        if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt
            && self.ui.sculpt.show_overlay
        {
            let active = self.doc.selection.active;
            if let Some((_, mesh)) = self
                .evaluated
                .iter_mut()
                .find(|(id, _)| Some(*id) == active)
            {
                if let Some(shown) = self.sculpt_display.show(mesh) {
                    *mesh = shown;
                }
            }
        }
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
        let sculpting = self.sculpt_stroke.is_some();
        let Some(gfx) = self.gfx.as_mut() else { return };
        // The stroke is over, however it ended: drop the reshaped mesh so it
        // is uploaded afresh, with a vertex split that matches what was
        // sculpted — a crease the stroke made shaded as one.
        if !sculpting {
            if let Some(view) = gfx.sculpt_view.take() {
                gfx.meshes
                    .retain(|(held, _)| !Arc::ptr_eq(held, &view.held));
            }
        }
        // Skinned objects first. Each is drawn from a GPU mesh of its rest
        // shape, skinned in place, so a new pose costs a matrix upload and a
        // dispatch rather than the extract and upload a changed mesh costs —
        // the difference between playing an animated character and not. The
        // CPU's skinned mesh is still what `evaluated` holds: picking,
        // snapping and the path tracer read it.
        let mut skinned = Vec::new();
        let mut encoder: Option<wgpu::CommandEncoder> = None;
        for (id, shown) in &self.evaluated {
            let Some((rest, matrices, preserve_volume)) = self.modifiers.skin_source(shown) else {
                continue;
            };
            let reused = gfx
                .skinned
                .iter()
                .position(|e| e.object == *id && Arc::ptr_eq(&e.rest, rest))
                .map(|i| gfx.skinned.swap_remove(i));
            let fresh = reused.is_none();
            let entry = reused.or_else(|| {
                let armature = self
                    .doc
                    .skin_matrices(*id)
                    .and_then(|(a, _)| self.doc.armature(a))?;
                let (offsets, bones, weights) = kreate_scene::skin::influences(rest, armature)?;
                let cpu = RenderMeshCpu::extract(rest);
                let mesh = GpuMesh::upload(
                    &gfx.gpu.device,
                    &gfx.viewport.layouts,
                    &cpu,
                    Vec4::new(0.72, 0.73, 0.76, 1.0),
                );
                let skin = kreate_render::Skin::new(
                    &gfx.gpu.device,
                    &gfx.deformer,
                    &mesh,
                    &cpu,
                    &kreate_render::Influences {
                        offsets,
                        bones,
                        weights,
                    },
                    matrices.len() as u32,
                )?;
                Some(SkinnedGpu {
                    object: *id,
                    rest: rest.clone(),
                    shown: shown.clone(),
                    mesh,
                    skin,
                })
            });
            let Some(mut entry) = entry else { continue };
            if fresh || !Arc::ptr_eq(&entry.shown, shown) {
                let rows: Vec<kreate_render::BoneRows> = matrices
                    .iter()
                    .map(|m| {
                        kreate_render::bone_rows(glam::Affine3A::from_mat4(
                            glam::DMat4::from(*m).as_mat4(),
                        ))
                    })
                    .collect();
                if !entry.skin.set_bones(&gfx.gpu.queue, &rows) {
                    // The armature gained or lost a bone: made again next
                    // frame, and drawn the ordinary way until then.
                    continue;
                }
                let encoder = encoder.get_or_insert_with(|| {
                    gfx.gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("skin"),
                        })
                });
                entry
                    .skin
                    .set_preserve_volume(&gfx.gpu.queue, preserve_volume);
                entry.skin.dispatch(encoder, &gfx.deformer);
                entry.shown = shown.clone();
            }
            skinned.push(entry);
        }
        if let Some(encoder) = encoder {
            gfx.gpu.queue.submit([encoder.finish()]);
        }
        gfx.skinned = skinned;
        // Nothing to draw is not uploaded: a baked gas domain, or a liquid's
        // domain on a frame it has run dry, is a mesh of no faces, and a GPU
        // buffer of no bytes cannot be bound.
        let ordinary: Vec<(ObjectId, Arc<MeshData>)> = self
            .evaluated
            .iter()
            .filter(|(id, mesh)| {
                mesh.num_faces() > 0
                    && !gfx
                        .skinned
                        .iter()
                        .any(|e| e.object == *id && Arc::ptr_eq(&e.shown, mesh))
            })
            .cloned()
            .collect();
        let held: Vec<Arc<MeshData>> = gfx.meshes.iter().map(|(arc, _)| arc.clone()).collect();
        let plan = gpu_sync::plan_sync(&held, &ordinary);
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
        let mut reshaped: Option<wgpu::CommandEncoder> = None;
        for (slot, mesh) in &plan.repositions {
            let Some(Some((held, gpu_mesh))) = old.get_mut(*slot) else {
                continue;
            };
            // A sculpt stroke after its first frame: positions in, normals
            // recomputed on the GPU, nothing extracted.
            if sculpting {
                if let Some(view) = gfx
                    .sculpt_view
                    .as_mut()
                    .filter(|v| Arc::ptr_eq(&v.held, held))
                {
                    if view
                        .reshape
                        .write(&gfx.gpu.queue, gpu_mesh, mesh.positions())
                    {
                        let encoder = reshaped.get_or_insert_with(|| {
                            gfx.gpu
                                .device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("reshape"),
                                })
                        });
                        view.reshape.dispatch(encoder, &gfx.deformer);
                        view.held = mesh.clone();
                        *held = mesh.clone();
                        continue;
                    }
                }
            }
            let cpu = RenderMeshCpu::extract(mesh);
            if cpu.positions.len() as u32 == gpu_mesh.render_verts {
                gpu_mesh.update_positions(&gfx.gpu.queue, &cpu);
                *held = mesh.clone();
                // The stroke's first frame: what the rest of it reshapes.
                if sculpting && gfx.sculpt_view.is_none() {
                    gfx.sculpt_view =
                        kreate_render::Reshape::new(&gfx.gpu.device, &gfx.deformer, gpu_mesh, &cpu)
                            .map(|reshape| SculptView {
                                held: mesh.clone(),
                                reshape,
                            });
                }
            } else {
                old[*slot] = None;
            }
        }
        if let Some(encoder) = reshaped {
            gfx.gpu.queue.submit([encoder.finish()]);
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
        if !self.op_ctx.mode.edits_elements() {
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

    /// The objects a region covers, through everything: what X-ray selects.
    ///
    /// Not read from the id buffer, which keeps one object a pixel even with
    /// X-ray on — so an object wholly behind another had no pixel left and a
    /// box over both took only the front one. Each shown object's evaluated
    /// mesh is projected instead: it is in the region if a vertex is, an
    /// edge crosses the region's outline, or the region lies inside a face.
    fn xray_region_objects(&self, shape: &kreate_render::SelectShape) -> Vec<u32> {
        let Some(region) = self.xray_region(shape) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (id, mesh) in &self.evaluated {
            if !self.doc.is_shown(*id) {
                continue;
            }
            let px = region.project(mesh, self.doc.world_transform(*id));
            let hit = px.iter().flatten().any(|p| region.inside(*p))
                || mesh.edges().iter().any(|[a, b]| {
                    matches!(
                        (px[*a as usize], px[*b as usize]),
                        (Some(a), Some(b)) if region.crosses(a, b)
                    )
                })
                || (0..mesh.num_faces()).any(|f| region.face_holds_anchor(mesh, &px, f));
            if hit {
                out.push(shell::object_key(*id));
            }
        }
        out
    }

    /// The mesh elements a region covers, through everything: what X-ray
    /// selects in edit mode, numbered as the region select's bitset is.
    ///
    /// The id buffer keeps one element a pixel even with X-ray on, so a back
    /// vertex that lands on the same pixels as a front one — every vertex of
    /// a cube seen straight on — was never collected. The cages are projected
    /// instead, as [`App::xray_region_objects`] does: a vertex is in if it is
    /// inside; an edge if an end is or it crosses the outline; a face if a
    /// corner is, a side crosses, or the region lies inside it.
    fn xray_region_elements(&self, shape: &kreate_render::SelectShape) -> Vec<u32> {
        let Some(region) = self.xray_region(shape) else {
            return Vec::new();
        };
        let domain = self.pick_domain();
        // Each mesh being edited and where its elements start.
        let meshes: Vec<(ObjectId, u32)> = match self.edit_bases() {
            Some((_, groups)) => kreate_ops::ops::mesh::edit_targets(&self.doc)
                .iter()
                .filter_map(|t| {
                    let key = shell::object_key(t.object);
                    groups.iter().find(|g| g.0 == key).map(|g| (t.object, g.1))
                })
                .collect(),
            None => self
                .doc
                .selection
                .active
                .into_iter()
                .map(|id| (id, 0))
                .collect(),
        };
        let mut out = Vec::new();
        for (id, base) in meshes {
            let Some(mesh) = self
                .doc
                .object(id)
                .and_then(|o| o.mesh())
                .and_then(|m| self.doc.mesh(m))
            else {
                continue;
            };
            let px = region.project(mesh, self.doc.world_transform(id));
            let vert_in: Vec<bool> = px
                .iter()
                .map(|p| p.is_some_and(|p| region.inside(p)))
                .collect();
            let edge_in = |a: u32, b: u32| {
                vert_in[a as usize]
                    || vert_in[b as usize]
                    || matches!(
                        (px[a as usize], px[b as usize]),
                        (Some(a), Some(b)) if region.crosses(a, b)
                    )
            };
            let (kind, found): (kreate_mesh::Domain, Vec<u32>) = match domain {
                kreate_render::PickDomain::Vert => (
                    kreate_mesh::Domain::Point,
                    (0..vert_in.len() as u32)
                        .filter(|v| vert_in[*v as usize])
                        .collect(),
                ),
                kreate_render::PickDomain::Edge => (
                    kreate_mesh::Domain::Edge,
                    (0..mesh.num_edges())
                        .filter(|e| {
                            let [a, b] = mesh.edges()[*e as usize];
                            edge_in(a, b)
                        })
                        .collect(),
                ),
                _ => (
                    kreate_mesh::Domain::Face,
                    (0..mesh.num_faces())
                        .filter(|f| {
                            let ring = mesh.face_verts(*f);
                            (0..ring.len()).any(|i| edge_in(ring[i], ring[(i + 1) % ring.len()]))
                                || region.face_holds_anchor(mesh, &px, *f)
                        })
                        .collect(),
                ),
            };
            let hidden = kreate_mesh::selection::hidden_flags(mesh, kind);
            out.extend(
                found
                    .into_iter()
                    .filter(|e| !hidden.get(*e as usize).copied().unwrap_or(false))
                    .map(|e| e + base),
            );
        }
        out
    }

    /// A region in the viewport's pixels, ready to test projected geometry
    /// against. `None` before the renderer exists.
    fn xray_region(&self, shape: &kreate_render::SelectShape) -> Option<XrayRegion> {
        use kreate_math::Vec2;
        let gfx = self.gfx.as_ref()?;
        let (w, h) = gfx.viewport.render_size();
        let (w, h) = (w.max(1) as f32, h.max(1) as f32);
        let outline: Vec<Vec2> = match shape {
            kreate_render::SelectShape::Box { from, to } => {
                let (a, b) = (Vec2::new(from.0, from.1), Vec2::new(to.0, to.1));
                let (lo, hi) = (a.min(b), a.max(b));
                vec![lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
            }
            kreate_render::SelectShape::Lasso(path) => {
                path.iter().map(|p| Vec2::new(p.0, p.1)).collect()
            }
            kreate_render::SelectShape::Circle { .. } => Vec::new(),
        };
        let circle = match shape {
            kreate_render::SelectShape::Circle { centre, radius } => {
                Some((Vec2::new(centre.0, centre.1), *radius))
            }
            _ => None,
        };
        Some(XrayRegion {
            view_projection: self.camera.view_projection(w as f64 / h as f64),
            eye: self.camera.eye(),
            size: Vec2::new(w, h),
            outline,
            circle,
        })
    }

    /// What a rubber band in progress should collect.
    ///
    /// `None` when nothing is being dragged, which is the overwhelmingly common
    /// case and costs nothing.
    fn region_request(&self) -> Option<(kreate_render::SelectShape, kreate_render::Collect)> {
        // A circle-select stroke tests the brush where the pointer is now.
        let shape = match (&self.circle, self.drag_select.as_ref()) {
            (
                Some(CircleSelect {
                    radius,
                    stroke: Some(_),
                }),
                _,
            ) => {
                let (x, y) = self.cursor_viewport_px()?;
                kreate_render::SelectShape::Circle {
                    centre: (x as f32, y as f32),
                    radius: *radius,
                }
            }
            (_, Some(drag)) if drag.moved => drag.shape(),
            _ => return None,
        };
        if !self.op_ctx.mode.edits_elements() {
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
            return Some((shape, kreate_render::Collect::Objects { count }));
        }

        // Several meshes in edit mode: their elements one after another, so a
        // box over two objects selects in both.
        if let Some((_, groups)) = self.edit_bases() {
            let count = groups.iter().map(|g| g.2).sum();
            return Some((shape, kreate_render::Collect::Many { count }));
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
            shape,
            kreate_render::Collect::Elements {
                object: shell::object_key(id),
                count,
            },
        ))
    }

    /// The revolve turns about `axis` now. With no profile picked, it takes
    /// what the sketch draws once closed against the axis — so an open line
    /// beside it is picked as soon as the axis is — and drops a region that
    /// was picked but is no longer one.
    fn set_revolve_axis(&mut self, axis: [u32; 2]) {
        self.sketching.axis = Some(axis);
        let Some((id, _)) = self.sketch_target() else {
            return;
        };
        let shape = kreate::sketching::revolve_shape(axis);
        let regions = kreate_scene::feature::shape_regions(&self.doc, id, &shape);
        self.sketching
            .seeds
            .retain(|s| kreate_mesh::regions::region_at(&regions, s.as_vec2()).is_some());
        if self.sketching.seeds.is_empty() {
            self.sketching.seeds =
                kreate_scene::feature::shape_profile_seeds(&self.doc, id, &shape)
                    .into_iter()
                    .map(|s| glam::DVec2::new(s[0], s[1]))
                    .collect();
        }
    }

    /// With more than one mesh in edit mode, where each object's elements
    /// start in a region select's bitset — indexed by object key, `u32::MAX`
    /// for one not being edited — and each one's (key, base, count), in
    /// order. `None` with one mesh or none.
    fn edit_bases(&self) -> Option<EditBases> {
        if !self.op_ctx.mode.edits_elements() {
            return None;
        }
        let targets = kreate_ops::ops::mesh::edit_targets(&self.doc);
        if targets.len() < 2 {
            return None;
        }
        let domain = self.pick_domain();
        let mut groups = Vec::new();
        let mut base = 0u32;
        for t in &targets {
            let Some(mesh) = self.doc.mesh(t.mesh) else {
                continue;
            };
            let count = match domain {
                kreate_render::PickDomain::Vert => mesh.num_verts(),
                kreate_render::PickDomain::Edge => mesh.num_edges(),
                _ => mesh.num_faces(),
            };
            groups.push((shell::object_key(t.object), base, count));
            base += count;
        }
        let len = groups.iter().map(|g| g.0 as usize + 1).max().unwrap_or(0);
        let mut bases = vec![u32::MAX; len];
        for (key, base, _) in &groups {
            bases[*key as usize] = *base;
        }
        Some((bases, groups))
    }

    /// Whether the cursor is inside the viewport rectangle.
    ///
    /// Window coordinates only — deliberately not [`App::cursor_viewport_px`],
    /// which converts into the offscreen texture's pixels and so needs the
    /// graphics state to exist. Whether a press belongs to the scene is a
    /// question about the layout, not about the renderer, and tying the two
    /// together makes it untestable and answers `false` before the first frame
    /// for a reason that has nothing to do with the layout.
    /// The extra viewport under the pointer, if there is one.
    fn extra_view_at_cursor(&self) -> Option<kreate_ui::AreaId> {
        let (x, y) = self.input.cursor?;
        let (x, y) = (x as f32, y as f32);
        self.gfx.as_ref()?.extra.iter().find_map(|(id, v)| {
            let (rx, ry, w, h) = v.rect_px;
            (x >= rx && x <= rx + w && y >= ry && y <= ry + h).then_some(*id)
        })
    }

    /// The extra viewports as the interface laid them out this frame: made
    /// when an area becomes one, sized to it, forgotten when it goes; and
    /// what their buttons asked.
    fn settle_extra_views(&mut self, pixels_per_point: f32, scale: f32) {
        let rects = self.ui.extra_views.clone();
        let commands = std::mem::take(&mut self.ui.extra_view_commands);
        let main_camera = self.camera;
        let Some(gfx) = self.gfx.as_mut() else { return };
        gfx.extra.retain(|id, _| rects.iter().any(|(r, _)| r == id));
        let mut changed = false;
        for (id, rect) in rects {
            let size = (
                ((rect.width() * pixels_per_point * scale) as u32).max(1),
                ((rect.height() * pixels_per_point * scale) as u32).max(1),
            );
            let view = gfx.extra.entry(id).or_insert_with(|| {
                changed = true;
                let target =
                    ViewportTarget::new(&gfx.gpu.device, &mut gfx.egui_renderer, size.0, size.1);
                let viewport = Viewport::new(
                    &gfx.gpu.device,
                    ViewportTarget::FORMAT,
                    target.size.0,
                    target.size.1,
                );
                ExtraView {
                    viewport,
                    target,
                    // The first viewport's view, to start from.
                    camera: main_camera,
                    rect_px: (0.0, 0.0, 0.0, 0.0),
                    size,
                    fresh: true,
                }
            });
            view.rect_px = (
                rect.min.x * pixels_per_point,
                rect.min.y * pixels_per_point,
                rect.width() * pixels_per_point,
                rect.height() * pixels_per_point,
            );
            if view.size != size {
                view.size = size;
                changed = true;
            }
            if view
                .target
                .resize(&gfx.gpu.device, &mut gfx.egui_renderer, size.0, size.1)
            {
                view.viewport
                    .resize(&gfx.gpu.device, view.target.size.0, view.target.size.1);
                changed = true;
            }
        }
        for (id, command) in commands {
            let Some(view) = gfx.extra.get_mut(&id) else {
                continue;
            };
            use kreate_render::camera::{Projection, ViewAxis};
            view.camera = match command {
                0 => view.camera.looking_along(ViewAxis::PosZ),
                1 => view.camera.looking_along(ViewAxis::NegY),
                2 => view.camera.looking_along(ViewAxis::PosX),
                _ => kreate_render::Camera {
                    projection: match view.camera.projection {
                        Projection::Perspective => Projection::Orthographic,
                        Projection::Orthographic => Projection::Perspective,
                    },
                    ..view.camera
                },
            };
            changed = true;
        }
        if changed {
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
    }

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
        // And a menu or list open over it — the mode list, a header menu,
        // Shift+A. Its click chose an entry; taken by the scene as well, it
        // was an object-mode click on empty space that cleared the selection,
        // and the mode just chosen then refused to start for want of a mesh.
        // Tooltips are left out: one under the pointer is not a thing to click.
        if self
            .egui_ctx
            .layer_id_at(at)
            .is_some_and(|l| matches!(l.order, egui::Order::Middle | egui::Order::Foreground))
        {
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
        self.input.command_down = false;
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
        for travel in &mut self.drag_travel {
            *travel += delta.0.abs() + delta.1.abs();
        }
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
        self.drag_travel[slot] = 0.0;
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
            // In weight paint mode a left drag is a brush stroke, not a box.
            if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Weights {
                self.begin_paint_stroke();
                return;
            }
            if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt {
                self.begin_sculpt_stroke();
                return;
            }
            if self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Texture {
                self.begin_texture_stroke();
                return;
            }
            // The header's Circle tool: a press paints, as C's brush does.
            if self.ui.tools.select_tool == 2 {
                if self.circle.is_none() {
                    self.toggle_circle_select();
                    self.circle_from_tool = self.circle.is_some();
                }
                if self.circle.is_some() {
                    self.begin_circle_stroke();
                    return;
                }
            }
            self.begin_drag_select();
        }
    }

    /// The header's tool settings, taken into the application's own.
    fn adopt_tool_settings(&mut self) {
        let t = self.ui.tools;
        self.snap.enabled = t.snap;
        self.snap.mode =
            kreate::snap::Mode::ALL[(t.snap_mode as usize).min(kreate::snap::Mode::ALL.len() - 1)];
        self.proportional = match (t.proportional, self.proportional) {
            (false, _) => None,
            (true, None) => Some(1.0),
            (true, radius) => radius,
        };
        self.proportional_falloff = t.falloff;
        // Circle select that the Circle tool turned on goes off with it.
        if t.select_tool != 2 && self.circle_from_tool {
            self.circle_from_tool = false;
            if self.circle.is_some() {
                self.end_circle_select();
            }
        }
    }

    /// Starts a brush stroke on the active mesh's active group.
    fn begin_paint_stroke(&mut self) {
        let Some(object) = self.doc.selection.active else {
            return;
        };
        let Some(group) = self.doc.object(object).and_then(|o| o.active_group.clone()) else {
            self.status("add a vertex group to paint (Data tab)");
            return;
        };
        let paint = self.ui.paint;
        let mut params = PropertyMap::new();
        params.set("group", PropertyValue::Text(group));
        params.set("vertices", PropertyValue::Ints(Vec::new()));
        params.set("weight", PropertyValue::Float(paint.weight as f64));
        params.set("strength", PropertyValue::Float(paint.strength as f64));
        match self.history.begin_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            "paint.weight_stroke",
            &params,
        ) {
            Ok(Some(session)) => {
                self.paint_stroke = Some(PaintStroke {
                    session,
                    touched: Default::default(),
                    normals: self.paint_normals(object),
                });
                self.drive_paint_stroke();
            }
            Ok(None) => {}
            Err(e) => self.status(&format!("paint: {e}")),
        }
    }

    /// Starts a physics bake on its own thread, so the window keeps drawing
    /// and the bake can be stopped. Its result goes in through
    /// `physics.bake` like any bake — one undo step, a script line — handed
    /// over by a token.
    fn start_bake(&mut self, only: u32) {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        if self.bake_job.is_some() {
            return;
        }
        if !self.doc.objects().any(|(_, o)| !o.physics.is_empty()) {
            self.status(
                "nothing has physics: give an object a rigid body, cloth or particles first",
            );
            return;
        }
        self.bake_tokens = self.bake_tokens.wrapping_add(1).max(1);
        let job = BakeJob {
            token: self.bake_tokens,
            only,
            progress: Arc::new(AtomicU64::new(0)),
            cancel: Arc::new(AtomicBool::new(false)),
            result: Arc::new(std::sync::Mutex::new(None)),
        };
        let doc = self.doc.clone();
        let (progress, cancel, result) =
            (job.progress.clone(), job.cancel.clone(), job.result.clone());
        let spawned = std::thread::Builder::new()
            .name("bake".into())
            .spawn(move || {
                let kind = kreate_scene::physics::bake::BakeOnly::from_index(only);
                let out = kreate_scene::physics::bake::bake_only_with(&doc, kind, |done, total| {
                    progress.store(((done as u64) << 32) | total as u64, Ordering::Relaxed);
                    !cancel.load(Ordering::Relaxed)
                });
                if let Ok(mut slot) = result.lock() {
                    *slot = Some(out);
                }
            });
        if spawned.is_ok() {
            self.bake_job = Some(job);
            self.redraw.set(RedrawReason::UI);
        }
    }

    /// Hands a finished bake in, or says it was stopped; and keeps the
    /// progress where the Physics tab reads it.
    /// The reads under way, for the interface's progress card; its Cancel
    /// passed on; and a frame asked for while any runs, so the card moves.
    fn publish_imports(&mut self) {
        if let Some(id) = self.ui.cancel_import.take() {
            if let Some(job) = self.importing.iter().find(|i| i.id == id) {
                job.progress.cancel();
            }
        }
        let was = !self.ui.importing.is_empty();
        self.ui.importing = self
            .importing
            .iter()
            .map(|i| {
                let (stage, fraction) = i.progress.get();
                kreate::shell::ImportStatus {
                    id: i.id,
                    name: i
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    opening: i.opening,
                    stage,
                    fraction,
                    seconds: i.started.elapsed().as_secs(),
                    cancelling: i.progress.cancelled(),
                }
            })
            .collect();
        if !self.importing.is_empty() {
            let soon = Instant::now() + std::time::Duration::from_millis(150);
            self.ui_repaint_at = Some(self.ui_repaint_at.map_or(soon, |at| at.min(soon)));
        } else if was {
            self.redraw.set(RedrawReason::UI);
        }
    }

    fn poll_bake(&mut self) {
        let Some(job) = &self.bake_job else {
            self.ui.bake_progress = None;
            return;
        };
        let p = job.progress.load(std::sync::atomic::Ordering::Relaxed);
        self.ui.bake_progress = Some(((p >> 32) as usize, (p & 0xffff_ffff) as usize));
        let finished = job.result.lock().ok().and_then(|mut r| r.take());
        let Some(outcome) = finished else {
            // Still running: draw again, so the bar moves.
            self.redraw.set(RedrawReason::UI);
            return;
        };
        let (token, only) = (job.token, job.only);
        self.bake_job = None;
        self.ui.bake_progress = None;
        match outcome {
            Some((cache, summary)) => {
                kreate_ops::ops::physics::stage_bake(token, cache, summary);
                let mut p = PropertyMap::new();
                p.set("staged", PropertyValue::Int(token as i64));
                p.set("only", PropertyValue::Enum(only));
                self.run_op("physics.bake", p);
            }
            None => self.status("Bake cancelled"),
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// H and Shift+H in sculpt: the face set under the pointer hidden, or
    /// shown alone.
    fn face_set_visibility(&mut self, action: u32) {
        let Some(object) = self.doc.selection.active else {
            return;
        };
        let Some(ndc) = self.cursor_ndc() else { return };
        let aspect = self.viewport_aspect();
        let Some(hit) = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect)
            .filter(|h| h.object == object)
        else {
            self.status("point at the face set to hide");
            return;
        };
        let mut p = PropertyMap::new();
        p.set("action", PropertyValue::Enum(action));
        p.set("face", PropertyValue::Int(hit.face as i64));
        self.run_op("sculpt.face_set_visibility", p);
    }

    /// The brush settings of the mode the viewport is in, if it paints or
    /// sculpts: radius in points, strength.
    fn brush_values(&mut self) -> Option<(&mut f32, &mut f32)> {
        use kreate_ops::mode::ActsOn;
        match self.op_ctx.mode.acts_on() {
            ActsOn::Sculpt => Some((&mut self.ui.sculpt.radius, &mut self.ui.sculpt.strength)),
            ActsOn::Weights => Some((&mut self.ui.paint.radius, &mut self.ui.paint.strength)),
            ActsOn::Texture => Some((
                &mut self.ui.texture_paint.radius,
                &mut self.ui.texture_paint.strength,
            )),
            _ => None,
        }
    }

    /// The pointer, in points.
    fn cursor_points(&self) -> Option<egui::Pos2> {
        let ppp = self.egui_ctx.pixels_per_point().max(0.01) as f64;
        self.input
            .cursor
            .map(|(x, y)| egui::pos2((x / ppp) as f32, (y / ppp) as f32))
    }

    /// `F` (radius) or `Shift+F` (strength): the ring holds still and the
    /// pointer's distance from its middle is the new value — Blender's way.
    fn begin_brush_adjust(&mut self, strength: bool) {
        if self.paint_stroke.is_some()
            || self.sculpt_stroke.is_some()
            || self.texture_stroke.is_some()
        {
            return;
        }
        let Some(pointer) = self.cursor_points() else {
            return;
        };
        let Some((radius, value)) = self.brush_values().map(|(r, s)| (*r, *s)) else {
            return;
        };
        let reach = if strength { radius * value } else { radius };
        self.ui.brush_adjust = Some(shell::BrushAdjust {
            strength,
            center: pointer - egui::vec2(reach, 0.0),
            radius,
            was: if strength { value } else { radius },
        });
        self.status(if strength {
            "Brush strength: move the pointer, click or Enter to keep, Esc to cancel"
        } else {
            "Brush size: move the pointer, click or Enter to keep, Esc to cancel"
        });
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    fn update_brush_adjust(&mut self) {
        let (Some(adjust), Some(pointer)) = (self.ui.brush_adjust, self.cursor_points()) else {
            return;
        };
        let distance = (pointer - adjust.center).length();
        if let Some((radius, strength)) = self.brush_values() {
            if adjust.strength {
                *strength = (distance / adjust.radius.max(1.0)).clamp(0.0, 1.0);
            } else {
                *radius = distance.clamp(4.0, 400.0);
            }
        }
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// Ends `F`: kept, or put back as it was.
    fn end_brush_adjust(&mut self, keep: bool) {
        let Some(adjust) = self.ui.brush_adjust.take() else {
            return;
        };
        if !keep {
            if let Some((radius, strength)) = self.brush_values() {
                if adjust.strength {
                    *strength = adjust.was;
                } else {
                    *radius = adjust.was;
                }
            }
        }
        self.status("");
        self.redraw.set(RedrawReason::HOVER_CHANGED);
    }

    /// The brush strokes in progress, once for every move of the pointer
    /// since the last frame.
    fn drive_strokes(&mut self) {
        if self.paint_stroke.is_some() {
            self.drive_paint_stroke();
        }
        if self.sculpt_stroke.is_some() {
            self.drive_sculpt_stroke();
        }
        if self.texture_stroke.is_some() {
            self.drive_texture_stroke();
        }
    }

    /// Where the pointer meets the active mesh, in the world, if it does.
    fn sculpt_hit(&mut self) -> Option<DVec3> {
        let object = self.doc.selection.active?;
        let ndc = self.cursor_ndc()?;
        let aspect = self.viewport_aspect();
        let hit = match self
            .sculpt_stroke
            .as_ref()
            .and_then(|s| s.pick_scene.as_ref())
        {
            Some(scene) => {
                self.stroke_pick
                    .raycast_ndc(&self.doc, scene, &self.camera, ndc, aspect)?
            }
            None => self
                .pick
                .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect)?,
        };
        (hit.object == object).then_some(hit.point)
    }

    /// A press in sculpt mode: a stroke starts where the pointer meets the
    /// mesh. Shift smooths whatever the brush is; Ctrl inverts it.
    fn begin_sculpt_stroke(&mut self) {
        let Some(object) = self
            .doc
            .selection
            .active
            .filter(|a| self.doc.object(*a).and_then(|o| o.mesh()).is_some())
        else {
            return;
        };
        let Some(at) = self.sculpt_hit() else { return };
        let settings = self.ui.sculpt.clone();
        let brush = if self.input.shift_down {
            1
        } else {
            settings.brush
        };
        let face_set = self
            .doc
            .object(object)
            .and_then(|o| o.mesh())
            .and_then(|m| self.doc.mesh(m))
            .map(|m| {
                kreate_mesh::sculpt::face_sets_of(m)
                    .into_iter()
                    .max()
                    .unwrap_or(0)
                    + 1
            })
            .unwrap_or(1);
        // The brush's screen radius as a world length at the depth it touched.
        let height = self
            .viewport_px
            .map(|(_, _, _, h)| h as f64)
            .unwrap_or(1.0)
            .max(1.0);
        let ppp = self.egui_ctx.pixels_per_point() as f64;
        let distance = (at - self.camera.eye()).length();
        let per_pixel = 2.0 * (self.camera.fov_y * 0.5).tan() * distance / height;
        let radius_world = settings.radius as f64 * ppp * per_pixel;
        self.sculpt_stroke = Some(SculptStroke {
            object,
            brush,
            invert: self.input.ctrl_down,
            dabs: Vec::new(),
            pressures: Vec::new(),
            face_set,
            pick_scene: self.sculpt_dyntopo(object, settings.dyntopo).then(|| {
                self.evaluated
                    .iter()
                    .filter(|(id, _)| *id == object)
                    .cloned()
                    .collect()
            }),
            last: at,
            radius_world,
            hold: at,
            sent: None,
            session: None,
        });
        self.add_sculpt_dab(at);
        self.drive_sculpt_stroke();
    }

    fn add_sculpt_dab(&mut self, at: DVec3) {
        let pressure = self.input.pen_pressure.unwrap_or(1.0).clamp(0.0, 1.0) as f64;
        let settings = &self.ui.sculpt;
        let (by_strength, by_size) = (settings.pressure_strength, settings.pressure_size);
        let Some(stroke) = self.sculpt_stroke.as_mut() else {
            return;
        };
        let to_local = self.doc.world_transform(stroke.object).inverse();
        let local = to_local.transform_point3(at);
        let size = if by_size { pressure.max(0.1) } else { 1.0 };
        let radius = to_local
            .transform_vector3(DVec3::X * stroke.radius_world * size)
            .length();
        stroke.dabs.extend([local.x, local.y, local.z, radius]);
        stroke
            .pressures
            .push(if by_strength { pressure } else { 1.0 });
        stroke.last = at;
    }

    /// The pointer moved during a stroke: a new dab once it has gone a
    /// quarter of the brush, or for a grab the drag so far — then the stroke
    /// re-run from where it began.
    fn drive_sculpt_stroke(&mut self) {
        let Some(stroke) = self.sculpt_stroke.as_ref() else {
            return;
        };
        let (object, brush, last, radius, hold) = (
            stroke.object,
            stroke.brush,
            stroke.last,
            stroke.radius_world,
            stroke.hold,
        );
        let to_local = self.doc.world_transform(object).inverse();
        let mut grab = DVec3::ZERO;
        if brush == 2 {
            // The pointer on the plane through where it took hold, facing the view.
            if let Some(ndc) = self.cursor_ndc() {
                let aspect = self.viewport_aspect();
                let eye = self.camera.eye();
                let dir = self.camera.ray_direction(ndc, aspect).as_dvec3();
                let n = self.camera.forward();
                let denom = dir.dot(n);
                if denom.abs() > 1e-9 {
                    let on = eye + dir * ((hold - eye).dot(n) / denom);
                    grab = to_local.transform_vector3(on - hold);
                }
            }
        } else if let Some(at) = if brush == 11 {
            // The snake hook pulls the surface off itself: the pointer on the
            // plane through the last dab, facing the view.
            self.cursor_ndc().and_then(|ndc| {
                let aspect = self.viewport_aspect();
                let eye = self.camera.eye();
                let dir = self.camera.ray_direction(ndc, aspect).as_dvec3();
                let n = self.camera.forward();
                let denom = dir.dot(n);
                (denom.abs() > 1e-9).then(|| eye + dir * ((last - eye).dot(n) / denom))
            })
        } else {
            self.sculpt_hit()
        } {
            // The stabiliser: the brush trails the pointer.
            let lazy = self.ui.sculpt.lazy.clamp(0.0, 0.95) as f64;
            let at = last.lerp(at, 1.0 - lazy);
            let step = radius * (self.ui.sculpt.spacing.max(0.02) as f64);
            let gap = at.distance(last);
            if gap >= step {
                // Driven once a frame, a fast pointer has gone several
                // spacings: the dabs between, on the straight line, so the
                // stroke stays a line rather than a row of beads.
                let n = ((gap / step).floor() as usize).clamp(1, 64);
                for k in 1..n {
                    self.add_sculpt_dab(last.lerp(at, k as f64 / n as f64));
                }
                self.add_sculpt_dab(at);
            }
        }
        let settings = self.ui.sculpt.clone();
        let view = to_local.transform_vector3(self.camera.forward());
        let Some(stroke) = self.sculpt_stroke.as_mut() else {
            return;
        };
        // Called every frame of the stroke; most frames the pointer has not
        // gone far enough for a new dab, and re-running would redo the same.
        let now = (stroke.dabs.len(), grab);
        if stroke.session.is_some() && stroke.sent == Some(now) {
            return;
        }
        stroke.sent = Some(now);
        let mut params = PropertyMap::new();
        params.set("brush", PropertyValue::Enum(stroke.brush));
        params.set("dabs", PropertyValue::Floats(stroke.dabs.clone()));
        params.set("strength", PropertyValue::Float(settings.strength as f64));
        params.set("invert", PropertyValue::Bool(stroke.invert));
        params.set("grab", PropertyValue::Vec3(grab.to_array()));
        params.set("mirror_x", PropertyValue::Bool(settings.mirror_x));
        params.set("mirror_y", PropertyValue::Bool(settings.mirror_y));
        params.set("mirror_z", PropertyValue::Bool(settings.mirror_z));
        params.set("radial", PropertyValue::Int(settings.radial.max(1) as i64));
        params.set("falloff", PropertyValue::Enum(settings.falloff));
        params.set("view", PropertyValue::Vec3(view.to_array()));
        params.set("front_only", PropertyValue::Bool(settings.front_only));
        params.set(
            "auto_smooth",
            PropertyValue::Float(settings.auto_smooth as f64),
        );
        params.set("alpha", PropertyValue::Text(settings.alpha.clone()));
        params.set("face_set", PropertyValue::Int(stroke.face_set as i64));
        if stroke.pressures.iter().any(|p| *p != 1.0) {
            params.set("pressures", PropertyValue::Floats(stroke.pressures.clone()));
        }
        if stroke.pick_scene.is_some() {
            params.set("dyntopo", PropertyValue::Bool(true));
            params.set("detail", PropertyValue::Float(settings.detail as f64));
        }
        match stroke.session.as_mut() {
            Some(session) => {
                if let Err(e) = self.history.update_modal(
                    &self.registry,
                    &mut self.doc,
                    None,
                    &self.op_ctx,
                    session,
                    &params,
                ) {
                    tracing::warn!(error = %e, "sculpt failed");
                }
            }
            None => {
                if let Ok(Some(session)) = self.history.begin_modal(
                    &self.registry,
                    &mut self.doc,
                    None,
                    &self.op_ctx,
                    "sculpt.stroke",
                    &params,
                ) {
                    stroke.session = Some(session);
                }
            }
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// What dynamic topology would throw away from an object's mesh: its
    /// layers that matter. UVs count only when a texture of its materials
    /// reads them — a primitive comes with UVs nothing uses.
    fn dyntopo_loses(&self, object: ObjectId) -> Vec<String> {
        let Some(o) = self.doc.object(object) else {
            return Vec::new();
        };
        let Some(mesh_id) = o.mesh() else {
            return Vec::new();
        };
        let Some(mesh) = self.doc.mesh(mesh_id) else {
            return Vec::new();
        };
        let textured = self
            .doc
            .mesh_slots(mesh_id)
            .iter()
            .chain(o.material_overrides.iter())
            .flatten()
            .filter_map(|m| self.doc.material(*m))
            .any(|m| !m.images().is_empty());
        let mut lost = kreate_mesh::dyntopo::loses(mesh);
        if !textured {
            lost.retain(|name| !name.starts_with("uv"));
        }
        lost
    }

    /// Whether this stroke cuts the surface finer as it goes. Said once per
    /// object when Auto holds back for what it would throw away.
    fn sculpt_dyntopo(&mut self, object: ObjectId, mode: u32) -> bool {
        match mode {
            kreate::shell::DYNTOPO_ON => true,
            kreate::shell::DYNTOPO_OFF => false,
            _ => {
                let lost = self.dyntopo_loses(object);
                if lost.is_empty() {
                    return true;
                }
                if self.dyntopo_told != Some(object) {
                    self.dyntopo_told = Some(object);
                    let message = self
                        .ui
                        .catalog
                        .get_args("sculpt-dyntopo-kept", &[("what", &lost.join(", "))]);
                    self.status(&message);
                }
                false
            }
        }
    }

    /// The image texture paint paints on: the active object's first
    /// material's base colour, by identity.
    fn paint_target(&self) -> Option<String> {
        let object = self.doc.selection.active?;
        let mesh = self.doc.object(object)?.mesh()?;
        let material = self.doc.mesh_slots(mesh).first().copied().flatten()?;
        let image = self.doc.material(material)?.base_color_image?;
        Some(self.doc.image_uid(image)?.raw().to_string())
    }

    /// Where the pointer meets the active mesh, as a UV and how many UV units
    /// a unit of the mesh's own length is there.
    fn texture_hit(&mut self) -> Option<(DVec3, glam::Vec2, f64)> {
        let object = self.doc.selection.active?;
        let ndc = self.cursor_ndc()?;
        let aspect = self.viewport_aspect();
        let hit = self
            .pick
            .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect)?;
        if hit.object != object {
            return None;
        }
        let mesh = self
            .evaluated
            .iter()
            .find(|(id, _)| *id == object)
            .map(|(_, m)| m.clone())?;
        let local = self
            .doc
            .world_transform(object)
            .inverse()
            .transform_point3(hit.point)
            .as_vec3();
        let (uv, scale) = uv_on_face(&mesh, hit.face, local)?;
        Some((hit.point, uv, scale))
    }

    /// A press in texture paint mode: a stroke starts where the pointer
    /// meets the mesh, on its texture.
    fn begin_texture_stroke(&mut self) {
        let Some(object) = self.doc.selection.active else {
            return;
        };
        let Some(image) = self.paint_target() else {
            let message = self.ui.catalog.get("texture-paint-no-image");
            self.status(&message);
            return;
        };
        let Some((at, _, _)) = self.texture_hit() else {
            return;
        };
        let height = self
            .viewport_px
            .map(|(_, _, _, h)| h as f64)
            .unwrap_or(1.0)
            .max(1.0);
        let ppp = self.egui_ctx.pixels_per_point() as f64;
        let distance = (at - self.camera.eye()).length();
        let per_pixel = 2.0 * (self.camera.fov_y * 0.5).tan() * distance / height;
        let radius_world = self.ui.texture_paint.radius as f64 * ppp * per_pixel;
        self.texture_stroke = Some(TextureStroke {
            object,
            image,
            dabs: Vec::new(),
            last: at,
            radius_world,
            session: None,
        });
        self.add_texture_dab();
        self.drive_texture_stroke();
    }

    fn add_texture_dab(&mut self) {
        let Some((at, uv, scale)) = self.texture_hit() else {
            return;
        };
        let strength = self.ui.texture_paint.strength as f64;
        let Some(stroke) = self.texture_stroke.as_mut() else {
            return;
        };
        let to_local = self.doc.world_transform(stroke.object).inverse();
        let radius_local = to_local
            .transform_vector3(DVec3::X * stroke.radius_world)
            .length();
        stroke
            .dabs
            .extend([uv.x as f64, uv.y as f64, radius_local * scale, strength]);
        stroke.last = at;
    }

    /// The pointer moved during a stroke: a new dab once it has gone a
    /// quarter of the brush, then the stroke re-run from where it began.
    fn drive_texture_stroke(&mut self) {
        let Some(stroke) = self.texture_stroke.as_ref() else {
            return;
        };
        let (last, radius) = (stroke.last, stroke.radius_world);
        if let Some((at, _, _)) = self.texture_hit() {
            if at.distance(last) >= radius * 0.25 {
                self.add_texture_dab();
            }
        }
        let color = self.ui.texture_paint.color;
        let Some(stroke) = self.texture_stroke.as_mut() else {
            return;
        };
        let mut params = PropertyMap::new();
        params.set("image", PropertyValue::Text(stroke.image.clone()));
        params.set("dabs", PropertyValue::Floats(stroke.dabs.clone()));
        params.set("color", PropertyValue::Vec3(color.map(|c| c as f64)));
        match stroke.session.as_mut() {
            Some(session) => {
                if let Err(e) = self.history.update_modal(
                    &self.registry,
                    &mut self.doc,
                    None,
                    &self.op_ctx,
                    session,
                    &params,
                ) {
                    tracing::warn!(error = %e, "texture paint failed");
                }
            }
            None => {
                if let Ok(Some(session)) = self.history.begin_modal(
                    &self.registry,
                    &mut self.doc,
                    None,
                    &self.op_ctx,
                    "paint.texture_stroke",
                    &params,
                ) {
                    stroke.session = Some(session);
                }
            }
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// The painted mesh's vertex normals, in the world, as it is drawn.
    fn paint_normals(&self, object: kreate_scene::document::ObjectId) -> Vec<DVec3> {
        let Some(mesh) = self
            .evaluated
            .iter()
            .find(|(id, _)| *id == object)
            .map(|(_, m)| m.clone())
        else {
            return Vec::new();
        };
        let p = mesh.positions();
        let mut normals = vec![DVec3::ZERO; p.len()];
        for f in 0..mesh.num_faces() {
            let ring = mesh.face_verts(f);
            let mut n = DVec3::ZERO;
            for i in 0..ring.len() {
                n += p[ring[i] as usize]
                    .as_dvec3()
                    .cross(p[ring[(i + 1) % ring.len()] as usize].as_dvec3());
            }
            for v in ring {
                normals[*v as usize] += n;
            }
        }
        let to_world = self
            .doc
            .world_transform(object)
            .matrix3
            .inverse()
            .transpose();
        normals
            .into_iter()
            .map(|n| (to_world * n).normalize_or_zero())
            .collect()
    }

    /// Adds the vertices under the brush to the stroke and re-applies it.
    ///
    /// On the evaluated mesh, as it is on screen — a posed character is
    /// painted where it stands — with the same vertex numbering as the
    /// document's, which a skin or a stack that keeps the topology has; a
    /// stack that changes it paints the cage instead.
    fn drive_paint_stroke(&mut self) {
        let (Some((x, y)), Some(projector), Some(object)) = (
            self.input.cursor,
            self.ui.projector,
            self.doc.selection.active,
        ) else {
            return;
        };
        let ppp = self.egui_ctx.pixels_per_point();
        let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
        let radius = self.ui.paint.radius;
        let Some(cage) = self
            .doc
            .object(object)
            .and_then(|o| o.mesh())
            .and_then(|m| self.doc.mesh(m))
        else {
            return;
        };
        let shown = self
            .evaluated
            .iter()
            .find(|(id, _)| *id == object)
            .map(|(_, m)| m.clone())
            .filter(|m| m.num_verts() == cage.num_verts());
        let world = self.doc.world_transform(object);
        let positions: Vec<glam::Vec3> = match &shown {
            Some(m) => m.positions().to_vec(),
            None => cage.positions().to_vec(),
        };
        let paint = self.ui.paint;
        let eye = self.camera.eye();
        let Some(stroke) = self.paint_stroke.as_mut() else {
            return;
        };
        let mut grew = stroke.touched.is_empty();
        for (v, p) in positions.iter().enumerate() {
            let w = world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
            let Some(d) = projector
                .project(w)
                .map(|s| s.distance(at))
                .filter(|d| *d <= radius)
            else {
                continue;
            };
            // Only when the normals are this mesh's: a stack that changed the
            // vertex count paints the cage, and has none here.
            let facing_away = stroke.normals.len() == positions.len()
                && stroke.normals.get(v).is_some_and(|n| n.dot(eye - w) <= 0.0);
            if paint.front_only && facing_away {
                continue;
            }
            // Blender's Smooth falloff: full in the middle, nothing at the rim.
            let amount = if paint.falloff {
                let t = 1.0 - d / radius.max(1e-3);
                t * t * (3.0 - 2.0 * t)
            } else {
                1.0
            };
            let slot = stroke.touched.entry(v as u32).or_insert(0.0);
            if amount > *slot + 1e-3 {
                *slot = amount;
                grew = true;
            }
        }
        if !grew {
            return;
        }
        let touched = &stroke.touched;
        let session = &mut stroke.session;
        let paint = self.ui.paint;
        let group = self
            .doc
            .object(object)
            .and_then(|o| o.active_group.clone())
            .unwrap_or_default();
        let mut params = PropertyMap::new();
        params.set("group", PropertyValue::Text(group));
        params.set(
            "vertices",
            PropertyValue::Ints(touched.keys().copied().collect()),
        );
        params.set(
            "amounts",
            PropertyValue::Floats(touched.values().map(|a| *a as f64).collect()),
        );
        params.set("weight", PropertyValue::Float(paint.weight as f64));
        params.set("strength", PropertyValue::Float(paint.strength as f64));
        if let Err(e) = self.history.update_modal(
            &self.registry,
            &mut self.doc,
            Some(&mut self.edit),
            &self.op_ctx,
            session,
            &params,
        ) {
            tracing::warn!(error = %e, "stroke failed");
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Ends whatever that button was doing.
    ///
    /// Runs even when the interface claimed the release — over a panel, a menu,
    /// the navigation gizmo — because a button that went down has to come up. It
    /// not doing so was what made navigation stick on and then appear broken.
    fn end_mouse_drag(&mut self, slot: usize) {
        // A stroke ends where the button comes up: one step to undo — with
        // the last movement in it, which waits for a frame otherwise.
        if slot == 0 && std::mem::take(&mut self.stroke_moved) {
            self.drive_strokes();
        }
        if slot == 0 {
            if self
                .sculpt_stroke
                .as_ref()
                .is_some_and(|s| s.pick_scene.is_some())
            {
                self.stroke_pick = kreate::pick::PickCache::new();
            }
            if let Some(SculptStroke {
                session: Some(session),
                ..
            }) = self.sculpt_stroke.take()
            {
                self.history.commit_modal(session);
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            if let Some(TextureStroke {
                session: Some(session),
                ..
            }) = self.texture_stroke.take()
            {
                self.history.commit_modal(session);
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            if let Some(PaintStroke { session, .. }) = self.paint_stroke.take() {
                self.history.commit_modal(session);
                if let Some(entry) = self.history.last() {
                    println!("{}", entry.to_script_preview(kreate::shell::SCRIPT_PREVIEW));
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
        }
        let was_navigating = self.drag_ops[slot].take().is_some();
        // Alt with the left button orbits when it drags; left where it was
        // pressed, in edit mode, it is Blender's loop select.
        if was_navigating
            && slot == 0
            && self.input.alt_down
            && self.drag_travel[0] < 4.0
            && self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements
        {
            self.loop_pick();
            return;
        }
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
            lasso: (self.ui.tools.select_tool == 1).then(|| vec![origin]),
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
        // A lasso keeps the path, a point every couple of pixels.
        if let Some(path) = drag.lasso.as_mut() {
            let last = *path.last().unwrap_or(&at);
            if (at.0 - last.0).hypot(at.1 - last.1) > 2.0 {
                path.push(at);
            }
        }
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
            // Blender's Ctrl+click in edit mode: the shortest path from the
            // selection to what was clicked.
            if drag.deselect && self.op_ctx.mode.edits_elements() {
                if let Some(hit) = self.hover {
                    let mode = match self.pick_domain() {
                        kreate_render::PickDomain::Vert => 0u32,
                        kreate_render::PickDomain::Edge => 1,
                        _ => 2,
                    };
                    let mut params = PropertyMap::new();
                    params.set("element", PropertyValue::Int(hit.element as i64));
                    params.set("mode", PropertyValue::Enum(mode));
                    self.run_op("mesh.select_shortest_path", params);
                }
                return;
            }
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

        if !self.op_ctx.mode.edits_elements() {
            // The reduction collected object ids, which are the same keys the
            // outliner round-trips — so this goes through the same command, and
            // the same resolution against the *current* document.
            let mut objects = resolved;
            if self.xray {
                for key in self.xray_region_objects(&drag.shape()) {
                    if !objects.contains(&key) {
                        objects.push(key);
                    }
                }
            }
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

        let mut elements = resolved;
        if self.xray {
            for e in self.xray_region_elements(&drag.shape()) {
                if !elements.contains(&e) {
                    elements.push(e);
                }
            }
        }
        // Several meshes: each bit back to its object and its own index.
        let mut objects = Vec::new();
        let mut counts = Vec::new();
        if let Some((_, groups)) = self.edit_bases() {
            let mut split = Vec::with_capacity(elements.len());
            for (key, base, count) in groups {
                let mine: Vec<u32> = elements
                    .iter()
                    .filter(|e| **e >= base && **e < base + count)
                    .map(|e| e - base)
                    .collect();
                objects.push(key + 1);
                counts.push(mine.len() as u32);
                split.extend(mine);
            }
            elements = split;
        }
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
        if !objects.is_empty() {
            params.set("objects", PropertyValue::Ints(objects));
            params.set("counts", PropertyValue::Ints(counts));
        }
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
        // Exhaustive on what the mode acts on, so a mode that acts on bones
        // has to say what a click does rather than landing in a default.
        match self.op_ctx.mode.acts_on() {
            // The sketch tools take their clicks before selection sees them.
            kreate_ops::mode::ActsOn::Sketch => {}
            kreate_ops::mode::ActsOn::MeshElements => {
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
            // A click in weight paint or sculpt mode is a stroke, handled on
            // the press.
            kreate_ops::mode::ActsOn::Weights
            | kreate_ops::mode::ActsOn::Sculpt
            | kreate_ops::mode::ActsOn::Texture => {}
            kreate_ops::mode::ActsOn::BoneJoints => {
                let Some(object) = kreate_ops::ops::armature::edited_object(&self.doc) else {
                    return true;
                };
                let (Some((x, y)), Some(projector)) = (self.input.cursor, self.ui.projector) else {
                    return true;
                };
                let ppp = self.egui_ctx.pixels_per_point();
                let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
                match shell::joint_under(&self.doc, object, &projector, at) {
                    Some((bone, part)) => {
                        let mut params = PropertyMap::new();
                        params.set("bone", PropertyValue::Text(bone));
                        params.set("part", PropertyValue::Enum(part));
                        params.set("extend", PropertyValue::Bool(extend));
                        self.run_op("armature.select_pick", params);
                    }
                    None if !extend => {
                        let mut params = PropertyMap::new();
                        params.set("action", PropertyValue::Enum(1));
                        self.run_op("armature.select_all", params);
                    }
                    None => {}
                }
            }
            kreate_ops::mode::ActsOn::Bones => {
                let Some(object) = kreate_ops::ops::pose::posed_object(&self.doc) else {
                    return true;
                };
                let (Some((x, y)), Some(projector)) = (self.input.cursor, self.ui.projector) else {
                    return true;
                };
                let ppp = self.egui_ctx.pixels_per_point();
                let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
                match shell::bone_under(&self.doc, object, &projector, at) {
                    Some((_, bone)) => {
                        let mut params = PropertyMap::new();
                        params.set("bone", PropertyValue::Text(bone));
                        params.set("extend", PropertyValue::Bool(extend));
                        self.run_op("pose.select_pick", params);
                    }
                    // A miss deselects, as it does for objects and elements —
                    // but not when extending.
                    None if !extend => {
                        let mut params = PropertyMap::new();
                        params.set("action", PropertyValue::Enum(1));
                        self.run_op("pose.select_all", params);
                    }
                    None => {}
                }
            }
            kreate_ops::mode::ActsOn::Objects => {
                // A lamp's or camera's glyph first, because it is drawn on
                // top. The id buffer has never heard of them — they have no
                // surface — so without this a click on a lamp selected
                // whatever was behind it, or nothing, and lights could not be
                // selected in the viewport at all.
                // Then a sketch's lines, drawn on top as well and as
                // invisible to the id buffer.
                let glyph = self
                    .glyph_under_cursor()
                    .or_else(|| self.sketch_under_cursor());
                let hit_key = match (glyph, hit) {
                    (Some(id), _) => Some(shell::object_key(id)),
                    (None, Some(hit)) => Some(hit.object),
                    (None, None) => None,
                };
                // Picking objects for a boolean or a field: the click chooses
                // rather than selects.
                if self.ui.object_pick.is_some() {
                    if let Some(key) = hit_key {
                        self.pick_object(key);
                    }
                    return true;
                }
                let (objects, active) = match hit_key {
                    // Through one rule for a plain click and a shift-click,
                    // the same one the outliner applies, so the two ways of
                    // selecting an object cannot disagree about what
                    // shift means.
                    Some(key) => {
                        let selected: Vec<u32> = self
                            .doc
                            .selection
                            .objects
                            .iter()
                            .map(|id| shell::object_key(*id))
                            .collect();
                        let active = self.doc.selection.active.map(shell::object_key);
                        shell::click_selection(&selected, active, key, extend)
                    }
                    // Clicking empty space clears the selection — but not when
                    // extending, where the user is adding to it and a miss
                    // should leave what they have alone.
                    None if extend => return true,
                    None => (Vec::new(), None),
                };
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

    /// Starts Fusion's Combine: the target and the tools chosen in the
    /// viewport, beginning from the selection — the active mesh the target,
    /// the other selected meshes its tools.
    fn begin_boolean_pick(&mut self, operation: u32) {
        let is_mesh = |doc: &Document, id: kreate_scene::ObjectId| {
            doc.object(id).and_then(|o| o.mesh()).is_some()
        };
        let target = self.doc.selection.active.filter(|a| is_mesh(&self.doc, *a));
        let tools: Vec<u32> = self
            .doc
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| Some(*id) != target && is_mesh(&self.doc, *id))
            .map(shell::object_key)
            .collect();
        self.ui.object_pick = Some(kreate::shell::ObjectPick::Boolean {
            operation: operation.min(2),
            target: target.map(shell::object_key),
            tools,
            apply: false,
        });
        self.show_pick_selection();
        let message = self.ui.catalog.get("pick-boolean-hint-target");
        self.status(&message);
    }

    /// A click on an object while picking.
    fn pick_object(&mut self, key: u32) {
        let Some(pick) = self.ui.object_pick.as_mut() else {
            return;
        };
        // Only a mesh can be cut or cut with.
        let is_mesh = shell::object_from_key(&self.doc, key)
            .and_then(|id| self.doc.object(id))
            .is_some_and(|o| o.mesh().is_some());
        if matches!(pick, kreate::shell::ObjectPick::Boolean { .. }) && !is_mesh {
            return;
        }
        if pick.click(key) {
            if let Some(kreate::shell::ObjectPick::Field { op_id, mut params }) =
                self.ui.object_pick.take()
            {
                if let Some(o) =
                    shell::object_from_key(&self.doc, key).and_then(|id| self.doc.object(id))
                {
                    params.set("object", PropertyValue::Text(o.name.clone()));
                    self.run_op(&op_id, params);
                }
            }
        } else {
            self.show_pick_selection();
        }
        self.redraw.set(RedrawReason::UI);
    }

    /// The boolean's choice, shown as the selection: the tools selected, the
    /// target active, so their outlines say which is which.
    fn show_pick_selection(&mut self) {
        let Some(kreate::shell::ObjectPick::Boolean { target, tools, .. }) = &self.ui.object_pick
        else {
            return;
        };
        let objects: Vec<u32> = tools.clone();
        let active = *target;
        self.ui
            .queue
            .push(kreate_ui::Command::SetSelection { objects, active });
    }

    /// Done: the boolean made from what was picked.
    fn finish_object_pick(&mut self) {
        let Some(kreate::shell::ObjectPick::Boolean {
            operation,
            target,
            tools,
            apply,
        }) = self.ui.object_pick.take()
        else {
            return;
        };
        let target = target.and_then(|k| shell::object_from_key(&self.doc, k));
        let tools: Vec<_> = tools
            .iter()
            .filter_map(|k| shell::object_from_key(&self.doc, *k))
            .filter(|id| Some(*id) != target)
            .collect();
        let Some(target) = target.filter(|_| !tools.is_empty()) else {
            return;
        };
        // The operator reads the selection: the tools, and the target last
        // so that it is active.
        self.doc.selection.clear();
        for id in tools {
            self.doc.selection.add(id);
        }
        self.doc.selection.add(target);
        self.run_op_with(
            "object.boolean",
            &[
                ("operation", PropertyValue::Enum(operation)),
                ("apply", PropertyValue::Bool(apply)),
            ],
        );
    }

    /// Runs what the console was given, and tells the console what happened.
    /// Asks for a text file of script lines and runs it, as the console would.
    ///
    /// Through `run_script`, so every line is its own undo step and a failing
    /// line is reported in the console and does not stop the rest.
    fn run_script_file(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Script", &["txt", "kscript", "py"])
            .set_title("Run Script")
            .pick_file()
        else {
            return;
        };
        match std::fs::read_to_string(&path) {
            Ok(source) => self.run_script(&source),
            Err(e) => self.status(&format!("cannot read {}: {e}", path.display())),
        }
    }

    fn run_script(&mut self, source: &str) {
        self.run_script_reporting(source);
    }

    /// Runs a script as `run_script` does, and says what became of each line:
    /// the line, the history entry it became or why it did not, and the
    /// objects it made.
    ///
    /// One line at a time rather than the whole script to the history at
    /// once, because a line may change the mode (see `kreate::appcall`) and
    /// every line after it has to run in the mode it left.
    fn run_script_reporting(&mut self, source: &str) -> Vec<ScriptLine> {
        use kreate_ui::console::{ConsoleLine, LineKind};
        let mut report = Vec::new();
        let mut errors = 0;
        let mut skipped = 0;
        for text in source.lines() {
            let trimmed = text.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // After a failure the rest is not run: every line after it was
            // written for a document the failed line would have made, and
            // running them anyway only buries the one error that matters
            // under a column of ones it caused.
            if errors > 0 {
                skipped += 1;
                report.push(ScriptLine {
                    text: trimmed.to_string(),
                    result: Err("skipped: an earlier line failed".to_string()),
                    created: Vec::new(),
                    materials: Vec::new(),
                    images: Vec::new(),
                });
                continue;
            }
            let before: std::collections::HashSet<_> = self
                .doc
                .objects()
                .filter_map(|(id, _)| self.doc.object_uid(id))
                .collect();
            let materials_before: std::collections::HashSet<_> = self
                .doc
                .materials()
                .filter_map(|(id, _)| self.doc.material_uid(id))
                .collect();
            let images_before: std::collections::HashSet<_> = self
                .doc
                .images()
                .filter_map(|(id, _)| self.doc.image_uid(id))
                .collect();
            self.ui.console_log.push(ConsoleLine {
                text: trimmed.to_string(),
                kind: LineKind::Input,
            });
            // What the line had done for it before it ran, said with its
            // result: the object it named selected, the mode it needed
            // entered.
            let mut notes: Vec<String> = Vec::new();
            let (line_text, target) =
                kreate_ops::script::take_object_target(&self.registry, trimmed);
            let targeted = match target {
                Some(name) => self.select_for_line(&name, &mut notes),
                None => Ok(()),
            };
            let result =
                match targeted.and_then(|()| self.enter_mode_for_line(&line_text, &mut notes)) {
                    Err(e) => Err(e),
                    Ok(()) => match kreate::appcall::parse(&line_text) {
                        Some(Ok(call)) => self.run_app_call(call),
                        Some(Err(e)) => Err(e),
                        None => {
                            let _ = kreate_ops::report::take();
                            let outcome = self
                                .history
                                .run_script(
                                    &self.registry,
                                    &mut self.doc,
                                    Some(&mut self.edit),
                                    &self.op_ctx,
                                    &line_text,
                                )
                                .pop();
                            let said = kreate_ops::report::take();
                            let with = |base: String| match &said {
                                Some(s) => format!("{base}: {s}"),
                                None => base,
                            };
                            match outcome.map(|o| o.result) {
                                Some(Ok(kreate_ops::OpResult::Finished)) => {
                                    Ok(with("finished".to_string()))
                                }
                                Some(Ok(other)) => Ok(with(format!("{other:?}"))),
                                Some(Err(e)) => Err(e.to_string()),
                                None => continue,
                            }
                        }
                    },
                };
            let result = match result {
                Ok(r) if !notes.is_empty() => Ok(format!("{r} ({})", notes.join(", "))),
                other => other,
            };
            match &result {
                Ok(r) if r == "finished" => {}
                Ok(other) => self.ui.console_log.push(ConsoleLine {
                    // i18n-exempt: an operator outcome is API text
                    text: other.clone(),
                    kind: LineKind::Note,
                }),
                Err(e) => {
                    errors += 1;
                    self.ui.console_log.push(ConsoleLine {
                        text: e.clone(),
                        kind: LineKind::Error,
                    });
                }
            }
            let created = self
                .doc
                .objects()
                .filter_map(|(id, o)| {
                    let uid = self.doc.object_uid(id)?;
                    (!before.contains(&uid)).then(|| (o.name.clone(), uid.raw()))
                })
                .collect();
            let made_materials = self
                .doc
                .materials()
                .filter_map(|(id, m)| {
                    let uid = self.doc.material_uid(id)?;
                    (!materials_before.contains(&uid)).then(|| (m.name.clone(), uid.raw()))
                })
                .collect();
            let made_images = self
                .doc
                .images()
                .filter_map(|(id, i)| {
                    let uid = self.doc.image_uid(id)?;
                    (!images_before.contains(&uid)).then(|| (i.name.clone(), uid.raw()))
                })
                .collect();
            report.push(ScriptLine {
                text: trimmed.to_string(),
                result,
                created,
                materials: made_materials,
                images: made_images,
            });
        }
        // The echo above is what was typed; the history line the console
        // draws beneath it is what it became. A line that ran is not shown
        // twice as the same thing.
        if skipped > 0 {
            self.ui.console_log.push(ConsoleLine {
                // i18n-exempt: script output is API text
                text: format!("{skipped} line(s) after it not run"),
                kind: LineKind::Note,
            });
        }
        if errors > 0 {
            self.status(&format!(
                "script: a line failed; {skipped} after it not run"
            ));
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
        report
    }

    /// A script line the application answers rather than an operator.
    /// Selects the object a script line named with `object=`, making it
    /// the active one, so the line acts on it. Out of edit mode first when
    /// it is another object than the one being edited: the mesh operators
    /// after it go back in by themselves.
    fn select_for_line(&mut self, name: &str, notes: &mut Vec<String>) -> Result<(), String> {
        let Some(id) = self
            .doc
            .objects()
            .find(|(_, o)| o.name == name)
            .map(|(id, _)| id)
        else {
            return Err(format!(
                "no object called {name:?}: get_scene lists the names (a second \"Cube\" is \"Cube.001\")"
            ));
        };
        if self.doc.selection.active == Some(id) && self.doc.selection.objects.len() == 1 {
            return Ok(());
        }
        if self.op_ctx.mode != OpMode::RESTING && self.doc.selection.active != Some(id) {
            self.run_app_call(kreate::appcall::AppCall::ModeSet("object".into()))?;
            notes.push("left edit mode".into());
        }
        let mut p = PropertyMap::new();
        p.set("name", PropertyValue::Text(name.to_string()));
        self.run_op("object.select", p);
        notes.push(format!("{name} selected"));
        Ok(())
    }

    /// Switches mode for a script line that needs another one: a mesh.* or
    /// uv.* line out of edit mode enters it on the active mesh, an object.*
    /// line in edit mode leaves it. The commonest error a script met, and
    /// one with only one fix.
    fn enter_mode_for_line(&mut self, line: &str, notes: &mut Vec<String>) -> Result<(), String> {
        let body = line
            .trim()
            .strip_prefix("kreate.ops.")
            .unwrap_or(line.trim());
        let Some(id) = body.split('(').next().map(str::trim) else {
            return Ok(());
        };
        let Some(info) = self.registry.get(id) else {
            return Ok(());
        };
        if info.poll(&self.op_ctx, &self.doc) {
            return Ok(());
        }
        let elements = self.op_ctx.mode.edits_elements();
        let wants_elements = id.starts_with("mesh.") || id.starts_with("uv.");
        let has_mesh = kreate_ops::mode::Has::of(&self.doc).mesh
            || self
                .doc
                .selection
                .objects
                .iter()
                .any(|o| self.doc.object(*o).and_then(|o| o.mesh()).is_some());
        if wants_elements && !elements && has_mesh {
            self.run_app_call(kreate::appcall::AppCall::ModeSet("edit".into()))?;
            notes.push("entered edit mode".into());
        } else if !wants_elements && id.starts_with("object.") && elements {
            self.run_app_call(kreate::appcall::AppCall::ModeSet("object".into()))?;
            notes.push("left edit mode".into());
        }
        Ok(())
    }

    fn run_app_call(&mut self, call: kreate::appcall::AppCall) -> Result<String, String> {
        use kreate::appcall::AppCall;
        match call {
            AppCall::ModeSet(name) => {
                let mode = match name.as_str() {
                    // Whatever Tab would enter from object mode: the edit
                    // mode of the kind of thing that is active.
                    "edit" => {
                        // An active object that cannot be edited, with a mesh
                        // among the selected: that mesh is what was meant.
                        let has = kreate_ops::mode::Has::of(&self.doc);
                        if !has.mesh && !has.armature {
                            let meant = self
                                .doc
                                .selection
                                .objects
                                .iter()
                                .copied()
                                .find(|id| self.doc.object(*id).and_then(|o| o.mesh()).is_some())
                                .and_then(|id| self.doc.object(id))
                                .map(|o| o.name.clone());
                            if let Some(name) = meant {
                                let mut p = PropertyMap::new();
                                p.set("name", PropertyValue::Text(name));
                                p.set("extend", PropertyValue::Bool(true));
                                self.run_op("object.select", p);
                            }
                        }
                        OpMode::RESTING.toggled(kreate_ops::mode::Has::of(&self.doc))
                    }
                    other => OpMode::from_tag(other).ok_or_else(|| {
                        format!(
                            "no mode called {other:?}: object, edit, sculpt, weight_paint or pose"
                        )
                    })?,
                };
                if self.op_ctx.mode == mode {
                    return Ok(format!("already in {} mode", mode.tag()));
                }
                self.set_mode(mode);
                if self.op_ctx.mode != mode {
                    let active = self
                        .doc
                        .selection
                        .active
                        .and_then(|id| self.doc.object(id))
                        .map(|o| format!("the active object is \"{}\"", o.name))
                        .unwrap_or_else(|| "nothing is selected".to_string());
                    return Err(format!(
                        "cannot enter {} mode: {active}, and it is not one that mode edits. \
                         Select one with kreate.ops.object.select(name=\"...\") first",
                        mode.tag()
                    ));
                }
                Ok(format!("mode {}", mode.tag()))
            }
            AppCall::OpenImage { path, color } => {
                let space = if color {
                    kreate_scene::ColorSpace::Srgb
                } else {
                    kreate_scene::ColorSpace::Raw
                };
                let image = kreate_io::load_image_file(std::path::Path::new(&path), space)
                    .map_err(|e| format!("cannot read {path}: {e}"))?;
                let (name, size) = (image.name.clone(), (image.width(), image.height()));
                let id = self.doc.add_image(image);
                let uid = self.doc.image_uid(id).ok_or("the image has no identity")?;
                Ok(format!(
                    "image {name:?} {}x{} uid={}",
                    size.0,
                    size.1,
                    uid.raw()
                ))
            }
        }
    }

    /// Ctrl+C: the selected objects, as text the application keeps.
    fn copy_selection(&mut self) {
        if self.doc.selection.is_empty() {
            self.status("nothing selected to copy");
            return;
        }
        let fragment = self.doc.extract_objects(&self.doc.selection.objects);
        match kreate_io::clipboard::encode(&fragment) {
            Ok(text) => {
                let count = fragment.object_count();
                self.clipboard = Some(text);
                self.status(&format!("copied {count} object(s)"));
            }
            Err(e) => self.status(&format!("could not copy: {e}")),
        }
    }

    /// Ctrl+V: whatever Ctrl+C took, through the paste operator.
    fn paste_clipboard(&mut self) {
        let Some(text) = self.clipboard.clone() else {
            self.status("nothing to paste");
            return;
        };
        let mut params = PropertyMap::new();
        params.set("data", PropertyValue::Text(text));
        self.run_op("object.paste", params);
    }

    /// The lamp or camera whose glyph is under the cursor, if any.
    ///
    /// Asked of the shell with the projector it drew the glyphs from, so a hit
    /// is a hit on what is on screen. `None` when the frame has not yet said
    /// where the picture is.
    fn glyph_under_cursor(&self) -> Option<kreate_scene::ObjectId> {
        // The rendered view draws no lamps or cameras, so none is clicked.
        if self
            .gfx
            .as_ref()
            .is_some_and(|g| g.viewport.shading == ShadingMode::Rendered)
        {
            return None;
        }
        let (x, y) = self.input.cursor?;
        let projector = self.ui.projector?;
        let ppp = self.egui_ctx.pixels_per_point();
        let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
        shell::glyph_under(&self.doc, &projector, at)
    }

    /// The sketch whose line is under the cursor, if any.
    fn sketch_under_cursor(&self) -> Option<kreate_scene::ObjectId> {
        let (x, y) = self.input.cursor?;
        let projector = self.ui.projector?;
        let ppp = self.egui_ctx.pixels_per_point();
        let at = egui::pos2(x as f32 / ppp, y as f32 / ppp);
        kreate::sketching::sketch_under(&self.doc, &projector, at, 6.0)
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
        if let Some(id) = self.nav_view {
            let input = &mut self.input;
            if let Some(view) = self.gfx.as_mut().and_then(|g| g.extra.get_mut(&id)) {
                let h = (view.rect_px.3 as f64).max(1.0);
                let (dx, dy) = std::mem::take(&mut input.orbit_delta);
                if (dx, dy) != (0.0, 0.0) {
                    let scale = std::f64::consts::TAU / (2.0 * h);
                    view.camera.orbit(-dx * scale, -dy * scale);
                }
                let (dx, dy) = std::mem::take(&mut input.pan_delta);
                if (dx, dy) != (0.0, 0.0) {
                    view.camera.pan(dx / h, dy / h);
                }
                let scroll = std::mem::take(&mut input.scroll);
                if scroll != 0.0 {
                    view.camera.dolly(scroll);
                }
                self.redraw.set(RedrawReason::CAMERA_MOVED);
            }
            if self.drag_ops.iter().all(Option::is_none) {
                self.nav_view = None;
            }
            return;
        }
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
            // The real time, so smoothing keeps up on a slow frame; a timed
            // move caps its own step (`MAX_TRANSITION_STEP`), so a long gap
            // does not teleport through it.
            .map_or(1.0 / 60.0, |dt| dt.clamp(1e-4, 1.0));
        let moving = self.motion.advance(&mut self.camera, dt);
        if moving {
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        moving
    }

    fn render(&mut self) {
        let began = std::time::Instant::now();
        self.render_frame();
        // A frame of more than a second is the window not answering: said
        // in the log, for working out afterwards what it was doing.
        let took = began.elapsed();
        if took.as_secs_f64() >= 1.0 {
            tracing::warn!(
                "slow frame: {:.1} s, mode {:?}, last step {:?}",
                took.as_secs_f64(),
                self.op_ctx.mode,
                self.history.last().map(|e| e.op_id.clone())
            );
        }
    }

    fn render_frame(&mut self) {
        // Every length the panels show and every number typed into one is in
        // the open document's unit.
        kreate_reflect::set_display_length_unit(self.doc.design.length_unit);
        self.publish_mass();
        // Before anything reads the camera: the eased position is what this
        // frame is drawn from, and it has to be settled before the viewport
        // projection, the ray queries and the interface all ask for it.
        self.advance_camera();
        self.follow_camera_view();
        self.read_picks();
        if self.circle.as_ref().is_some_and(|c| c.stroke.is_some()) {
            self.drive_circle_stroke();
            // The reduction answers a frame or two late; keep frames coming
            // while the button is down, or the last stretch of a stroke that
            // ends without moving would never land.
            self.redraw.set(RedrawReason::HOVER_CHANGED);
        }
        self.evaluate_scene();
        self.sync_gpu();
        // Before the interface borrows anything: the panel reads this, and it
        // has to be settled while `self` is still free.
        self.publish_uv_view();
        // The sketch overlay and the feature list, likewise.
        self.update_sketch_view();
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
        self.ui.select_band = self
            .drag_select
            .as_ref()
            .filter(|d| d.moved && d.lasso.is_none())
            .map(|d| {
                let scale = self.egui_ctx.pixels_per_point();
                let to_points = |p: (f32, f32)| egui::pos2(p.0 / scale, p.1 / scale);
                (to_points(d.origin), to_points(d.current))
            });
        self.ui.select_lasso = self
            .drag_select
            .as_ref()
            .filter(|d| d.moved)
            .and_then(|d| d.lasso.as_ref())
            .map(|path| {
                let scale = self.egui_ctx.pixels_per_point();
                path.iter()
                    .map(|p| egui::pos2(p.0 / scale, p.1 / scale))
                    .collect()
            })
            .unwrap_or_default();
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
                // So the tool column lights the button for the drag in
                // progress: what the header says in words, the column says
                // by which picture is lit.
                tool: Some(match g.kind {
                    kreate::modal::Kind::Move => kreate_ui::toolbar::Tool::Move,
                    kreate::modal::Kind::Rotate => kreate_ui::toolbar::Tool::Rotate,
                    kreate::modal::Kind::Scale => kreate_ui::toolbar::Tool::Scale,
                    kreate::modal::Kind::Extrude | kreate::modal::Kind::ExtrudeFree => {
                        kreate_ui::toolbar::Tool::Extrude
                    }
                    kreate::modal::Kind::Inset => kreate_ui::toolbar::Tool::Inset,
                    kreate::modal::Kind::Bevel => kreate_ui::toolbar::Tool::Bevel,
                    kreate::modal::Kind::LoopCut => kreate_ui::toolbar::Tool::LoopCut,
                    kreate::modal::Kind::EdgeSlide => kreate_ui::toolbar::Tool::Move,
                }),
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
                    // The 3D tool column has no button for a UV drag.
                    tool: None,
                })
            });

        // --- run the interface, which tells us how big the viewport is -------
        let mut raw_input = state.take_egui_input(&window);
        shell::keep_tab_for_the_keymap(&self.egui_ctx, &mut raw_input);
        let image = self.gfx.as_ref().map(|g| shell::ViewportImage {
            texture: g.target.id,
            uv: g.viewport.uv_extent(),
        });
        self.ui.extra_images = self
            .gfx
            .as_ref()
            .map(|g| {
                g.extra
                    .iter()
                    .map(|(id, v)| {
                        (
                            *id,
                            shell::ViewportImage {
                                texture: v.target.id,
                                uv: v.viewport.uv_extent(),
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        // What the interface needs of the camera this frame: the gizmo's
        // rotation, and enough to project a lamp's position onto the picture.
        // From last frame's viewport rectangle, which is the one the glyphs
        // will be drawn into until the layout changes — and the shell replaces
        // the rectangle with this frame's as it draws.
        self.ui.camera_rotation = self.camera.rotation;
        // The lens as a 35 mm camera would have it: the vertical field of
        // view across a 24 mm-high frame.
        self.ui.view_lens_mm = 12.0 / (self.camera.fov_y * 0.5).tan();
        self.ui.projector = self.viewport_px.map(|(x, y, w, h)| {
            let ppp = self.egui_ctx.pixels_per_point();
            let aspect = if h > 0.0 { (w / h) as f64 } else { 1.0 };
            kreate_ui::object_glyph::Projector {
                eye: self.camera.eye(),
                rotation: self.camera.rotation,
                projection: self.camera.projection_matrix(aspect).as_dmat4(),
                rect: egui::Rect::from_min_size(
                    egui::pos2(x / ppp, y / ppp),
                    egui::vec2(w / ppp, h / ppp),
                ),
            }
        });
        // The header's tool settings show what the keys last set.
        self.ui.tools.snap = self.snap.enabled;
        self.ui.tools.snap_mode = kreate::snap::Mode::ALL
            .iter()
            .position(|m| *m == self.snap.mode)
            .unwrap_or(0) as u32;
        self.ui.tools.proportional = self.proportional.is_some();
        self.ui.tools.falloff = self.proportional_falloff;
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
        // And take back what the header changed.
        self.adopt_tool_settings();
        // The interface's own request to be drawn again, and when. A long delay
        // is egui saying "never" in a number.
        let delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|v| v.repaint_delay)
            .unwrap_or(std::time::Duration::MAX);
        self.ui_repaint_at =
            (delay < std::time::Duration::from_secs(60)).then(|| Instant::now() + delay);

        // The single input gate. Scattered `if !typing` checks are how a rename
        // field ends up firing tool shortcuts again.
        self.ui.egui_wants_keyboard = self.egui_ctx.egui_wants_keyboard_input();
        self.ui.ime_active = output.platform_output.ime.is_some();

        self.apply_keymap_edits();
        self.save_workspaces();
        self.drain_ui_commands();
        // The asset browser's requests go out from here too, not only when a
        // job wakes the loop: the first list and the first pictures are
        // asked for by drawing the panel, with nothing yet running to wake it.
        self.serve_assets();
        self.serve_imports();
        // Again after the panels: an edit they made this frame — a material
        // value dragged — is drawn this frame, and the reasons are cleared
        // when it ends. Checked only before, a settled view showed the old
        // picture until something else asked for a frame.
        self.restart_mean_if_changed();
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
        // At the preference's fraction of the panel's pixels, stretched to
        // fit: the cursor is mapped through the drawn size, so picking keeps
        // working at any scale.
        let scale = self
            .ui
            .resolution_scale
            .clamp(kreate::prefs::MIN_RESOLUTION_SCALE, 1.0);
        let wanted = (
            (viewport_rect.width().max(0.0) * pixels_per_point * scale) as u32,
            (viewport_rect.height().max(0.0) * pixels_per_point * scale) as u32,
        );
        // The canvas follows the chrome: light under the light theme, as
        // Fusion draws it.
        if let Some(gfx) = self.gfx.as_mut() {
            let light = self.ui.appearance == kreate_ui::Appearance::Light;
            if gfx.viewport.is_light() != light {
                gfx.viewport.set_light(light);
                self.redraw.set(RedrawReason::SETTINGS);
            }
        }
        // Not while a render runs: the picture is the render's size, not the
        // panel's, and following the panel restarted its samples every frame.
        if self.render_job.is_none()
            && wanted != self.wanted_viewport
            && wanted.0 > 0
            && wanted.1 > 0
        {
            self.wanted_viewport = wanted;
            self.redraw.set(RedrawReason::RESIZE);
        }
        self.settle_extra_views(pixels_per_point, scale);
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
        let bases = self.edit_bases().map(|(b, _)| b);
        let preview_state = region_request.as_ref().and_then(|(_, collect)| {
            let kreate_render::Collect::Elements { count, .. } = collect else {
                return None;
            };
            Some(kreate_render::PreviewSelection {
                domain: self.pick_domain(),
                deselect: self.drag_select.as_ref().is_some_and(|d| d.deselect)
                    || self
                        .circle
                        .as_ref()
                        .and_then(|c| c.stroke.as_ref())
                        .is_some_and(|s| s.deselect),
                elements: *count,
            })
        });

        let Some(gfx) = self.gfx.as_mut() else { return };

        // Texture changes go up before the surface is asked for, not with the
        // paint. egui sends each texture whole once and only patches after, so
        // a frame that gives up on the surface below must not take the whole
        // one with it: the next frame's patch would land on a texture that was
        // never made, which egui-wgpu treats as fatal. Seen on a first frame
        // whose surface was not ready yet.
        for (id, delta) in &output.textures_delta.set {
            gfx.egui_renderer
                .update_texture(&gfx.gpu.device, &gfx.gpu.queue, *id, delta);
        }

        use wgpu::CurrentSurfaceTexture as Acquired;
        let frame = match gfx.surface.get_current_texture() {
            Acquired::Success(f) => Some(f),
            Acquired::Suboptimal(f) => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                Some(f)
            }
            Acquired::Outdated | Acquired::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                self.redraw.set(RedrawReason::RESIZE);
                None
            }
            Acquired::Occluded => None,
            Acquired::Timeout => {
                tracing::warn!("timed out acquiring a surface texture");
                None
            }
            Acquired::Validation => {
                tracing::error!("validation error acquiring a surface texture");
                None
            }
        };
        // A hidden window has no frame to draw into, and normally that means
        // nothing is drawn. Not while a capture or a render waits on the
        // viewport: it draws offscreen regardless, and only the window's own
        // picture is skipped.
        if frame.is_none() && !self.capturing {
            // Nothing was drawn with them, so what egui retired can go now.
            for id in &output.textures_delta.free {
                gfx.egui_renderer.free_texture(id);
            }
            return;
        }
        let surface_view = frame.as_ref().map(|f| {
            f.texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        // One draw entry per *object*, sharing GPU geometry by the identity of the
        // evaluated mesh — so a thousand linked duplicates cost a thousand
        // transforms. Driven from the evaluated scene rather than from the
        // document, which is what stops the renderer from being able to draw
        // something the picker cannot hit.
        let preview_target = region_request.as_ref().and(self.doc.selection.active);
        // One material per mesh slot, per object, built before the draws so
        // that a draw can borrow it — which is what keeps `Draw` `Copy` and a
        // frame free of per-object allocation in the usual case. Empty for an
        // object whose mesh has no slots, which means "every slot takes
        // `material`".
        let slot_materials: Vec<Vec<u32>> = self
            .evaluated
            .iter()
            .map(|(id, _)| {
                self.materials
                    .as_ref()
                    .map(|m| m.slots_of(&self.doc, *id))
                    .unwrap_or_default()
            })
            .collect();
        // The studio and flat colour of each slot: its material's base
        // colour, as Blender's solid mode shows it. Read straight from the
        // document, because those modes compile no materials.
        let slot_colors: Vec<Vec<[f32; 4]>> = self
            .evaluated
            .iter()
            .map(|(id, _)| {
                let slots = self
                    .doc
                    .drawn_slot_count(*id)
                    .clamp(1, kreate_render::viewport::SOLID_SLOTS);
                (0..slots)
                    .map(|slot| {
                        self.doc
                            .material_id_at(*id, slot)
                            .and_then(|m| self.doc.material(m))
                            .map(|m| [m.base_color.r, m.base_color.g, m.base_color.b, 1.0])
                            .unwrap_or([0.0; 4])
                    })
                    .collect()
            })
            .collect();
        // The image each slot shows in the flat mode, as a layer of the
        // viewport's solid textures; -1 for its colour. Studio shows the
        // material's colour, to read shape by. The images go up only when
        // the set, or one of them, changes.
        let textured =
            self.solid_textures && gfx.viewport.shading == kreate_render::ShadingMode::Flat;
        let mut images: Vec<kreate_scene::document::ImageId> = Vec::new();
        // A cut-out's picture goes up whatever the mode: the rendered view's
        // shadow pass reads it to cast a leaf's shadow rather than its card's.
        let cuts = |id: &kreate_scene::ObjectId, slot: usize| {
            self.doc
                .material_id_at(*id, slot)
                .and_then(|m| self.doc.material(m))
                .is_some_and(|m| m.alpha_clip.is_some())
        };
        let slot_layers: Vec<Vec<i32>> = self
            .evaluated
            .iter()
            .map(|(id, _)| {
                if !textured && !cuts(id, 0) {
                    return Vec::new();
                }
                let slots = self
                    .doc
                    .drawn_slot_count(*id)
                    .clamp(1, kreate_render::viewport::SOLID_SLOTS);
                (0..slots)
                    .map(|slot| {
                        let Some(image) = self
                            .doc
                            .material_id_at(*id, slot)
                            .and_then(|m| self.doc.material(m))
                            .and_then(|m| m.base_color_image)
                            .filter(|i| self.doc.image(*i).is_some())
                        else {
                            return -1;
                        };
                        match images.iter().position(|i| *i == image) {
                            Some(at) => at as i32,
                            None if images.len() < kreate_render::solid_textures::MAX_LAYERS => {
                                images.push(image);
                                images.len() as i32 - 1
                            }
                            None => -1,
                        }
                    })
                    .collect()
            })
            .collect();
        // And where each slot is cut away, read from its image's alpha as the
        // rendered view reads it.
        let slot_clips: Vec<Vec<f32>> = self
            .evaluated
            .iter()
            .map(|(id, _)| {
                if !textured && !cuts(id, 0) {
                    return Vec::new();
                }
                (0..kreate_render::viewport::SOLID_SLOTS)
                    .map(|slot| {
                        self.doc
                            .material_id_at(*id, slot)
                            .and_then(|m| self.doc.material(m))
                            .and_then(|m| m.alpha_clip)
                            .unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect();
        let want: Vec<(kreate_scene::document::ImageId, u64)> = images
            .iter()
            .filter_map(|i| Some((*i, self.doc.image(*i)?.revision())))
            .collect();
        if (textured || !images.is_empty()) && want != self.solid_images {
            let pixels: Vec<(u32, u32, Vec<u8>)> = images
                .iter()
                .filter_map(|i| self.doc.image(*i))
                .map(|img| {
                    (
                        img.width(),
                        img.height(),
                        kreate::materials::encoded_rgba(img, img.color_space),
                    )
                })
                .collect();
            let sources: Vec<kreate_render::solid_textures::SolidImage<'_>> = pixels
                .iter()
                .map(|(w, h, p)| kreate_render::solid_textures::SolidImage {
                    width: *w,
                    height: *h,
                    pixels: p,
                })
                .collect();
            gfx.viewport
                .set_solid_images(&gfx.gpu.device, &gfx.gpu.queue, &sources);
            self.solid_images = want;
        }
        // A boolean's cutter is drawn as edges, or it would cover the very
        // cut it makes. See `boolean_cutters`.
        let cutters = kreate_scene::modifier::boolean_cutters(&self.doc);
        let mut draws: Vec<Draw> = Vec::new();
        for (at, (id, mesh)) in self.evaluated.iter().enumerate() {
            let resident = gfx
                .meshes
                .iter()
                .find(|(m, _)| Arc::ptr_eq(m, mesh))
                .map(|(_, g)| g)
                .or_else(|| {
                    gfx.skinned
                        .iter()
                        .find(|e| e.object == *id && Arc::ptr_eq(&e.shown, mesh))
                        .map(|e| &e.mesh)
                });
            if let Some(gpu_mesh) = resident {
                // In edit mode the object-level highlight stands down: the
                // whole wireframe going orange would drown out the element
                // selection, which is the only thing the user is looking at.
                draws.push(
                    Draw::new(gpu_mesh, self.doc.world_transform(*id))
                        // The painted mesh, by its weights.
                        .weights(
                            self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Weights
                                && self.doc.selection.active == Some(*id),
                        )
                        // The sculpted mesh, with its mask and face sets.
                        .sculpt_overlay(
                            self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt
                                && self.ui.sculpt.show_overlay
                                && self.doc.selection.active == Some(*id),
                        )
                        // The object's first material, which is what every
                        // slot a mesh does not name falls back to. Zero when
                        // the materials have not been compiled yet, which is
                        // the studio mode: that pass does not read a material,
                        // and the rendered one compiles them before it builds
                        // any draws.
                        .with_material(
                            self.materials
                                .as_ref()
                                .map(|m| m.index_of(&self.doc, *id))
                                .unwrap_or(0),
                        )
                        // And one per slot, for a mesh whose faces are not all
                        // on the same one. `material_index` has reached the
                        // glTF exporter since it was written and never reached
                        // the screen; this is the other end of it.
                        .with_materials(&slot_materials[at])
                        .with_colors(&slot_colors[at])
                        .with_layers(&slot_layers[at])
                        .with_clips(&slot_clips[at])
                        .selected(
                            self.op_ctx.mode.acts_on() == kreate_ops::mode::ActsOn::Objects
                                && self.doc.selection.contains(*id),
                        )
                        .active(self.doc.selection.active == Some(*id))
                        .wire_only(cutters.contains(id))
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
                            self.op_ctx.mode.edits_elements()
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
        gfx.viewport.show_elements = self.op_ctx.mode.shows_elements() && !self.rendering_still;
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

        // The scene's lights, repacked every frame because every position
        // is rebased onto the camera and the camera moves.
        //
        // Only in the mode that reads them: packing is cheap but not free, and
        // a scene with three hundred lamps should not pay for them while the
        // user is modelling under the studio lights.
        // The overlays as they are, for the menu's check marks.
        let overlays = kreate::shell::Overlays {
            flat: gfx.viewport.shading == kreate_render::ShadingMode::Flat,
            grid: gfx.viewport.show_grid,
            ground_shadow: gfx.viewport.show_ground_shadow,
            wireframe: gfx.viewport.show_wireframe,
            face_orientation: gfx.viewport.face_orientation,
            statistics: self.ui.show_stats,
            dimensions: self.ui.show_dimensions,
            cursor: self.ui.show_cursor,
            ambient_occlusion: gfx.viewport.ambient_occlusion,
            xray: gfx.viewport.xray,
            outline: gfx.viewport.show_outline,
            bloom: gfx.viewport.show_bloom,
            reflections: gfx.viewport.show_reflections,
            contact_shadows: gfx.viewport.contact_shadows,
            path_trace: gfx.viewport.path_trace,
            textures: self.solid_textures,
        };
        // The application's studio: always in the material preview, and in
        // the rendered view when the scene brings no light of its own.
        let studio_rendered = gfx.viewport.shading == kreate_render::ShadingMode::Rendered
            && scenelights::uses_studio(&self.doc);
        gfx.viewport.studio_lit = studio_rendered;
        gfx.viewport.path_trace_when_still = self.ui.trace_when_still;
        for v in gfx.extra.values_mut() {
            v.viewport.studio_lit = studio_rendered;
            v.viewport.path_trace_when_still = self.ui.trace_when_still;
        }
        let preview = match gfx.viewport.shading {
            kreate_render::ShadingMode::Material => {
                Some(scenelights::material_preview(self.camera.rotation))
            }
            kreate_render::ShadingMode::Rendered if scenelights::uses_studio(&self.doc) => {
                Some(scenelights::rendered_studio(&self.doc))
            }
            _ => None,
        };
        if gfx.viewport.shading.lit() {
            // The extra viewports are lit as this one is: each setting below
            // goes to them too, and one made since the last lit frame is
            // handed the sky, the materials and the traced scene whole.
            let fresh = gfx.extra.values().any(|v| v.fresh);
            // The material preview's own light: an even pale sky and one
            // soft key from over the viewer's left shoulder, turning with the
            // view as Blender's preview light does — the scene's lamps and
            // sky are the rendered view's.
            if let Some(preview) = &preview {
                // The studio's room, uploaded once and kept while it is the
                // one in use: six faces and their mips are not a frame's cost.
                self.environment = None;
                let (device, queue) = (&gfx.gpu.device, &gfx.gpu.queue);
                let upload = !self.studio_environment;
                let pixels = (upload || fresh).then(kreate_render::studio::equirect);
                gfx.viewport.set_world(preview.world);
                if upload {
                    if let Some(pixels) = &pixels {
                        gfx.viewport.set_environment(
                            device,
                            queue,
                            kreate_render::studio::WIDTH,
                            kreate_render::studio::HEIGHT,
                            pixels,
                        );
                    }
                }
                for v in gfx.extra.values_mut() {
                    v.viewport.set_world(preview.world);
                    if upload || v.fresh {
                        if let Some(pixels) = &pixels {
                            v.viewport.set_environment(
                                device,
                                queue,
                                kreate_render::studio::WIDTH,
                                kreate_render::studio::HEIGHT,
                                pixels,
                            );
                        }
                    }
                }
                self.studio_environment = true;
            } else {
                // Out of the studio: its room goes, and the scene's own sky
                // is rebuilt below as if it had never been there.
                if std::mem::take(&mut self.studio_environment) {
                    let (device, queue) = (&gfx.gpu.device, &gfx.gpu.queue);
                    gfx.viewport.clear_environment(device, queue);
                    for v in gfx.extra.values_mut() {
                        v.viewport.clear_environment(device, queue);
                    }
                    self.environment = None;
                    self.environment_stale = true;
                }
                let world = scenelights::world_radiance(&self.doc);
                gfx.viewport.set_world(world);
                for v in gfx.extra.values_mut() {
                    v.viewport.set_world(world);
                }
            }
            // The environment, rebuilt only when it is a different one. Six
            // faces and their mips is not a per-frame cost, and the key is what
            // tells "the same sky again" from "a sky that finished loading".
            let want = kreate::environment::key_of(&self.doc);
            let stale = std::mem::take(&mut self.environment_stale);
            if preview.is_none() && (want != self.environment || fresh || stale) {
                let (device, queue) = (&gfx.gpu.device, &gfx.gpu.queue);
                let viewports = std::iter::once(&mut gfx.viewport)
                    .chain(gfx.extra.values_mut().map(|v| &mut v.viewport));
                match kreate::environment::background_image(&self.doc) {
                    Some(image) => {
                        let pixels = kreate::environment::pixels_of(image);
                        for v in viewports {
                            v.set_environment(
                                device,
                                queue,
                                image.width(),
                                image.height(),
                                &pixels,
                            );
                        }
                    }
                    None => {
                        for v in viewports {
                            v.clear_environment(device, queue);
                        }
                    }
                }
                self.environment = want;
            }
            // And its two dials, every frame: they cost a uniform write, and a
            // slider whose effect waits for something else to change is a
            // slider that looks broken.
            let (rotation, strength) = self
                .doc
                .active_world()
                .and_then(|id| self.doc.world(id))
                .map(|w| (w.rotation as f32, w.strength as f32))
                .unwrap_or((0.0, 1.0));
            // The studio turns with the view in the preview; the scene's sky
            // is turned by its own dial.
            let (rotation, strength) = preview
                .as_ref()
                .map_or((rotation, strength), |p| (p.yaw, 1.0));
            gfx.viewport
                .set_world_shape(&gfx.gpu.queue, rotation, strength);
            for v in gfx.extra.values_mut() {
                v.viewport
                    .set_world_shape(&gfx.gpu.queue, rotation, strength);
            }

            // The materials. Compiled every frame because it is string
            // building; uploaded only when the *code* changed, because that is
            // a pipeline and an atlas. Dragging a colour writes sixteen bytes
            // per material and nothing else — which is the whole reason the
            // compiler keeps constants out of the text.
            let materials = kreate::materials::Materials::compile(&self.doc);
            let transmissive: Vec<f32> = materials
                .compiled
                .iter()
                .map(|c| c.transmission())
                .collect();
            gfx.viewport.set_transmissive(&transmissive);
            for v in gfx.extra.values_mut() {
                v.viewport.set_transmissive(&transmissive);
            }
            let same = self
                .materials
                .as_ref()
                .is_some_and(|held| held.same_code_as(&materials));
            if same && !fresh {
                for (i, compiled) in materials.compiled.iter().enumerate() {
                    gfx.viewport.set_material_constants(
                        &gfx.gpu.queue,
                        i,
                        &compiled.constants,
                        &compiled.image_slots,
                    );
                    for v in gfx.extra.values_mut() {
                        v.viewport.set_material_constants(
                            &gfx.gpu.queue,
                            i,
                            &compiled.constants,
                            &compiled.image_slots,
                        );
                    }
                }
            } else {
                // The pixels first and the borrows after, because a material
                // source borrows them — untiling into a temporary inside the
                // map would hand over a reference to something dropped at the
                // end of the closure.
                let pixels: Vec<Vec<kreate::materials::TextureImage>> = materials
                    .compiled
                    .iter()
                    .map(|c| kreate::materials::pixels_for(&self.doc, c))
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
                gfx.viewport
                    .set_materials(&gfx.gpu.device, &gfx.gpu.queue, &sources);
                for v in gfx.extra.values_mut() {
                    v.viewport
                        .set_materials(&gfx.gpu.device, &gfx.gpu.queue, &sources);
                }
            }
            self.materials = Some(materials);
            // Through a camera with an aperture, the path tracer is that lens.
            gfx.viewport.path_trace_lens = self
                .camera_view
                .and(self.camera_view_object)
                .and_then(|id| match self.doc.object(id).map(|o| &o.data) {
                    Some(kreate_scene::document::ObjectData::Camera(c)) => self.doc.camera(*c),
                    _ => None,
                })
                .and_then(|c| {
                    let f = c.f_stop?;
                    Some(kreate_render::pathtrace::Lens {
                        radius: (c.focal_length_mm / 1000.0 / f / 2.0) as f32,
                        focus: c.focus_distance as f32,
                    })
                });
            // The path tracer's scene, rebuilt when what it is made of
            // changed — not when the camera moved.
            if gfx.viewport.wants_path_trace_scene() {
                if let Some(materials) = self.materials.as_ref() {
                    let traced_materials = materials.for_tracer(&self.doc);
                    let images = &traced_materials.images;
                    let mut surfaces = traced_materials.surfaces.clone();
                    // Each area light as the glowing panel it is, one-sided,
                    // each with a surface of its own after the materials'.
                    let panels = scenelights::area_panels(&self.doc);
                    let first_panel = surfaces.len() as u32;
                    surfaces.extend(panels.iter().map(|p| kreate_render::pathtrace::PtSurface {
                        base_color: [0.0; 3],
                        roughness: 1.0,
                        emission: p.radiance,
                        one_sided: true,
                        ..Default::default()
                    }));
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    // Cutters have no surface in the picture, as in the
                    // viewport's own passes.
                    let cutters = kreate_scene::modifier::boolean_cutters(&self.doc);
                    let traced = || {
                        self.evaluated
                            .iter()
                            .filter(|(id, _)| !cutters.contains(id))
                    };
                    for (id, mesh) in traced() {
                        (Arc::as_ptr(mesh) as usize).hash(&mut h);
                        for v in self.doc.world_transform(*id).to_cols_array() {
                            v.to_bits().hash(&mut h);
                        }
                        materials.index_of(&self.doc, *id).hash(&mut h);
                        materials.slots_of(&self.doc, *id).hash(&mut h);
                    }
                    format!("{surfaces:?}").hash(&mut h);
                    // The code and its numbers: a colour typed into a node
                    // changes the second and must reach the picture.
                    traced_materials.code.hash(&mut h);
                    for v in traced_materials.values.iter().flatten() {
                        v.to_bits().hash(&mut h);
                    }
                    for p in &panels {
                        format!("{:?}{:?}", p.positions, p.transform).hash(&mut h);
                    }
                    for image in images {
                        (image.width, image.height, image.pixels.len()).hash(&mut h);
                        image.pixels.iter().step_by(97).for_each(|b| b.hash(&mut h));
                    }
                    let key = h.finish();
                    if self.path_trace_key != Some(key) || fresh {
                        self.path_trace_key = Some(key);
                        let mut extracted: Vec<_> = traced()
                            .map(|(id, mesh)| {
                                let cpu = kreate_render::RenderMeshCpu::extract(mesh);
                                let positions: Vec<[f32; 3]> =
                                    cpu.positions.iter().map(|p| p.0).collect();
                                let uvs = cpu.uvs.clone();
                                let normals: Vec<kreate_math::Vec3> = cpu
                                    .normals
                                    .iter()
                                    .map(|n| kreate_render::mesh_gpu::oct_decode(*n))
                                    .collect();
                                let runs = materials.runs_of(&self.doc, *id, &cpu.submeshes);
                                (
                                    positions,
                                    uvs,
                                    normals,
                                    cpu.indices,
                                    self.doc.world_transform(*id),
                                    materials.index_of(&self.doc, *id),
                                    runs,
                                    cpu.colors,
                                )
                            })
                            .collect();
                        extracted.extend(panels.iter().enumerate().map(|(i, p)| {
                            (
                                p.positions.clone(),
                                vec![[0.0f32; 2]; p.positions.len()],
                                p.normals.clone(),
                                p.indices.clone(),
                                p.transform,
                                first_panel + i as u32,
                                Vec::new(),
                                Vec::new(),
                            )
                        }));
                        let meshes: Vec<kreate_render::pathtrace::PtMesh<'_>> = extracted
                            .iter()
                            .map(
                                |(
                                    positions,
                                    uvs,
                                    normals,
                                    indices,
                                    transform,
                                    surface,
                                    runs,
                                    colors,
                                )| {
                                    kreate_render::pathtrace::PtMesh {
                                        positions,
                                        uvs,
                                        normals,
                                        colors,
                                        indices,
                                        transform: *transform,
                                        surface: *surface,
                                        runs,
                                    }
                                },
                            )
                            .collect();
                        let pt_images: Vec<kreate_render::pathtrace::PtImage<'_>> = images
                            .iter()
                            .map(|i| kreate_render::pathtrace::PtImage {
                                width: i.width,
                                height: i.height,
                                pixels: &i.pixels,
                                color: i.kind == kreate_render::material::TextureKind::Color,
                            })
                            .collect();
                        gfx.viewport.set_path_trace_scene(
                            &gfx.gpu.device,
                            &gfx.gpu.queue,
                            &meshes,
                            &surfaces,
                            &pt_images,
                            &traced_materials.materials(),
                        );
                        for v in gfx.extra.values_mut() {
                            v.viewport.set_path_trace_scene(
                                &gfx.gpu.device,
                                &gfx.gpu.queue,
                                &meshes,
                                &surfaces,
                                &pt_images,
                                &traced_materials.materials(),
                            );
                        }
                    }
                }
            }
            // With the distance each lamp is worth computing to, which is
            // what the tile pass culls against.
            let lights = match &preview {
                Some(preview) => preview.lights.clone(),
                None => scenelights::culled_lights(&self.doc, self.camera.pivot),
            };
            gfx.viewport
                .set_lights(&gfx.gpu.device, &gfx.gpu.queue, self.camera.eye(), &lights);
            for v in gfx.extra.values_mut() {
                v.viewport
                    .set_lights(&gfx.gpu.device, &gfx.gpu.queue, v.camera.eye(), &lights);
            }
            // The bridge between §2.2's photometric lights and §2.1's display
            // transform: without it a sunlit scene saturates every pixel. The
            // scene camera's, not the viewport camera's — the exposure is a
            // property of the shot and not of how the user is looking around.
            // What the user is looking at. A spot reading on the subject,
            // which is what a photographer takes and what keeps the walls from
            // changing brightness as the camera turns.
            let shot = scenelights::shot(&self.doc, self.time.rate(), self.camera.pivot);
            // The preview's light is its own, so its exposure is too.
            let exposure = preview
                .as_ref()
                .map_or(shot.exposure as f32, |p| p.exposure);
            let display = kreate_render::display::Display {
                view: if self.ui.punchy {
                    kreate_math::ViewTransform::AgXPunchy
                } else {
                    kreate_math::ViewTransform::AgX
                },
                exposure,
            };
            gfx.viewport.set_display(&gfx.gpu.queue, display);
            for v in gfx.extra.values_mut() {
                v.viewport.set_display(&gfx.gpu.queue, display);
                v.fresh = false;
            }
            // What the header reports, from the same answer the picture was
            // made with.
            self.ui.view_status = kreate::shell::ViewStatus {
                overlays,
                shading: gfx.viewport.shading,
                exposure,
                metered: shot.metered && preview.is_none(),
                samples: gfx.viewport.path_trace_samples(),
                orthographic: matches!(
                    self.camera.projection,
                    kreate_render::Projection::Orthographic
                ),
            };
        } else {
            // The other modes light themselves, so there is no exposure to
            // report and the header says nothing rather than a stale number.
            self.ui.view_status = kreate::shell::ViewStatus {
                overlays,
                shading: gfx.viewport.shading,
                orthographic: matches!(
                    self.camera.projection,
                    kreate_render::Projection::Orthographic
                ),
                ..Default::default()
            };
        }

        // The lamps and cameras, as lines and rings for the renderer to draw
        // against its depth — see `glyphs`. Rebuilt each frame: a handful of
        // objects, and a copy of their transforms would be one more thing to
        // keep in step with undo.
        let shown = match self.op_ctx.mode.acts_on() {
            kreate_ops::mode::ActsOn::Bones => kreate_ops::ops::pose::posed_object(&self.doc)
                .map(|id| (id, kreate::glyphs::Bones::Pose)),
            kreate_ops::mode::ActsOn::BoneJoints => {
                kreate_ops::ops::armature::edited_object(&self.doc)
                    .map(|id| (id, kreate::glyphs::Bones::Edit))
            }
            _ => None,
        };
        let local: Option<Vec<ObjectId>> = self.local_view.as_ref().map(|(aside, _)| {
            self.doc
                .objects()
                .map(|(id, _)| id)
                .filter(|id| !aside.contains(id))
                .collect()
        });
        let (mut marker_lines, marker_points) = if self.rendering_still {
            Default::default()
        } else {
            kreate::glyphs::markers_only(&self.doc, shown, local.as_deref())
        };
        if !self.rendering_still {
            marker_lines.extend(collab::peer_lines(self.session.as_ref(), &self.doc));
        }
        gfx.viewport.set_markers(
            &gfx.gpu.device,
            &gfx.gpu.queue,
            self.camera.eye(),
            &marker_lines,
            &marker_points,
        );

        // 3D into the offscreen target...
        gfx.viewport
            .prepare(&gfx.gpu.device, &gfx.gpu.queue, &self.camera, &mut draws);
        // Baked smoke and fire at this frame: each shown gas domain's grid.
        // After `prepare`, which settles the size and depth they march to.
        let now = self.doc.playback.frame;
        let volumes: Vec<kreate_render::volume::VolumeData<'_>> = self
            .doc
            .objects()
            .filter(|(id, _)| self.doc.is_shown(*id))
            .filter_map(|(_, o)| {
                let Some(kreate_scene::physics::Fluid::Domain(d)) = o.physics.fluid else {
                    return None;
                };
                let (track, at) = self.doc.physics_cache.gas_at(o.uid, now)?;
                Some(kreate_render::volume::VolumeData {
                    cells: track.cells.map(|c| c as u32),
                    smoke: &at.smoke,
                    flame: &at.flame,
                    min: track.bounds.min.as_dvec3(),
                    max: track.bounds.max.as_dvec3(),
                    smoke_color: d.smoke_color.as_vec3().to_array(),
                    key: std::sync::Arc::as_ptr(&at.smoke) as *const u8 as usize,
                    detail: d.detail as f32,
                    time: (now / self.doc.playback.fps.max(1.0)) as f32,
                })
            })
            .collect();
        gfx.viewport
            .set_volumes(&gfx.gpu.device, &gfx.gpu.queue, &self.camera, &volumes);
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
                if let Some(bases) = &bases {
                    gfx.region.set_bases(&gfx.gpu.device, &gfx.gpu.queue, bases);
                }
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

        // The extra viewports: the same scene through their own cameras.
        // After the first's passes, whose picking and outlines read the
        // draws as that camera left them.
        for view in gfx.extra.values_mut() {
            let v = &mut view.viewport;
            let main = &gfx.viewport;
            // Shaded as the first one is, and lit by what it was handed above.
            v.shading = main.shading;
            v.path_trace = main.path_trace;
            v.path_trace_denoise = main.path_trace_denoise;
            v.path_trace_noise_threshold = main.path_trace_noise_threshold;
            v.show_bloom = main.show_bloom;
            v.show_reflections = main.show_reflections;
            v.contact_shadows = main.contact_shadows;
            v.ambient_occlusion = main.ambient_occlusion;
            v.show_grid = main.show_grid;
            v.show_ground_shadow = main.show_ground_shadow;
            v.show_wireframe = main.show_wireframe;
            v.show_elements = main.show_elements;
            v.show_outline = main.show_outline;
            v.xray = main.xray;
            v.face_orientation = main.face_orientation;
            v.set_render_size(view.size.0, view.size.1);
            v.prepare(&gfx.gpu.device, &gfx.gpu.queue, &view.camera, &mut draws);
            v.render(&mut encoder, &view.target.view, &draws);
        }

        // ...then the interface over the surface, with the target as an image.
        let paint_jobs = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);
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
        if let Some(surface_view) = &surface_view {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
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
        if let Some(frame) = frame {
            frame.present();
        }
        // An EXR render's object ids, once its camera has settled: the
        // picking pass over this frame's depth, read back. Once a picture,
        // because what is where does not change while the samples add up.
        if let Some(job) = self.render_job.as_mut() {
            if job.settings.format == kreate_scene::render::OutputFormat::Exr
                && job.drawn >= 3
                && job.ids.is_none()
            {
                match gfx.viewport.read_object_ids(
                    &gfx.gpu.device,
                    &gfx.gpu.queue,
                    &self.camera,
                    &draws,
                ) {
                    Ok(ids) => job.ids = Some(ids),
                    Err(e) => tracing::warn!(error = %e, "no object ids for the render"),
                }
            }
        }
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
        if self.pointer_in_timeline() {
            return KeyContext::Timeline;
        }
        // Over the shader editor only the global keys — undo, save — reach the
        // keymap. The graph handles its own `X` and `Shift+A`, and the
        // viewport's meaning of them (delete the object, add a cube) is the
        // last thing somebody pointing at a node wants.
        if self.pointer_in_node_editor() {
            return KeyContext::Global;
        }
        mode_key_context(self.op_ctx.mode)
    }

    /// Whether the pointer is over a timeline.
    fn pointer_in_timeline(&self) -> bool {
        let Some((x, y)) = self.input.cursor else {
            return false;
        };
        let scale = f64::from(self.egui_ctx.pixels_per_point().max(0.01));
        self.ui
            .timeline_at(egui::pos2((x / scale) as f32, (y / scale) as f32))
    }

    /// Whether the pointer is over a shader editor.
    fn pointer_in_node_editor(&self) -> bool {
        let Some((x, y)) = self.input.cursor else {
            return false;
        };
        let scale = f64::from(self.egui_ctx.pixels_per_point().max(0.01));
        self.ui
            .node_editor_at(egui::pos2((x / scale) as f32, (y / scale) as f32))
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
        // A render takes Escape first: it is the one way out of a long one.
        if code == KeyCode::Escape && self.render_job.is_some() {
            self.finish_render_job("render stopped");
            return;
        }
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
        // A curve point being dragged: Esc puts it back.
        if code == KeyCode::Escape && self.cancel_curve_drag() {
            return;
        }
        if self.circle.is_some()
            && matches!(
                code,
                KeyCode::Escape | KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space
            )
        {
            self.end_circle_select();
            return;
        }
        if self.text_place && code == KeyCode::Escape {
            self.text_place = false;
            self.status("");
            return;
        }
        if self.mate.is_some() && code == KeyCode::Escape {
            self.mate = None;
            self.status("Align Faces cancelled");
            return;
        }
        // Esc stops a bake that is running, before it means anything else.
        if code == KeyCode::Escape {
            if let Some(job) = &self.bake_job {
                job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        }
        if self.ui.brush_adjust.is_some() {
            match code {
                KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space => {
                    self.end_brush_adjust(true);
                    return;
                }
                KeyCode::Escape => {
                    self.end_brush_adjust(false);
                    return;
                }
                _ => {}
            }
        }
        if self.measure.is_some() {
            match code {
                KeyCode::Escape => {
                    self.end_measure("Measure cancelled");
                    return;
                }
                KeyCode::Backspace => {
                    if let Some(tool) = &mut self.measure {
                        tool.points.pop();
                        tool.angle = tool.angle && tool.points.len() >= 2;
                    }
                    self.update_measure();
                    return;
                }
                _ => {}
            }
        }
        if self.knife.is_some() {
            match code {
                KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space => {
                    self.commit_knife();
                    return;
                }
                KeyCode::Escape => {
                    self.end_knife("Knife cancelled");
                    return;
                }
                KeyCode::Backspace => {
                    if let Some((points, _)) = &mut self.knife {
                        points.pop();
                    }
                    self.update_knife();
                    return;
                }
                _ => {}
            }
        }
        if self.op_ctx.mode == OpMode::Sketch && self.sketch_key(code) {
            return;
        }
        if self.loop_cut_hover.is_some() && code == KeyCode::Escape {
            self.end_loop_cut_hover();
            self.status("Loop Cut cancelled");
            return;
        }
        // Quitting and F1 are deliberately outside the keymap: there must
        // always be a way out and a way to ask, even from a keymap the user has
        // broken.
        match code {
            // `Escape` cancels, and outside a gesture, a pie or a search there
            // is nothing to cancel — so it does nothing, as in Blender. It used
            // to quit, which is what a Blender user's hand does after every
            // transform: a session lost to the key that means "never mind".
            KeyCode::Escape => return,
            // Quit on the key every application quits on.
            KeyCode::KeyQ
                if self.input.ctrl_down && !self.input.shift_down && !self.input.alt_down =>
            {
                self.quit(event_loop);
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
        self.dispatching_key = Some(name);
        self.dispatch(&action);
        self.dispatching_key = None;
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
            "object.translate" | "mesh.translate" | "pose.translate" => {
                self.begin_gesture(kreate::modal::Kind::Move)
            }
            "mesh.edge_slide" => self.begin_gesture(kreate::modal::Kind::EdgeSlide),
            "object.rotate" | "mesh.rotate" | "pose.rotate" => {
                self.begin_gesture(kreate::modal::Kind::Rotate)
            }
            "object.scale" | "mesh.scale" | "pose.scale" => {
                self.begin_gesture(kreate::modal::Kind::Scale)
            }
            // Blender's E: extrude, then drag the new faces out along their
            // normal with the pointer, rather than by a fixed amount.
            // Faces go out along their normal; edges and vertices, with no
            // face to give a direction, move freely.
            // Alt+E's rows: the same gesture, along the region's normal or
            // each face's own.
            // Add > Text: the next click puts it where it lands.
            "app.place_text" => {
                self.text_place = true;
                let message = self.ui.catalog.get("text-place-prompt");
                self.status(&message);
            }
            // A sketch on a face: the next click picks the face.
            "app.sketch_on_face" => {
                self.sketch_face_pick = true;
                self.sketch_waiting = true;
                let message = self.ui.catalog.get("sketch-pick-face");
                self.status(&message);
            }
            "app.extrude" if self.op_ctx.mode.edits_elements() => {
                let along = matches!(
                    action.params.get("along_normals"),
                    Some(PropertyValue::Bool(true))
                );
                if along {
                    if kreate_ops::ops::mesh::any_face_selected(&self.doc) {
                        self.extrude_along_normals = true;
                        self.begin_gesture(kreate::modal::Kind::Extrude);
                        self.extrude_along_normals = false;
                    } else {
                        self.status("select faces to extrude");
                    }
                } else {
                    self.dispatch(&Action::new("mesh.extrude_region"));
                }
            }
            "mesh.extrude_region" if self.op_ctx.mode.edits_elements() => {
                if kreate_ops::ops::mesh::selected_faces_normal(&self.doc).is_some() {
                    self.begin_gesture(kreate::modal::Kind::Extrude)
                } else {
                    self.begin_gesture(kreate::modal::Kind::ExtrudeFree)
                }
            }
            // I, Ctrl+B and Ctrl+R the same way: the pointer sets how much.
            "mesh.inset" if self.op_ctx.mode.edits_elements() => {
                self.begin_gesture(kreate::modal::Kind::Inset)
            }
            "mesh.bevel" if self.op_ctx.mode.edits_elements() => {
                self.begin_gesture(kreate::modal::Kind::Bevel)
            }
            "mesh.loop_cut" if self.op_ctx.mode.edits_elements() => self.begin_loop_cut_hover(),
            "mesh.knife" if self.op_ctx.mode.edits_elements() => self.begin_knife(),
            // The revolve's axis as the sketch's own horizontal or vertical.
            "app.sketch_axis" => {
                use kreate_scene::feature::{AXIS_X, AXIS_Y};
                let axis = match action.params.get("axis") {
                    Some(PropertyValue::Text(t)) if t == "y" => AXIS_Y,
                    _ => AXIS_X,
                };
                self.set_revolve_axis([axis; 2]);
                self.redraw.set(RedrawReason::UI);
            }
            "app.sketch_tool" => {
                let name = match action.params.get("tool") {
                    Some(PropertyValue::Text(t)) => t.clone(),
                    _ => String::new(),
                };
                if let Some(tool) = kreate::sketching::Tool::from_name(&name) {
                    if self.op_ctx.mode != OpMode::Sketch {
                        self.set_mode(OpMode::Sketch);
                    }
                    // The pattern works on what is selected: it is kept.
                    let picked = (self.sketching.points.clone(), self.sketching.lines.clone());
                    self.sketching.take(tool);
                    // An extrude nobody has picked for takes what the sketch
                    // draws — a rectangle on a face, a plate with its holes,
                    // a line of text — at once; a click puts a region back
                    // or adds one, as before.
                    if matches!(
                        tool,
                        kreate::sketching::Tool::Extrude | kreate::sketching::Tool::Revolve
                    ) {
                        if let Some((id, _)) = self.sketch_target() {
                            self.sketching.seeds =
                                kreate_scene::feature::profile_seeds(&self.doc, id)
                                    .into_iter()
                                    .map(|s| glam::DVec2::new(s[0], s[1]))
                                    .collect();
                        }
                    }
                    if matches!(
                        tool,
                        kreate::sketching::Tool::Pattern
                            | kreate::sketching::Tool::Mirror
                            | kreate::sketching::Tool::Offset
                    ) {
                        (self.sketching.points, self.sketching.lines) = picked;
                        self.ui.pattern_dialog.picking_centre = false;
                    }
                    self.redraw.set(RedrawReason::UI);
                }
            }
            "app.sketch_free" => {
                // The selected points, and the ends of the selected lines.
                let mut points = self.sketching.points.clone();
                if let Some(o) = kreate_ops::ops::sketch_draw::active_sketch(&self.doc)
                    .and_then(|id| self.doc.object(id))
                {
                    for line in &self.sketching.lines {
                        if let Some([a, b]) = o.sketch_lines.get(*line as usize) {
                            points.extend([*a, *b]);
                        }
                    }
                }
                points.sort_unstable();
                points.dedup();
                if points.is_empty() {
                    self.status(&self.ui.catalog.get("sketch-free-nothing"));
                } else {
                    self.run_op_with(
                        "sketch.constraint_free",
                        &[("points", PropertyValue::Ints(points))],
                    );
                }
            }
            // The selected lines made construction, or ordinary again when
            // every one of them already is.
            "app.sketch_construction" if self.op_ctx.mode == OpMode::Sketch => {
                let lines = self.sketching.lines.clone();
                if let Some(id) = kreate_ops::ops::sketch_draw::active_sketch(&self.doc) {
                    if let Some(o) = self.doc.object(id) {
                        let all = !lines.is_empty()
                            && lines.iter().all(|l| {
                                o.sketch_lines.get(*l as usize).is_some_and(|line| {
                                    kreate_scene::sketch::Constraint::is_construction(
                                        &o.sketch, *line,
                                    )
                                })
                            });
                        if !lines.is_empty() {
                            self.run_op_with(
                                "sketch.construction",
                                &[
                                    ("lines", PropertyValue::Ints(lines)),
                                    ("on", PropertyValue::Bool(!all)),
                                ],
                            );
                        }
                    }
                }
            }
            "app.sketch_delete" if self.op_ctx.mode == OpMode::Sketch => {
                let (points, lines) = (self.sketching.points.clone(), self.sketching.lines.clone());
                if !points.is_empty() || !lines.is_empty() {
                    self.run_op_with(
                        "sketch.delete",
                        &[
                            ("points", PropertyValue::Ints(points)),
                            ("lines", PropertyValue::Ints(lines)),
                        ],
                    );
                    self.sketching.reset();
                }
            }
            "mesh.select_linked_pick" if self.op_ctx.mode.edits_elements() => {
                let deselect = action
                    .params
                    .get("deselect")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.linked_pick(deselect);
            }
            "view.circle_select" => self.toggle_circle_select(),
            "sculpt.face_set_hide" => self.face_set_visibility(0),
            "sculpt.face_set_isolate" => self.face_set_visibility(1),
            "sculpt.remesh" => {
                let mut p = PropertyMap::new();
                p.set(
                    "voxel_size",
                    PropertyValue::Float(self.ui.sculpt.voxel_size as f64),
                );
                self.run_op("sculpt.voxel_remesh", p);
            }
            "sculpt.quad" => {
                let mut p = PropertyMap::new();
                p.set(
                    "faces",
                    PropertyValue::Int(self.ui.sculpt.quad_faces as i64),
                );
                self.run_op("sculpt.quad_remesh", p);
            }
            "brush.radius_adjust" => self.begin_brush_adjust(false),
            "brush.strength_adjust" => self.begin_brush_adjust(true),
            "sculpt.brush" => {
                if let Some(PropertyValue::Enum(b)) = action.params.get("brush") {
                    self.ui.sculpt.brush =
                        (*b).min(kreate_mesh::sculpt::Brush::ALL.len() as u32 - 1);
                }
            }
            "view.local_view" => self.toggle_local_view(),
            "view.camera" => self.toggle_camera_view(),
            // The keys' boolean, like the menu's: chosen in the viewport.
            "object.boolean" => {
                let operation = action.params.get("operation").and_then(|v| match v {
                    PropertyValue::Enum(n) => Some(*n),
                    PropertyValue::Int(n) => Some(*n as u32),
                    _ => None,
                });
                self.begin_boolean_pick(operation.unwrap_or(0));
            }
            // Ctrl+Numpad 0: the active camera becomes the scene's, and the
            // view goes through it.
            "view.camera_set_active" => {
                let Some(active) = self.doc.selection.active.filter(|a| self.doc.is_camera(*a))
                else {
                    let message = self.ui.catalog.get("camera-not-active");
                    self.status(&message);
                    return;
                };
                self.run_op("scene.camera_set", PropertyMap::new());
                // Out of the camera view, toggling goes in through the scene
                // camera; through another camera, it goes to the active one.
                if self.camera_view_object != Some(active) {
                    self.toggle_camera_view();
                }
            }
            "view.measure" => self.begin_measure(),
            "view.walk" => self.begin_walk(),
            "object.align_faces" => {
                self.end_loop_cut_hover();
                self.mate = Some(None);
                self.status("Align Faces: click the face of the part that moves, then the face it goes against (Esc stops)");
            }
            "file.export_drawing" => {
                let view = action
                    .params
                    .get("view")
                    .and_then(|v| v.as_int())
                    .unwrap_or(0);
                self.export_drawing(view as u32);
            }
            // The Object tab's Mesh Check: what is wrong with the active
            // mesh, in the world, and then where.
            "app.mesh_check" => {
                let thickness = action
                    .params
                    .get("thickness")
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.001);
                self.run_mesh_check(thickness);
            }
            "app.mesh_check_select" => {
                if !self.op_ctx.mode.edits_elements() {
                    self.set_mode(OpMode::RESTING.toggled(kreate_ops::mode::Has::of(&self.doc)));
                }
                self.run_op("mesh.check_select", action.params.clone());
                if self
                    .history
                    .last()
                    .is_some_and(|e| e.op_id == "mesh.check_select")
                {
                    self.set_select_mode(
                        if action
                            .params
                            .get("issue")
                            .and_then(|v| v.as_int())
                            .is_some_and(|i| matches!(i, 3 | 5 | 6))
                        {
                            2
                        } else if action.params.get("issue").and_then(|v| v.as_int()) == Some(4) {
                            0
                        } else {
                            1
                        },
                    );
                }
            }
            "view.frame_selected" => self.frame_selected(),
            "view.toggle_statistics" => {
                self.ui.show_stats = !self.ui.show_stats;
                self.redraw.set(RedrawReason::UI);
            }
            "view.toggle_dimensions" => {
                self.ui.show_dimensions = !self.ui.show_dimensions;
                self.redraw.set(RedrawReason::UI);
            }
            "view.toggle_cursor" => {
                self.ui.show_cursor = !self.ui.show_cursor;
                self.redraw.set(RedrawReason::UI);
            }
            "app.rename" => {
                // The active object's name, in a field under the pointer.
                let name = self
                    .doc
                    .selection
                    .active
                    .and_then(|a| self.doc.object(a))
                    .map(|o| o.name.clone());
                if let (Some(name), Some((x, y))) = (name, self.input.cursor) {
                    let scale = self.egui_ctx.pixels_per_point().max(0.01);
                    self.ui.rename = Some((egui::pos2(x as f32 / scale, y as f32 / scale), name));
                    self.redraw.set(RedrawReason::UI);
                }
            }
            "view.toggle_xray" => {
                // One switch for both halves, as Blender's: what shows through
                // can be picked through.
                self.xray = !self.xray;
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.xray = self.xray;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_path_trace" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.path_trace = !gfx.viewport.path_trace;
                }
                // Built afresh when it is next wanted.
                self.path_trace_key = None;
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_reflections" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_reflections = !gfx.viewport.show_reflections;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_contact_shadows" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.contact_shadows = !gfx.viewport.contact_shadows;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_bloom" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_bloom = !gfx.viewport.show_bloom;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_outline" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_outline = !gfx.viewport.show_outline;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_ao" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.ambient_occlusion = !gfx.viewport.ambient_occlusion;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "physics.bake_start" => {
                let only = match action.params.get("only") {
                    Some(PropertyValue::Enum(i)) => *i,
                    _ => 0,
                };
                self.start_bake(only)
            }
            "physics.bake_cancel" => {
                if let Some(job) = &self.bake_job {
                    job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            "view.toggle_textures" => {
                self.solid_textures = !self.solid_textures;
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            "view.toggle_face_orientation" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.face_orientation = !gfx.viewport.face_orientation;
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            // The 3D cursor.
            "view.cursor_place" => self.place_cursor(),
            "view.cursor_to_origin" => {
                self.place_cursor_turned(DVec3::ZERO, kreate_math::DQuat::IDENTITY)
            }
            "view.cursor_to_selected" => self.cursor_to_selected(),
            "view.selection_to_cursor" => self.selection_to_cursor(),
            "armature.translate" => self.begin_gesture(kreate::modal::Kind::Move),
            "armature.rotate" => self.begin_gesture(kreate::modal::Kind::Rotate),
            "armature.scale" => self.begin_gesture(kreate::modal::Kind::Scale),
            // Blender's E: grow the bones, then the drag that places them. Two
            // steps, as Extrude and Move are in Blender's history too.
            "armature.extrude" => {
                let before = self.history.entries().len();
                self.run_op("armature.extrude", PropertyMap::new());
                if self.history.entries().len() > before {
                    self.begin_gesture(kreate::modal::Kind::Move);
                }
            }
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
            // Copy is not an edit: it turns the selection into a fragment and
            // keeps the text. Paste is, and goes through the history with that
            // text as its parameter — see `object.paste` on why.
            "edit.copy" => self.copy_selection(),
            "edit.paste" => self.paste_clipboard(),
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
            // The clock. Not operators: moving through time changes what the
            // scene shows and leaves nothing to undo, as Blender's frame
            // change does not.
            "time.play" => {
                self.playing = match self.playing {
                    Some(_) => None,
                    None => Some(std::time::Instant::now()),
                };
                self.redraw.set(RedrawReason::PLAYING);
            }
            "time.toggle_autokey" => {
                self.autokey = !self.autokey;
                self.status(if self.autokey {
                    "auto-keying on"
                } else {
                    "auto-keying off"
                });
                self.redraw.set(RedrawReason::UI);
            }
            "time.set_frame" => {
                if let Some(PropertyValue::Float(frame)) = action.params.get("frame") {
                    self.set_frame(*frame);
                }
            }
            "time.step" => {
                let delta = match action.params.get("delta") {
                    Some(PropertyValue::Int(d)) => *d as f64,
                    _ => 1.0,
                };
                self.set_frame(self.doc.playback.frame.round() + delta);
            }
            "time.jump" => {
                let to_end = matches!(action.params.get("to_end"), Some(PropertyValue::Bool(true)));
                let p = self.doc.playback;
                self.set_frame(if to_end { p.end } else { p.start } as f64);
            }
            "time.set_range" => {
                let get = |name: &str| match action.params.get(name) {
                    Some(PropertyValue::Int(v)) => Some(*v as i32),
                    _ => None,
                };
                if let (Some(start), Some(end)) = (get("start"), get("end")) {
                    self.doc.playback.start = start;
                    self.doc.playback.end = end.max(start + 1);
                    self.redraw.set(RedrawReason::UI);
                }
            }
            "app.search" => {
                self.ui.palette.open();
                self.redraw.set(RedrawReason::UI);
            }
            // A pie, at the pointer, closed by letting go of the key that
            // opened it.
            "app.pie" => {
                let Some(PropertyValue::Text(name)) = action.params.get("menu") else {
                    return;
                };
                let scale = f64::from(self.egui_ctx.pixels_per_point().max(0.01));
                let at = self
                    .input
                    .cursor
                    .map(|(x, y)| egui::pos2((x / scale) as f32, (y / scale) as f32))
                    .unwrap_or(egui::pos2(200.0, 200.0));
                let key = self.dispatching_key.and_then(egui_key);
                let now = self.egui_ctx.input(|i| i.time);
                self.ui.pie.open(name, at, key, now);
                self.redraw.set(RedrawReason::UI);
            }
            // The history on the system clipboard as script lines, for a text
            // editor, a bug report or File ▸ Run Script later.
            "edit.copy_script" => {
                let script = self.history.to_script();
                let lines = script.lines().count();
                self.egui_ctx.copy_text(script);
                self.status(&format!("copied {lines} script lines"));
            }
            "file.run_script" => self.run_script_file(),
            "view.frame_all" => self.frame_all(),
            "view.toggle_edit_mode" => self.toggle_edit_mode(),
            "render.image" => self.start_render_job(false),
            "render.animation" => self.start_render_job(true),
            "render.view" => self.show_last_render(),
            // The render window's Stop, and the menu's: what Escape does.
            "render.stop" => {
                if self.render_job.is_some() {
                    self.finish_render_job("render stopped");
                } else {
                    self.status("nothing is rendering");
                }
            }
            // The sidebar's View tab: the viewport's focal length, in mm.
            "view.set_lens" => {
                if let Some(PropertyValue::Float(mm)) = action.params.get("lens") {
                    let mm = mm.clamp(1.0, 1000.0);
                    self.camera.fov_y = 2.0 * (12.0 / mm).atan();
                    // And where the easing is headed, or it eases straight
                    // back to the old lens: a settled camera is set to its
                    // target every frame.
                    self.motion.target.fov_y = self.camera.fov_y;
                    self.motion.interrupt(&self.camera);
                    self.redraw.set(RedrawReason::CAMERA_MOVED);
                }
            }
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
            "view.toggle_ground_shadow" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_ground_shadow = !gfx.viewport.show_ground_shadow;
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
                    // The three, in order: wireframe, preview, rendered.
                    // Rendered last because it is the one that needs a
                    // scene with lights in it.
                    gfx.viewport.shading = match gfx.viewport.shading {
                        ShadingMode::Wireframe => ShadingMode::Material,
                        ShadingMode::Studio | ShadingMode::Flat | ShadingMode::Material => {
                            ShadingMode::Rendered
                        }
                        ShadingMode::Rendered => ShadingMode::Wireframe,
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
                // The easing's target too: a settled camera is set to its
                // target every frame, which put the old projection straight
                // back.
                self.motion.target.projection = self.camera.projection;
                self.redraw.set(RedrawReason::CAMERA_MOVED);
            }
            // Navigation drags carry their motion in the input state rather than
            // in parameters, so they are handled where the deltas are.
            "view.orbit" | "view.pan" | "view.dolly" => {}
            // Blender's Shift+A and Ctrl+A: a menu under the pointer.
            "app.menu" => {
                if let Some(PropertyValue::Text(name)) = action.params.get("menu") {
                    let name = name.clone();
                    self.open_popup_menu(&name);
                }
            }
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
                        ("on_floor", PropertyValue::Bool(true)),
                    ],
                );
            }
            _ => {
                // An operator. The element mode is supplied where the binding
                // did not name one, so `X` deletes whatever the user is
                // selecting rather than always vertices.
                let mut params = action.params.clone();
                // A key goes on the frame the timeline is on, unless the
                // binding or the script named one. Whole, because the clock can
                // be between frames while playing and a key at 12.37 is one
                // nobody can land on again.
                if action.op_id.starts_with("anim.keyframe") && params.get("frame").is_none() {
                    params.set(
                        "frame",
                        PropertyValue::Float(self.doc.playback.frame.round()),
                    );
                }
                if action.op_id.starts_with("mesh.")
                    && params.get("mode").is_none()
                    && matches!(
                        action.op_id.as_str(),
                        "mesh.delete" | "mesh.dissolve" | "mesh.select" | "mesh.hide"
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
        KeyCode::Backquote => "Backquote",
        KeyCode::Space => "Space",
        KeyCode::ArrowLeft => "ArrowLeft",
        KeyCode::ArrowRight => "ArrowRight",
        KeyCode::ArrowUp => "ArrowUp",
        KeyCode::ArrowDown => "ArrowDown",
        // F1 is not here: it is outside the keymap on purpose, so help is
        // reachable from any keymap however broken.
        KeyCode::F2 => "F2",
        KeyCode::F3 => "F3",
        KeyCode::F4 => "F4",
        KeyCode::F5 => "F5",
        KeyCode::F6 => "F6",
        KeyCode::F7 => "F7",
        KeyCode::F8 => "F8",
        KeyCode::F9 => "F9",
        KeyCode::F10 => "F10",
        KeyCode::F11 => "F11",
        KeyCode::F12 => "F12",
        KeyCode::BracketLeft => "BracketLeft",
        KeyCode::BracketRight => "BracketRight",
        KeyCode::NumpadAdd => "NumpadAdd",
        KeyCode::NumpadDivide => "NumpadDivide",
        KeyCode::NumpadDecimal => "NumpadDecimal",
        KeyCode::Numpad0 => "Numpad0",
        KeyCode::Slash => "Slash",
        KeyCode::NumpadSubtract => "NumpadSubtract",
        KeyCode::Equal => "Equal",
        KeyCode::Minus => "Minus",
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

/// Whether two viewport cameras show the same view, to within what rounding
/// in the easing leaves.
fn same_view(a: &kreate_render::Camera, b: &kreate_render::Camera) -> bool {
    let reach = a.distance.max(b.distance).max(1e-3);
    a.eye().distance(b.eye()) <= reach * 1e-6
        && a.pivot.distance(b.pivot) <= reach * 1e-6
        && a.rotation.dot(b.rotation).abs() >= 1.0 - 1e-9
        && (a.fov_y - b.fov_y).abs() <= 1e-9
        && a.projection == b.projection
}

/// Below this many vertices a mesh is too coarse for a brush to shape
/// visibly, and entering sculpt mode says so.
const SCULPT_DENSE_ENOUGH: u32 = 1000;

/// A scene file read for importing: its path, ready to add, or why not.
/// The connector's search: one kind from one source, matched by words, and
/// with `thumbnails` the first few's pictures — on whichever thread asks,
/// since a source may be slow to answer and the pictures slower.
fn assets_search(
    params: &serde_json::Value,
    folders: &[std::path::PathBuf],
) -> Result<serde_json::Value, String> {
    use serde_json::json;
    let text = |key: &str| params.get(key).and_then(|v| v.as_str()).map(str::to_string);
    use kreate::assets::{Kind, Source};
    let kind = text("kind")
        .and_then(|k| Kind::from_tag(&k))
        .ok_or("`kind` is model, material or hdri")?;
    let source = Source::from_tag(&text("source").unwrap_or("polyhaven".into()))
        .ok_or("`source` is polyhaven, ambientcg, objaverse, sketchfab, polypizza or local")?;
    let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let query = text("query").unwrap_or_default();
    let animated = params
        .get("animated")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let asked = kreate::assets::Query {
        text: query.clone(),
        animated,
    };
    let entries = kreate::assets::list_now(source, kind, folders, &asked)?;
    // Pictures for the first few, when asked: what lets a model choose by
    // looking rather than by name.
    let pictures = params
        .get("thumbnails")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let found: Vec<serde_json::Value> = entries
        .iter()
        .filter(|e| source.searched() || e.matches(&query))
        .filter(|e| !animated || e.animated)
        .take(limit.clamp(1, 200))
        .enumerate()
        .map(|(i, e)| {
            let mut item = json!({
                "id": e.id,
                "name": e.name,
                "tags": e.tags,
                "authors": e.authors,
                "licence": e.licence,
                "animated": e.animated,
            });
            if pictures && i < 6 {
                if let Some(png) = kreate::assets::thumbnail_now(
                    &e.id,
                    e.source,
                    e.path.clone(),
                    e.thumbnail_url.clone(),
                )
                .and_then(|p| kreate::assets::picture_png(&p))
                {
                    item["png_base64"] = json!(kreate::bridge::base64(&png));
                }
            }
            item
        })
        .collect();
    Ok(json!({ "source": source.tag(), "kind": kind.tag(), "found": found }))
}

/// Where an import is to land and how tall, and who is waiting for it.
#[derive(Default)]
struct ImportWish {
    /// The middle of its base, in the world; beside what is there if none.
    location: Option<DVec3>,
    /// Its height in metres; the file's if none.
    height: Option<f64>,
    /// A connector call waiting to be told what arrived.
    reply: Option<kreate::bridge::Request>,
}

impl ImportWish {
    /// Tells whoever waits, once.
    fn answer(&mut self, result: Result<serde_json::Value, String>) {
        if let Some(reply) = self.reply.take() {
            reply.answer(result);
        }
    }
}

/// One answer in the chat panel, by the undo history around it.
struct ChatTurn {
    /// The history's cursor when the message was sent.
    start: usize,
    /// And when the answer ended; `None` while it is still answering.
    end: Option<usize>,
    /// The objects there were before, by uid, to say what it added and took
    /// away.
    objects: std::collections::HashSet<u64>,
}

/// A file read by an import thread: where it is and how much room the
/// model takes, or why it could not be read.
/// A read finished: the file, what it holds the room of, and whether it
/// replaces the document (Open) rather than joining it (Import).
type ImportReady =
    Result<(std::path::PathBuf, kreate_math::Aabb, bool), (std::path::PathBuf, String)>;

/// A scene file being read on a thread of its own.
struct Importing {
    id: u64,
    path: std::path::PathBuf,
    started: Instant,
    opening: bool,
    progress: std::sync::Arc<kreate_io::scene_file::ReadProgress>,
}

/// What became of one line of a script.
struct ScriptLine {
    text: String,
    /// What the line became, or why it did not run.
    result: Result<String, String>,
    /// The objects it made, by name and uid.
    created: Vec<(String, u64)>,
    /// And the materials and images.
    materials: Vec<(String, u64)>,
    images: Vec<(String, u64)>,
}

impl ApplicationHandler for App {
    /// The mouse's own movement, which Walk Navigation turns the head by:
    /// the pointer is held still meanwhile and reports nothing.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.walk_look(delta.0, delta.1);
        }
    }

    /// A connector request is waiting. See `kreate::bridge`.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        self.serve_bridge();
        self.serve_chat();
        self.serve_outbox();
        self.serve_assets();
        self.serve_imports();
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // The icon in the title bar and the taskbar: a cube with a K on it,
        // 64 pixels square, straight RGBA so nothing has to be decoded.
        let icon = winit::window::Icon::from_rgba(
            include_bytes!("../assets/kreate-64.rgba").to_vec(),
            64,
            64,
        )
        .ok();
        let attrs = Window::default_attributes()
            .with_title("Kreate")
            .with_window_icon(icon)
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0))
            // Hidden until the loading screen is in it: shown at once, the
            // window is a blank white rectangle for as long as the graphics
            // take to come up.
            .with_visible(false);
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
        kreate_ui::Theme::new(self.ui.appearance, kreate_ui::Density::default())
            .install(&self.egui_ctx);
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

        // The loading screen, as soon as there is something to draw it with.
        // What follows — every pipeline compiled, the shaders warmed — is the
        // bulk of the start-up, and without this the window sits blank for it.
        let mut splash = Splash::new(&self.egui_ctx, &self.ui.catalog);
        // egui's input as the window has it now — its size, its scale — kept
        // to draw every splash frame from, re-timed: the frames below are
        // drawn off the main thread, where the window's event state is not.
        let template = self
            .egui_state
            .as_mut()
            .expect("made just above")
            .take_egui_input(&window);
        let progress = SplashProgress::new(self.ui.catalog.get("splash-shaders"), 0.5);
        splash.draw(
            &self.egui_ctx,
            &template,
            &window,
            &gpu,
            &surface,
            &config,
            &mut egui_renderer,
            &progress,
        );
        window.set_visible(true);

        let target = ViewportTarget::new(&gpu.device, &mut egui_renderer, size.width, size.height);
        let warming = self.ui.catalog.get("splash-warming");
        let opening = self.ui.catalog.get("splash-opening");
        // The compiling holds this thread for seconds, so the splash is drawn
        // from another meanwhile: about sixty frames a second, the bar easing
        // toward each step's mark. The device is shared between threads by
        // design, and nothing here touches the surface or the renderer while
        // the painter has them.
        let viewport = std::thread::scope(|scope| {
            let painter = scope.spawn(|| {
                while !progress.finished() {
                    splash.draw(
                        &self.egui_ctx,
                        &template,
                        &window,
                        &gpu,
                        &surface,
                        &config,
                        &mut egui_renderer,
                        &progress,
                    );
                    std::thread::sleep(std::time::Duration::from_millis(16));
                }
            });
            let mut viewport = Viewport::new(
                &gpu.device,
                ViewportTarget::FORMAT,
                target.size.0,
                target.size.1,
            );
            // A moving view smoothed as a still one is: see
            // `Viewport::temporal_aa`.
            viewport.temporal_aa = true;
            // The preview: the materials under a light of their own.
            viewport.shading = ShadingMode::Material;
            // Pay for shader compilation now, while the window is still coming
            // up, rather than on the frame where the user first presses Tab or
            // moves the mouse over the scene. Creating a pipeline and compiling
            // it are not the same thing: the driver defers the second until a
            // draw arrives, and the overlay and picking pipelines are not
            // touched by an ordinary first frame. Measured cold, that first
            // touch costs around ten milliseconds — a dropped frame, landing
            // exactly when the user starts interacting.
            progress.step(warming, 0.9);
            viewport.warm_up(&gpu.device, &gpu.queue);
            progress.step(opening, 1.0);
            progress.finish();
            let _ = painter.join();
            viewport
        });
        // The last frame, full: the painter may have stopped short of it.
        splash.draw(
            &self.egui_ctx,
            &template,
            &window,
            &gpu,
            &surface,
            &config,
            &mut egui_renderer,
            &progress,
        );
        drop(splash);

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
            skinned: Vec::new(),
            pick_target,
            region,
            deformer,
            preview: None,
            sculpt_view: None,
            egui_renderer,
            target,
            extra: std::collections::BTreeMap::new(),
        });
        self.window = Some(window);
        // A document opened at launch keeps the view it was saved with; a
        // new one is framed.
        match self.file.view() {
            Some(view) => {
                self.camera = view;
                self.motion = kreate_render::CameraMotion::new(view);
            }
            None => self.frame_all(),
        }
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
        // A walk owns the keys, the buttons and the wheel while it runs.
        if self.walk_event(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => self.quit(event_loop),
            WindowEvent::DroppedFile(path) => {
                self.open_dropped(&path);
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }

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
                // Putting text: left puts it under the pointer, right gives up.
                if self.text_place && pressed {
                    match button {
                        MouseButton::Left if self.press_belongs_to_viewport() => {
                            self.place_text_under_cursor();
                            return;
                        }
                        MouseButton::Right => {
                            self.text_place = false;
                            self.status("");
                            return;
                        }
                        _ => {}
                    }
                }
                // Picking a face to sketch on: left takes the face under
                // the pointer, right gives up.
                if self.sketch_face_pick && pressed {
                    match button {
                        MouseButton::Left if self.press_belongs_to_viewport() => {
                            self.sketch_on_face_under_cursor();
                            return;
                        }
                        MouseButton::Right => {
                            self.sketch_face_pick = false;
                            self.sketch_waiting = false;
                            self.status("");
                            return;
                        }
                        _ => {}
                    }
                }
                // C: left strokes the brush, right stops it.
                if self.circle.is_some() {
                    match (button, pressed) {
                        (MouseButton::Left, true) if self.press_belongs_to_viewport() => {
                            self.begin_circle_stroke();
                            return;
                        }
                        (MouseButton::Left, false) => {
                            self.finish_circle_stroke();
                            return;
                        }
                        (MouseButton::Right, true) => {
                            self.end_circle_select();
                            return;
                        }
                        _ => {}
                    }
                }
                // Align Faces: left picks a face, right gives up.
                if self.mate.is_some() && pressed {
                    match button {
                        MouseButton::Left if self.press_belongs_to_viewport() => {
                            self.mate_click();
                            return;
                        }
                        MouseButton::Right => {
                            self.mate = None;
                            self.status("Align Faces cancelled");
                            return;
                        }
                        _ => {}
                    }
                }
                // F sizing the brush: left keeps, right puts it back.
                if self.ui.brush_adjust.is_some() && pressed {
                    match button {
                        MouseButton::Left => {
                            self.end_brush_adjust(true);
                            return;
                        }
                        MouseButton::Right => {
                            self.end_brush_adjust(false);
                            return;
                        }
                        _ => {}
                    }
                }
                // The Measure tool: left takes a point, right puts it away.
                if self.measure.is_some() && pressed {
                    match button {
                        MouseButton::Left if self.press_belongs_to_viewport() => {
                            self.measure_click();
                            return;
                        }
                        MouseButton::Right => {
                            self.end_measure("Measure cancelled");
                            return;
                        }
                        _ => {}
                    }
                }
                // K drawing a cut: left adds a point, right gives up.
                if self.knife.is_some() && pressed {
                    match button {
                        MouseButton::Left if self.press_belongs_to_viewport() => {
                            if let Some((points, Some(hover))) = &mut self.knife {
                                points.push(*hover);
                            }
                            self.update_knife();
                            return;
                        }
                        MouseButton::Right => {
                            self.end_knife("Knife cancelled");
                            return;
                        }
                        _ => {}
                    }
                }
                // The sketch tools: left draws or picks, right finishes.
                // The middle button falls through, so the view still turns.
                if self.op_ctx.mode == OpMode::Sketch && self.sketch_mouse(button, pressed) {
                    return;
                }
                // A curve's points and handles, in object mode, before a
                // click would select something else.
                if self.curve_mouse(button, pressed) {
                    return;
                }
                // Ctrl+R looking for an edge: left cuts, right gives up.
                if self.loop_cut_hover.is_some() && pressed {
                    match button {
                        MouseButton::Left if self.press_belongs_to_viewport() => {
                            self.commit_loop_cut_hover();
                            return;
                        }
                        // A click on a panel is somewhere else to be.
                        MouseButton::Left => self.end_loop_cut_hover(),
                        MouseButton::Right => {
                            self.end_loop_cut_hover();
                            self.status("Loop Cut cancelled");
                            return;
                        }
                        _ => {}
                    }
                }
                if self.gesture.is_some() && pressed {
                    match button {
                        // While a snap base is picked, the click picks it.
                        MouseButton::Left
                            if self.gesture.as_ref().is_some_and(|(g, _)| g.picking_base()) =>
                        {
                            self.finish_base_pick();
                            return;
                        }
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
                // A press the keymap names — Shift+Right-click for the 3D
                // cursor. Drags are resolved when they start moving.
                if pressed
                    && button == MouseButton::Right
                    && self.gesture.is_none()
                    && self.press_belongs_to_viewport()
                {
                    if let Some(action) = self.keymap.resolve(
                        self.key_context(),
                        &KeyInput::Mouse(KeyMouse::Right),
                        self.modifiers(),
                        EventKind::Press,
                    ) {
                        self.dispatch(&action);
                        return;
                    }
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
                        } else if let Some(view) = self.extra_view_at_cursor() {
                            // Only navigation: picking and editing are the
                            // first viewport's.
                            if let Some(action) = self.drag_action(mapped).filter(|a| {
                                matches!(a.op_id.as_str(), "view.orbit" | "view.pan" | "view.dolly")
                            }) {
                                self.drag_travel[slot] = 0.0;
                                self.drag_ops[slot] = Some(action.op_id);
                                self.nav_view = Some(view);
                            }
                        }
                    } else if !pressed && self.buttons[slot] {
                        self.end_mouse_drag(slot);
                    }
                    self.buttons[slot] = pressed;
                }
            }

            // A pen: Windows reports it as a touch with a force, alongside
            // the pointer it also moves. Its pressure is all that is taken
            // from here — the pointer drives the stroke as a mouse would —
            // and it is forgotten when the pen lifts, so a mouse afterwards
            // presses at full strength.
            WindowEvent::Touch(touch) => {
                use winit::event::TouchPhase;
                self.input.pen_pressure = match touch.phase {
                    TouchPhase::Ended | TouchPhase::Cancelled => None,
                    _ => touch.force.map(|f| f.normalized() as f32),
                };
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
                if self.paint_stroke.is_some()
                    || self.sculpt_stroke.is_some()
                    || self.texture_stroke.is_some()
                {
                    self.stroke_moved = true;
                    self.redraw.set(RedrawReason::SCENE_CHANGED);
                }
                if self.loop_cut_hover.is_some() {
                    self.update_loop_cut_hover();
                }
                if self.knife.is_some() {
                    self.update_knife();
                }
                if self.op_ctx.mode == OpMode::Sketch {
                    self.sketch_hover();
                }
                if self.curve_drag.is_some() {
                    self.drive_curve_drag();
                }
                if self.measure.is_some() {
                    self.update_measure();
                }
                if self.ui.brush_adjust.is_some() {
                    self.update_brush_adjust();
                }
                if self.circle.is_some() {
                    self.redraw.set(RedrawReason::HOVER_CHANGED);
                }
                // The brush circle follows the pointer.
                if matches!(
                    self.op_ctx.mode.acts_on(),
                    kreate_ops::mode::ActsOn::Weights
                        | kreate_ops::mode::ActsOn::Sculpt
                        | kreate_ops::mode::ActsOn::Texture
                ) {
                    self.redraw.set(RedrawReason::HOVER_CHANGED);
                }
            }

            // Owned by geometry, like a press. `consumed` is worse than useless
            // here: with no button held it is *always* true over the viewport,
            // so the wheel was being dropped exactly where zooming is wanted.
            // Over a panel the cursor is outside the viewport and the interface
            // keeps its own scrolling.
            WindowEvent::MouseWheel { delta, .. }
                if !self.cursor_in_viewport() && self.extra_view_at_cursor().is_some() =>
            {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
                self.nav_view = self.extra_view_at_cursor();
                self.input.note_event();
                self.input.scroll += y;
            }
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
                if let Some(c) = &mut self.circle {
                    c.radius = (c.radius * 1.15f32.powf(y as f32)).clamp(4.0, 800.0);
                    self.show_circle();
                } else if !self.adjust_loop_cuts(y as f32) && !self.adjust_proportional(y as f32) {
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
                self.input.command_down = if cfg!(target_os = "macos") {
                    m.state().super_key()
                } else {
                    m.state().control_key()
                };
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
                // Cmd+V on a Mac, Ctrl+V elsewhere, in the chat's box: a
                // picture or files on the clipboard are attached. egui pastes
                // only text, and does it itself.
                if self.ui.chat_focused
                    && self.input.command_down
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyV)
                {
                    self.paste_into_chat();
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input.note_event();
                    self.handle_key(code, event_loop);
                }
            }

            WindowEvent::RedrawRequested => {
                if std::mem::take(&mut self.stroke_moved) {
                    self.drive_strokes();
                }
                self.advance_playback();
                self.advance_walk();
                self.sync_timeline();
                self.apply_input();
                self.run_drivers();
                if self.redraw.any() {
                    self.restart_mean_if_changed();
                    self.render();
                    // Cleared before the render job steps, not after: the
                    // step moves the clock on to the next frame, or the next
                    // moment of a blurred one, and asks for that to be drawn.
                    // Cleared after, the ask was wiped, and a render whose
                    // last picture had all its samples waited for some other
                    // event to wake it — a mouse move — between every frame.
                    self.redraw.clear();
                    self.advance_render_job();
                    if self.render_job.is_some()
                        || self.gfx.as_ref().is_some_and(|g| g.viewport.accumulating())
                    {
                        self.redraw.set(RedrawReason::ACCUMULATE);
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.collab_tick();
        self.poll_bake();
        self.publish_imports();
        let downloaded = self
            .update_ready
            .as_ref()
            .and_then(|slot| slot.lock().ok().and_then(|mut v| v.take()));
        if let Some(version) = downloaded {
            self.status(&format!(
                "Kreate {version} is downloaded and will be used the next time Kreate starts"
            ));
            self.redraw.set(RedrawReason::UI);
        }
        // While the user is dragging, stay on Poll so motion is continuous;
        // otherwise Wait, so an idle window costs nothing — or until the
        // interface asked to be drawn again.
        if self.ui_repaint_at.is_some_and(|at| at <= Instant::now()) {
            self.ui_repaint_at = None;
            self.redraw.set(RedrawReason::UI);
        }
        if self.input.has_pending()
            || self.redraw.any()
            || self.motion.is_animating()
            || !self.ui.queue.is_empty()
            || self.playing.is_some()
            // A key held in a walk moves the view every frame, not only when
            // the keyboard repeats: that came half a second late and then
            // thirty times a second, a walk in stumbles.
            || self.walk.as_ref().is_some_and(|w| w.moving())
        {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Poll);
        } else if let Some(at) = self.ui_repaint_at {
            let at = if self.session.is_some() {
                at.min(Instant::now() + std::time::Duration::from_millis(30))
            } else {
                at
            };
            event_loop.set_control_flow(ControlFlow::WaitUntil(at));
        } else if self.session.is_some() {
            // Others' edits arrive without an event here to wake for.
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + std::time::Duration::from_millis(30),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// A render's EXR: the picture as R, G, B and A, its depth as Z and its
/// normals as N.X, N.Y and N.Z — the names compositors look for first.
fn exr_of(
    p: &kreate_render::viewport::RenderPasses,
    objects: Option<&[Option<String>]>,
) -> Vec<u8> {
    use kreate_io::exr::Channel;
    let pick = |f: &dyn Fn(usize) -> f32| -> Vec<f32> { (0..p.combined.len()).map(f).collect() };
    let r = pick(&|i| p.combined[i][0]);
    let g = pick(&|i| p.combined[i][1]);
    let b = pick(&|i| p.combined[i][2]);
    let a = pick(&|i| p.combined[i][3]);
    let nx = pick(&|i| p.normal[i][0]);
    let ny = pick(&|i| p.normal[i][1]);
    let nz = pick(&|i| p.normal[i][2]);
    fn channel<'a>(name: &'a str, values: &'a [f32], full_float: bool) -> Channel<'a> {
        Channel {
            name,
            values,
            full_float,
        }
    }
    // Cryptomatte, rank zero only: each pixel's object and full coverage.
    // The picking pass gives one object a pixel, so an edge is the object at
    // the pixel's centre; the next rank is left empty, which every reader
    // takes as nothing behind.
    let ids: Vec<f32> = objects
        .map(|o| {
            o.iter()
                .map(|n| n.as_deref().map_or(0.0, kreate_io::exr::cryptomatte_id))
                .collect()
        })
        .unwrap_or_default();
    let coverage: Vec<f32> = objects
        .map(|o| {
            o.iter()
                .map(|n| if n.is_some() { 1.0 } else { 0.0 })
                .collect()
        })
        .unwrap_or_default();
    let nothing = vec![0.0f32; ids.len()];
    let mut names: Vec<&str> = objects
        .map(|o| o.iter().filter_map(|n| n.as_deref()).collect())
        .unwrap_or_default();
    names.sort_unstable();
    names.dedup();
    let attributes = if objects.is_some() {
        kreate_io::exr::cryptomatte_attributes("CryptoObject", &names)
    } else {
        Vec::new()
    };
    let strings: Vec<(&str, &str)> = attributes
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let mut extra = Vec::new();
    if objects.is_some() && ids.len() == p.combined.len() {
        extra.extend([
            channel("CryptoObject00.R", &ids, true),
            channel("CryptoObject00.G", &coverage, true),
            channel("CryptoObject00.B", &nothing, true),
            channel("CryptoObject00.A", &nothing, true),
        ]);
    }
    let mut channels = vec![
        channel("R", &r, false),
        channel("G", &g, false),
        channel("B", &b, false),
        channel("A", &a, false),
        // Full floats: a depth in halves loses centimetres at a few metres.
        channel("Z", &p.depth, true),
        channel("N.X", &nx, false),
        channel("N.Y", &ny, false),
        channel("N.Z", &nz, false),
    ];
    channels.extend(extra);
    kreate_io::exr::write(p.width, p.height, &channels, &strings)
}

/// A render under way: F12's still, or Ctrl+F12's frames.
///
/// Run a frame at a time by the ordinary frame loop rather than in a loop of
/// its own, so the window stays alive: the picture refines in the viewport,
/// the status line counts the samples, and Escape stops it.
struct RenderJob {
    camera: kreate_scene::document::ObjectId,
    settings: kreate_scene::render::RenderSettings,
    /// The frame being rendered and the last one, for an animation.
    frames: Option<(i32, i32)>,
    /// Where files go when the settings name no place.
    folder: std::path::PathBuf,
    saved: SavedView,
    started: Instant,
    /// When sampling began and how far along it was then, for the time left.
    sampling: Option<(Instant, f32)>,
    /// Whether the viewport has been given the render's look: a frame after
    /// the start, so "preparing" is on screen while the renderer is built.
    look: bool,
    /// Frames drawn since this picture started: its first frames still show
    /// the last one's samples.
    drawn: u32,
    /// Which object each pixel shows, read once the camera has settled, for
    /// an EXR's Cryptomatte. See `Viewport::read_object_ids`.
    ids: Option<Vec<u32>>,
    /// Motion blur, when the settings ask for it and the camera's shutter is
    /// open: the picture as the mean of pictures across the shutter.
    blur: Option<Blur>,
    written: Vec<std::path::PathBuf>,
    /// An animation being written as one MP4, opened at its first frame.
    video: Option<kreate::video::Video>,
}

/// Motion blur in progress: which moment of the shutter is being drawn, and
/// the sum of those drawn so far.
struct Blur {
    step: u32,
    /// How many moments this render takes: see `Blur::steps_for`.
    steps: u32,
    /// How long the shutter is open, as a fraction of a frame.
    shutter: f64,
    /// The frame the shutter is centred on.
    base: f64,
    sum: Vec<[f32; 4]>,
}

impl Blur {
    /// The most moments across the shutter. Each gets an equal share of the
    /// samples, so the render costs what it did without blur.
    const STEPS: u32 = 16;

    /// The fewest samples a moment gets: what the denoiser needs to clean
    /// a picture, which it does from the samples' own spread. Sixteen
    /// moments of one sample each averaged to sixteen samples of noise no
    /// denoiser had seen whole.
    const MIN_SAMPLES: u32 = 16;

    /// How many moments a render of `samples` takes: four to sixteen, each
    /// with a share of the samples where they are many enough.
    fn steps_for(samples: u32) -> u32 {
        (samples / Self::MIN_SAMPLES).clamp(4, Self::STEPS)
    }

    /// How many samples each moment of a render of `samples` takes: its
    /// share, and never fewer than the denoiser needs — so a draft with
    /// motion blur costs more than one without, rather than speckling.
    fn samples_each(samples: u32) -> u32 {
        (samples / Self::steps_for(samples)).max(Self::MIN_SAMPLES)
    }

    /// The time of moment `step` of `steps`: the middles of equal slices of
    /// the shutter, centred on `base`.
    fn time(base: f64, shutter: f64, step: u32, steps: u32) -> f64 {
        base + shutter * ((step as f64 + 0.5) / steps.max(1) as f64 - 0.5)
    }
}

/// The view a render borrows, to give back.
struct SavedView {
    camera: kreate_render::Camera,
    motion: kreate_render::CameraMotion,
    camera_view: Option<kreate_render::Camera>,
    camera_view_object: Option<kreate_scene::document::ObjectId>,
    wanted_viewport: (u32, u32),
    look: (
        kreate_render::ShadingMode,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        Option<u32>,
        bool,
    ),
    frame: f64,
}

/// How far the start-up has come, shared between the thread doing it and the
/// one drawing the splash.
///
/// Each step names what it is doing and the mark the bar reaches when it is
/// done. Between steps the bar eases toward that mark and never quite gets
/// there — the work has no finer progress to report, and a bar that stops
/// dead reads as a hang — and when the step ends it moves on from wherever it
/// had got to.
struct SplashProgress {
    state: std::sync::Mutex<SplashStep>,
    start: std::time::Instant,
    done: std::sync::atomic::AtomicBool,
}

struct SplashStep {
    message: String,
    from: f32,
    to: f32,
    since: std::time::Instant,
}

impl SplashProgress {
    fn new(message: String, to: f32) -> Self {
        let now = std::time::Instant::now();
        Self {
            state: std::sync::Mutex::new(SplashStep {
                message,
                from: 0.0,
                to,
                since: now,
            }),
            start: now,
            done: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Where the bar is now, nought to one, and what is being done.
    fn now(&self) -> (f32, String) {
        let step = self.state.lock().expect("splash progress");
        let t = step.since.elapsed().as_secs_f32();
        // Most of the way in a second and a half, the rest ever more slowly.
        let shown = step.from + (step.to - step.from) * (1.0 - (-t / 1.5).exp());
        (shown, step.message.clone())
    }

    /// On to the next step: from wherever the bar has got to.
    fn step(&self, message: String, to: f32) {
        let (shown, _) = self.now();
        let mut step = self.state.lock().expect("splash progress");
        *step = SplashStep {
            message,
            from: shown,
            to,
            since: std::time::Instant::now(),
        };
    }

    /// The work is done: the bar full, and the painter to stop.
    fn finish(&self) {
        {
            let mut step = self.state.lock().expect("splash progress");
            step.from = 1.0;
            step.to = 1.0;
        }
        self.done.store(true, std::sync::atomic::Ordering::Release);
    }

    fn finished(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Seconds since the splash first showed, for its motion.
    fn time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

/// The loading screen: the logo, the name, what is being done, a progress bar
/// and the version, drawn while the window is being made ready — from a
/// thread of its own while the main one compiles, so it moves.
struct Splash {
    logo: Option<egui::TextureHandle>,
    title: String,
    version: String,
}

impl Splash {
    fn new(ctx: &egui::Context, catalog: &kreate_i18n::Catalog) -> Self {
        let logo = image::load_from_memory(include_bytes!("../assets/kreate.png"))
            .ok()
            .map(|img| {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                ctx.load_texture(
                    "splash-logo",
                    egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw()),
                    egui::TextureOptions::LINEAR,
                )
            });
        Self {
            logo,
            title: catalog.get("splash-title"),
            version: catalog.format(
                "splash-version",
                kreate_i18n::TextKind::Ui,
                &[("version", &kreate::version::describe())],
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        ctx: &egui::Context,
        template: &egui::RawInput,
        window: &Window,
        gpu: &kreate_render::Gpu,
        surface: &wgpu::Surface<'_>,
        config: &wgpu::SurfaceConfiguration,
        renderer: &mut egui_wgpu::Renderer,
        progress: &SplashProgress,
    ) {
        let t = progress.time();
        let (fraction, message) = progress.now();
        let mut input = template.clone();
        input.events.clear();
        input.time = Some(t as f64);
        let output = ctx.run_ui(input, |ui| {
            let theme = kreate_ui::theme::from_ui(ui);
            let background = ui.visuals().panel_fill;
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(background))
                .show(ui, |ui| {
                    // Behind everything: the ground running, the solids
                    // turning, the motes rising.
                    let backdrop =
                        kreate::splash::backdrop(ui.max_rect(), t, theme.text_dim, theme.accent);
                    ui.painter().extend(backdrop);
                    ui.vertical_centered(|ui| {
                        ui.add_space((ui.available_height() * 0.5 - 150.0).max(0.0));
                        // The logo breathing: a few per cent larger and back
                        // over two seconds, in a box that does not move.
                        let (slot, _) =
                            ui.allocate_exact_size(egui::vec2(140.0, 140.0), egui::Sense::hover());
                        if let Some(logo) = &self.logo {
                            let size = 128.0 * (1.0 + 0.03 * (t * std::f32::consts::PI).sin());
                            egui::Image::new(logo).paint_at(
                                ui,
                                egui::Rect::from_center_size(slot.center(), egui::vec2(size, size)),
                            );
                        }
                        ui.add_space(12.0);
                        ui.heading(&self.title);
                        ui.add_space(6.0);
                        // What is being done, and dots that keep counting
                        // while it is.
                        let dots = ".".repeat(1 + (t * 2.5) as usize % 3);
                        // i18n-exempt: a translated message and its dots
                        ui.label(format!("{message}{dots:<3}"));
                        ui.add_space(14.0);
                        // The bar: the track, how far, and a glint running
                        // along the filled part so it is seen to be alive.
                        let (bar, _) =
                            ui.allocate_exact_size(egui::vec2(260.0, 4.0), egui::Sense::hover());
                        let painter = ui.painter();
                        painter.rect_filled(bar, 2.0, theme.border);
                        let filled = egui::Rect::from_min_size(
                            bar.min,
                            egui::vec2(bar.width() * fraction.clamp(0.0, 1.0), bar.height()),
                        );
                        painter.rect_filled(filled, 2.0, theme.accent);
                        let sweep = (t * 0.7).fract();
                        let x = bar.left() + (sweep * 1.4 - 0.2) * bar.width();
                        let glint = egui::Rect::from_min_max(
                            egui::pos2((x - 24.0).max(filled.left()), bar.top()),
                            egui::pos2((x + 24.0).min(filled.right()), bar.bottom()),
                        );
                        if glint.width() > 0.0 {
                            painter.rect_filled(glint, 2.0, egui::Color32::from_white_alpha(70));
                        }
                        ui.add_space(16.0);
                        ui.weak(&self.version);
                    });
                });
        });
        for (id, delta) in &output.textures_delta.set {
            renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
        }
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            // No frame to draw into yet; the next step will try again.
            _ => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: output.pixels_per_point,
        };
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("splash"),
            });
        let extra = renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splash"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        gpu.queue
            .submit(extra.into_iter().chain(Some(encoder.finish())));
        window.pre_present_notify();
        frame.present();
        for id in &output.textures_delta.free {
            renderer.free_texture(id);
        }
    }
}

/// A selection region in viewport pixels, and the camera that projects
/// geometry onto it: what X-ray tests meshes against.
struct XrayRegion {
    view_projection: kreate_math::Mat4,
    eye: kreate_math::DVec3,
    size: kreate_math::Vec2,
    /// A box's or lasso's outline; empty for a circle.
    outline: Vec<kreate_math::Vec2>,
    circle: Option<(kreate_math::Vec2, f32)>,
}

impl XrayRegion {
    /// Each vertex of `mesh` in pixels, `None` for one behind the eye.
    fn project(
        &self,
        mesh: &MeshData,
        world: kreate_math::DAffine3,
    ) -> Vec<Option<kreate_math::Vec2>> {
        mesh.positions()
            .iter()
            .map(|p| {
                let rel = (world.transform_point3(p.as_dvec3()) - self.eye).as_vec3();
                let c = self.view_projection * rel.extend(1.0);
                (c.w > 1e-6).then(|| {
                    let n = c.truncate() / c.w;
                    kreate_math::Vec2::new(
                        (n.x * 0.5 + 0.5) * self.size.x,
                        (0.5 - n.y * 0.5) * self.size.y,
                    )
                })
            })
            .collect()
    }

    fn inside(&self, p: kreate_math::Vec2) -> bool {
        match self.circle {
            Some((c, r)) => p.distance(c) <= r,
            None => in_polygon(&self.outline, p),
        }
    }

    /// Whether the segment `ab` crosses the region's edge.
    fn crosses(&self, a: kreate_math::Vec2, b: kreate_math::Vec2) -> bool {
        match self.circle {
            Some((c, r)) => {
                let ab = b - a;
                let t = ((c - a).dot(ab) / ab.length_squared().max(f32::EPSILON)).clamp(0.0, 1.0);
                (a + ab * t).distance(c) <= r
            }
            None => (0..self.outline.len()).any(|i| {
                segments_cross(
                    a,
                    b,
                    self.outline[i],
                    self.outline[(i + 1) % self.outline.len()],
                )
            }),
        }
    }

    /// Whether face `f` holds a point of the region — for a region wholly
    /// inside one face, which touches none of its corners or sides.
    fn face_holds_anchor(&self, mesh: &MeshData, px: &[Option<kreate_math::Vec2>], f: u32) -> bool {
        let anchor = match self.circle {
            Some((c, _)) => c,
            None => self
                .outline
                .first()
                .copied()
                .unwrap_or(kreate_math::Vec2::ZERO),
        };
        let ring: Option<Vec<kreate_math::Vec2>> =
            mesh.face_verts(f).iter().map(|v| px[*v as usize]).collect();
        ring.is_some_and(|r| in_polygon(&r, anchor))
    }
}

/// Whether `p` is inside the closed outline `ring` (even-odd).
fn in_polygon(ring: &[kreate_math::Vec2], p: kreate_math::Vec2) -> bool {
    let mut inside = false;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}

/// Whether segments `ab` and `cd` cross.
fn segments_cross(
    a: kreate_math::Vec2,
    b: kreate_math::Vec2,
    c: kreate_math::Vec2,
    d: kreate_math::Vec2,
) -> bool {
    let side =
        |p: kreate_math::Vec2, q: kreate_math::Vec2, r: kreate_math::Vec2| (q - p).perp_dot(r - p);
    let (d1, d2) = (side(c, d, a), side(c, d, b));
    let (d3, d4) = (side(a, b, c), side(a, b, d));
    (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
}

/// A move's snapping step in the document's unit: a round number of what
/// the panels show — 1 mm, 1 cm, 0.1 m, 1/10 in, 1 in — so a part drawn in
/// millimetres does not jump a tenth of a metre at a time.
fn snap_step(unit: kreate_scene::design::LengthUnit) -> f64 {
    use kreate_scene::design::LengthUnit as U;
    match unit {
        U::Millimetres => 0.001,
        U::Centimetres => 0.01,
        U::Inches => 0.00254,
        U::Feet => 0.0254,
        U::Metres => 0.1,
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
        None => kreate::startup::starting_scene(&mut doc),
    }
    doc
}

/// Files an export names, written in the folder beside it: a stage's
/// pictures, a robot's meshes.
fn write_beside(path: &std::path::Path, files: &[(String, Vec<u8>)]) -> Result<(), String> {
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    for (name, bytes) in files {
        let target = dir.join(name);
        if let Some(folder) = target.parent() {
            std::fs::create_dir_all(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
        }
        std::fs::write(&target, bytes).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    Ok(())
}

fn main() {
    // To the console and to the day's file in the settings folder's `logs`.
    kreate::logs::init();

    // The computer's fonts, listed on the side: the first text's font menu
    // then opens at once rather than after a second's look through them.
    let _ = std::thread::Builder::new().name("fonts".into()).spawn(|| {
        kreate_mesh::text::fonts();
    });

    // The interface is in English. `--lang ko` still switches it, for the
    // Korean catalogue that is kept and tested.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut locale = "en-US".to_string();
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

    // An update downloaded last time goes in before anything opens; when it
    // did, the new version is already starting and this one steps aside.
    if kreate::update::apply_staged(&args) {
        return;
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
    // The connector's way in. A failure costs the connector and nothing else.
    let chat_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    let chat = kreate::claude::Chat::new(move || {
        if let Ok(proxy) = chat_proxy.lock() {
            let _ = proxy.send_event(());
        }
    });
    // Asked once at start, off the main thread: the panel shows who it will
    // talk as before anything is typed.
    chat.check_account();
    app.chat = Some(chat);
    app.restore_chat();
    let outbox_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    app.outbox = Some(kreate::contact::Outbox::new(move || {
        if let Ok(proxy) = outbox_proxy.lock() {
            let _ = proxy.send_event(());
        }
    }));
    let import_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    app.import_wake = Some(std::sync::Arc::new(move || {
        if let Ok(proxy) = import_proxy.lock() {
            let _ = proxy.send_event(());
        }
    }));
    let assets_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    app.assets = Some(kreate::assets::Library::new(move || {
        if let Ok(proxy) = assets_proxy.lock() {
            let _ = proxy.send_event(());
        }
    }));
    app.ui.asset_folders = kreate::assets::load_folders();
    let proxy = std::sync::Mutex::new(event_loop.create_proxy());
    if let Some(info) = kreate::bridge::info_path() {
        match kreate::bridge::start(&info, move || {
            if let Ok(proxy) = proxy.lock() {
                let _ = proxy.send_event(());
            }
        }) {
            Ok(bridge) => {
                tracing::info!(port = bridge.port, "connector bridge listening");
                app.bridge = Some(bridge);
            }
            Err(e) => tracing::warn!(error = %e, "connector bridge not started"),
        }
    }
    // Off the main thread: a slow network never holds up the window.
    let update_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    app.update_ready = Some(kreate::update::check_in_background(move || {
        if let Ok(proxy) = update_proxy.lock() {
            let _ = proxy.send_event(());
        }
    }));
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
        text.push_str(&format!("kreate {}\n", kreate::version::describe()));
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

/// How far to move a model whose box is `model` so it overlaps none of
/// `boxes`: nothing when it is clear already, else along +X past whatever it
/// meets, with a tenth of its width between. Every asset is made at the
/// origin, so without this the second lands inside the first — a bottle
/// imported after a barrel was in the outliner and nowhere to be seen.
fn clear_offset(model: (DVec3, DVec3), boxes: &[(DVec3, DVec3)]) -> DVec3 {
    let (lo, hi) = model;
    let gap = ((hi.x - lo.x) * 0.1).max(0.05);
    let mut offset = DVec3::ZERO;
    for _ in 0..256 {
        let (a, b) = (lo + offset, hi + offset);
        let Some((_, other)) = boxes
            .iter()
            .find(|(c, d)| a.cmplt(*d).all() && b.cmpgt(*c).all())
        else {
            break;
        };
        offset.x = other.x + gap - lo.x;
    }
    offset
}

/// `params` with the auto-key parameters set, when auto-keying is on and the
/// operator transforms objects.
///
/// Not when the caller already named them: a script line or a replayed step
/// that says `autokey=False` means it, whatever the record button says now.
/// `at`, or the first point along +X from it clear of every box (world
/// bounds, low and high corner). A new shape is two units across, so it is
/// clear of a box by its half and a little room.
fn spot_beside(at: DVec3, boxes: &[(DVec3, DVec3)]) -> DVec3 {
    let mut spot = at;
    for _ in 0..64 {
        let Some((_, hi)) = boxes
            .iter()
            .find(|(lo, hi)| spot.cmpge(*lo - 1.0).all() && spot.cmple(*hi + 1.0).all())
        else {
            break;
        };
        spot.x = hi.x + 1.5;
    }
    spot
}

/// What the application adds to an operator's parameters from its own state:
/// the auto-key frame, the pivot point, and the 3D cursor for a new object.
/// Taken as a copy, so it can be read before a gesture's session is borrowed.
#[derive(Clone, Copy, Debug)]
struct AppParams {
    key: Option<f64>,
    pivot: u32,
    cursor: DVec3,
    /// The work plane's turn, when the Cursor orientation is chosen: new
    /// shapes are set square to it.
    plane: Option<kreate_math::DQuat>,
    /// The header's Auto Merge, for the edit mode's move, rotate and scale.
    auto_merge: bool,
    /// The frame the timeline is on, for the markers' operators.
    frame: f64,
}

/// A turn whose Z is `normal` and whose X is level (along the ground) where
/// the normal allows, so a work plane on a wall reads with X across it.
fn work_plane_turn(normal: DVec3) -> kreate_math::DQuat {
    let z = normal.normalize_or(DVec3::Z);
    let x = DVec3::Z.cross(z);
    let x = if x.length() < 1e-6 {
        DVec3::X
    } else {
        x.normalize()
    };
    let y = z.cross(x);
    kreate_math::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z)).normalize()
}

impl AppParams {
    fn apply(&self, op_id: &str, params: &PropertyMap) -> PropertyMap {
        let mut out = with_autokey(self.key, op_id, params);
        // The header's pivot point, in each operator's own numbering; a
        // parameter already given — a replay, a script — wins.
        let pivot = match op_id {
            "object.rotate" | "object.scale" => Some([0, 1, 2, 4][self.pivot.min(3) as usize]),
            "mesh.rotate" | "mesh.scale" => Some(u32::from(self.pivot == 3)),
            _ => None,
        };
        if let (Some(pivot), None) = (pivot, out.get("pivot")) {
            out.set("pivot", PropertyValue::Enum(pivot));
        }
        // New objects appear at the 3D cursor. A lamp and a camera keep their
        // own offset from it — above and back — so a new one lights and sees
        // the thing at the cursor rather than sitting inside it.
        let offset = match op_id {
            "object.add_primitive"
            | "object.add_empty"
            | "object.add_armature"
            | "object.add_gear"
            | "object.add_text"
            | "object.add_wall"
            | "object.add_door"
            | "object.add_window"
            | "object.add_stairs"
            | "object.add_column"
            | "object.add_railing"
            | "object.add_slab"
            | "object.add_roof" => Some(DVec3::ZERO),
            "object.add_light" => Some(DVec3::new(0.0, 0.0, 4.0)),
            "object.add_camera" => Some(DVec3::new(0.0, -8.0, 2.0)),
            _ => None,
        };
        if let (Some(offset), None) = (offset, out.get("location")) {
            let at = self.cursor + offset;
            out.set("location", PropertyValue::Vec3(at.to_array()));
        }
        // A marker goes on the frame the timeline is on, unless one is named.
        if matches!(op_id, "marker.add" | "marker.remove" | "marker.camera_bind")
            && out.get("frame").is_none()
        {
            out.set("frame", PropertyValue::Float(self.frame));
        }
        if matches!(op_id, "mesh.translate" | "mesh.rotate" | "mesh.scale")
            && out.get("auto_merge").is_none()
        {
            out.set("auto_merge", PropertyValue::Bool(self.auto_merge));
        }
        if let (Some(q), "object.add_primitive", None) = (self.plane, op_id, out.get("rotation")) {
            let euler = kreate_scene::animation::to_euler(q);
            out.set("rotation", PropertyValue::Vec3(euler.to_array()));
        }
        out
    }
}

fn with_autokey(frame: Option<f64>, op_id: &str, params: &PropertyMap) -> PropertyMap {
    let mut out = params.clone();
    let transforms = matches!(
        op_id,
        "object.translate"
            | "object.rotate"
            | "object.scale"
            | "object.set_transform"
            | "pose.translate"
            | "pose.rotate"
            | "pose.scale"
    );
    if let (Some(frame), true) = (frame, transforms) {
        if out.get("autokey").is_none() {
            out.set("autokey", PropertyValue::Bool(true));
            out.set("key_frame", PropertyValue::Float(frame));
        }
    }
    out
}

/// The egui key for a keymap key name, for a widget that watches the key
/// itself — a pie waiting for its key to be let go.
fn egui_key(name: &str) -> Option<egui::Key> {
    match name {
        "Backquote" => Some(egui::Key::Backtick),
        "BracketLeft" => Some(egui::Key::OpenBracket),
        "BracketRight" => Some(egui::Key::CloseBracket),
        other => egui::Key::from_name(other.strip_prefix("Digit").unwrap_or(other)),
    }
}

/// The keymap context a mode's keys resolve in, as its card names it.
///
/// Object mode's for a tag the keymap does not know — a mode from a newer
/// build. A test holds every shipped mode's tag to a real context.
fn mode_key_context(mode: OpMode) -> KeyContext {
    KeyContext::from_tag(mode.info().keymap).unwrap_or(KeyContext::ObjectMode)
}

/// How far `p` is from the segment `a`–`b`, on screen.
fn segment_distance(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a + ab * t).distance(p)
}

/// The viewing modes offered: wireframe, the preview, rendered. The solid
/// view's studio and flat lights are the preview now.
fn offered(mode: ShadingMode) -> ShadingMode {
    match mode {
        ShadingMode::Studio | ShadingMode::Flat => ShadingMode::Material,
        m => m,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_xray_region_tests_see_inside_and_across() {
        use kreate_math::Vec2;
        let square = [
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        assert!(in_polygon(&square, Vec2::new(5.0, 5.0)));
        assert!(!in_polygon(&square, Vec2::new(15.0, 5.0)));
        // A long edge through the box with no end in it still crosses it.
        assert!(segments_cross(
            Vec2::new(-5.0, 5.0),
            Vec2::new(5.0, 5.0),
            square[3],
            square[0]
        ));
        assert!(!segments_cross(
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
            square[0],
            square[1]
        ));
    }

    #[test]
    fn x_ray_reaches_the_corners_hidden_behind_the_front_ones() {
        use super::*;
        use kreate_math::Vec2;
        // Looking straight down an axis, a cube's back corners fall on the
        // very pixels of its front ones — and still all eight are found.
        let cube = kreate_mesh::primitives::cube(1.0);
        let region = XrayRegion {
            view_projection: kreate_math::Mat4::IDENTITY,
            eye: DVec3::ZERO,
            size: Vec2::new(100.0, 100.0),
            outline: vec![
                Vec2::new(10.0, 10.0),
                Vec2::new(90.0, 10.0),
                Vec2::new(90.0, 90.0),
                Vec2::new(10.0, 90.0),
            ],
            circle: None,
        };
        let px = region.project(&cube, kreate_math::DAffine3::IDENTITY);
        assert_eq!(
            px.iter().flatten().filter(|p| region.inside(**p)).count(),
            8
        );
        // A box inside one face, touching none of its corners, holds it.
        let small = XrayRegion {
            outline: vec![
                Vec2::new(48.0, 48.0),
                Vec2::new(52.0, 48.0),
                Vec2::new(52.0, 52.0),
                Vec2::new(48.0, 52.0),
            ],
            ..region
        };
        assert!((0..cube.num_faces()).any(|f| small.face_holds_anchor(&cube, &px, f)));
        assert!(px.iter().flatten().all(|p| !small.inside(*p)));
    }

    #[test]
    fn a_new_object_is_set_beside_what_the_cursor_is_inside() {
        use super::*;
        let cube = (DVec3::splat(-1.0), DVec3::splat(1.0));
        let next = (DVec3::new(1.5, -1.0, -1.0), DVec3::new(3.5, 1.0, 1.0));
        // Free: where the cursor is.
        assert_eq!(
            spot_beside(DVec3::new(10.0, 0.0, 0.0), &[cube]),
            DVec3::new(10.0, 0.0, 0.0)
        );
        // Inside the cube: past it.
        assert_eq!(spot_beside(DVec3::ZERO, &[cube]), DVec3::new(2.5, 0.0, 0.0));
        // And past the one beside it too.
        assert_eq!(
            spot_beside(DVec3::ZERO, &[cube, next]),
            DVec3::new(5.0, 0.0, 0.0)
        );
    }

    #[test]
    fn an_imported_model_lands_clear_of_what_is_there() {
        use super::*;
        // A barrel at the origin, and a bottle made at the origin too.
        let barrel = (DVec3::new(-0.3, -0.3, 0.0), DVec3::new(0.3, 0.3, 0.9));
        let bottle = (DVec3::new(-0.05, -0.05, 0.0), DVec3::new(0.05, 0.05, 0.3));
        // An empty scene, or one clear of it: where the file puts it.
        assert_eq!(clear_offset(bottle, &[]), DVec3::ZERO);
        let far = (DVec3::splat(5.0), DVec3::splat(6.0));
        assert_eq!(clear_offset(bottle, &[far]), DVec3::ZERO);
        // Inside the barrel: moved along X until it touches nothing.
        let offset = clear_offset(bottle, &[barrel]);
        assert!(offset.y == 0.0 && offset.z == 0.0, "{offset}");
        assert!(bottle.0.x + offset.x > barrel.1.x, "{offset}");
        // Past a second one standing beside the first, too.
        let crate_ = (DVec3::new(0.35, -0.2, 0.0), DVec3::new(1.0, 0.2, 0.5));
        let offset = clear_offset(bottle, &[barrel, crate_]);
        assert!(bottle.0.x + offset.x > crate_.1.x, "{offset}");
    }

    #[test]
    fn the_pivot_point_and_the_cursor_reach_the_operators() {
        use super::*;
        let fill = AppParams {
            key: None,
            pivot: 3,
            cursor: DVec3::new(1.0, 2.0, 3.0),
            plane: None,
            auto_merge: false,
            frame: 12.0,
        };
        let none = PropertyMap::new();
        assert_eq!(
            fill.apply("object.rotate", &none).get("pivot"),
            Some(&PropertyValue::Enum(4))
        );
        assert_eq!(
            fill.apply("mesh.scale", &none).get("pivot"),
            Some(&PropertyValue::Enum(1))
        );
        assert_eq!(fill.apply("mesh.translate", &none).get("pivot"), None);
        // A given pivot is not overruled.
        let mut given = PropertyMap::new();
        given.set("pivot", PropertyValue::Enum(0));
        assert_eq!(
            fill.apply("object.scale", &given).get("pivot"),
            Some(&PropertyValue::Enum(0))
        );
        // New things at the cursor; a lamp above it.
        assert_eq!(
            fill.apply("object.add_primitive", &none).get("location"),
            Some(&PropertyValue::Vec3([1.0, 2.0, 3.0]))
        );
        assert_eq!(
            fill.apply("object.add_light", &none).get("location"),
            Some(&PropertyValue::Vec3([1.0, 2.0, 7.0]))
        );
        let median = AppParams { pivot: 0, ..fill };
        assert_eq!(
            median.apply("mesh.rotate", &none).get("pivot"),
            Some(&PropertyValue::Enum(0))
        );
    }

    #[test]
    fn every_key_that_opens_a_pie_can_be_watched_for_its_release() {
        // The pie closes when its key is let go, which it can only see if the
        // keymap's name for the key has an egui key. One without would open a
        // pie that hold-and-release never closes.
        for preset in kreate_input::presets::ALL {
            for binding in kreate_input::presets::load(*preset).bindings() {
                if binding.action.op_id != "app.pie" {
                    continue;
                }
                let kreate_input::Input::Key(name) = &binding.input else {
                    panic!("{preset:?}: a pie opened by something that is not a key");
                };
                assert!(
                    egui_key(name).is_some(),
                    "{preset:?}: `{name}` has no egui key"
                );
            }
        }
    }

    use super::*;

    /// An application with no window, which is all the input state machine needs.
    fn app() -> App {
        // Nothing a test does lands in the user's own folders.
        kreate::prefs::keep_to_a_test_folder();
        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        let mut app = App::empty(Document::new(), "en-US", registry);
        app.keymap = presets::load(Preset::Kreate);
        app
    }

    /// The middle button, which the default preset orbits with.
    const ORBIT: usize = 1;

    /// A rotation read from get_scene goes back into set_transform as it
    /// is: both in radians, and the units said.
    #[test]
    fn get_scene_gives_rotations_in_the_unit_operators_take() {
        let mut app = app();
        let id = app
            .doc
            .add_mesh_object("Part", kreate_mesh::primitives::cube(0.02));
        app.doc.selection.add(id);
        app.doc.selection.active = Some(id);
        let set = |app: &mut App, r: [f64; 3]| {
            let mut p = PropertyMap::new();
            p.set("rotation", PropertyValue::Vec3(r));
            app.history
                .dispatch(
                    &app.registry,
                    &mut app.doc,
                    &app.op_ctx,
                    "object.set_transform",
                    &p,
                )
                .unwrap();
        };
        let wanted = [0.3, -0.7, 1.2];
        set(&mut app, wanted);
        let scene = app.scene_json_with(&connector::SceneFilter::default());
        assert_eq!(scene["units"]["angle"], "rad");
        assert_eq!(scene["units"]["length"], "m");
        let read: Vec<f64> = scene["objects"][0]["rotation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for (r, w) in read.iter().zip(wanted) {
            assert!((r - w).abs() < 1e-9, "{read:?}");
        }
        let before = app.doc.object(id).unwrap().transform.rotation;
        set(&mut app, [read[0], read[1], read[2]]);
        let after = app.doc.object(id).unwrap().transform.rotation;
        assert!(before.dot(after).abs() > 1.0 - 1e-12);
    }

    /// Parts that only scrape — pressed together by a hundredth of a
    /// millimetre — are touching, not crossing; parts that really run into
    /// each other are crossing, with the volume they share.
    #[test]
    fn check_overlaps_tells_a_scrape_from_an_overlap() {
        let mut app = app();
        let cube = |s: f32| kreate_mesh::primitives::cube(s);
        // A cradle with a 10 mm pocket, and a servo pressed into it a
        // hundredth of a millimetre too wide: its sides scrape the walls.
        let mut pocket = cube(0.01);
        for p in pocket.positions_mut() {
            p.z += 0.006;
        }
        let cradle = kreate_mesh::modifiers::boolean::boolean(
            &cube(0.02),
            &pocket,
            kreate_mesh::modifiers::BooleanOp::Difference,
        );
        app.doc.add_mesh_object("Cradle", cradle);
        let servo = app.doc.add_mesh_object("Servo", cube(0.01 + 2e-5));
        app.doc.object_mut(servo).unwrap().transform.translation = DVec3::new(0.0, 0.0, 0.006);
        // And a bracket that really runs 5 mm into the cradle's side.
        let bracket = app.doc.add_mesh_object("Bracket", cube(0.02));
        app.doc.object_mut(bracket).unwrap().transform.translation = DVec3::new(-0.015, 0.0, 0.0);
        let out = app.overlaps_json(0.001, true, None, std::time::Duration::from_secs(60));
        let names = |key: &str| -> Vec<String> {
            out[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| format!("{}-{}", p["a"].as_str().unwrap(), p["b"].as_str().unwrap()))
                .collect()
        };
        assert_eq!(names("touching"), vec!["Cradle-Servo"], "{out}");
        assert_eq!(names("overlapping"), vec!["Cradle-Bracket"], "{out}");
        assert_eq!(out["overlapping"][0]["contact"], "crossing", "{out}");
        let v = out["overlapping"][0]["volume"].as_f64().unwrap();
        assert!((v - 0.005 * 0.02 * 0.02).abs() < 1e-9, "{v}");
    }

    /// The connector finds an asset and adds it: search by words, add by
    /// the id the search gave, the model arriving in the scene once fetched.
    #[test]
    fn the_connector_searches_the_asset_folders_and_adds_what_it_found() {
        let dir =
            std::env::temp_dir().join(format!("kreate-connector-assets-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let mesh = kreate_mesh::primitives::cube(1.0);
        std::fs::write(
            dir.join("wooden_crate.obj"),
            kreate_io::write_obj(&[("Crate".to_string(), &mesh)], Default::default()),
        )
        .expect("written");
        let mut app = app();
        app.assets = Some(kreate::assets::Library::new(|| {}));
        app.ui.asset_folders = vec![dir.clone()];
        let found = app
            .bridge_call(
                "assets",
                &serde_json::json!({"kind": "model", "source": "local", "query": "crate"}),
            )
            .expect("searched");
        let id = found["found"][0]["id"]
            .as_str()
            .expect("one found")
            .to_string();
        assert!(app
            .bridge_call(
                "assets",
                &serde_json::json!({"kind": "model", "source": "local", "query": "barrel"}),
            )
            .expect("searched")["found"]
            .as_array()
            .is_some_and(|a| a.is_empty()));

        let before = app.doc.objects().count();
        app.bridge_call(
            "add_asset",
            &serde_json::json!({"kind": "model", "source": "local", "id": id}),
        )
        .expect("asked for");
        let start = std::time::Instant::now();
        while app.doc.objects().count() == before && start.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            app.serve_assets();
            app.serve_imports();
        }
        assert_eq!(app.doc.objects().count(), before + 1);
        assert!(app
            .bridge_call(
                "add_asset",
                &serde_json::json!({"kind": "model", "source": "local", "id": "nope"})
            )
            .is_err());

        // Again, placed and sized: its base's middle at (3, 1, 0), two
        // metres tall.
        let before = app.doc.objects().count();
        app.bridge_call(
            "add_asset",
            &serde_json::json!({"kind": "model", "source": "local", "id": id,
                                "location": [3.0, 1.0, 0.0], "height": 2.0}),
        )
        .expect("asked for");
        let start = std::time::Instant::now();
        while app.doc.objects().count() == before && start.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            app.serve_assets();
            app.serve_imports();
        }
        app.evaluate_scene();
        let arrived = app.arrived_json();
        let first = &arrived["added"][0];
        let size = first["size"].as_array().expect("a size");
        let min = first["min"].as_array().expect("a corner");
        let f = |v: &serde_json::Value| v.as_f64().expect("a number");
        assert!((f(&size[2]) - 2.0).abs() < 1e-3, "two metres tall: {first}");
        assert!((f(&min[2])).abs() < 1e-3, "standing on the ground: {first}");
        assert!(
            (f(&min[0]) + f(&size[0]) / 2.0 - 3.0).abs() < 1e-3,
            "centred on x = 3: {first}"
        );
        assert!(
            (f(&min[1]) + f(&size[1]) / 2.0 - 1.0).abs() < 1e-3,
            "and y = 1: {first}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// File ▸ Import reads the file off the main thread and adds it as one
    /// step whose script line names the file — not the file's bytes.
    #[test]
    fn an_import_is_read_aside_and_recorded_by_its_path() {
        let dir = std::env::temp_dir().join(format!("kreate-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let path = dir.join("pair.obj");
        let mesh = kreate_mesh::primitives::cube(1.0);
        std::fs::write(
            &path,
            kreate_io::write_obj(
                &[("A".to_string(), &mesh), ("B".to_string(), &mesh)],
                Default::default(),
            ),
        )
        .expect("written");
        let mut app = app();
        let before = app.doc.objects().count();
        app.import_into(&path);
        let start = std::time::Instant::now();
        while app.doc.objects().count() == before && start.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            app.serve_imports();
        }
        assert_eq!(app.doc.objects().count(), before + 2);
        let line = app.history.last().expect("a step").to_script_call();
        assert!(line.starts_with("kreate.ops.object.import("), "{line}");
        assert!(line.len() < 400, "the line names the file: {line}");
        // And undoing it takes both away.
        assert!(app.history.undo(&mut app.doc), "undone");
        assert_eq!(app.doc.objects().count(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Several sides, or some objects alone: a bad side or name is said, and
    /// whatever happens the objects hidden for the pictures come back and a
    /// saved document stays saved.
    #[test]
    fn a_capture_of_some_objects_leaves_the_scene_as_it_was() {
        let mut app = app();
        app.doc
            .add_mesh_object("Keep", kreate_mesh::primitives::cube(1.0));
        let other = app
            .doc
            .add_mesh_object("Other", kreate_mesh::primitives::cube(1.0));
        app.file.still_saved(&app.doc);
        let ask = |app: &mut App, p: serde_json::Value| app.bridge_call("capture", &p);
        let bad = ask(&mut app, serde_json::json!({ "views": ["sideways"] }));
        assert!(bad.is_err_and(|e| e.contains("sideways")));
        let missing = ask(&mut app, serde_json::json!({ "only": ["Nobody"] }));
        assert!(missing.is_err_and(|e| e.contains("Nobody")));
        // No window here, so the picture itself fails — after the hiding.
        let shot = ask(
            &mut app,
            serde_json::json!({ "views": ["front", "top"], "only": ["Keep"] }),
        );
        assert!(shot.is_err(), "no window to draw in");
        assert!(app.doc.object(other).expect("other").visible, "shown again");
        assert!(!app.file.is_modified(&app.doc), "still saved");
    }

    #[test]
    fn the_context_says_what_is_selected_and_where_the_view_looks() {
        let mut app = app();
        let id = app
            .doc
            .add_mesh_object("Chosen", kreate_mesh::primitives::cube(1.0));
        app.doc.selection.clear();
        app.doc.selection.add(id);
        app.doc.selection.active = Some(id);
        let context = app
            .bridge_call("context", &serde_json::json!({}))
            .expect("context");
        assert_eq!(context["active"], "Chosen");
        assert_eq!(context["selected"][0], "Chosen");
        assert_eq!(context["mode"], "object");
        let looking = &context["view"]["looking"];
        let length = (0..3)
            .map(|i| looking[i].as_f64().unwrap_or(0.0).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((length - 1.0).abs() < 1e-6, "a direction: {looking}");
    }

    /// A box seen from the side against a picture of a taller one: they
    /// disagree, and where is put at the box's own depth.
    #[test]
    fn a_silhouette_is_compared_and_its_differences_placed_on_the_model() {
        let mut app = app();
        let ids: Vec<_> = app.doc.objects().map(|(id, _)| id).collect();
        for id in ids {
            app.doc.remove_object(id);
        }
        let id = app
            .doc
            .add_mesh_object("Block", kreate_mesh::primitives::cube(1.0));
        app.doc.object_mut(id).expect("block").transform.translation = DVec3::new(3.0, 0.0, 0.5);
        let dir = std::env::temp_dir().join(format!("kreate-outline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let path = dir.join("tall.png");
        // A shape twice as tall as wide, on white: a block with a tower.
        let mut img = image::RgbImage::from_pixel(100, 200, image::Rgb([255, 255, 255]));
        for y in 20..180 {
            for x in 30..70 {
                img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        img.save(&path).expect("saved");
        let answer = app
            .bridge_call(
                "silhouette",
                &serde_json::json!({ "reference": path.display().to_string(), "view": "right" }),
            )
            .expect("compared");
        // A square against a shape four times as tall as wide, at one height:
        // a quarter of it shared.
        let agreement = answer["agreement"].as_f64().expect("a number");
        assert!((agreement - 0.25).abs() < 0.03, "{agreement}");
        let first = &answer["differences"][0];
        assert!(
            (first["at"][0].as_f64().expect("x") - 3.0).abs() < 0.01,
            "on the box's middle across X: {first}"
        );
        assert!(answer["png_base64"].as_str().is_some_and(|p| !p.is_empty()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn making_a_model_wants_words_or_a_picture_that_is_there() {
        let mut app = app();
        let none = app.bridge_call("generate", &serde_json::json!({}));
        assert!(none.is_err_and(|e| e.contains("prompt")));
        let missing = app.bridge_call(
            "generate",
            &serde_json::json!({ "image": "/nowhere/at/all.png" }),
        );
        assert!(missing.is_err_and(|e| e.contains("no picture")));
    }

    /// The connector's clock: read, moved, and a take switched on with its
    /// timing kept.
    #[test]
    fn the_timeline_tool_reads_moves_and_switches_takes() {
        use kreate_scene::animation::{Action, Interpolation};
        let mut app = app();
        let id = app
            .doc
            .add_mesh_object("Walker", kreate_mesh::primitives::cube(1.0));
        let mut walk = Action::new("Walk");
        walk.curve_mut("location", 0)
            .insert(1.0, 0.0, Interpolation::Linear);
        walk.curve_mut("location", 0)
            .insert(25.0, 4.0, Interpolation::Linear);
        let walk = app.doc.add_action(walk);
        let run = app.doc.add_action(Action::new("Run"));
        let mut strip = kreate_scene::nla::NlaStrip::new("Run", run, 40.0, (1.0, 13.0));
        strip.mute = true;
        strip.scale = 2.0;
        {
            let o = app.doc.object_mut(id).expect("walker");
            o.action = Some(walk);
            o.nla.push(strip);
        }
        let read = app
            .bridge_call("timeline", &serde_json::json!({}))
            .expect("read");
        let walker = &read["animated"][0];
        assert_eq!(walker["object"], "Walker");
        assert_eq!(walker["action"]["name"], "Walk");
        assert_eq!(walker["strips"][0]["muted"], true);

        let moved = app
            .bridge_call("timeline", &serde_json::json!({ "frame": 13.0 }))
            .expect("moved");
        assert_eq!(moved["frame"], 13.0);
        let x = app.doc.object(id).expect("walker").transform.translation.x;
        assert!((x - 2.0).abs() < 1e-6, "halfway along: {x}");

        app.bridge_call(
            "timeline",
            &serde_json::json!({ "object": "Walker", "strip": 0, "mute": false }),
        )
        .expect("switched");
        let strip = &app.doc.object(id).expect("walker").nla[0];
        assert!(!strip.mute, "switched on");
        assert_eq!((strip.start, strip.scale), (40.0, 2.0), "its timing kept");
    }

    /// One answer that added two shapes: said under it, and taken back in
    /// one step — but not over an edit the user made afterwards.
    #[test]
    fn an_answer_is_summed_up_and_undone_whole() {
        use kreate_ui::chat::ChatRole;
        let mut app = app();
        let add = |app: &mut App| {
            let mut params = PropertyMap::new();
            params.set("location", PropertyValue::Vec3([5.0, 0.0, 0.0]));
            app.run_op("object.add_primitive", params);
        };
        let before = app.doc.objects().count();
        let begin = |app: &mut App| {
            app.chat_turns.push(ChatTurn {
                start: app.history.cursor(),
                end: None,
                objects: app.doc.objects().map(|(_, o)| o.uid.raw()).collect(),
            })
        };
        begin(&mut app);
        add(&mut app);
        add(&mut app);
        app.finish_chat_turn();
        let summary = app.ui.chat_log.last().expect("a summary");
        assert_eq!(summary.role, ChatRole::Turn(0));
        assert!(summary.text.contains('2'), "{}", summary.text);
        app.revert_chat_turn(0);
        assert_eq!(app.doc.objects().count(), before, "both shapes gone");
        assert!(
            !app.ui.chat_log.iter().any(|m| m.role == ChatRole::Turn(0)),
            "the button goes once used"
        );

        // A second answer, then an edit of the user's: not undone.
        begin(&mut app);
        add(&mut app);
        app.finish_chat_turn();
        add(&mut app);
        app.revert_chat_turn(1);
        assert_eq!(app.doc.objects().count(), before + 2, "nothing taken back");
    }

    /// A model imported twice, both times made at the origin, lands in two
    /// places: the second is not hidden inside the first.
    #[test]
    fn imports_land_beside_each_other_not_inside() {
        let dir = std::env::temp_dir().join(format!("kreate-import-beside-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let path = dir.join("box.obj");
        let mesh = kreate_mesh::primitives::cube(1.0);
        std::fs::write(
            &path,
            kreate_io::write_obj(&[("Box".to_string(), &mesh)], Default::default()),
        )
        .expect("written");
        let mut app = app();
        for _ in 0..2 {
            let before = app.doc.objects().count();
            app.import_into(&path);
            let start = std::time::Instant::now();
            while app.doc.objects().count() == before && start.elapsed().as_secs() < 10 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                app.serve_imports();
            }
            app.evaluate_scene();
        }
        let boxes = app.mesh_boxes();
        assert!(boxes.len() >= 2, "{} meshes", boxes.len());
        for (i, (a, b)) in boxes.iter().enumerate() {
            for (c, d) in &boxes[i + 1..] {
                assert!(
                    !(a.cmplt(*d).all() && b.cmpgt(*c).all()),
                    "{a}..{b} overlaps {c}..{d}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An empty scene, changed or not, is left without the save question:
    /// there is nothing in it to keep.
    #[test]
    fn an_empty_scene_is_left_without_asking() {
        let mut app = app();
        let cube = app
            .doc
            .add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));
        app.doc.remove_object(cube);
        assert!(app.file.is_modified(&app.doc));
        assert_eq!(app.doc.object_count(), 0);
        // Would block on a dialog if it asked.
        assert!(app.confirm_discarding_changes("unsaved-what-quit"));
        assert!(app.confirm_discarding_changes("unsaved-what-open"));
    }

    /// While the save question is on screen, a second request to leave —
    /// another click on the window's close button — is not a second
    /// question: it is refused at once, without a dialog.
    #[test]
    fn a_second_close_while_asking_does_not_ask_again() {
        let mut app = app();
        app.doc
            .add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));
        assert!(app.file.is_modified(&app.doc));
        app.asking = true;
        // Would block on a dialog if it asked; returns at once instead.
        assert!(!app.confirm_discarding_changes("unsaved-what-quit"));
    }

    /// A new document starts stopped at its first frame: the last one's
    /// playback does not run on into it, nor its clock.
    #[test]
    fn a_new_document_starts_stopped_at_its_first_frame() {
        let mut app = app();
        app.set_frame(120.0);
        app.playing = Some(std::time::Instant::now());
        app.new_document();
        assert!(app.playing.is_none(), "still playing");
        let start = app.doc.playback.start as f64;
        assert_eq!(app.doc.playback.frame, start);
        assert_eq!(app.time.frame(), start);
    }

    /// File ▸ Open shows a document from where it was looked at when it was
    /// saved — not framed afresh, not the last document's view.
    #[test]
    fn an_opened_document_is_seen_from_its_saved_view() {
        let dir = std::env::temp_dir().join(format!("kreate-open-view-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let path = dir.join("looked.kreate");
        let saved = Camera {
            pivot: DVec3::new(2.0, 1.0, 0.0),
            distance: 6.5,
            projection: kreate_render::Projection::Orthographic,
            ..Camera::default()
        };
        let mut file = DocumentFile::new();
        file.set_view(&saved);
        file.save(&path, &load_document(None)).expect("saved");

        let mut app = app();
        app.camera.distance = 40.0;
        app.load_from(&path);
        assert_eq!(app.camera.pivot, saved.pivot);
        assert_eq!(app.camera.distance, 6.5);
        assert_eq!(
            app.camera.projection,
            kreate_render::Projection::Orthographic
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// File ▸ New shows the view a fresh start shows: the default angle,
    /// lens and projection, out of any scene camera — not the last
    /// document's view.
    #[test]
    fn a_new_document_is_seen_from_the_default_view() {
        let mut app = app();
        let fresh = kreate::startup::starting_camera();
        assert_eq!(fresh.projection, kreate_render::Projection::Orthographic);
        app.camera.rotation = glam::DQuat::from_rotation_z(1.2) * app.camera.rotation;
        app.camera.fov_y = 0.3;
        app.camera.projection = kreate_render::Projection::Perspective;
        app.camera_view = Some(app.camera);
        app.new_document();
        assert!(app.camera.rotation.abs_diff_eq(fresh.rotation, 1e-9));
        assert_eq!(app.camera.fov_y, fresh.fov_y);
        assert_eq!(app.camera.projection, fresh.projection);
        assert!(app.camera_view.is_none());
    }

    /// The Lasso tool's drag selects with its path; without one, a box.
    #[test]
    fn a_lasso_drag_selects_with_its_path() {
        let mut drag = DragSelect {
            origin: (0.0, 0.0),
            current: (40.0, 30.0),
            moved: true,
            extend: false,
            deselect: false,
            lasso: Some(vec![(0.0, 0.0), (40.0, 0.0), (40.0, 30.0)]),
        };
        assert!(matches!(drag.shape(), kreate_render::SelectShape::Lasso(p) if p.len() == 3));
        drag.lasso = None;
        assert!(matches!(
            drag.shape(),
            kreate_render::SelectShape::Box { .. }
        ));
    }

    /// Blender's Ctrl+P and Alt+P menus: plain parenting first, the armature
    /// binds after; clearing the parent, and the armature.
    #[test]
    fn the_parent_menus_offer_object_and_armature() {
        let mut app = app();
        let ops = |app: &App| -> Vec<String> {
            app.ui
                .context_menu
                .groups
                .iter()
                .flat_map(|g| g.entries.iter().map(|e| e.op_id.clone()))
                .collect()
        };
        app.open_popup_menu("parent");
        assert_eq!(
            ops(&app),
            [
                "object.parent",
                "object.bind_armature",
                "object.bind_armature"
            ]
        );
        app.open_popup_menu("parent_clear");
        assert_eq!(ops(&app), ["object.parent", "object.unbind_armature"]);
    }

    /// A press on a list or menu open over the scene is the list's: the
    /// mode list's entries sit over the viewport, and their click must not
    /// also reach the scene.
    #[test]
    fn a_press_on_a_menu_over_the_scene_is_the_menus() {
        let mut app = app();
        app.viewport_px = Some((0.0, 0.0, 800.0, 600.0));
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..Default::default()
        };
        let ctx = app.egui_ctx.clone();
        for _ in 0..2 {
            let _ = ctx.run_ui(input.clone(), |ui| {
                egui::CentralPanel::default().show(ui, |_| {});
                egui::Area::new(egui::Id::new("a menu"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(egui::pos2(100.0, 100.0))
                    .show(ui.ctx(), |ui| {
                        ui.set_min_size(egui::vec2(120.0, 80.0));
                    });
            });
        }
        let ppp = ctx.pixels_per_point() as f64;
        app.input.cursor = Some((150.0 * ppp, 130.0 * ppp));
        assert!(!app.press_belongs_to_viewport(), "on the menu");
        app.input.cursor = Some((500.0 * ppp, 400.0 * ppp));
        assert!(app.press_belongs_to_viewport(), "on the scene beside it");
    }

    /// A script stops at its first failing line, says why and what would
    /// fix it, and runs nothing after it.
    #[test]
    fn a_script_stops_at_its_first_failure_and_says_why() {
        let mut app = app();
        let report = app.run_script_reporting(
            "kreate.ops.mesh.select_nothing()
kreate.ops.object.add_primitive()",
        );
        assert_eq!(report.len(), 2);
        let why = report[0].result.as_ref().unwrap_err();
        assert!(why.contains("edit mode"), "{why}");
        assert!(why.contains("mode_set"), "the fix is named: {why}");
        assert!(report[1].result.as_ref().unwrap_err().contains("skipped"));
        assert_eq!(app.doc.objects().count(), 0, "the second line did not run");
    }

    /// A sketch read back by the indices its operators take: a rectangle is
    /// four points, four lines and one region of its area.
    #[test]
    fn a_sketch_is_read_out_with_its_points_lines_and_regions() {
        let mut app = app();
        app.run_script_reporting(
            "kreate.ops.sketch.new()
kreate.ops.sketch.rectangle(a=(0, 0, 0), b=(0.02, 0.01, 0))",
        );
        let out = app.bridge_call("sketch", &serde_json::json!({})).unwrap();
        // The sketch's origin point, and the rectangle's four.
        assert_eq!(out["points"].as_array().unwrap().len(), 5);
        assert_eq!(out["lines"].as_array().unwrap().len(), 4);
        let regions = out["regions"].as_array().unwrap();
        assert_eq!(regions.len(), 1);
        assert!((regions[0]["area"].as_f64().unwrap() - 0.0002).abs() < 1e-7);
        assert_eq!(out["plane"]["normal"][2], 1.0);
        let err = app
            .bridge_call("sketch", &serde_json::json!({ "object": "Nobody" }))
            .unwrap_err();
        assert!(err.contains("Nobody"));
    }

    /// Two cubes measured: each its volume and mass, and between them the
    /// gap their surfaces leave — none where they touch.
    #[test]
    fn two_objects_are_measured_and_the_gap_between_them() {
        let mut app = app();
        app.run_script_reporting(
            "kreate.ops.object.add_primitive()
kreate.ops.object.add_primitive(location=(2.5, 0, 0))",
        );
        let out = app
            .bridge_call(
                "measure",
                &serde_json::json!({ "objects": ["Cube", "Cube.001"], "density": 1000.0 }),
            )
            .unwrap();
        let a = &out["objects"][0];
        assert_eq!(a["closed"], true);
        assert!((a["volume_m3"].as_f64().unwrap() - 8.0).abs() < 1e-6);
        assert!((a["mass_kg"].as_f64().unwrap() - 8000.0).abs() < 1e-3);
        let between = &out["between"];
        assert!(
            (between["surface_gap_m"].as_f64().unwrap() - 0.5).abs() < 1e-5,
            "{between}"
        );
        assert_eq!(between["overlapping"], false);
        assert!((between["box_gap"][0].as_f64().unwrap() - 0.5).abs() < 1e-9);
        // Pushed together: no gap, and said to overlap.
        app.run_script_reporting(
            "kreate.ops.object.set_transform(object=\"Cube.001\", location=(1.5, 0, 0))",
        );
        let out = app
            .bridge_call(
                "measure",
                &serde_json::json!({ "objects": ["Cube", "Cube.001"] }),
            )
            .unwrap();
        assert_eq!(out["between"]["overlapping"], true);
        assert_eq!(out["between"]["surface_gap_m"], 0.0);
        assert!((out["density_kg_m3"].as_f64().unwrap() - app.doc.design.density).abs() < 1e-9);
    }

    /// The history read out: the steps as the lines they were, and what
    /// was undone.
    #[test]
    fn the_history_is_read_out_as_its_lines() {
        let mut app = app();
        app.run_script_reporting(
            "kreate.ops.object.add_primitive()
kreate.ops.object.set_transform(location=(1, 0, 0))",
        );
        app.undo();
        let out = app
            .bridge_call("history", &serde_json::json!({ "limit": 10 }))
            .unwrap();
        assert_eq!(out["steps"], 2);
        assert_eq!(out["cursor"], 1);
        assert_eq!(out["can_redo"], true);
        assert!(out["done"][0]["line"]
            .as_str()
            .unwrap()
            .contains("add_primitive"));
        assert!(out["undone"][0]["line"]
            .as_str()
            .unwrap()
            .contains("set_transform"));
    }

    /// The scene listed in part: by name, by prefix, by selection, a few,
    /// and in short past a size — each said with how many there are.
    #[test]
    fn the_scene_is_listed_in_part_when_asked_and_in_short_when_big() {
        let mut app = app();
        for _ in 0..3 {
            app.run_op("object.add_primitive", PropertyMap::new());
        }
        let scene =
            |app: &mut App, params: serde_json::Value| app.bridge_call("scene", &params).unwrap();
        let all = scene(&mut app, serde_json::json!({}));
        assert_eq!(all["object_count"], 3);
        assert_eq!(all["objects"].as_array().unwrap().len(), 3);
        assert!(all["objects"][0].get("rotation").is_some(), "full detail");
        assert!(all["note"].is_null());
        let some = scene(&mut app, serde_json::json!({ "prefix": "Cube.0" }));
        assert_eq!(some["objects"].as_array().unwrap().len(), 2);
        assert_eq!(some["object_count"], 3);
        let named = scene(&mut app, serde_json::json!({ "objects": ["Cube.002"] }));
        assert_eq!(named["objects"][0]["name"], "Cube.002");
        let chosen = scene(&mut app, serde_json::json!({ "selected": true }));
        assert_eq!(chosen["objects"].as_array().unwrap().len(), 1);
        let few = scene(&mut app, serde_json::json!({ "limit": 1 }));
        assert_eq!(few["objects_listed"], 1);
        assert!(few["note"].as_str().unwrap().contains("1 of 3"));
        let short = scene(&mut app, serde_json::json!({ "compact": true }));
        assert!(short["objects"][0].get("rotation").is_none());
        assert!(short["objects"][0].get("dimensions").is_some());
        // Past the size, short by itself.
        for _ in 0..connector::SceneFilter::COMPACT_FROM {
            app.run_op("object.add_empty", PropertyMap::new());
        }
        let big = scene(&mut app, serde_json::json!({}));
        assert!(
            big["objects"][0].get("rotation").is_none(),
            "{}",
            big["note"]
        );
        assert!(big["note"].as_str().unwrap().contains("in short"));
    }

    /// Overlaps asked about some objects are tested among those alone.
    #[test]
    fn overlaps_are_checked_among_the_objects_named() {
        let mut app = app();
        app.run_script_reporting(
            "kreate.ops.object.add_primitive()
kreate.ops.object.add_primitive(location=(1, 0, 0))
kreate.ops.object.add_primitive(location=(0, 1, 0))",
        );
        let all = app.bridge_call("overlaps", &serde_json::json!({})).unwrap();
        assert_eq!(all["overlapping"].as_array().unwrap().len(), 3);
        assert_eq!(all["pairs_tested"], 3);
        let two = app
            .bridge_call(
                "overlaps",
                &serde_json::json!({ "objects": ["Cube", "Cube.002"] }),
            )
            .unwrap();
        assert_eq!(two["overlapping"].as_array().unwrap().len(), 1);
        assert_eq!(two["checked"], 2);
        let err = app
            .bridge_call("overlaps", &serde_json::json!({ "objects": ["Nobody"] }))
            .unwrap_err();
        assert!(err.contains("Nobody"));
    }

    /// A line names its object with `object=` and is run on it; a mesh line
    /// out of edit mode enters it and an object line in edit mode leaves it,
    /// each said with the result.
    #[test]
    fn a_line_selects_its_object_and_enters_the_mode_it_needs() {
        let mut app = app();
        let report = app.run_script_reporting(
            "kreate.ops.object.add_primitive()
kreate.ops.object.add_primitive(location=(3, 0, 0))
kreate.ops.mesh.select(action=\"all\", object=\"Cube\")
kreate.ops.mesh.translate(translation=(0, 0, 1))
kreate.ops.object.set_transform(object=\"Cube.001\", location=(3, 0, 2))",
        );
        for line in &report {
            assert!(line.result.is_ok(), "{}: {:?}", line.text, line.result);
        }
        let third = report[2].result.as_ref().unwrap();
        assert!(
            third.contains("Cube selected") && third.contains("entered edit mode"),
            "{third}"
        );
        assert!(
            !report[3].result.as_ref().unwrap().contains("mode"),
            "already there"
        );
        let fifth = report[4].result.as_ref().unwrap();
        assert!(
            fifth.contains("left edit mode") && fifth.contains("Cube.001 selected"),
            "{fifth}"
        );
        assert_eq!(app.op_ctx.mode, OpMode::RESTING);
        let cube = app.doc.objects().find(|(_, o)| o.name == "Cube").unwrap().1;
        let mesh = app.doc.mesh(cube.mesh().unwrap()).unwrap();
        assert!(
            mesh.positions().iter().all(|p| p.z >= -1e-6),
            "the two-unit cube was moved up by one: {:?}",
            mesh.positions()
        );
        let other = app
            .doc
            .objects()
            .find(|(_, o)| o.name == "Cube.001")
            .unwrap()
            .1;
        assert_eq!(other.transform.translation.z, 2.0);
        // A name nobody has is an error that names it.
        let report = app.run_script_reporting("kreate.ops.object.delete(object=\"Nobody\")");
        assert!(report[0].result.as_ref().unwrap_err().contains("Nobody"));
    }

    /// A line that changed nothing is answered as cancelled with why, and
    /// the lines after it still run: a script is not stopped by a no-op.
    #[test]
    fn the_connector_says_why_a_line_changed_nothing() {
        let mut app = app();
        let out = app
            .bridge_call(
                "run",
                &serde_json::json!({ "script": "kreate.ops.object.add_primitive()
kreate.ops.object.set_transform(location=(0, 0, 0))
kreate.ops.object.add_primitive()" }),
            )
            .unwrap();
        let lines = out["lines"].as_array().unwrap();
        assert_eq!(lines[1]["result"], "cancelled");
        assert_eq!(lines[1]["ok"], true);
        let warning = lines[1]["warning"].as_str().unwrap();
        assert!(warning.contains("already"), "{warning}");
        assert_eq!(lines[2]["result"], "finished");
        assert_eq!(app.doc.objects().count(), 2);
        assert!(out.get("warnings").is_none(), "no boolean gave up");
    }

    /// Entering edit mode with a light active and a mesh selected edits the
    /// mesh: that is what was meant.
    #[test]
    fn edit_mode_finds_the_selected_mesh_when_the_active_object_is_not_one() {
        let mut app = app();
        app.run_op("object.add_primitive", PropertyMap::new());
        let cube = app.doc.selection.active.unwrap();
        app.run_op("object.add_light", PropertyMap::new());
        let light = app.doc.selection.active.unwrap();
        app.doc.selection.add(cube);
        app.doc.selection.active = Some(light);
        let report = app.run_script_reporting("kreate.ops.object.mode_set(mode=\"edit\")");
        assert!(report[0].result.is_ok(), "{:?}", report[0].result);
        assert!(app.op_ctx.mode.edits_elements());
        assert_eq!(app.doc.selection.active, Some(cube));
    }

    /// STEP, the parts list and a drawing, all in the scene's unit; two
    /// objects sharing a mesh are one part twice in the list.
    #[test]
    fn step_parts_list_and_drawing_are_written_in_the_scenes_unit() {
        let mut app = app();
        app.run_op("object.add_primitive", PropertyMap::new());
        app.run_op("object.duplicate", {
            let mut p = PropertyMap::new();
            p.set("linked", PropertyValue::Bool(true));
            p
        });
        let mut p = PropertyMap::new();
        p.set("length", PropertyValue::Enum(2));
        app.run_op("scene.units_set", p);
        app.evaluate_scene();
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let read = |name: &str| {
            let path = dir.join(format!("kreate-test-{pid}-{name}"));
            let text = std::fs::read_to_string(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            text
        };
        app.export_to(&dir.join(format!("kreate-test-{pid}-a.step")))
            .unwrap();
        let step = read("a.step");
        assert!(step.contains("SI_UNIT(.MILLI.,.METRE.)"));
        assert_eq!(step.matches("FACETED_BREP(").count(), 2);
        app.export_to(&dir.join(format!("kreate-test-{pid}-b.csv")))
            .unwrap();
        let csv = read("b.csv");
        let row = csv.lines().nth(1).expect("one part");
        assert!(row.contains(",2,,2000.0000,2000.0000,2000.0000,"), "{csv}");
        assert_eq!(csv.lines().count(), 2, "{csv}");
        app.write_drawing(
            &dir.join(format!("kreate-test-{pid}-c.dxf")),
            kreate_io::drawing::View::Top,
        )
        .unwrap();
        assert!(read("c.dxf").contains(
            "$INSUNITS
70
4
"
        ));
    }

    /// File > Export to .3mf writes each shown part where it stands, in the
    /// scene's unit — a 2 m cube in a millimetre scene is 2000 across.
    #[test]
    fn a_3mf_export_is_in_the_scenes_unit() {
        let mut app = app();
        app.run_op("object.add_primitive", PropertyMap::new());
        let mut p = PropertyMap::new();
        p.set("length", PropertyValue::Enum(2));
        app.run_op("scene.units_set", p);
        app.evaluate_scene();
        let path = std::env::temp_dir().join(format!("kreate-test-{}.3mf", std::process::id()));
        app.export_to(&path).expect("written");
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let mut model = String::new();
        std::io::Read::read_to_string(&mut zip.by_name("3D/3dmodel.model").unwrap(), &mut model)
            .unwrap();
        assert!(model.contains("unit=\"millimeter\""));
        assert!(
            model.contains("x=\"1000\"") && model.contains("x=\"-1000\""),
            "{model}"
        );
    }

    /// The section view cuts what is drawn and picked, half a cube for a
    /// plane through it, while the Object tab still weighs the whole part.
    #[test]
    fn a_section_cuts_the_view_but_not_the_part() {
        let mut app = app();
        app.run_op("object.add_primitive", PropertyMap::new());
        let mut p = PropertyMap::new();
        p.set("enabled", PropertyValue::Bool(true));
        p.set("axis", PropertyValue::Enum(0));
        app.run_op("scene.section_set", p);
        app.evaluate_scene();
        let (_, shown) = &app.evaluated[0];
        assert!(shown.bounds().max.x < 1e-3, "{:?}", shown.bounds());
        app.publish_mass();
        assert!((app.ui.mass.unwrap().volume - 8.0).abs() < 1e-9);
        // Off again, and the whole is drawn.
        let mut p = PropertyMap::new();
        p.set("enabled", PropertyValue::Bool(false));
        app.run_op("scene.section_set", p);
        app.evaluate_scene();
        assert!(app.evaluated[0].1.bounds().max.x > 0.9);
    }

    /// Mesh Check finds a plane's open rim, and Select takes the user into
    /// edit mode with exactly those edges selected.
    #[test]
    fn mesh_check_counts_and_then_selects_the_fault() {
        let mut app = app();
        let mut p = PropertyMap::new();
        p.set("shape", PropertyValue::Enum(0));
        app.run_op("object.add_primitive", p);
        app.run_mesh_check(0.001);
        let (_, counts) = app.ui.mesh_check.expect("checked");
        assert_eq!(counts[0], 4, "{counts:?}");
        let mut p = PropertyMap::new();
        p.set("issue", PropertyValue::Enum(0));
        app.dispatch(&Action {
            op_id: "app.mesh_check_select".into(),
            params: p,
        });
        assert!(app.op_ctx.mode.edits_elements());
        let id = app.doc.selection.active.unwrap();
        let mesh = app
            .doc
            .mesh(app.doc.object(id).unwrap().mesh().unwrap())
            .unwrap();
        let edges = kreate_mesh::selection::selected(mesh, kreate_mesh::attribute::Domain::Edge);
        assert_eq!(edges.len(), 4);
    }

    /// The Object tab's physical properties are the active object's as
    /// evaluated, in the world: a 2 m cube scaled by half in Z is 4 m³.
    #[test]
    fn the_active_object_is_measured_as_it_stands() {
        let mut app = app();
        app.run_op("object.add_primitive", PropertyMap::new());
        let id = app.doc.selection.active.unwrap();
        app.doc.object_mut(id).unwrap().transform.scale = DVec3::new(1.0, 1.0, 0.5);
        app.evaluate_scene();
        app.publish_mass();
        let mass = app.ui.mass.expect("a mesh is active");
        assert!((mass.volume - 4.0).abs() < 1e-9, "{}", mass.volume);
        assert!(mass.closed);
    }

    /// A work plane on a face: the cursor turns to it, a new shape is set
    /// square to it while the Cursor orientation is chosen, and not otherwise.
    #[test]
    fn a_work_plane_turns_the_cursor_and_squares_new_shapes() {
        let wall = DVec3::new(1.0, 0.0, 0.0);
        let q = work_plane_turn(wall);
        assert!((q * DVec3::Z - wall).length() < 1e-12, "Z is the normal");
        assert!((q * DVec3::X).z.abs() < 1e-12, "X is level");
        let up = work_plane_turn(DVec3::Z);
        assert!(
            up.abs_diff_eq(kreate_math::DQuat::IDENTITY, 1e-12),
            "the ground is no turn"
        );

        let mut app = app();
        app.place_cursor_turned(DVec3::new(0.0, 0.0, 1.0), q);
        assert!(app.doc.cursor_rotation.abs_diff_eq(q, 1e-9));
        app.ui.tools.orientation = 2;
        app.run_op("object.add_primitive", PropertyMap::new());
        let added = app.doc.selection.active.unwrap();
        let turn = app.doc.object(added).unwrap().transform.rotation;
        assert!(
            (turn * DVec3::Z - wall).length() < 1e-9,
            "square to the plane"
        );
        app.ui.tools.orientation = 0;
        app.run_op("object.add_primitive", PropertyMap::new());
        let added = app.doc.selection.active.unwrap();
        let turn = app.doc.object(added).unwrap().transform.rotation;
        assert!(turn.abs_diff_eq(kreate_math::DQuat::IDENTITY, 1e-12));
    }

    #[test]
    #[ignore]
    fn sculpt_frame_cost() {
        use std::time::Instant;
        let mut app = app();
        let mesh = app
            .doc
            .add_mesh(kreate_mesh::primitives::uv_sphere(1.0, 512, 256));
        let id = app.doc.add_object(kreate_scene::document::Object::new(
            "Ball",
            kreate_scene::document::ObjectData::Mesh(mesh),
        ));
        app.doc.selection.add(id);
        app.set_mode(OpMode::from_tag("sculpt").unwrap());
        app.evaluate_scene();
        // The tree is built once, when the stroke starts; what is measured is
        // the frames after.
        let _ = app.pick.raycast(
            &app.doc,
            &app.evaluated,
            DVec3::new(0.0, -5.0, 0.0),
            DVec3::Y,
            f64::INFINITY,
        );
        let mut dabs: Vec<f64> = Vec::new();
        let mut session: Option<kreate_ops::ModalSession> = None;
        let (mut op, mut eval, mut pick) = (0.0, 0.0, 0.0);
        let frames = 60;
        for i in 0..frames {
            let a = i as f64 * 0.02;
            dabs.extend([a.cos(), a.sin(), 0.0, 0.1]);
            let mut params = PropertyMap::new();
            params.set("brush", PropertyValue::Enum(0));
            params.set("dabs", PropertyValue::Floats(dabs.clone()));
            let t = Instant::now();
            match session.as_mut() {
                Some(s) => {
                    app.history
                        .update_modal(&app.registry, &mut app.doc, None, &app.op_ctx, s, &params)
                        .unwrap();
                }
                None => {
                    session = app
                        .history
                        .begin_modal(
                            &app.registry,
                            &mut app.doc,
                            None,
                            &app.op_ctx,
                            "sculpt.stroke",
                            &params,
                        )
                        .unwrap();
                }
            }
            op += t.elapsed().as_secs_f64();
            let t = Instant::now();
            app.evaluate_scene();
            eval += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let _ = app.pick.raycast(
                &app.doc,
                &app.evaluated,
                DVec3::new(0.0, -5.0, 0.0),
                DVec3::Y,
                f64::INFINITY,
            );
            pick += t.elapsed().as_secs_f64();
        }
        let ms = |s: f64| s * 1000.0 / frames as f64;
        eprintln!(
            "per frame: operator {:.2} ms, evaluate {:.2} ms, pick {:.2} ms",
            ms(op),
            ms(eval),
            ms(pick)
        );
        let t = Instant::now();
        app.history.commit_modal(session.unwrap());
        eprintln!("commit {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    /// F sizes the brush by the pointer's distance from a ring that stays
    /// put, Shift+F its strength the same way; Esc puts either back.
    #[test]
    fn f_sizes_the_brush_and_shift_f_weights_it() {
        let mut app = app();
        let mesh = app.doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let cube = app.doc.add_object(kreate_scene::document::Object::new(
            "Box",
            kreate_scene::document::ObjectData::Mesh(mesh),
        ));
        app.doc.selection.add(cube);
        app.set_mode(OpMode::from_tag("sculpt").unwrap());
        let ppp = app.egui_ctx.pixels_per_point() as f64;
        let at = |x: f64| Some((x * ppp, 300.0 * ppp));
        app.ui.sculpt.radius = 50.0;
        app.ui.sculpt.strength = 0.5;

        app.input.cursor = at(400.0);
        app.begin_brush_adjust(false);
        app.input.cursor = at(430.0);
        app.update_brush_adjust();
        assert!(
            (app.ui.sculpt.radius - 80.0).abs() < 1e-3,
            "{}",
            app.ui.sculpt.radius
        );
        app.end_brush_adjust(true);
        assert!(app.ui.brush_adjust.is_none());
        assert!((app.ui.sculpt.radius - 80.0).abs() < 1e-3, "kept");

        // The middle is 0.5 x 80 to the left of 430, at 390.
        app.begin_brush_adjust(true);
        app.input.cursor = at(420.0);
        app.update_brush_adjust();
        assert!(
            (app.ui.sculpt.strength - 0.375).abs() < 1e-3,
            "{}",
            app.ui.sculpt.strength
        );
        app.end_brush_adjust(false);
        assert!(
            (app.ui.sculpt.strength - 0.5).abs() < 1e-6,
            "Esc puts it back"
        );
    }

    /// A bake on its own thread: the bar moves, the result goes in as one
    /// undo step through `physics.bake`, and a cancelled one leaves nothing.
    /// A rectangle drawn on a cube's top: the Extrude tool takes it without
    /// a click, and the dialog joins it onto the cube. The sketch panel's
    /// Extrude button is this tool; it made an empty solid of a drawn sketch.
    #[test]
    fn extruding_a_rectangle_drawn_on_a_face_joins_it_without_a_pick() {
        let mut app = app();
        let part = app
            .doc
            .add_mesh_object("Part", kreate_mesh::primitives::cube(0.04));
        app.doc.selection.add(part);
        app.doc.selection.active = Some(part);
        app.run_op_with(
            "sketch.new",
            &[
                ("plane", PropertyValue::Enum(4)),
                ("at", PropertyValue::Vec3([0.0, 0.0, 0.02])),
            ],
        );
        app.run_op_with(
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([-0.01, -0.01, 0.0])),
                ("b", PropertyValue::Vec3([0.01, 0.01, 0.0])),
            ],
        );
        let sketch = kreate_ops::ops::sketch_draw::active_sketch(&app.doc).unwrap();
        app.evaluate_scene();
        let mut p = PropertyMap::new();
        p.set("tool", PropertyValue::Text("extrude".into()));
        app.dispatch(&Action {
            op_id: "app.sketch_tool".into(),
            params: p,
        });
        assert_eq!(app.sketching.tool, kreate::sketching::Tool::Extrude);
        assert_eq!(app.sketching.seeds.len(), 1, "the rectangle, not the face");
        app.ui.length_unit = 0.001;
        app.ui.solid_dialog.distance = 5.0;
        app.choose_solid_operation(sketch);
        assert_eq!(app.ui.solid_dialog.operation, 1, "a join");
        assert_eq!(app.ui.solid_dialog.target, "Part");
    }

    #[test]
    fn an_extrude_into_an_object_cuts_it_and_out_of_one_joins_it() {
        let mut app = app();
        // A box from z = 0 to 2, and a sketch on the ground under it.
        let mut mesh = kreate_mesh::primitives::cube(2.0);
        for p in mesh.positions_mut() {
            p.z += 1.0;
        }
        app.doc.add_mesh_object("Box", mesh);
        app.run_op_with("sketch.new", &[]);
        let sketch = kreate_ops::ops::sketch_draw::active_sketch(&app.doc).unwrap();
        app.evaluate_scene();
        app.sketching.tool = kreate::sketching::Tool::Extrude;
        app.sketching.seeds = vec![glam::DVec2::new(0.0, 0.0)];
        app.ui.length_unit = 1.0;
        // Up, into the box: a cut from it.
        app.ui.solid_dialog.distance = 0.5;
        app.choose_solid_operation(sketch);
        assert_eq!(app.ui.solid_dialog.operation, 2);
        assert_eq!(app.ui.solid_dialog.target, "Box");
        // Down, away from it with the box behind: a join.
        app.ui.solid_dialog.distance = -0.5;
        app.choose_solid_operation(sketch);
        assert_eq!(app.ui.solid_dialog.operation, 1);
        // Picked by hand, it stays.
        app.ui.solid_dialog.chosen = true;
        app.ui.solid_dialog.operation = 0;
        app.ui.solid_dialog.distance = 0.5;
        app.choose_solid_operation(sketch);
        assert_eq!(app.ui.solid_dialog.operation, 0);
        // In the open: a new body.
        app.ui.solid_dialog.chosen = false;
        app.sketching.seeds = vec![glam::DVec2::new(5.0, 5.0)];
        app.choose_solid_operation(sketch);
        assert_eq!(app.ui.solid_dialog.operation, 0);
    }

    #[test]
    fn drawing_catches_another_sketchs_ends_and_centres_not_its_circles_points() {
        let mut app = app();
        app.run_op_with("sketch.new", &[]);
        app.run_op_with(
            "sketch.line",
            &[("points", PropertyValue::Floats(vec![0.01, 0.0, 0.02, 0.0]))],
        );
        app.run_op_with(
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.05, 0.0, 0.0])),
                ("radius", PropertyValue::Float(0.01)),
            ],
        );
        app.run_op_with("sketch.new", &[("plane", PropertyValue::Enum(1))]);
        let second = kreate_ops::ops::sketch_draw::active_sketch(&app.doc).unwrap();
        let mut caught: Vec<(i64, i64)> = kreate::sketching::other_points(&app.doc, Some(second))
            .iter()
            .map(|p| ((p.x * 1e4).round() as i64, (p.y * 1e4).round() as i64))
            .collect();
        caught.sort();
        assert_eq!(caught, vec![(100, 0), (200, 0), (500, 0)]);
    }

    /// A sketch is edited again: Tab on it goes into the sketch tools, not
    /// its vertices, and Tab again comes out.
    #[test]
    fn tab_on_a_sketch_goes_back_into_it() {
        let mut app = app();
        app.run_op_with("sketch.new", &[]);
        app.run_op_with(
            "sketch.line",
            &[(
                "points",
                PropertyValue::Floats(vec![0.01, 0.01, 0.02, 0.01]),
            )],
        );
        app.set_mode(OpMode::Object);
        assert_eq!(app.op_ctx.mode, OpMode::Object);
        app.toggle_edit_mode();
        assert_eq!(app.op_ctx.mode, OpMode::Sketch);
        app.toggle_edit_mode();
        assert_eq!(app.op_ctx.mode, OpMode::Object);
    }

    #[test]
    fn the_pattern_tool_keeps_the_selection_and_shows_its_copies() {
        let mut app = app();
        app.run_op_with("sketch.new", &[]);
        app.run_op_with(
            "sketch.line",
            &[(
                "points",
                PropertyValue::Floats(vec![0.01, 0.01, 0.02, 0.01]),
            )],
        );
        let id = kreate_ops::ops::sketch_draw::active_sketch(&app.doc).unwrap();
        app.set_mode(OpMode::Sketch);
        app.sketching.lines = vec![0];
        app.dispatch(
            &Action::new("app.sketch_tool").with("tool", PropertyValue::Text("pattern".into())),
        );
        assert_eq!(app.sketching.tool, kreate::sketching::Tool::Pattern);
        assert_eq!(
            app.sketching.lines,
            vec![0],
            "what was selected is what repeats"
        );
        app.ui.pattern_dialog.count = 4;
        app.ui.pattern_dialog.spacing = 5.0;
        let world = app.doc.world_transform(id);
        app.ui.sketch_view = Some(Default::default());
        app.preview_sketch_pattern(id, world);
        let view = app.ui.sketch_view.as_ref().unwrap();
        assert_eq!(view.preview.len(), 3, "three copies of the one line");
    }

    /// A curve point dragged across the view: moved live with its handles,
    /// one step when let go, put back on Esc.
    #[test]
    fn a_curve_point_dragged_moves_with_its_handles_and_is_one_step() {
        let mut app = app();
        app.run_op_with(
            "object.add_curve",
            &[
                ("shape", PropertyValue::Enum(2)),
                ("size", PropertyValue::Float(1.0)),
            ],
        );
        let id = kreate_ops::ops::curve::active_curve(&app.doc).expect("a curve");
        let point = app.doc.object(id).unwrap().curve.as_ref().unwrap().splines[0].points[2];
        assert_eq!(point.position, DVec3::ZERO);
        let steps = app.history.entries().len();
        // The press: the point at the origin, the pointer's plane through it.
        app.curve_drag = Some(CurveDrag {
            object: id,
            spline: 0,
            index: 2,
            part: CurvePart::Point,
            point,
            plane: DVec3::ZERO,
            from: DVec3::ZERO,
            session: None,
        });
        // Without a window there is no ray to drive the drag by: the move
        // is applied the way `drive_curve_drag` applies it, through the
        // same operator as the drag's own step.
        let mut p = point;
        p.translate(DVec3::new(0.0, 0.0, 1.0));
        let params =
            kreate_reflect::OpParams::to_map(&kreate_ops::ops::curve::point_params(0, 2, &p));
        let session = app
            .history
            .begin_modal(
                &app.registry,
                &mut app.doc,
                None,
                &app.op_ctx,
                "curve.point_set",
                &params,
            )
            .unwrap()
            .unwrap();
        app.curve_drag.as_mut().unwrap().session = Some(session);
        let moved = app.doc.object(id).unwrap().curve.as_ref().unwrap().splines[0].points[2];
        assert_eq!(moved.position, DVec3::new(0.0, 0.0, 1.0));
        assert_eq!(moved.left, point.left + DVec3::Z, "handles came along");
        // Esc: back where it was, no step.
        assert!(app.cancel_curve_drag());
        let back = app.doc.object(id).unwrap().curve.as_ref().unwrap().splines[0].points[2];
        assert_eq!(back.position, DVec3::ZERO);
        assert_eq!(app.history.entries().len(), steps);
        assert!(!app.cancel_curve_drag(), "nothing left to cancel");
    }

    #[test]
    fn a_sketch_line_dragged_moves_live_from_where_it_began_and_is_one_step() {
        let mut app = app();
        app.run_op_with("sketch.new", &[]);
        app.run_op_with(
            "sketch.line",
            &[("points", PropertyValue::Floats(vec![1.0, 1.0, 2.0, 1.0]))],
        );
        let id = kreate_ops::ops::sketch_draw::active_sketch(&app.doc).unwrap();
        let steps = app.history.entries().len();
        app.sketching.drag = Some(kreate::sketching::SketchDrag {
            points: vec![1, 2],
            anchor: glam::DVec2::new(1.5, 1.0),
            single: None,
        });
        // Two moves: the second from where the drag began, not from the first.
        app.drive_sketch_drag(glam::DVec2::new(1.5, 2.0));
        app.drive_sketch_drag(glam::DVec2::new(1.5, 3.0));
        let p = kreate_ops::ops::sketch_draw::points(&app.doc, id);
        assert!((p[1] - glam::DVec2::new(1.0, 3.0)).length() < 1e-9, "{p:?}");
        assert!((p[2] - glam::DVec2::new(2.0, 3.0)).length() < 1e-9, "{p:?}");
        let session = app.sketch_drag_session.take().expect("a live step");
        app.history.commit_modal(session);
        assert_eq!(app.history.entries().len(), steps + 1, "one step");

        // Esc puts a drag back.
        app.sketching.drag = Some(kreate::sketching::SketchDrag {
            points: vec![1],
            anchor: glam::DVec2::new(1.0, 3.0),
            single: Some(1),
        });
        app.drive_sketch_drag(glam::DVec2::new(0.0, 0.0));
        assert!(app.cancel_sketch_drag());
        let p = kreate_ops::ops::sketch_draw::points(&app.doc, id);
        assert!((p[1] - glam::DVec2::new(1.0, 3.0)).length() < 1e-9, "{p:?}");
    }

    #[test]
    fn a_background_bake_lands_as_one_step_and_a_cancelled_one_does_not() {
        let mut app = app();
        let mesh = app.doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let mut o = kreate_scene::document::Object::new(
            "Box",
            kreate_scene::document::ObjectData::Mesh(mesh),
        );
        o.transform.translation = DVec3::new(0.0, 0.0, 3.0);
        o.physics.rigid = Some(kreate_scene::physics::RigidBody::default());
        app.doc.add_object(o);
        app.doc.physics.frame_end = 30;
        let wait = |app: &mut App| {
            for _ in 0..2000 {
                app.poll_bake();
                if app.bake_job.is_none() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            panic!("the bake never finished");
        };
        let steps = app.history.entries().len();
        app.start_bake(0);
        assert!(app.bake_job.is_some());
        wait(&mut app);
        assert!(!app.doc.physics_cache.is_empty());
        assert_eq!(app.history.entries().len(), steps + 1);
        assert!(app
            .history
            .last()
            .unwrap()
            .to_script_call()
            .contains("physics.bake"));

        app.start_bake(0);
        app.bake_job
            .as_ref()
            .unwrap()
            .cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        wait(&mut app);
        assert_eq!(
            app.history.entries().len(),
            steps + 1,
            "cancelled: nothing new"
        );
    }

    /// What a frame of a dynamic-topology stroke costs, piece by piece.
    #[test]
    #[ignore]
    fn dyntopo_frame_cost() {
        use std::time::Instant;
        let mut app = app();
        let mesh = app
            .doc
            .add_mesh(kreate_mesh::primitives::uv_sphere(1.0, 128, 64));
        let id = app.doc.add_object(kreate_scene::document::Object::new(
            "Ball",
            kreate_scene::document::ObjectData::Mesh(mesh),
        ));
        app.doc.selection.add(id);
        app.set_mode(OpMode::from_tag("sculpt").unwrap());
        app.evaluate_scene();
        // As the application does: the stroke finds the surface on the mesh
        // as it began, whose tree is built once.
        let start: Vec<_> = app.evaluated.clone();
        let mut stroke_pick = kreate::pick::PickCache::new();
        let _ = stroke_pick.raycast(
            &app.doc,
            &start,
            DVec3::new(0.0, -5.0, 0.0),
            DVec3::Y,
            f64::INFINITY,
        );
        let mut dabs: Vec<f64> = Vec::new();
        let mut session: Option<kreate_ops::ModalSession> = None;
        let (mut op, mut eval, mut pick) = (0.0, 0.0, 0.0);
        let frames = 40;
        for i in 0..frames {
            let a = i as f64 * 0.03;
            dabs.extend([a.cos(), a.sin(), 0.0, 0.15]);
            let mut params = PropertyMap::new();
            params.set("dabs", PropertyValue::Floats(dabs.clone()));
            params.set("dyntopo", PropertyValue::Bool(true));
            params.set("detail", PropertyValue::Float(0.15));
            let t = Instant::now();
            match session.as_mut() {
                Some(s) => {
                    app.history
                        .update_modal(&app.registry, &mut app.doc, None, &app.op_ctx, s, &params)
                        .unwrap();
                }
                None => {
                    session = app
                        .history
                        .begin_modal(
                            &app.registry,
                            &mut app.doc,
                            None,
                            &app.op_ctx,
                            "sculpt.stroke",
                            &params,
                        )
                        .unwrap();
                }
            }
            op += t.elapsed().as_secs_f64();
            let t = Instant::now();
            app.evaluate_scene();
            eval += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let _ = stroke_pick.raycast(
                &app.doc,
                &start,
                DVec3::new(0.0, -5.0, 0.0),
                DVec3::Y,
                f64::INFINITY,
            );
            pick += t.elapsed().as_secs_f64();
        }
        let ms = |s: f64| s * 1000.0 / frames as f64;
        let faces = app.doc.mesh(mesh).unwrap().num_faces();
        eprintln!(
            "faces {faces}; per frame: operator {:.2} ms, evaluate {:.2} ms, pick {:.2} ms",
            ms(op),
            ms(eval),
            ms(pick)
        );
    }

    /// A sketch point dragged by hand settles back where the constraints
    /// let it, without a step of its own.
    #[test]
    fn a_sketch_solves_itself_after_an_edit() {
        let mut app = app();
        let mesh = app.doc.add_mesh(kreate_mesh::MeshData::from_faces(
            vec![
                glam::Vec3::ZERO,
                glam::Vec3::X,
                glam::Vec3::new(1.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2]],
        ));
        let mut o = kreate_scene::document::Object::new(
            "Sketch",
            kreate_scene::document::ObjectData::Mesh(mesh),
        );
        o.sketch = vec![
            kreate_scene::sketch::Constraint::Fixed {
                a: 0,
                at: [0.0, 0.0],
            },
            kreate_scene::sketch::Constraint::Distance {
                a: 0,
                b: 1,
                value: 1.0,
            },
            kreate_scene::sketch::Constraint::Horizontal { a: 0, b: 1 },
        ];
        let id = app.doc.add_object(o);
        app.doc.selection.add(id);
        app.doc.selection.active = Some(id);
        app.run_drivers();
        // Dragged off its line, too far.
        app.doc.mesh_mut(mesh).unwrap().positions_mut()[1] = glam::Vec3::new(1.6, 0.4, 0.0);
        let steps = app.history.entries().len();
        app.run_drivers();
        let p = app.doc.mesh(mesh).unwrap().positions()[1];
        assert!(
            (p.length() - 1.0).abs() < 1e-4 && p.y.abs() < 1e-4,
            "back on its constraints: {p:?}"
        );
        assert_eq!(app.history.entries().len(), steps);
    }

    /// The Measure tool: two clicks leave a distance, Ctrl on the second
    /// asks for a third and leaves an angle; Esc puts the tool away.
    #[test]
    fn the_measure_tool_leaves_distances_and_angles() {
        let mut app = app();
        app.begin_measure();
        let click = |app: &mut App, at: DVec3, ctrl: bool| {
            app.input.ctrl_down = ctrl;
            app.measure.as_mut().expect("the tool is out").hover = Some(at);
            app.measure_click();
        };
        click(&mut app, DVec3::ZERO, false);
        click(&mut app, DVec3::new(0.0, 0.03, 0.04), false);
        assert_eq!(app.doc.design.measurements.len(), 1);
        assert!((app.doc.design.measurements[0].length().unwrap() - 0.05).abs() < 1e-12);

        click(&mut app, DVec3::X, false);
        click(&mut app, DVec3::ZERO, true);
        assert_eq!(
            app.doc.design.measurements.len(),
            1,
            "an angle wants a third point"
        );
        click(&mut app, DVec3::Y, false);
        let angle = app.doc.design.measurements[1].angle().unwrap();
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        // Each is one undo step, and the tool stays out until put away.
        assert!(app.measure.is_some());
        app.end_measure("");
        assert!(app.measure.is_none());
        assert!(app.ui.measure_preview.is_empty());
    }

    /// Blender's Alt+E: the region, or each face along its own normal.
    #[test]
    fn the_extrude_menu_offers_region_and_along_normals() {
        let mut app = app();
        app.open_popup_menu("extrude");
        let rows: Vec<_> = app
            .ui
            .context_menu
            .groups
            .iter()
            .flat_map(|g| g.entries.iter())
            .map(|e| (e.op_id.clone(), e.params.get("along_normals").cloned()))
            .collect();
        assert_eq!(
            rows,
            [
                ("app.extrude".to_string(), None),
                ("app.extrude".to_string(), Some(PropertyValue::Bool(true))),
            ]
        );
    }

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
    fn local_view_shows_only_the_selection_and_comes_back() {
        use kreate_scene::document::{Object, ObjectData};
        let mut app = app();
        let mesh = app.doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let a = app.doc.add_object(Object::new("A", ObjectData::Mesh(mesh)));
        let b = app.doc.add_object(Object::new("B", ObjectData::Mesh(mesh)));
        app.doc.selection.add(a);
        app.evaluate_scene();
        assert_eq!(app.evaluated.len(), 2);
        app.toggle_local_view();
        assert!(
            app.evaluated.iter().all(|(id, _)| *id == a),
            "B is still drawn"
        );
        // Made while in it: shown, as Blender shows it.
        let c = app.doc.add_object(Object::new("C", ObjectData::Mesh(mesh)));
        app.evaluate_scene();
        assert!(
            app.evaluated.iter().any(|(id, _)| *id == c),
            "C is not drawn"
        );
        assert!(app.evaluated.iter().all(|(id, _)| *id != b), "B is drawn");
        app.toggle_local_view();
        app.evaluate_scene();
        assert!(app.evaluated.iter().any(|(id, _)| *id == b));
        // A new document leaves it.
        app.doc.selection.add(a);
        app.toggle_local_view();
        assert!(app.local_view.is_some());
        app.adopt(kreate_scene::Document::new());
        assert!(app.local_view.is_none());
        // Nothing selected: no local view to enter.
        app.doc.selection.clear();
        app.toggle_local_view();
        assert!(app.local_view.is_none());
    }

    #[test]
    fn numpad_zero_goes_through_another_camera_when_it_is_active() {
        let mut app = app();
        app.run_op("object.add_camera", PropertyMap::new());
        let first = app.doc.selection.active.unwrap();
        app.toggle_camera_view();
        assert_eq!(app.camera_view_object, Some(first));
        app.run_op("object.add_camera", PropertyMap::new());
        let second = app.doc.selection.active.unwrap();
        app.toggle_camera_view();
        assert!(app.camera_view.is_some(), "still in a camera view");
        assert_eq!(app.camera_view_object, Some(second), "the other camera's");
        app.toggle_camera_view();
        assert!(app.camera_view.is_none(), "the same camera again leaves");
    }

    #[test]
    fn orthographic_stays_orthographic_once_the_camera_settles() {
        // A settled camera is set to the easing's target every frame; a switch
        // that changed only the drawn camera was undone on the next one.
        let mut app = app();
        // From perspective, which is not where a fresh start is.
        app.camera.projection = kreate_render::Projection::Perspective;
        app.motion = kreate_render::CameraMotion::new(app.camera);
        app.dispatch(&Action::new("view.toggle_projection"));
        assert_eq!(
            app.camera.projection,
            kreate_render::Projection::Orthographic
        );
        for _ in 0..5 {
            app.motion.advance(&mut app.camera, 1.0 / 60.0);
        }
        assert_eq!(
            app.camera.projection,
            kreate_render::Projection::Orthographic,
            "it went back"
        );

        // And the lens, the same way.
        let mut lens = PropertyMap::new();
        lens.set("lens", PropertyValue::Float(100.0));
        app.dispatch(&Action {
            op_id: "view.set_lens".into(),
            params: lens,
        });
        let fov = app.camera.fov_y;
        for _ in 0..5 {
            app.motion.advance(&mut app.camera, 1.0 / 60.0);
        }
        assert_eq!(app.camera.fov_y, fov, "the lens went back");
    }

    #[test]
    fn numpad_zero_needs_a_camera_and_toggles_back() {
        let mut app = app();
        app.toggle_camera_view();
        assert!(
            app.camera_view.is_none(),
            "there is no camera to look through"
        );
        app.run_op("object.add_camera", PropertyMap::new());
        app.toggle_camera_view();
        assert!(app.camera_view.is_some());
        app.toggle_camera_view();
        assert!(app.camera_view.is_none());
    }

    #[test]
    fn the_camera_view_follows_the_camera_until_the_user_takes_the_view() {
        let mut app = app();
        app.run_op("object.add_camera", PropertyMap::new());
        let camera = app.doc.selection.active.expect("the camera is active");
        app.toggle_camera_view();
        // The move into the camera finishes, and the view is the camera's.
        while app.motion.advance(&mut app.camera, 0.1) {}
        app.follow_camera_view();
        let start = app.camera.eye();

        // The camera moves — an animation playing, or a script.
        app.doc.object_mut(camera).unwrap().transform.translation += DVec3::new(2.0, 0.0, 1.0);
        app.follow_camera_view();
        assert!(
            (app.camera.eye() - start - DVec3::new(2.0, 0.0, 1.0)).length() < 1e-6,
            "the view stayed at {start:?} while the camera moved"
        );
        assert!(app.camera_view.is_some());

        // The user orbits: the view is theirs, and the camera view is left.
        app.motion.target.orbit(0.3, 0.0);
        while app.motion.advance(&mut app.camera, 0.1) {}
        app.follow_camera_view();
        assert!(
            app.camera_view.is_none(),
            "orbiting did not leave the camera view"
        );
        let theirs = app.camera.eye();
        app.doc.object_mut(camera).unwrap().transform.translation += DVec3::new(1.0, 0.0, 0.0);
        app.follow_camera_view();
        assert_eq!(
            app.camera.eye(),
            theirs,
            "a view the user took was pulled back"
        );
    }

    #[test]
    fn a_dropped_scene_is_added_to_this_one_as_one_step() {
        let mut app = app();
        let cube = app.doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        app.doc.add_object(kreate_scene::document::Object::new(
            "Cube",
            kreate_scene::document::ObjectData::Mesh(cube),
        ));
        let (bytes, _) = kreate_io::write_stl(
            &kreate_mesh::primitives::cube(2.0),
            &kreate_io::StlOptions::default(),
        );
        let dir = std::env::temp_dir().join(format!("kreate-drop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dropped.stl");
        std::fs::write(&path, bytes).unwrap();
        let (objects, steps) = (app.doc.object_count(), app.history.entries().len());
        app.open_dropped(&path);
        // Read on a thread, added when it is ready.
        let start = std::time::Instant::now();
        while app.doc.object_count() == objects && start.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            app.serve_imports();
        }
        assert_eq!(
            app.doc.object_count(),
            objects + 1,
            "the scene was replaced or nothing came"
        );
        assert_eq!(app.history.entries().len(), steps + 1);
        assert!(app.history.undo(&mut app.doc));
        assert_eq!(app.doc.object_count(), objects);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn c_turns_the_selection_brush_on_and_off_and_a_stroke_over_nothing_leaves_no_step() {
        let mut app = app();
        app.toggle_circle_select();
        assert!(app.circle.is_some());
        assert!(app.ui.circle_select.is_some(), "the brush is not drawn");
        let depth = app.history.entries().len();
        app.begin_circle_stroke();
        app.drive_circle_stroke();
        app.finish_circle_stroke();
        assert_eq!(app.history.entries().len(), depth);
        if let Some(c) = &mut app.circle {
            c.radius = 90.0;
        }
        app.toggle_circle_select();
        assert!(app.circle.is_none() && app.ui.circle_select.is_none());
        // The next one starts the size the last one ended.
        app.toggle_circle_select();
        assert_eq!(app.circle.as_ref().map(|c| c.radius), Some(90.0));
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
            lasso: None,
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

    /// A conversation is kept as it goes and, after a restart, shown again
    /// and carried on in the same session; New Chat leaves nothing to bring
    /// back.
    #[test]
    fn the_chat_is_kept_and_comes_back_after_a_restart() {
        use kreate_ui::chat::{ChatMessage, ChatRole};
        let mut first = app();
        let dir = kreate::chatlog::chat_dir().expect("a test settings folder");
        kreate::chatlog::forget_last_in(&dir);
        let mut chat = kreate::claude::Chat::new(|| {});
        chat.resume("session-1".into());
        first.chat = Some(chat);
        first.ui.chat_log = vec![
            ChatMessage {
                role: ChatRole::User,
                text: "a bracket, token: s3cr3t".into(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                text: "run_script: kreate.ops.object.add_primitive()".into(),
            },
            ChatMessage {
                role: ChatRole::Claude,
                text: "Made it.".into(),
            },
        ];
        first.save_chat();
        let page = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "md"))
            .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
            .expect("a page");
        let md = std::fs::read_to_string(&page).unwrap();
        assert!(md.contains("Made it."), "{md}");
        assert!(!md.contains("s3cr3t"), "{md}");

        let mut again = app();
        again.chat = Some(kreate::claude::Chat::new(|| {}));
        again.restore_chat();
        assert_eq!(again.ui.chat_log.len(), 4, "three said and a note");
        assert_eq!(again.ui.chat_log[2].text, "Made it.");
        // Claude has no such session on this machine: shown, not resumed.
        assert_eq!(again.chat.as_ref().unwrap().session(), None);

        again.ui.queue.push(kreate_ui::Command::ChatNew);
        again.drain_ui_commands();
        let mut third = app();
        third.chat = Some(kreate::claude::Chat::new(|| {}));
        third.restore_chat();
        assert!(third.ui.chat_log.is_empty());
    }

    /// Picking "on a face" and clicking a part: a sketch on the face under
    /// the pointer, its outline in it; clicking nothing makes nothing.
    #[test]
    fn a_click_on_a_face_starts_a_sketch_on_it() {
        let mut app = app();
        with_viewport(&mut app);
        let part = app
            .doc
            .add_mesh_object("Part", kreate_mesh::primitives::cube(2.0));
        app.doc.selection.add(part);
        app.doc.selection.active = Some(part);
        app.camera = kreate::startup::starting_camera();
        app.sketch_face_pick = true;
        app.sketch_on_face_under_cursor();
        let s = kreate_ops::ops::sketch_draw::active_sketch(&app.doc).expect("a sketch");
        assert_ne!(s, part);
        let o = app.doc.object(s).unwrap();
        assert_eq!(o.sketch_lines.len(), 4, "the face's square");
        // On the face: the sketch's origin is on the cube's surface.
        let at = app.doc.world_transform(s).translation;
        assert_eq!(o.parent, Some(part));
        assert!((at.abs().max_element() - 1.0).abs() < 1e-6, "{at}");
        assert!(!app.sketch_face_pick);
        // Off every part: nothing.
        let before = app.doc.objects().count();
        app.input.cursor = Some((110.0, 60.0));
        app.sketch_face_pick = true;
        app.sketch_on_face_under_cursor();
        assert_eq!(app.doc.objects().count(), before);
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
    fn sketch_mode_from_a_tab_above_the_viewport_asks_for_a_plane_inside_it() {
        // The mode tabs sit above the viewport, and the plane menu is drawn by
        // the viewport alone: opened at the pointer, it was off the view and
        // never seen, and the Sketch tab looked as if it did nothing.
        let mut app = app();
        with_viewport(&mut app);
        app.input.cursor = Some((300.0, 10.0));
        app.set_mode(OpMode::Sketch);
        assert!(!app.ui.context_menu.groups.is_empty(), "no plane menu");
        let at = app.ui.context_menu.at.expect("the plane menu is not open");
        let scale = app.egui_ctx.pixels_per_point();
        let view = egui::Rect::from_min_size(
            egui::pos2(100.0 / scale, 50.0 / scale),
            egui::vec2(800.0 / scale, 600.0 / scale),
        );
        assert!(view.contains(at), "{at:?} is outside the viewport {view:?}");
    }

    #[test]
    fn edit_mode_on_a_body_from_a_sketch_has_something_to_select() {
        // A body is built from its features and holds no mesh; edit mode on
        // it had nothing to pick. Entering makes it a plain mesh first.
        let mut app = app();
        app.run_op_with("sketch.new", &[]);
        app.run_op_with(
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.04, 0.02, 0.0])),
                ("join", PropertyValue::Int(1)),
            ],
        );
        app.run_op_with(
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.002, 0.002])),
                ("distance", PropertyValue::Float(0.01)),
            ],
        );
        let body = app
            .doc
            .objects()
            .find(|(_, o)| !o.features.is_empty())
            .map(|(id, _)| id)
            .expect("a body");
        app.doc.selection.clear();
        app.doc.selection.add(body);
        app.doc.selection.active = Some(body);
        app.set_mode(OpMode::EditMesh);
        assert_eq!(app.op_ctx.mode, OpMode::EditMesh);
        let object = app.doc.object(body).unwrap();
        assert!(object.features.is_empty());
        let mesh = app.doc.mesh(object.mesh().unwrap()).unwrap();
        assert_eq!(mesh.num_faces(), 6);
    }

    /// Add > Text waits for a click and puts the text there: on the
    /// surface under the pointer, lying on it facing the view; on the
    /// floor where there is none. One step each.
    #[test]
    fn text_is_put_where_it_is_clicked() {
        let mut app = app();
        with_viewport(&mut app);
        app.camera.pivot = DVec3::ZERO;
        app.doc
            .add_mesh_object("Block", kreate_mesh::primitives::cube(2.0));
        app.evaluate_scene();
        app.dispatch(&Action::new("app.place_text"));
        assert!(app.text_place);
        app.place_text_under_cursor();
        assert!(!app.text_place);
        let text = app.doc.selection.active.expect("text added");
        let o = app.doc.object(text).unwrap();
        assert!(o.name.starts_with("Text"), "{}", o.name);
        let at = o.transform.translation;
        let up = o.transform.rotation * DVec3::Z;
        // On the block's surface: one coordinate at its half-size, and the
        // text's Z out of that face, toward the eye.
        let on_face = [at.x, at.y, at.z]
            .iter()
            .any(|c| (c.abs() - 1.0).abs() < 1e-6);
        assert!(on_face, "{at}");
        assert!(up.dot(app.camera.eye() - at) > 0.0, "facing the view: {up}");
        assert!((up.length() - 1.0).abs() < 1e-9);

        // Nothing under the pointer: on the floor, lying flat.
        let block = app
            .doc
            .objects()
            .find(|(_, o)| o.name == "Block")
            .map(|(id, _)| id)
            .unwrap();
        app.doc.remove_object(block);
        app.evaluate_scene();
        app.dispatch(&Action::new("app.place_text"));
        app.place_text_under_cursor();
        let text = app.doc.selection.active.unwrap();
        let o = app.doc.object(text).unwrap();
        assert!(
            o.transform.translation.z.abs() < 1e-6,
            "{}",
            o.transform.translation
        );
        assert!((o.transform.rotation * DVec3::Z - DVec3::Z).length() < 1e-9);
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
    /// A part moved with its children carries them: none of them is a place
    /// to snap to, and all of them are places to take the snap base from.
    #[test]
    fn a_selected_parents_children_move_with_it_for_snapping() {
        let mut app = app();
        let parent = app
            .doc
            .add_mesh_object("Body", kreate_mesh::primitives::cube(1.0));
        let child = app
            .doc
            .add_mesh_object("Leg", kreate_mesh::primitives::cube(0.5));
        let other = app
            .doc
            .add_mesh_object("Floor", kreate_mesh::primitives::plane(4.0));
        app.doc.set_parent(child, Some(parent)).unwrap();
        app.doc.object_mut(child).unwrap().transform.translation = DVec3::new(3.0, 0.0, 0.0);
        app.doc.object_mut(other).unwrap().transform.translation = DVec3::new(0.0, 0.0, -5.0);
        app.doc.selection.clear();
        app.doc.selection.add(parent);
        app.evaluate_scene();
        // Only the floor's four corners to snap to.
        let targets = app.build_snap_world(kreate::snap::Mode::Element);
        assert_eq!(targets.verts.len(), 4);
        assert!(targets.verts.iter().all(|v| (v.z + 5.0).abs() < 1e-9));
        // The body's and the leg's corners to take the base from.
        let own = app.build_own_snap_world();
        assert_eq!(own.verts.len(), 16);
    }
}
