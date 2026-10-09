//! The rendered mode, end to end.
//!
//! Every ingredient has been checked against the CPU on a device already. What
//! this checks is the *assembly*: that a scene with a light in it comes out
//! lit, that the display transform is applied once, that the overlays are not
//! put through it, and that switching modes changes the picture.
//!
//! Rendered offscreen and read back. A pixel test would confirm that something
//! appeared; these look at what the pixels *are*, because the failures worth
//! catching here — a mode that shades but never resolves, a resolve applied
//! twice, an unlit scene that comes out grey anyway — all produce a picture.

use kreate_render::display::Display;
use kreate_render::lights::{Falloff, PlacedLight};
use kreate_render::viewport::{Draw, ShadingMode, Viewport, VIEWPORT_DEPTH_FORMAT};
use kreate_render::{brdf::Surface, Camera, GpuOptions, RenderMeshCpu};

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// This test's turn on the device, held for as long as the test runs.
    ///
    /// One test at a time: many threads driving one device at once — path
    /// tracing, readbacks, maps and waits — ended this binary now and then in
    /// a heap corruption inside the driver (0xc0000374), with every test
    /// passing on its own. The CI runner's Windows build already runs the
    /// application's GPU tests singly for the same family of crash.
    _turn: std::sync::MutexGuard<'static, ()>,
}

fn gpu() -> Option<Gpu> {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
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
    let turn = TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    Some(Gpu {
        device,
        queue,
        _turn: turn,
    })
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

const SIZE: u32 = 64;
/// What a window would give, near enough.
///
/// Not a float target, which was the first attempt: the overlay pipelines
/// blend, and `Rgba32Float` is not blendable in the WebGPU core. Eight bits is
/// what a screen has anyway, and it turns out to sharpen the test rather than
/// blunt it — see `the_display_transform_is_applied_and_applied_once`, which
/// counts *saturated* pixels, a measurement only a bounded format can make.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A cube filling most of the view, a camera looking at it, and whatever
/// lights are given.
fn shoot(
    gpu: &Gpu,
    shading: ShadingMode,
    surface: Surface,
    lights: &[PlacedLight],
    display: Display,
) -> Vec<[f32; 4]> {
    let graph = kreate_scene::shader::ShaderGraph::standard(
        "test",
        surface.base_color,
        surface.metallic,
        surface.roughness,
    );
    shoot_graph(gpu, shading, &graph, lights, display)
}

/// [`shoot`], with the material a graph rather than a flat surface.
fn shoot_graph(
    gpu: &Gpu,
    shading: ShadingMode,
    graph: &kreate_scene::shader::ShaderGraph,
    lights: &[PlacedLight],
    display: Display,
) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = shading;
    viewport.set_display(&gpu.queue, display);
    // The grid and the wireframe are interface, and this test is about the
    // shading. Left on, they would be the brightest thing in the frame.
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);

    // Looking at the origin from six metres, which frames a two-metre cube.
    let camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };

    graph_material(gpu, &mut viewport, graph);
    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY)];
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), lights);
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
    assert_eq!(VIEWPORT_DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);

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

/// An exposure that frames a physically lit scene.
///
/// **A gap, recorded rather than hidden.** The lights are photometric — a sun
/// is twenty thousand lux — and the display transform expects scene-linear
/// values where 0.18 is mid grey. Nothing bridges the two yet, so a physically
/// correct scene at the default exposure saturates every pixel, which is
/// exactly what the first version of these tests measured.
///
/// The bridge belongs on the camera: §2.5 already stores the f-stop, the
/// shutter and the focus distance, and the standard photographic exposure
/// falls out of those plus a sensitivity. Until that is wired up, a test has
/// to supply the number, and so would a user.
///
/// Twelve stops down puts a sunlit grey card near one, which is where AgX
/// wants it.
const FRAMED: Display = Display {
    view: kreate_math::ViewTransform::AgX,
    exposure: -12.0,
};

/// Gives the viewport one material, compiled from a surface.
///
/// A surface *is* a graph now — `ShaderGraph::standard` with nothing connected
/// — and the rendered pass calls the function that graph compiles to. These
/// tests used to hand a `Surface` straight to a draw; that they still produce
/// the same pictures is the whole claim of the change.
fn one_material(gpu: &Gpu, viewport: &mut Viewport, surface: Surface) {
    let graph = kreate_scene::shader::ShaderGraph::standard(
        "test",
        surface.base_color,
        surface.metallic,
        surface.roughness,
    );
    graph_material(gpu, viewport, &graph);
}

fn graph_material(gpu: &Gpu, viewport: &mut Viewport, graph: &kreate_scene::shader::ShaderGraph) {
    let compiled = kreate_shader::compile(graph).expect("compiles");
    viewport.set_materials(
        &gpu.device,
        &gpu.queue,
        &[kreate_render::viewport::MaterialSource {
            wgsl: &compiled.wgsl,
            constants: &compiled.constants,
            image_slots: &compiled.image_slots,
            images: &[],
        }],
    );
}

fn grey() -> Surface {
    Surface {
        base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        metallic: 0.0,
        roughness: 0.5,
    }
}

/// A sun pointing away from the camera, so it lights the side that faces it.
fn key_light(lux: f32) -> PlacedLight {
    PlacedLight {
        falloff: Falloff::Parallel,
        direction: glam::Vec3::new(-0.3, 0.4, -0.87).normalize(),
        amount: [lux; 3],
        ..PlacedLight::default()
    }
}

/// The brightest pixel, how many are lit, and how many are at the top.
///
/// The third is what says whether the display transform ran: an unresolved
/// linear image of a sunlit surface is thousands of times over the display's
/// range, so every pixel of it saturates. AgX leaves a gradient.
struct Shot {
    brightest: f32,
    lit: usize,
    saturated: usize,
}

fn summary(pixels: &[[f32; 4]]) -> Shot {
    Shot {
        brightest: pixels.iter().fold(0.0f32, |m, p| m.max(p[0])),
        lit: pixels.iter().filter(|p| p[0] > 0.01).count(),
        // 254 rather than 255, so a value one code short of the top still
        // counts: the question is "clipped", not "exactly at the maximum".
        saturated: pixels.iter().filter(|p| p[0] >= 254.0 / 255.0).count(),
    }
}

/// A shadow scene: boxes standing on a floor under a tilted sun.
///
/// Every choice here is against a way the first version of this fixture could
/// not fail. It was a single cube floating over a floor, lit straight down,
/// and a falsification sweep walked through it: sixteen of forty-two breaks
/// survived, almost all of them because this scene could not express the
/// failure.
///
/// - **The sun is tilted**, not straight down. A surface square to the light
///   stores exactly the depth it later compares, and reversed-Z's `>=` passes
///   on equality — so with no bias at all there was no acne to find. The tilt
///   also puts a slope under the bias, and breaks the picture's symmetry, which
///   is what makes a flipped texture coordinate visible.
/// - **The boxes stand on the floor.** A shadow that starts at its caster's
///   base is the only one that can detach from it, which is the failure an
///   over-large bias produces and the reason the bias is not simply enormous.
/// - **There are nine of them**, so the draw list is longer than the per-draw
///   uniform ring starts out, and each box's shadow is somewhere different.
///   One shared model matrix put every shadow in one place; with one box that
///   is the right place.
///
/// The lights are switches rather than a list, because each one exists to make
/// one thing observable and the test that wants it should say which.
#[derive(Clone, Copy, Default)]
struct ShadowScene {
    /// The key sun casts.
    cast: bool,
    /// A second sun, from the other side.
    second: bool,
    /// And whether *it* casts.
    ///
    /// Separate from `cast` because the two lights' shadows have to be
    /// measurable one at a time: adding a light brightens the whole picture,
    /// so "darker than before" says nothing unless the only thing that changed
    /// is which of them casts.
    second_casts: bool,
    /// A light that is present and casts nothing, *before* the casters.
    ///
    /// The entries are parallel to the lights, so an entry written by layer
    /// rather than by light would land on this one.
    idle: bool,
    /// Ranges so long the window never bites, which is the same arithmetic as
    /// no cutoff at all.
    ///
    /// The control for `ending_a_light_where_it_goes_black_changes_no_pixel`:
    /// the two pictures have to be identical, so the difference between them
    /// has to be the cutoff and nothing else.
    enormous_range: bool,
    /// The end the display says each lamp deserves.
    ///
    /// What a viewport will fill in once something culls against it. Applied
    /// here rather than there because nothing culls yet, and a wire with
    /// nothing on the other end of it is a wire no test can pull.
    derived_range: bool,
    /// A range shorter than the scene, which very much does change the picture.
    ///
    /// There so the test above cannot pass by measuring nothing. A lamp's
    /// derived range is far beyond a viewport scene — see the note in
    /// `ending_a_light_where_it_goes_black_changes_no_pixel` — so "the picture
    /// did not move" needs a demonstration that this rig would have noticed if
    /// it had.
    short_range: bool,
    /// And that front light starting to cast.
    ///
    /// Which pushes everything after it one place down the queue for the eight
    /// shadow maps there are — so the last caster loses its map while going on
    /// lighting the scene. That is the one arrangement in which an entry left
    /// over from the previous frame can be seen: everywhere else a light with
    /// no frame is a light that reaches nothing.
    idle_casts: bool,
    /// Enough casting lights to run past `MAX_SHADOWS`.
    crowd: bool,
    /// How big the key light's source is, as a multiple of the sun's own.
    ///
    /// Zero is the hard light nothing real is. One is the sun at its own
    /// half-degree; larger is the overcast sky, and what it should produce is a
    /// wider soft edge rather than a blurrier picture.
    source: f32,
    /// The boxes hovering well above the floor rather than standing on it.
    ///
    /// The control for the contact test: the same scene with the only
    /// difference being how far a caster is from what it falls on.
    lifted: bool,
    /// Two dim lamps lighting their own pools of floor and nothing else.
    ///
    /// The only arrangement in the file where one tile's light list differs
    /// from another's. Everything else here is lit by suns and by lamps bright
    /// enough that their range covers the whole scene — so every tile holds
    /// every light, and a shading pass reading the wrong tile gets the right
    /// answer by accident. A sweep that made it read tile zero for every
    /// fragment walked through the lot.
    pools: bool,
    /// A floor stretching past where a sun's cascades reach.
    ///
    /// The one arrangement in which a fragment can be lit by a sun and outside
    /// every one of its frames: cascades stop at `CASCADE_REACH`, and anything
    /// further is a distance where a shadow is a few pixels wide and the map
    /// would be a stain. What happens there is the same decision as outside any
    /// frustum — lit rather than black — and this is where it is reachable.
    long_floor: bool,
    /// A point light down among the boxes rather than above them.
    ///
    /// The only arrangement that puts fragments on more than one face of the
    /// cube. A lamp hung overhead has nearly the whole floor on its downward
    /// face, so picking the wrong face for a direction moves a handful of
    /// corner pixels and nothing notices — a sweep that reversed the ±X faces
    /// walked straight through the overhead test.
    point_inside: bool,
    /// A point light overhead instead of the key sun.
    ///
    /// A bare point light gets one of its six faces, which is a documented
    /// limitation and the only way a fragment can be lit by a light and still
    /// be outside that light's shadow frustum.
    point: bool,
    /// A spot overhead instead of the key sun.
    ///
    /// The only light that puts a *perspective* shadow frustum on the device.
    /// A sun is orthographic and the point light above casts nothing, so
    /// without this the whole perspective path — the projection, and the texel
    /// that grows with distance — was checked by arithmetic alone.
    spot: bool,
    /// And that spot turned to face away from everything.
    ///
    /// A lamp swung away from the scene is the reachable way for a light to go
    /// on casting while its frustum stops seeing anything, which is the one
    /// case where an entry not reset between frames leaves the previous
    /// frame's matrix shadowing the scene.
    spot_away: bool,
}

struct ShadowShot {
    pixels: Vec<[f32; 4]>,
    /// How many shadow maps the frame rendered.
    passes: usize,
}

const SHADOW_SIZE: u32 = 128;

fn shadow_lights(scene: ShadowScene) -> Vec<PlacedLight> {
    let mut lights: Vec<PlacedLight> = Vec::new();
    if scene.pools {
        // Fifty candela apiece, which at this exposure is invisible beyond
        // about five metres — `cull_range` says so, and the range below is that
        // number. So the window takes away only light already under the black
        // point, and what is left is two separate pools with a dark gap.
        for x in [-6.0f64, 6.0] {
            lights.push(PlacedLight {
                position: glam::DVec3::new(x, 0.0, 2.0),
                // Bright enough to photograph, and stopped short by hand. Not
                // the distance `cull_range` derives: that is where a lamp stops
                // being *visible*, and AgX shows twelve and a half stops of
                // shadow, so for anything bright enough to see it comes out at
                // tens of metres and covers the whole scene. An artist-set
                // distance is what makes a tile's list short — which is the
                // same conclusion the plan reaches from the other direction.
                amount: [2_000.0; 3],
                cast_shadow: false,
                range: Some(5.0),
                ..PlacedLight::default()
            });
        }
        return lights;
    }
    if scene.idle {
        lights.push(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.0, -0.2, -0.98).normalize(),
            amount: [2_000.0; 3],
            cast_shadow: scene.idle_casts,
            ..PlacedLight::default()
        });
    }
    if scene.crowd {
        // Before the key light, so that *it* is the one the layer budget runs
        // out on. The cap is first-come, and a bright lamp losing faces is what
        // makes the denied-face decision visible — a dim fill losing one is
        // not.
        //
        // Fifteen, which leaves exactly one layer for whatever asks next. A
        // point light then gets one face of its six, and the other five are
        // denied — which is what makes the decision about a face with no map
        // cover enough of the picture for a test to see it. Twelve left four
        // faces granted, the two denied ones happened to face sky and floor
        // directly under the lamp, and blacking them out moved six pixels.
        for i in 0..15 {
            let a = i as f32 * 0.7;
            // Spots rather than suns, because a spot is exactly one frame and
            // a sun is as many cascades as the view reaches. What this crowd is
            // for is filling the layer budget to a known depth, and a light
            // whose cost depends on the camera cannot do that.
            lights.push(PlacedLight {
                position: glam::DVec3::new((a.cos() * 9.0) as f64, (a.sin() * 9.0) as f64, 7.0),
                direction: glam::Vec3::new(-a.cos() * 0.5, -a.sin() * 0.5, -0.85).normalize(),
                cone: Some((80f32.to_radians(), 0.3)),
                amount: [60_000.0; 3],
                cast_shadow: true,
                ..PlacedLight::default()
            });
        }
    }
    lights.push(if scene.spot || scene.spot_away {
        PlacedLight {
            // Off to one side and well tilted, for the same reason the sun
            // is: a lamp hanging straight over a floor stores the depth it
            // later compares and finds no acne to prevent, so a scene lit that
            // way cannot tell a working bias from a missing one.
            position: glam::DVec3::new(9.0, 7.0, 11.0),
            direction: if scene.spot_away {
                // Straight up, at nothing.
                glam::Vec3::Z
            } else {
                glam::Vec3::new(-9.0, -7.0, -11.0).normalize()
            },
            // Wide enough to reach the boxes and narrow enough that the
            // frustum is not a sun in disguise.
            cone: Some((70f32.to_radians(), 0.25)),
            amount: [2_600_000.0; 3],
            cast_shadow: scene.cast,
            // A lamp with a bulb rather than a point. The perspective half of
            // the penumbra arithmetic runs nowhere else — a sun is orthographic
            // and uses the other branch entirely.
            //
            // Small enough that the filter does not reach its widest setting:
            // at the clamp every scale looks the same, so a test up there
            // cannot tell a right one from a wrong one.
            radius: 0.05 * scene.source,
            ..PlacedLight::default()
        }
    } else if scene.point_inside {
        PlacedLight {
            // Between the boxes and at their height, so the scene wraps around
            // it and every face of the cube has something in it.
            position: glam::DVec3::new(2.25, 2.25, 0.8),
            amount: [60_000.0; 3],
            cast_shadow: scene.cast,
            radius: 0.05 * scene.source,
            ..PlacedLight::default()
        }
    } else if scene.point {
        PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 9.0),
            amount: [900_000.0; 3],
            cast_shadow: scene.cast,
            ..PlacedLight::default()
        }
    } else {
        PlacedLight {
            falloff: Falloff::Parallel,
            // Tilted on both axes: a light square to the floor makes a scene
            // in which no bias is needed and a flipped map looks the same.
            direction: glam::Vec3::new(-0.50, -0.35, -0.79).normalize(),
            amount: [18_000.0; 3],
            cast_shadow: scene.cast,
            // Half of 0.526°, which is the sun's own apparent radius.
            angular_radius: 0.0046 * scene.source,
            ..PlacedLight::default()
        }
    });
    if scene.second {
        lights.push(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.55, 0.30, -0.78).normalize(),
            amount: [14_000.0; 3],
            cast_shadow: scene.second_casts,
            ..PlacedLight::default()
        });
    }
    if scene.derived_range {
        let black = FRAMED.on_cpu().black_radiance();
        for light in &mut lights {
            let brightest = light.amount.iter().copied().fold(0.0f32, f32::max);
            light.range = Some(kreate_render::lights::cull_range(brightest, black));
        }
    }
    if scene.short_range {
        for light in &mut lights {
            light.range = Some(8.0);
        }
    }
    if scene.enormous_range {
        // Further than the scene is wide by a wide margin, so the window is one
        // everywhere the geometry is. Set rather than left as `None`, because
        // `None` is what the viewport fills in — the control has to be a range
        // that does nothing, not the absence of one.
        for light in &mut lights {
            light.range = Some(1.0e6);
        }
    }
    lights
}

/// Renders the scene, optionally twice on the same viewport.
///
/// The second frame exists for one failure: a light that stops casting between
/// frames. The entries are reused every frame, so a renderer that filled in the
/// casters without first clearing the rest would keep the previous frame's
/// matrix — and a viewport built fresh for each shot, which is what every other
/// test here does, can never see it.
fn shadow_frames(gpu: &Gpu, scenes: &[ShadowScene]) -> Vec<ShadowShot> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SHADOW_SIZE, SHADOW_SIZE);
    // Shadows are measured by their edges and their light, and the glow
    // around bright light softens edges and the occlusion darkens corners —
    // different claims, with their own tests.
    viewport.show_bloom = false;
    viewport.ambient_occlusion = false;
    viewport.shading = ShadingMode::Rendered;
    viewport.set_display(&gpu.queue, FRAMED);
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    // The geometry is built once for every frame this call renders, so the
    // floor's size comes from the first: a pair of frames that disagreed about
    // how big the world is would be comparing two scenes.
    let long = scenes.first().is_some_and(|s| s.long_floor);
    // Standing on the floor, or well above it. Like the floor's size, taken
    // from the first frame: the geometry is built once and a pair of frames
    // that disagreed about where the boxes are would be comparing two scenes.
    let height = if scenes.first().is_some_and(|s| s.lifted) {
        5.0
    } else {
        0.8
    };
    let floor_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::plane(if long {
        1600.0
    } else {
        16.0
    }));
    let box_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.6));
    // A sphere for the angles a box does not have. Every incidence from
    // head-on to grazing appears on one, and the band where it turns away from
    // the light is where acne lives in every renderer that has it.
    let ball_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::uv_sphere(0.9, 24, 16));
    let floor = kreate_render::GpuMesh::upload(
        &gpu.device,
        &viewport.layouts,
        &floor_mesh,
        glam::Vec4::ONE,
    );
    let boxes =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &box_mesh, glam::Vec4::ONE);
    let balls =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &ball_mesh, glam::Vec4::ONE);

    // Closer for the pools, which are a couple of metres across: at the
    // framing the rest of the file uses they are nine pixels, and nine pixels
    // is not a measurement.
    let camera = Camera {
        distance: if scenes.first().is_some_and(|s| s.pools) {
            14.0
        } else {
            22.0
        },
        ..Camera::default()
    };

    // Nine objects standing on the floor, none of them at the middle: a
    // shadow under a single central object is where a shadow would land however
    // the model matrices were mixed up.
    //
    // Three kinds, and the mix is the point. Upright boxes are the plain case.
    // *Rotated* boxes are the only draws whose bounding box is not their mesh's
    // — a scene of translations alone never asks whether the fitting rotates
    // what it fits. Spheres carry every incidence angle at once.
    one_material(gpu, &mut viewport, grey());
    let mut draws = vec![Draw::new(&floor, glam::DAffine3::IDENTITY)];
    for row in 0..3 {
        for col in 0..3 {
            let n = row * 3 + col;
            let x = -4.5 + col as f64 * 4.5;
            let y = -4.5 + row as f64 * 4.5;
            let at = glam::DVec3::new(x, y, height);
            let draw = match n % 3 {
                // Standing on the floor, not hovering over it: half the cube's
                // height up puts its underside at zero.
                0 => Draw::new(&boxes, glam::DAffine3::from_translation(at)),
                1 => Draw::new(
                    &boxes,
                    glam::DAffine3::from_rotation_translation(
                        glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.6, 0.4, n as f64 * 0.5),
                        at,
                    ),
                ),
                _ => Draw::new(&balls, glam::DAffine3::from_translation(at)),
            };
            draws.push(draw);
        }
    }

    let mut out = Vec::new();
    for scene in scenes {
        viewport.set_world([0.0; 3]);
        viewport.set_lights(
            &gpu.device,
            &gpu.queue,
            camera.eye(),
            &shadow_lights(*scene),
        );
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow shot"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
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

        let row = (SHADOW_SIZE * 4).div_ceil(256) * 256;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row * SHADOW_SIZE) as u64,
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
                    rows_per_image: Some(SHADOW_SIZE),
                },
            },
            wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
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

        let mut pixels = Vec::with_capacity((SHADOW_SIZE * SHADOW_SIZE) as usize);
        for y in 0..SHADOW_SIZE {
            let base = (y * row) as usize;
            let line = &data[base..base + (SHADOW_SIZE * 4) as usize];
            for x in 0..SHADOW_SIZE as usize {
                pixels.push([
                    line[x * 4] as f32 / 255.0,
                    line[x * 4 + 1] as f32 / 255.0,
                    line[x * 4 + 2] as f32 / 255.0,
                    line[x * 4 + 3] as f32 / 255.0,
                ]);
            }
        }
        out.push(ShadowShot {
            pixels,
            passes: viewport.shadow_pass_count(),
        });
    }
    out
}

fn shadow_shot(gpu: &Gpu, scene: ShadowScene) -> ShadowShot {
    shadow_frames(gpu, &[scene]).pop().expect("one frame")
}

/// How many pixels got darker, and how many got brighter.
fn difference(before: &ShadowShot, after: &ShadowShot) -> (usize, usize) {
    assert_eq!(before.pixels.len(), after.pixels.len());
    let darker = before
        .pixels
        .iter()
        .zip(&after.pixels)
        .filter(|(a, b)| b[0] + 0.02 < a[0])
        .count();
    let brighter = before
        .pixels
        .iter()
        .zip(&after.pixels)
        .filter(|(a, b)| b[0] > a[0] + 0.02)
        .count();
    (darker, brighter)
}

#[test]
fn boxes_cast_shadows_on_the_floor() {
    // The whole point of the two commits, in one measurement: the same scene
    // with the light's `cast_shadow` on and off, and the difference has to be
    // darkness somewhere.
    //
    // A comparison rather than an absolute, because "is this pixel dark" has a
    // dozen innocent reasons to be true and "did turning shadows on darken it"
    // has one.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(&gpu, ShadowScene::default());
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            cast: true,
            ..ShadowScene::default()
        },
    );
    let (darker, brighter) = difference(&lit, &shadowed);

    // Nine boxes on a floor at this framing put around a thousand pixels of
    // the sixteen thousand into shadow. The bound is well under that and well
    // over what one box would give, which is the number that matters: a
    // renderer drawing every box from one model matrix piles all nine shadows
    // into one place, and the total collapses.
    assert!(
        darker > 400,
        "only {darker} pixels got darker when shadows were turned on"
    );
    // And nothing got *brighter*, which a bias pushed the wrong way would do.
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");
}

#[test]
fn the_lit_parts_of_the_scene_are_not_darkened() {
    // Shadow acne: a surface shadowing itself because its own depth compares as
    // nearer than what the map stored. It shows as a dirty stipple over
    // everything the light touches, and the bias is what prevents it.
    //
    // This needs the sun to be *tilted*: square to a surface, the map stores
    // the depth the fragment later compares, and reversed-Z's `>=` passes on
    // equality — so the first version of this test measured a scene in which
    // acne could not occur, and a sweep that deleted the bias entirely walked
    // straight through it.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(&gpu, ShadowScene::default());
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            cast: true,
            ..ShadowScene::default()
        },
    );
    let changed = lit
        .pixels
        .iter()
        .zip(&shadowed.pixels)
        .filter(|(a, b)| (a[0] - b[0]).abs() > 0.02)
        .count();
    let visible = lit.pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        changed * 3 < visible,
        "{changed} of {visible} lit pixels changed; that is acne, not a shadow"
    );
}

/// Which pixels one light's shadows darken.
///
/// Both shots have both lights in them and differ only in which one casts, so
/// what comes back is that light's shadows and nothing else. Compared with the
/// same scene and not with a brighter or dimmer one, because adding or removing
/// a light moves every pixel and buries the thing being measured.
fn shadow_mask(lit: &ShadowShot, shadowed: &ShadowShot) -> Vec<bool> {
    lit.pixels
        .iter()
        .zip(&shadowed.pixels)
        .map(|(a, b)| b[0] + 0.02 < a[0])
        .collect()
}

#[test]
fn two_casting_lights_shadow_two_different_places() {
    // Two casters need two layers of the array, and the layer index is the only
    // thing that says which. A shader that read layer zero for both would
    // compare the second light's depths against the first light's map.
    //
    // "Does it get darker" would not catch that — garbage compares dark often
    // enough. What catches it is *where*: the two suns are on opposite sides, so
    // their shadows fall on opposite sides of the boxes, and a second light
    // reading the first one's map can only produce the first one's shadow.
    let gpu = gpu_or_skip!(gpu());
    let neither = shadow_shot(
        &gpu,
        ShadowScene {
            second: true,
            ..ShadowScene::default()
        },
    );
    let key_only = shadow_shot(
        &gpu,
        ShadowScene {
            second: true,
            cast: true,
            ..ShadowScene::default()
        },
    );
    let fill_only = shadow_shot(
        &gpu,
        ShadowScene {
            second: true,
            second_casts: true,
            ..ShadowScene::default()
        },
    );
    let both = shadow_shot(
        &gpu,
        ShadowScene {
            second: true,
            cast: true,
            second_casts: true,
            ..ShadowScene::default()
        },
    );
    assert!(
        both.passes > key_only.passes,
        "the second caster added no maps: {} against {}",
        both.passes,
        key_only.passes
    );

    let key = shadow_mask(&neither, &key_only);
    let fill = shadow_mask(&neither, &fill_only);
    let together = shadow_mask(&neither, &both);
    let key_count = key.iter().filter(|x| **x).count();
    let fill_count = fill.iter().filter(|x| **x).count();
    assert!(key_count > 200, "the key casts nothing: {key_count}");
    assert!(fill_count > 100, "the fill casts nothing: {fill_count}");

    // The two lights shadow different places, so each one's shadows are its
    // own and not a copy of the other's.
    let shared = key.iter().zip(&fill).filter(|(a, b)| **a && **b).count();
    assert!(
        shared * 2 < fill_count.min(key_count),
        "{shared} of {fill_count} and {key_count} shadowed pixels coincide"
    );

    // And the shot with *both* lights casting has to contain both sets. This
    // is where the layer index is actually asked about: with a single caster
    // the layer is zero whatever the code does, so the two shots above agree
    // with a renderer that always reads layer zero. Only the picture that
    // needs two layers can tell.
    let fill_kept = fill
        .iter()
        .zip(&together)
        .filter(|(a, b)| **a && **b)
        .count();
    assert!(
        fill_kept * 10 > fill_count * 7,
        "with both lights casting only {fill_kept} of the fill's {fill_count}          shadowed pixels survive; the second light is reading the first's map"
    );
}

#[test]
fn a_light_that_casts_nothing_costs_no_shadow_map() {
    // `cast_shadow` is §2.2's switch. The shading shader checks it too, so a
    // renderer that ignored it here would draw exactly the same picture — and
    // pay a full pass over the scene's geometry for every fill light in the
    // setup. Only a count can see that, so there is one.
    let gpu = gpu_or_skip!(gpu());
    let idle = shadow_shot(
        &gpu,
        ShadowScene {
            idle: true,
            ..ShadowScene::default()
        },
    );
    assert_eq!(idle.passes, 0, "{} maps for no casters", idle.passes);

    let one = shadow_shot(
        &gpu,
        ShadowScene {
            cast: true,
            idle: true,
            ..ShadowScene::default()
        },
    );
    // More than none, and not a fixed number: a sun is split into as many
    // cascades as the view reaches, so what is being checked is the switch and
    // not the count.
    assert!(one.passes > 0, "a caster drew no map");
}

#[test]
fn more_casters_than_layers_are_dropped_and_not_wrapped() {
    // `MAX_SHADOWS` is a cap on passes, not on lights. The lights past it still
    // light the scene; what they lose is their shadow. The failure worth
    // catching is the other resolution — wrapping round to layer zero, so the
    // ninth light shadows the scene with the first light's map.
    let gpu = gpu_or_skip!(gpu());
    let crowd = shadow_shot(
        &gpu,
        ShadowScene {
            cast: true,
            crowd: true,
            // The cube is what takes the count past the cap: twelve suns and
            // six faces is eighteen frames for sixteen layers.
            point_inside: true,
            ..ShadowScene::default()
        },
    );
    assert_eq!(
        crowd.passes,
        kreate_render::shadow::MAX_SHADOWS,
        "{} maps for eighteen frames",
        crowd.passes
    );
    // And the scene is still lit by all of them: a light denied a map is not a
    // light denied.
    let brightest = crowd.pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(brightest > 0.2, "the crowd lit nothing: {brightest}");
}

#[test]
fn a_light_that_stops_casting_leaves_no_shadow_behind() {
    // The entries are reused frame to frame, so the ones the casters do not
    // write have to be reset — otherwise a light that stops casting keeps the
    // matrix it had, and the scene goes on being shadowed by a lamp the user
    // turned off.
    //
    // Two frames on *one* viewport. Every other test here builds a fresh one,
    // which is exactly why none of them could see this.
    let gpu = gpu_or_skip!(gpu());
    let frames = shadow_frames(
        &gpu,
        &[
            ShadowScene {
                cast: true,
                ..ShadowScene::default()
            },
            ShadowScene::default(),
        ],
    );
    assert_eq!(frames[1].passes, 0);

    // The second frame has to match a scene that never cast at all.
    let never = shadow_shot(&gpu, ShadowScene::default());
    let (darker, brighter) = difference(&never, &frames[1]);
    assert_eq!(
        (darker, brighter),
        (0, 0),
        "the turned-off light left {darker} shadowed pixels behind"
    );
}

/// How many pixels are neither fully lit nor fully dark, between two shots.
///
/// The width of a shadow's edge, counted. A hard edge is a step from one to the
/// other across a texel and leaves almost none of these; a soft one is made of
/// them.
fn edge_pixels(lit: &ShadowShot, shadowed: &ShadowShot) -> usize {
    lit.pixels
        .iter()
        .zip(&shadowed.pixels)
        .filter(|(a, b)| {
            let drop = a[0] - b[0];
            // Changed, but not all the way: the part of the shadow that is
            // partly in and partly out.
            drop > 0.02 && drop < a[0] * 0.8
        })
        .count()
}

#[test]
fn a_bigger_source_casts_a_softer_shadow() {
    // What `radius` and `angular_radius` are for, and what they did until now:
    // nothing. Both were packed and no shader read either, so the light panel
    // offered a control labelled "what makes a shadow soft" that made no
    // difference to any picture.
    //
    // Counted as the width of the edge rather than as blur, because those are
    // different things and only one of them is a penumbra: a blurrier shadow
    // is softer everywhere, and a bigger source is softer only where the
    // caster is far from what it falls on.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(&gpu, ShadowScene::default());

    let hard = shadow_shot(
        &gpu,
        ShadowScene {
            cast: true,
            source: 0.0,
            ..ShadowScene::default()
        },
    );
    let soft = shadow_shot(
        &gpu,
        ShadowScene {
            cast: true,
            // A source thirty times the sun's, which is roughly a bright
            // overcast sky and casts the shadow everybody recognises as soft.
            source: 30.0,
            ..ShadowScene::default()
        },
    );

    let hard_edge = edge_pixels(&lit, &hard);
    let soft_edge = edge_pixels(&lit, &soft);
    assert!(
        soft_edge > hard_edge * 2,
        "a source thirty times wider gave {soft_edge} edge pixels against          {hard_edge}"
    );

    // And it is still a shadow: a filter wide enough to wash it out would pass
    // the test above and fail the eye.
    let (darker, brighter) = difference(&lit, &soft);
    assert!(darker > 300, "the soft shadow faded away: {darker}");
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");
}

#[test]
fn a_shadow_is_sharp_where_it_touches_and_soft_where_it_does_not() {
    // The property that makes this a penumbra rather than a blur, and the only
    // one that tells the two apart. A box standing on a floor has its shadow
    // *at its base* however large the source is — there is no gap for the
    // source's size to spread across — and the further the shadow falls from
    // the box, the wider its edge.
    //
    // Lifting the boxes changes more than that, though: the shadow moves, and
    // with a tilted sun it can move onto a different part of the floor or
    // behind another box. Counting the edge alone would be measuring that as
    // much as the penumbra.
    //
    // So the same change is made twice, once with a source that has size and
    // once with one that has none, and it is the *difference between the two
    // differences* that is the penumbra. Everything geometric cancels: the hard
    // pair moves exactly as far, onto exactly the same floor.
    let gpu = gpu_or_skip!(gpu());
    let edge_at = |source: f32, lifted: bool| {
        let base = ShadowScene {
            source,
            lifted,
            ..ShadowScene::default()
        };
        let lit = shadow_shot(&gpu, base);
        let shadowed = shadow_shot(&gpu, ShadowScene { cast: true, ..base });
        let (darker, _) = difference(&lit, &shadowed);
        assert!(darker > 100, "nothing to measure at {source}/{lifted}");
        edge_pixels(&lit, &shadowed) as f32
    };

    // As a ratio and not a difference. A hard shadow's edge grows when the
    // casters are lifted too — the shadow is further away and covers more of
    // the frame — and that growth is pure geometry. Anything a soft one does
    // *above* it is the penumbra.
    //
    // A difference would not do: pinning the filter at its widest makes the
    // touching shadow already as soft as it can be, so it has less room to grow
    // and the difference comes out *smaller* than the correct one rather than
    // larger. The ratio falls below the hard shadow's, which is the shape of
    // the failure and what this compares.
    let hard = edge_at(0.0, true) / edge_at(0.0, false);
    let soft = edge_at(30.0, true) / edge_at(30.0, false);
    assert!(
        soft > hard * 1.05,
        "lifting the casters grew a soft shadow's edge {soft}-fold and a hard          one's {hard}-fold; the penumbra does not grow with distance"
    );
}

#[test]
fn past_the_cascades_a_sun_stops_shadowing_rather_than_going_black() {
    // A floor stretching a kilometre away. The sun's slices cover the first
    // three hundred metres of it and the rest has no map at all — which is a
    // decision, not an oversight: out there a shadow is a few pixels across and
    // a map fitted to cover it would be a metre a texel.
    //
    // What must not happen is the far floor going black at the boundary. That
    // is the same call the sampler makes outside any frustum, and this is the
    // one scene in the file that reaches it: everywhere else the frames are
    // fitted to the scene, so nothing lit is ever outside them.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(
        &gpu,
        ShadowScene {
            long_floor: true,
            ..ShadowScene::default()
        },
    );
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            long_floor: true,
            cast: true,
            ..ShadowScene::default()
        },
    );

    let (darker, brighter) = difference(&lit, &shadowed);
    assert!(
        darker > 100,
        "nothing near the camera is shadowed: {darker}"
    );
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");

    // The far floor is most of the frame at this framing, so treating it as
    // shadowed would be most of the picture rather than a corner of it.
    let visible = lit.pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        darker * 4 < visible,
        "{darker} of {visible} lit pixels went dark; past the last cascade was          treated as shadowed rather than lit"
    );
}

#[test]
fn a_lamp_in_a_tile_that_is_not_the_first_still_lights_it() {
    // What a picture can say about the tiled path, which is less than it looks.
    //
    // The list is a pure optimisation: the falloff's window already takes a
    // lamp to zero past its range, so a lamp wrongly *in* a tile costs time and
    // changes nothing. Only a lamp wrongly *left out* shows, and whether the
    // cull leaves the right ones out is checked directly in `tests/tiles.rs` by
    // reading the lists back.
    //
    // What is left for a render to check is the other end: that the shading
    // pass looks up the tile the fragment is actually in. Two lamps in separate
    // pools, neither of them in the corner tile — so a pass that read one
    // tile's list for every fragment loses them and the picture goes dark.
    let gpu = gpu_or_skip!(gpu());
    let shot = shadow_shot(
        &gpu,
        ShadowScene {
            pools: true,
            ..ShadowScene::default()
        },
    );

    let lit = shot.pixels.iter().filter(|p| p[0] > 0.02).count();
    let dark = shot.pixels.iter().filter(|p| p[0] <= 0.02).count();
    assert!(lit > 100, "the pools lit nothing: {lit}");
    assert!(dark > 100, "the pools lit everything: {dark} dark pixels");

    // And on both sides of the frame, which is the part a single tile's list
    // cannot satisfy: whichever tile it were, one of the two lamps would be
    // missing from it.
    let half = SHADOW_SIZE as usize / 2;
    let left = shot
        .pixels
        .iter()
        .enumerate()
        .filter(|(i, p)| (*i % (SHADOW_SIZE as usize)) < half && p[0] > 0.02)
        .count();
    let right = lit - left;
    assert!(left > 30 && right > 30, "lit {left} left and {right} right");
}

#[test]
fn a_spot_casts_through_a_perspective_frustum() {
    // The only perspective shadow on the device. A sun's frustum is
    // orthographic and the point light overhead casts nothing, so without this
    // the perspective projection — and the texel that has to grow with
    // distance, which an orthographic one never does — were checked by
    // arithmetic and never by a picture.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(
        &gpu,
        ShadowScene {
            spot: true,
            ..ShadowScene::default()
        },
    );
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            spot: true,
            cast: true,
            ..ShadowScene::default()
        },
    );
    assert_eq!(shadowed.passes, 1);

    let (darker, brighter) = difference(&lit, &shadowed);
    assert!(darker > 200, "the spot casts nothing: {darker} pixels");
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");

    // Acne is the failure worth watching for here: a perspective frustum's
    // texels are not all one size, and a fix that assumed they were stipples
    // whatever is furthest from the lamp.
    let visible = lit.pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        darker * 3 < visible,
        "{darker} of {visible} lit pixels went dark; that is acne, not a shadow"
    );
}

#[test]
fn a_frame_does_not_inherit_the_last_one() {
    // The entries live across frames, so the ones this frame's casters do not
    // write have to be reset. The general form of that, which is also the
    // strongest: a scene rendered on a viewport that has just rendered
    // something else has to come out the same as one rendered on a fresh
    // viewport.
    //
    // The two scenes differ by the *front* light starting to cast, which pushes
    // every other caster one place down the queue for the eight maps there are.
    // The last one loses its map and goes on lighting the scene, and that is
    // the only way a left-over entry can be seen at all: everywhere else a
    // light with no frame is a light that reaches nothing — `spot_view`
    // returns nothing exactly when the scene is behind the lamp, which is also
    // when the lamp illuminates none of it.
    let gpu = gpu_or_skip!(gpu());
    let first = ShadowScene {
        cast: true,
        crowd: true,
        idle: true,
        ..ShadowScene::default()
    };
    let second = ShadowScene {
        idle_casts: true,
        ..first
    };
    let pair = shadow_frames(&gpu, &[first, second]);
    let fresh = shadow_shot(&gpu, second);
    assert_eq!(pair[1].passes, fresh.passes);
    let differing = pair[1]
        .pixels
        .iter()
        .zip(&fresh.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "{differing} pixels differ from the same scene on a fresh viewport"
    );
}

#[test]
fn a_lamp_swung_away_from_the_scene_stops_shadowing_it() {
    // The reachable way for a light to keep casting while its frustum stops
    // seeing anything: a spot turned to face the ceiling. `spot_view` has
    // nothing to frame and returns nothing, so the light's entry has to go back
    // to empty — and if the entries are filled in without being reset first,
    // it keeps the matrix it had and goes on shadowing the scene from where it
    // used to point.
    //
    // Two frames on *one* viewport, because that is the only place a stale
    // entry can exist. Turning `cast_shadow` off would not do: the shading
    // shader checks that too, so the picture would come out right for the
    // wrong reason.
    let gpu = gpu_or_skip!(gpu());
    let frames = shadow_frames(
        &gpu,
        &[
            ShadowScene {
                spot: true,
                cast: true,
                ..ShadowScene::default()
            },
            ShadowScene {
                spot_away: true,
                cast: true,
                ..ShadowScene::default()
            },
        ],
    );
    assert_eq!(frames[0].passes, 1);
    assert_eq!(
        frames[1].passes, 0,
        "a lamp aimed at nothing rendered a map"
    );

    let never = shadow_shot(
        &gpu,
        ShadowScene {
            spot_away: true,
            ..ShadowScene::default()
        },
    );
    let (darker, brighter) = difference(&never, &frames[1]);
    assert_eq!(
        (darker, brighter),
        (0, 0),
        "the lamp left {darker} shadowed pixels behind after turning away"
    );
}

#[test]
fn only_the_rendered_mode_pays_for_shadows() {
    // The other shading modes light themselves, so a frame fitted for them is
    // a full pass over the scene's geometry per lamp with no effect on the
    // picture at all. Invisible in the pixels, by construction — so it is a
    // count, like `cast_shadow`.
    let gpu = gpu_or_skip!(gpu());
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SHADOW_SIZE, SHADOW_SIZE);
    // Shadows are measured by their edges and their light, and the glow
    // around bright light softens edges and the occlusion darkens corners —
    // different claims, with their own tests.
    viewport.show_bloom = false;
    viewport.ambient_occlusion = false;
    viewport.shading = ShadingMode::Studio;
    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
    let camera = Camera::default();
    let mut draws = vec![Draw::new(&cube, glam::DAffine3::IDENTITY)];
    viewport.set_lights(
        &gpu.device,
        &gpu.queue,
        camera.eye(),
        &shadow_lights(ShadowScene {
            cast: true,
            ..ShadowScene::default()
        }),
    );
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    assert_eq!(
        viewport.shadow_pass_count(),
        0,
        "the studio mode rendered a shadow map"
    );
}

#[test]
fn a_point_light_overhead_casts_a_shadow() {
    // A lamp hung over a model, which is where anybody puts the first one they
    // add. It used to cast nothing at all: a point light was given one of its
    // six faces and that face looks along +X, so a lamp above a scene pointed
    // its only map sideways and caught nothing. The pass was drawn and the
    // picture was identical.
    //
    // Six faces now, and only the ones with anything in them — a lamp over a
    // floor pays for the four or five that see it and not for the one aimed at
    // empty sky.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(
        &gpu,
        ShadowScene {
            point: true,
            ..ShadowScene::default()
        },
    );
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            point: true,
            cast: true,
            ..ShadowScene::default()
        },
    );
    assert!(
        shadowed.passes > 1,
        "a point light drew {} maps; a cube is more than one",
        shadowed.passes
    );

    let (darker, brighter) = difference(&lit, &shadowed);
    assert!(
        darker > 200,
        "the lamp overhead casts nothing: {darker} pixels"
    );
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");

    // And not acne: a cube's faces meet at right angles and a bias that worked
    // on one of them has to work on all six.
    let visible = lit.pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        darker * 3 < visible,
        "{darker} of {visible} lit pixels went dark; that is acne, not a shadow"
    );

    // The perspective half of the penumbra arithmetic is *not* measured here,
    // and that is recorded rather than faked. A picture test for it was
    // written, passed with the filter forced to one texel, and was therefore
    // measuring something other than the penumbra — so it went, rather than
    // staying as a test that agrees with whatever the code does. The arithmetic
    // is a closed form and is checked exactly in `shadow::tests`, which is the
    // better place for it; a picture test wants a fixture built for it.
}

#[test]
fn a_lamp_among_the_boxes_shadows_in_every_direction() {
    // The cube proper. A lamp between the boxes and at their height has scene
    // on every side of it, so its six frames are all used and the sampler has
    // to pick the right one per fragment — which a lamp hung overhead never
    // asks, because nearly everything it lights is on one face.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(
        &gpu,
        ShadowScene {
            point_inside: true,
            ..ShadowScene::default()
        },
    );
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            point_inside: true,
            cast: true,
            ..ShadowScene::default()
        },
    );
    assert_eq!(
        shadowed.passes, 6,
        "a lamp surrounded by scene used {} faces",
        shadowed.passes
    );

    let (darker, brighter) = difference(&lit, &shadowed);
    assert!(
        darker > 300,
        "the lamp casts almost nothing: {darker} pixels"
    );
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");

    // A face picked wrongly compares a fragment against a map drawn from
    // somewhere else, which darkens far more of the picture than a shadow
    // does. The same bound acne is caught by, and it catches this for the same
    // reason: both are "most of what is lit went dark".
    let visible = lit.pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        darker * 3 < visible,
        "{darker} of {visible} lit pixels went dark; that is a face picked          wrongly, not a shadow"
    );
}

#[test]
fn a_light_denied_a_face_is_lit_where_it_has_none() {
    // The decision the sampler makes for a face with no frame, and the one
    // arrangement that still reaches it now that a point light gets a cube: the
    // layer budget running out part way through one. The faces that were
    // afforded still shadow; the rest are lit rather than black, for the same
    // reason as outside a frustum — the map says nothing about them, and a hard
    // black edge where the budget ran out would read as a bug and be one.
    let gpu = gpu_or_skip!(gpu());
    let lit = shadow_shot(
        &gpu,
        ShadowScene {
            point_inside: true,
            crowd: true,
            ..ShadowScene::default()
        },
    );
    let shadowed = shadow_shot(
        &gpu,
        ShadowScene {
            point_inside: true,
            cast: true,
            crowd: true,
            ..ShadowScene::default()
        },
    );
    // Fifteen suns took their layer each before the lamp asked, so one face of
    // its cube is drawn and five are denied. That is the arrangement, and it is
    // worth pinning because it is the only one that reaches the decision below
    // across enough of the picture to be measurable.
    assert_eq!(shadowed.passes, kreate_render::shadow::MAX_SHADOWS);

    // The one face it was given points away from most of the scene, so it casts
    // almost nothing — and that is the point rather than a disappointment: five
    // sixths of what this lamp lights has no map behind it, which is exactly
    // the region the decision below governs.

    // So the property, stated directly: a lamp with almost no map darkens
    // almost nothing. Treating a denied face as shadowed instead removes this
    // lamp's whole contribution over five sixths of what it lights — a quarter
    // of the light there, across a thousand-odd pixels.
    //
    // A count rather than a fraction of the frame, because a fraction is the
    // bound that let this through: a third of the picture is a lot of room for
    // a hard black edge to hide in.
    let (darker, brighter) = difference(&lit, &shadowed);
    assert_eq!(brighter, 0, "{brighter} pixels got brighter");
    assert!(
        darker < 100,
        "{darker} pixels went dark; the faces that were denied a map were          treated as shadowed rather than lit"
    );
}

#[test]
fn ending_a_light_where_it_goes_black_changes_no_pixel() {
    // The claim the culling will rest on, checked where it matters — in the
    // picture, byte for byte.
    //
    // Every lamp now carries a distance past which it is not computed, chosen
    // from this viewport's own display: the point below which a value comes out
    // as black whatever else is going on. A window brings the falloff to
    // exactly zero there, so nothing is dropped abruptly. If any of that is off
    // — the range too short, the window biting too early, the black point
    // measured before the transfer function instead of after — a pixel moves,
    // and this notices.
    //
    // Rendered twice: once with the ranges the viewport works out, and once
    // with ranges so large the window cannot reach the scene at all, which is
    // the same arithmetic as having no cutoff.
    // The spot, because a sun does not diverge and so has no end to be given:
    // in a scene lit only by parallel light this would be comparing two
    // identical arrangements and proving nothing.
    let gpu = gpu_or_skip!(gpu());
    let lit = ShadowScene {
        spot: true,
        cast: true,
        ..ShadowScene::default()
    };
    let cut = shadow_shot(
        &gpu,
        ShadowScene {
            derived_range: true,
            ..lit
        },
    );
    let uncut = shadow_shot(
        &gpu,
        ShadowScene {
            enormous_range: true,
            ..lit
        },
    );

    let moved = cut
        .pixels
        .iter()
        .zip(&uncut.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        moved, 0,
        "{moved} pixels moved when the lights were given ends"
    );

    // And this rig can see a range when one bites, which is the half that
    // stops the assertion above from being satisfied by measuring nothing. It
    // needs saying: a lamp's derived range is the distance past which it cannot
    // change a pixel, and for a bright lamp under a shot exposure that is
    // hundreds of metres — far outside a scene this size. The cutoff is exact
    // and it is also, at viewport scale, rarely reached.
    let cropped = shadow_shot(
        &gpu,
        ShadowScene {
            short_range: true,
            ..lit
        },
    );
    let changed = cropped
        .pixels
        .iter()
        .zip(&uncut.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        changed > 100,
        "a range shorter than the scene changed {changed} pixels; this rig          cannot see a cutoff at all"
    );
}

#[test]
fn a_lit_scene_comes_out_lit() {
    // The assembly, at its bluntest. Every ingredient was already checked
    // against the CPU; this says they were wired together.
    let gpu = gpu_or_skip!(gpu());
    let shot = shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(20_000.0)],
        FRAMED,
    );
    let shot = summary(&shot);
    assert!(shot.brightest > 0.1, "nothing was lit: {}", shot.brightest);
    // The cube covers a good part of a 64-pixel frame.
    assert!(shot.lit > 400, "only {} pixels were lit", shot.lit);
}

#[test]
fn an_unlit_scene_is_black_and_not_grey() {
    // A renderer that supplied a default light, or that left the studio
    // shading in the pass, would light this — and the user would have no way
    // to turn it off. Black is the honest answer for a scene with no lights.
    let gpu = gpu_or_skip!(gpu());
    let shot = shoot(&gpu, ShadingMode::Rendered, grey(), &[], FRAMED);
    let shot = summary(&shot);
    assert!(
        shot.brightest < 0.01,
        "an unlit scene was lit: {}",
        shot.brightest
    );
}

#[test]
fn the_display_transform_is_applied_and_applied_once() {
    // The clue that it ran at all: a sun at twenty thousand lux on a grey
    // surface is far outside the display's range, so an unresolved image
    // would read as saturated white everywhere the cube is. AgX leaves
    // headroom.
    let gpu = gpu_or_skip!(gpu());
    let shot = shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(20_000.0)],
        FRAMED,
    );
    let shot = summary(&shot);
    assert!(
        shot.brightest > 0.5,
        "a sunlit surface should be bright: {}",
        shot.brightest
    );
    // The measurement the format makes possible: a linear image of a surface
    // under twenty thousand lux is thousands of times over the display's
    // range, so reaching the target unresolved would clip nearly every pixel
    // of the cube. A transform that ran leaves a gradient.
    assert!(
        shot.saturated * 4 < shot.lit,
        "{} of {} lit pixels are clipped; the image looks unresolved",
        shot.saturated,
        shot.lit
    );

    // And a dimmer scene is dimmer, which a transform applied twice or not at
    // all would both still manage — this is the weaker half, and it is here
    // because a transform that ignored its input would pass everything above.
    let dim = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(200.0)],
        FRAMED,
    ));
    assert!(dim.brightest < shot.brightest);
    assert!(dim.brightest > 0.0);
}

#[test]
fn exposure_moves_the_picture() {
    // The uniform reaches the fragment stage of the resolve, through a
    // viewport that owns the buffer rather than allocating one per frame.
    let gpu = gpu_or_skip!(gpu());
    let flat = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(2_000.0)],
        FRAMED,
    ))
    .brightest;
    let raised = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(2_000.0)],
        Display {
            view: kreate_math::ViewTransform::AgX,
            exposure: FRAMED.exposure + 2.0,
        },
    ))
    .brightest;
    assert!(raised > flat * 1.05, "{flat} then {raised}");
}

#[test]
fn the_material_reaches_the_shader() {
    // Per draw, through the ring the pass reads. A pipeline that ignored it
    // would shade everything as the default grey, and a scene of one object
    // would look correct.
    let gpu = gpu_or_skip!(gpu());
    let light = [key_light(5_000.0)];
    let dark = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        Surface {
            base_color: kreate_math::SceneRgb::new(0.05, 0.05, 0.05),
            metallic: 0.0,
            roughness: 0.5,
        },
        &light,
        FRAMED,
    ))
    .brightest;
    let bright = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        Surface {
            base_color: kreate_math::SceneRgb::new(0.9, 0.9, 0.9),
            metallic: 0.0,
            roughness: 0.5,
        },
        &light,
        FRAMED,
    ))
    .brightest;
    assert!(bright > dark * 1.3, "{dark} against {bright}");
}

#[test]
fn the_three_modes_are_three_different_pictures() {
    // Guards every test above: if the mode were ignored and the studio pass
    // ran throughout, the scene would be lit, the transform would appear to
    // clamp, and only this would notice.
    let gpu = gpu_or_skip!(gpu());
    let light = [key_light(20_000.0)];
    let studio = summary(&shoot(&gpu, ShadingMode::Studio, grey(), &light, FRAMED));
    let flat = summary(&shoot(&gpu, ShadingMode::Flat, grey(), &light, FRAMED));
    let rendered = summary(&shoot(&gpu, ShadingMode::Rendered, grey(), &light, FRAMED));

    // All three draw the cube.
    for (name, shot) in [
        ("studio", &studio),
        ("flat", &flat),
        ("rendered", &rendered),
    ] {
        assert!(shot.lit > 400, "{name} drew {} pixels", shot.lit);
        assert!(shot.brightest > 0.0, "{name} drew nothing");
    }
    // And the rendered one is not either of the others: the studio pass
    // ignores the scene's lights and this one is lit by a sun through a
    // display transform.
    assert!(
        (rendered.brightest - studio.brightest).abs() > 0.02,
        "rendered {} against studio {}",
        rendered.brightest,
        studio.brightest
    );
}

#[test]
fn a_resize_keeps_the_intermediate_in_step() {
    // The bind group holds the old view, so a resize that remade the texture
    // and not the bind group would resolve last frame's image at the new
    // size — a picture, and the wrong one.
    let gpu = gpu_or_skip!(gpu());
    let mut viewport = Viewport::new(&gpu.device, FORMAT, 32, 32);
    viewport.shading = ShadingMode::Rendered;
    viewport.resize(&gpu.device, 96, 48);

    // Drawing after the resize must not be a validation error, which is what
    // a stale bind group would be once the texture behind it was dropped.
    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
    // Looking at the origin from six metres, which frames a two-metre cube.
    let camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY)];
    viewport.set_lights(
        &gpu.device,
        &gpu.queue,
        camera.eye(),
        &[key_light(10_000.0)],
    );
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resized"),
        size: wgpu::Extent3d {
            width: 96,
            height: 48,
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
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
}

#[test]
fn more_lights_than_the_buffer_started_with_still_light_the_scene() {
    // The buffer grows by being replaced, and the bind group holds the old
    // one. A grow without a rebuild is a validation error at best and last
    // frame's lights at worst.
    let gpu = gpu_or_skip!(gpu());
    let many: Vec<PlacedLight> = (0..24)
        .map(|i| PlacedLight {
            position: glam::DVec3::new(i as f64 * 0.1 - 1.2, -4.0, 2.0),
            amount: [400.0; 3],
            ..PlacedLight::default()
        })
        .collect();
    let shot = shoot(&gpu, ShadingMode::Rendered, grey(), &many, FRAMED);
    let shot = summary(&shot);
    assert!(
        shot.brightest > 0.05,
        "the grown buffer lit nothing: {}",
        shot.brightest
    );
    assert!(shot.lit > 400, "only {} pixels were lit", shot.lit);
}

/// Bloom: light past white spills into the dark around it — the background
/// beside a hard-lit cube is brighter with the glow than without — and a
/// picture with nothing past white is left as it was.
#[test]
fn light_past_white_glows_and_the_rest_does_not() {
    let gpu = gpu_or_skip!(gpu());
    let shot = |bloom: bool, lux: f32| {
        let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
        viewport.shading = ShadingMode::Rendered;
        viewport.show_bloom = bloom;
        viewport.show_grid = false;
        viewport.show_wireframe = false;
        viewport.set_display(&gpu.queue, FRAMED);
        let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
        let uploaded =
            kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
        let camera = Camera {
            distance: 6.0,
            ..Camera::default()
        };
        one_material(&gpu, &mut viewport, grey());
        let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY)];
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[key_light(lux)]);
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        read_frame(&gpu, &mut viewport, &draws)
    };
    // Far past white: the glow brightens the background around the cube.
    let (plain, glowing) = (shot(false, 2_000_000.0), shot(true, 2_000_000.0));
    let lift: f32 = plain
        .iter()
        .zip(&glowing)
        .map(|(a, b)| (b[0] - a[0]).max(0.0))
        .sum();
    assert!(lift > 1.0, "the glow reached nothing: {lift}");
    // Nothing past white: the same picture either way.
    let (dim_plain, dim_glow) = (shot(false, 2.0), shot(true, 2.0));
    let most: f32 = dim_plain
        .iter()
        .zip(&dim_glow)
        .map(|(a, b)| (a[0] - b[0]).abs())
        .fold(0.0, f32::max);
    assert!(most < 0.02, "a dim picture changed by {most}");
}

/// Draws the prepared viewport and reads the frame back.
fn read_frame(gpu: &Gpu, viewport: &mut Viewport, draws: &[Draw]) -> Vec<[f32; 4]> {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
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
    viewport.render(&mut encoder, &view, draws);
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
    (0..SIZE as usize)
        .flat_map(|y| {
            let line = data[y * row as usize..y * row as usize + (SIZE * 4) as usize].to_vec();
            (0..SIZE as usize).map(move |x| {
                [
                    line[x * 4] as f32 / 255.0,
                    line[x * 4 + 1] as f32 / 255.0,
                    line[x * 4 + 2] as f32 / 255.0,
                    line[x * 4 + 3] as f32 / 255.0,
                ]
            })
        })
        .collect()
}

/// Screen-space reflections: a mirror floor under a lit cube shows the cube
/// in it — the floor changes where the reflection falls — and with them off
/// it does not.
#[test]
fn a_mirror_floor_reflects_what_stands_on_it() {
    let gpu = gpu_or_skip!(gpu());
    let shot = |reflections: bool| {
        let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
        viewport.shading = ShadingMode::Rendered;
        viewport.show_reflections = reflections;
        viewport.show_bloom = false;
        viewport.ambient_occlusion = false;
        viewport.show_grid = false;
        viewport.show_wireframe = false;
        viewport.set_display(&gpu.queue, FRAMED);
        let cube = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.0));
        let floor = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(12.0, 2, 2));
        let cube =
            kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cube, glam::Vec4::ONE);
        let floor =
            kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor, glam::Vec4::ONE);
        // A matte cube, lit, on a polished metal floor — whose picture is all
        // reflection: black sky and a lit cube.
        let compile = |base: f32, metallic: f32, roughness: f32| {
            let graph = kreate_scene::shader::ShaderGraph::standard(
                "test",
                kreate_math::SceneRgb::new(base, base, base),
                metallic,
                roughness,
            );
            kreate_shader::compile(&graph).expect("compiles")
        };
        let (matte, mirror) = (compile(0.8, 0.0, 0.5), compile(0.95, 1.0, 0.05));
        fn source(c: &kreate_shader::Compiled) -> kreate_render::viewport::MaterialSource<'_> {
            kreate_render::viewport::MaterialSource {
                wgsl: &c.wgsl,
                constants: &c.constants,
                image_slots: &c.image_slots,
                images: &[],
            }
        }
        viewport.set_materials(&gpu.device, &gpu.queue, &[source(&matte), source(&mirror)]);
        let camera = Camera {
            pivot: glam::DVec3::new(0.0, 0.0, 0.3),
            distance: 5.0,
            rotation: glam::DQuat::from_rotation_x(1.15),
            ..Camera::default()
        };
        let mut draws = vec![
            Draw::new(&floor, glam::DAffine3::IDENTITY).with_material(1),
            Draw::new(
                &cube,
                glam::DAffine3::from_translation(glam::DVec3::new(0.0, 0.0, 0.8)),
            ),
        ];
        viewport.set_lights(
            &gpu.device,
            &gpu.queue,
            camera.eye(),
            &[key_light(20_000.0)],
        );
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        read_frame(&gpu, &mut viewport, &draws)
    };
    let (without, with) = (shot(false), shot(true));
    let changed = without
        .iter()
        .zip(&with)
        .filter(|(a, b)| (0..3).any(|i| (a[i] - b[i]).abs() > 0.04))
        .count();
    assert!(changed > 40, "the reflection changed {changed} pixels");
    // The top rows are the far floor and the sky behind it, which nothing is
    // reflected into from this angle: left alone.
    let top: usize = (0..SIZE as usize * 4)
        .filter(|i| (0..3).any(|c| (without[*i][c] - with[*i][c]).abs() > 0.04))
        .count();
    assert!(top < 8, "{top} pixels changed in the far rows");
}

/// The cube on a floor under a sun, rasterised or path traced over `frames`
/// frames, through `lens` when it has one.
fn traced_shot(
    gpu: &Gpu,
    traced: bool,
    frames: u32,
    lens: Option<kreate_render::pathtrace::Lens>,
) -> Vec<[f32; 4]> {
    traced_shot_with(gpu, traced, frames, lens, false)
}

fn traced_shot_with(
    gpu: &Gpu,
    traced: bool,
    frames: u32,
    lens: Option<kreate_render::pathtrace::Lens>,
    denoise: bool,
) -> Vec<[f32; 4]> {
    let grey = kreate_render::pathtrace::PtSurface {
        base_color: [0.8, 0.8, 0.8],
        metallic: 0.0,
        roughness: 0.5,
        ..Default::default()
    };
    traced_scene(gpu, traced, frames, lens, denoise, grey, &[])
}

#[allow(clippy::too_many_arguments)]
fn traced_scene(
    gpu: &Gpu,
    traced: bool,
    frames: u32,
    lens: Option<kreate_render::pathtrace::Lens>,
    denoise: bool,
    surface: kreate_render::pathtrace::PtSurface,
    images: &[kreate_render::pathtrace::PtImage<'_>],
) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.path_trace = traced;
    viewport.path_trace_denoise = denoise;
    viewport.path_trace_lens = lens;
    viewport.contact_shadows = CONTACT.with(|c| c.get());
    viewport.show_bloom = false;
    viewport.show_reflections = false;
    viewport.ambient_occlusion = false;
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.set_display(&gpu.queue, FRAMED);
    let cube_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.0));
    let floor_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(12.0, 2, 2));
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cube_cpu, glam::Vec4::ONE);
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_cpu, glam::Vec4::ONE);
    let cube_kind = CUBE.with(|c| c.get());
    if cube_kind == CubeKind::Glass {
        let floor_graph =
            kreate_scene::shader::ShaderGraph::standard("floor", grey().base_color, 0.0, 0.5);
        let mut glass_graph = floor_graph.clone();
        if let kreate_scene::shader::NodeKind::Surface {
            base_color,
            transmission,
            roughness,
            ..
        } = &mut glass_graph.nodes[0].kind
        {
            *base_color = kreate_math::SceneRgb::new(1.0, 1.0, 1.0);
            *transmission = GLASS.with(|g| g.get());
            *roughness = 0.05;
        }
        let compiled = [
            kreate_shader::compile(&floor_graph).expect("compiles"),
            kreate_shader::compile(&glass_graph).expect("compiles"),
        ];
        let sources: Vec<_> = compiled
            .iter()
            .map(|c| kreate_render::viewport::MaterialSource {
                wgsl: &c.wgsl,
                constants: &c.constants,
                image_slots: &c.image_slots,
                images: &[],
            })
            .collect();
        viewport.set_materials(&gpu.device, &gpu.queue, &sources);
        viewport.set_transmissive(&[0.0, GLASS.with(|g| g.get())]);
    } else {
        one_material(gpu, &mut viewport, grey());
    }
    let lift = if cube_kind == CubeKind::None {
        glam::DAffine3::from_scale_rotation_translation(
            glam::DVec3::splat(1e-6),
            glam::DQuat::IDENTITY,
            glam::DVec3::new(0.0, 0.0, 0.5),
        )
    } else {
        glam::DAffine3::from_translation(glam::DVec3::new(0.0, 0.0, 0.5))
    };
    let camera = Camera {
        pivot: glam::DVec3::new(0.0, 0.0, 0.3),
        distance: 5.0,
        rotation: glam::DQuat::from_rotation_x(1.0),
        ..Camera::default()
    };
    let parts = [(&cube_cpu, lift), (&floor_cpu, glam::DAffine3::IDENTITY)];
    let uvs: Vec<Vec<[f32; 2]>> = parts.iter().map(|(m, _)| m.uvs.clone()).collect();
    let data: Vec<(Vec<[f32; 3]>, Vec<glam::Vec3>)> = parts
        .iter()
        .map(|(m, _)| {
            (
                m.positions.iter().map(|p| p.0).collect(),
                m.normals
                    .iter()
                    .map(|n| kreate_render::mesh_gpu::oct_decode(*n))
                    .collect(),
            )
        })
        .collect();
    let meshes: Vec<kreate_render::pathtrace::PtMesh<'_>> = parts
        .iter()
        .zip(&data)
        .zip(&uvs)
        .map(|(((m, t), (p, n)), uv)| kreate_render::pathtrace::PtMesh {
            positions: p,
            uvs: uv,
            normals: n,
            indices: &m.indices,
            transform: *t,
            surface: 0,
            runs: &[],
        })
        .collect();
    let mut meshes = meshes;
    if cube_kind == CubeKind::Glass {
        meshes[0].surface = 1;
    }
    let glass = kreate_render::pathtrace::PtSurface {
        base_color: [1.0, 1.0, 1.0],
        roughness: 0.05,
        transmission: GLASS.with(|g| g.get()),
        ior: 1.5,
        ..Default::default()
    };
    viewport.set_path_trace_scene(
        &gpu.device,
        &gpu.queue,
        &meshes,
        &[surface, glass],
        images,
        &Default::default(),
    );
    let mut draws = vec![
        Draw::new(&cube, lift),
        Draw::new(&floor, glam::DAffine3::IDENTITY),
    ];
    if cube_kind == CubeKind::Glass {
        draws[0].material = 1;
    }
    let mut out = Vec::new();
    for _ in 0..frames {
        // Faint lamps far off first, when a test asks: each takes a shadow
        // map, so the sun after them is left with none.
        let mut lights: Vec<PlacedLight> = (0..DECOYS.with(|d| d.get()))
            .map(|i| PlacedLight {
                falloff: Falloff::InverseSquare,
                position: glam::DVec3::new(40.0 + i as f64, 30.0, 20.0),
                amount: [0.01; 3],
                ..PlacedLight::default()
            })
            .collect();
        lights.push(key_light(20_000.0));
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &lights);
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        out = read_frame(gpu, &mut viewport, &draws);
    }
    out
}

/// The path tracer: the same scene, the same lights, the same units — so its
/// picture is about as bright as the rasteriser's — and light reaches what the
/// lamp does not, by bouncing off what it does.
#[test]
fn the_path_tracer_agrees_with_the_rasteriser_and_bounces_light() {
    let gpu = gpu_or_skip!(gpu());
    let raster = traced_shot(&gpu, false, 1, None);
    let traced = traced_shot(&gpu, true, 64, None);
    let mean =
        |p: &Vec<[f32; 4]>| p.iter().map(|c| c[0] + c[1] + c[2]).sum::<f32>() / p.len() as f32;
    let (r, t) = (mean(&raster), mean(&traced));
    assert!(t > 0.05, "the traced picture is black: {t}");
    // The same light in the same units: within a factor of two overall — the
    // tracer adds the bounced light the rasteriser does not have.
    assert!(
        t > r * 0.5 && t < r * 2.0,
        "traced {t} against rasterised {r}"
    );
}

/// Hard steps between neighbouring pixels: what a sharp edge is made of.
fn sharp_steps(p: &[[f32; 4]]) -> usize {
    let lum = |c: [f32; 4]| c[0] + c[1] + c[2];
    let w = SIZE as usize;
    (0..p.len())
        .filter(|i| i % w + 1 < w && (lum(p[*i]) - lum(p[i + 1])).abs() > 0.3)
        .count()
}

/// Depth of field: through a wide lens focused far beyond it, the cube's
/// edges blur — fewer sharp steps than through a pinhole.
#[test]
fn a_wide_lens_focused_away_blurs_the_cube() {
    let gpu = gpu_or_skip!(gpu());
    let pinhole = traced_shot(&gpu, true, 96, None);
    let lens = kreate_render::pathtrace::Lens {
        radius: 0.25,
        focus: 40.0,
    };
    let blurred = traced_shot(&gpu, true, 96, Some(lens));
    let (p, b) = (sharp_steps(&pinhole), sharp_steps(&blurred));
    assert!(p > 10, "the pinhole picture has no edges to blur: {p}");
    assert!(
        b * 2 < p,
        "{b} sharp steps through the lens against {p} through a pinhole"
    );
}

/// Denoising: eight samples, filtered, are nearer the converged picture than
/// eight samples raw.
#[test]
fn a_young_mean_denoised_is_nearer_the_converged_picture() {
    let gpu = gpu_or_skip!(gpu());
    let converged = traced_shot_with(&gpu, true, 256, None, false);
    let raw = traced_shot_with(&gpu, true, 8, None, false);
    let filtered = traced_shot_with(&gpu, true, 8, None, true);
    let error = |p: &Vec<[f32; 4]>| -> f32 {
        p.iter()
            .zip(&converged)
            .map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>())
            .sum()
    };
    if let Ok(dir) = std::env::var("KREATE_SHOT_DIR") {
        for (name, img) in [
            ("dn-converged", &converged),
            ("dn-raw", &raw),
            ("dn-filtered", &filtered),
        ] {
            let bytes: Vec<u8> = img
                .iter()
                .flat_map(|c| c.iter().map(|v| (v.clamp(0.0, 1.0).sqrt() * 255.0) as u8))
                .collect();
            let _ = image::RgbaImage::from_raw(SIZE, SIZE, bytes)
                .map(|i| i.save(std::path::Path::new(&dir).join(format!("{name}.png"))));
        }
    }
    let (r, f) = (error(&raw), error(&filtered));
    assert!(f < r * 0.7, "filtered error {f} against raw {r}");
}

/// Textures: a floor whose base colour image is red on one half and blue on
/// the other is traced red and blue, where without it the floor is grey.
#[test]
fn the_path_tracer_reads_the_base_colour_image() {
    let gpu = gpu_or_skip!(gpu());
    // Two texels: red, then blue — sRGB-encoded, as a colour image is.
    let pixels = [230u8, 20, 20, 255, 20, 20, 230, 255];
    let image = kreate_render::pathtrace::PtImage {
        width: 2,
        height: 1,
        pixels: &pixels,
        color: true,
    };
    let white = kreate_render::pathtrace::PtSurface {
        base_color: [1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 0.5,
        base_color_image: Some(0),
        ..Default::default()
    };
    let shot = traced_scene(&gpu, true, 32, None, true, white, &[image]);
    let reddish = shot.iter().filter(|p| p[0] > p[2] + 0.15).count();
    let bluish = shot.iter().filter(|p| p[2] > p[0] + 0.15).count();
    assert!(
        reddish > 100 && bluish > 100,
        "{reddish} red and {bluish} blue pixels"
    );
}

/// Emission: a glowing cube is seen with no light in the scene at all, and
/// the same cube not glowing is black.
#[test]
fn a_glowing_surface_is_seen_in_the_dark() {
    let gpu = gpu_or_skip!(gpu());
    let mut graph = kreate_scene::shader::ShaderGraph::standard(
        "glow",
        kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        0.0,
        0.5,
    );
    if let kreate_scene::shader::NodeKind::Surface { emission, .. } = &mut graph.nodes[0].kind {
        *emission = kreate_math::SceneRgb::new(5000.0, 1000.0, 200.0);
    }
    let shot = shoot_graph(&gpu, ShadingMode::Rendered, &graph, &[], FRAMED);
    let lit = shot.iter().filter(|p| p[0] > 0.2).count();
    assert!(lit > 400, "only {lit} pixels glow");
    // Its colour, not a grey: red over green over blue.
    let warm = shot.iter().filter(|p| p[0] > p[1] && p[1] > p[2]).count();
    assert!(warm > 400, "only {warm} pixels have the glow's colour");
}

/// The path tracer adds a glowing surface's light where a path meets it.
#[test]
fn the_path_tracer_sees_a_glow() {
    let gpu = gpu_or_skip!(gpu());
    let plain = kreate_render::pathtrace::PtSurface {
        base_color: [0.8, 0.8, 0.8],
        roughness: 0.5,
        ..Default::default()
    };
    let glowing = kreate_render::pathtrace::PtSurface {
        emission: [8000.0, 2000.0, 400.0],
        ..plain
    };
    let mean = |p: &Vec<[f32; 4]>| p.iter().map(|c| c[0] - c[2]).sum::<f32>() / p.len() as f32;
    let a = traced_scene(&gpu, true, 16, None, false, plain, &[]);
    let b = traced_scene(&gpu, true, 16, None, false, glowing, &[]);
    assert!(
        mean(&b) > mean(&a) + 0.1,
        "glowing {} against plain {}",
        mean(&b),
        mean(&a)
    );
}

thread_local! {
    /// Whether `traced_scene`'s rasterised view has contact shadows.
    static CONTACT: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

thread_local! {
    /// How many faint far lamps `traced_scene` puts ahead of its sun.
    static DECOYS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Contact shadows only take light away, and where the cube stands on the
/// floor they take some.
///
/// With the sun's shadow map taken by lamps ahead of it in the budget, so
/// that what darkens the floor by the cube is the march and nothing else —
/// with a map, the two shadows land on the same pixels and the march has
/// nothing left to add.
#[test]
fn contact_shadows_darken_where_things_touch_and_nowhere_brighten() {
    let gpu = gpu_or_skip!(gpu());
    DECOYS.with(|d| d.set(kreate_render::shadow::MAX_SHADOWS));
    CONTACT.with(|c| c.set(false));
    let without = traced_shot(&gpu, false, 1, None);
    CONTACT.with(|c| c.set(true));
    let with = traced_shot(&gpu, false, 1, None);
    DECOYS.with(|d| d.set(0));
    let lum = |c: &[f32; 4]| c[0] + c[1] + c[2];
    let darker = with
        .iter()
        .zip(&without)
        .filter(|(a, b)| lum(a) < lum(b) - 0.03)
        .count();
    let brighter = with
        .iter()
        .zip(&without)
        .filter(|(a, b)| lum(a) > lum(b) + 0.03)
        .count();
    assert!(darker > 0, "contact shadows changed nothing");
    assert!(
        brighter <= darker / 20,
        "{brighter} pixels brighter against {darker} darker"
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CubeKind {
    None,
    Solid,
    Glass,
}

thread_local! {
    /// How much light the glass cube lets through.
    static GLASS: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
    /// What `traced_scene` stands on its floor.
    static CUBE: std::cell::Cell<CubeKind> = const { std::cell::Cell::new(CubeKind::Solid) };
}

/// The cube as `kind`, rasterised or traced.
fn cube_shot(gpu: &Gpu, kind: CubeKind, traced: bool, frames: u32) -> Vec<[f32; 4]> {
    CUBE.with(|c| c.set(kind));
    let shot = traced_shot_with(gpu, traced, frames, None, traced);
    CUBE.with(|c| c.set(CubeKind::Solid));
    shot
}

fn distance(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).sum::<f32>())
        .sum()
}

/// Glass lets the light through: where a solid cube throws its shadow, a
/// glass one throws none — in the rasterised view and the path tracer. And
/// in the rasterised view, where glass is screen-space refraction, a clear
/// cube looks far more like the floor with no cube than a solid one does.
///
/// Not that last for the path tracer, because there it is false and should
/// be: through the cube's lower front face, total internal reflection turns
/// the view up into the sky — black here — as it would in a real block.
#[test]
fn a_glass_cube_lets_the_light_through() {
    let gpu = gpu_or_skip!(gpu());
    let lum = |c: &[f32; 4]| c[0] + c[1] + c[2];
    for (traced, frames) in [(false, 1), (true, 48)] {
        let none = cube_shot(&gpu, CubeKind::None, traced, frames);
        let solid = cube_shot(&gpu, CubeKind::Solid, traced, frames);
        let glass = cube_shot(&gpu, CubeKind::Glass, traced, frames);
        if let Ok(dir) = std::env::var("KREATE_SHOT_DIR") {
            for (name, img) in [("none", &none), ("solid", &solid), ("glass", &glass)] {
                let bytes: Vec<u8> = img
                    .iter()
                    .flat_map(|c| c.iter().map(|v| (v.clamp(0.0, 1.0).sqrt() * 255.0) as u8))
                    .collect();
                let _ = image::RgbaImage::from_raw(SIZE, SIZE, bytes).map(|i| {
                    i.save(std::path::Path::new(&dir).join(format!("glass-{traced}-{name}.png")))
                });
            }
        }
        assert!(glass.iter().all(|p| p.iter().all(|v| v.is_finite())));
        // The solid's shadow: floor that is lit with no cube and dark with one.
        let shadow: Vec<usize> = (0..none.len())
            .filter(|&i| lum(&solid[i]) < 0.3 * lum(&none[i]))
            .collect();
        assert!(
            shadow.len() > 20,
            "traced {traced}: the solid cube casts no shadow to test against"
        );
        let lit = shadow
            .iter()
            .filter(|&&i| lum(&glass[i]) > 0.6 * lum(&none[i]))
            .count();
        assert!(
            lit * 10 > shadow.len() * 8,
            "traced {traced}: {lit} of {} shadow pixels lit through glass",
            shadow.len()
        );
        let to_glass = distance(&glass, &none);
        assert!(
            to_glass > 0.0,
            "traced {traced}: the glass cube is invisible"
        );
        if !traced {
            let to_solid = distance(&solid, &none);
            assert!(
                to_glass < to_solid * 0.6,
                "the glass cube is {to_glass} from no cube, the solid one {to_solid}"
            );
        }
    }
}

/// A pan moves the eye and turns the view not at all, so the eye-relative
/// projection is the same before and after — and the mean of a still view
/// must still start over, or every frame of the drag is averaged into one
/// smeared picture.
#[test]
fn a_pan_starts_the_mean_over() {
    let gpu = gpu_or_skip!(gpu());
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.0));
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
    one_material(&gpu, &mut viewport, grey());
    let mut draws = vec![Draw::new(&cube, glam::DAffine3::IDENTITY)];
    let mut camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };
    for _ in 0..64 {
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    }
    assert!(
        !viewport.accumulating(),
        "a still view never finished its mean"
    );
    camera.pivot += glam::DVec3::new(0.5, 0.0, 0.25);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    assert!(
        viewport.accumulating(),
        "the pan was averaged into the old picture"
    );
}

/// The cube on the floor under a lamp beside and above it, rendered through
/// each camera in turn as the application would frame by frame; the last
/// frame comes back.
fn orbit_frames(gpu: &Gpu, cameras: &[Camera]) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.show_bloom = false;
    viewport.ambient_occlusion = false;
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.set_display(&gpu.queue, FRAMED);
    let cube_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.0));
    let floor_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(12.0, 2, 2));
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cube_cpu, glam::Vec4::ONE);
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_cpu, glam::Vec4::ONE);
    one_material(gpu, &mut viewport, grey());
    let lift = glam::DAffine3::from_translation(glam::DVec3::new(0.0, 0.0, 0.5));
    let mut draws = vec![
        Draw::new(&cube, lift),
        Draw::new(&floor, glam::DAffine3::IDENTITY),
    ];
    let lamp = PlacedLight {
        falloff: Falloff::InverseSquare,
        position: glam::DVec3::new(2.0, 1.0, 3.0),
        amount: [3.0e5; 3],
        radius: 0.1,
        ..PlacedLight::default()
    };
    let mut out = Vec::new();
    for camera in cameras {
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[lamp]);
        viewport.prepare(&gpu.device, &gpu.queue, camera, &mut draws);
        out = read_frame(gpu, &mut viewport, &draws);
    }
    out
}

/// Shadows stay while the view moves: a frame drawn mid-orbit is the frame a
/// fresh viewport draws from the same place.
#[test]
fn shadows_stay_while_the_view_moves() {
    let gpu = gpu_or_skip!(gpu());
    let at = |turn: f64| Camera {
        pivot: glam::DVec3::new(0.0, 0.0, 0.3),
        distance: 6.0,
        rotation: glam::DQuat::from_rotation_z(turn) * glam::DQuat::from_rotation_x(1.0),
        ..Camera::default()
    };
    let still = orbit_frames(&gpu, &[at(0.3)]);
    let mut path: Vec<Camera> = (0..6).map(|_| at(0.0)).collect();
    path.extend((1..=3).map(|i| at(0.1 * i as f64)));
    let moved = orbit_frames(&gpu, &path);
    let lum = |c: &[f32; 4]| c[0] + c[1] + c[2];
    let dark = |p: &Vec<[f32; 4]>| p.iter().filter(|c| lum(c) < 0.05).count();
    let differ = still
        .iter()
        .zip(&moved)
        .filter(|(a, b)| (lum(a) - lum(b)).abs() > 0.1)
        .count();
    assert!(
        dark(&still) > 50,
        "the still frame has no shadow: {}",
        dark(&still)
    );
    assert!(
        differ < 40,
        "{differ} pixels differ mid-orbit; dark {} still against {} moving",
        dark(&still),
        dark(&moved)
    );
}

/// The path tracer's first frame — what every frame of a moving view is —
/// keeps the shadow: one sample, filtered, is dark where many samples are.
#[test]
fn one_traced_sample_filtered_keeps_the_shadow() {
    let gpu = gpu_or_skip!(gpu());
    let converged = traced_shot_with(&gpu, true, 128, None, true);
    let first = traced_shot_with(&gpu, true, 1, None, true);
    let lum = |c: &[f32; 4]| c[0] + c[1] + c[2];
    let shadow: Vec<usize> = (0..converged.len())
        .filter(|&i| lum(&converged[i]) < 0.2)
        .collect();
    let kept = shadow.iter().filter(|&&i| lum(&first[i]) < 0.4).count();
    eprintln!("TRACED shadow {} kept {kept}", shadow.len());
    if let Ok(dir) = std::env::var("KREATE_SHOT_DIR") {
        for (name, img) in [("pt-1", &first), ("pt-128", &converged)] {
            let bytes: Vec<u8> = img
                .iter()
                .flat_map(|c| c.iter().map(|v| (v.clamp(0.0, 1.0).sqrt() * 255.0) as u8))
                .collect();
            let _ = image::RgbaImage::from_raw(SIZE, SIZE, bytes)
                .map(|i| i.save(std::path::Path::new(&dir).join(format!("{name}.png"))));
        }
    }
    assert!(shadow.len() > 30, "no shadow to keep");
    assert!(
        kept * 10 >= shadow.len() * 8,
        "{kept} of {} shadow pixels dark after one sample",
        shadow.len()
    );
}

/// A cube that lets a little light through keeps most of its shadow — in the
/// rasterised view and the path tracer. It used to lose all of it: any
/// transmission at all and the object stopped casting.
#[test]
fn a_mostly_solid_cube_keeps_most_of_its_shadow() {
    let gpu = gpu_or_skip!(gpu());
    let lum = |c: &[f32; 4]| c[0] + c[1] + c[2];
    for (traced, frames) in [(false, 16), (true, 64)] {
        let none = cube_shot(&gpu, CubeKind::None, traced, frames);
        let solid = cube_shot(&gpu, CubeKind::Solid, traced, frames);
        GLASS.with(|g| g.set(0.2));
        let faint = cube_shot(&gpu, CubeKind::Glass, traced, frames);
        GLASS.with(|g| g.set(1.0));
        // The floor the solid cube shadows: dark with it, lit without.
        let shadow: Vec<usize> = (0..none.len())
            .filter(|&i| lum(&solid[i]) < 0.3 * lum(&none[i]))
            .collect();
        assert!(shadow.len() > 20, "traced {traced}: no shadow to compare");
        let mean = |p: &Vec<[f32; 4]>| {
            shadow.iter().map(|&i| lum(&p[i])).sum::<f32>() / shadow.len() as f32
        };
        let (dark, lit, here) = (mean(&solid), mean(&none), mean(&faint));
        // Four fifths of the light stopped: much nearer the solid's shadow
        // than the open floor, and not quite as dark.
        let depth = (lit - here) / (lit - dark).max(1e-6);
        assert!(
            depth > 0.6,
            "traced {traced}: a fifth-transmissive cube keeps {depth:.2} of the shadow"
        );
        assert!(
            depth < 0.97,
            "traced {traced}: its shadow is as dark as a solid's ({depth:.2})"
        );
    }
}

/// A material value changed on a settled view starts its mean again, so the
/// picture shows the new value at once and not after the next camera move.
#[test]
fn a_material_change_on_a_settled_view_is_drawn() {
    let gpu = gpu_or_skip!(gpu());
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.0));
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
    let compiled = |r: f32| {
        let graph = kreate_scene::shader::ShaderGraph::standard(
            "m",
            kreate_math::SceneRgb::new(r, 0.5, 0.5),
            0.0,
            0.5,
        );
        kreate_shader::compile(&graph).expect("compiles")
    };
    let (grey, red) = (compiled(0.5), compiled(0.9));
    graph_material(
        &gpu,
        &mut viewport,
        &kreate_scene::shader::ShaderGraph::standard(
            "m",
            kreate_math::SceneRgb::new(0.5, 0.5, 0.5),
            0.0,
            0.5,
        ),
    );
    let mut draws = vec![Draw::new(&cube, glam::DAffine3::IDENTITY)];
    let camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };
    // The frame loop hands the constants over every frame.
    for _ in 0..64 {
        viewport.set_material_constants(&gpu.queue, 0, &grey.constants, &grey.image_slots);
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    }
    assert!(!viewport.accumulating(), "the still view never settled");
    viewport.set_material_constants(&gpu.queue, 0, &red.constants, &red.image_slots);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    assert!(viewport.accumulating(), "the new colour was not drawn");
}

/// A render's passes, read back: the picture where the cube is, its normal
/// facing the camera, and its distance along the view — and nothing in the
/// empty corner.
#[test]
fn a_renders_passes_read_back_with_depth_and_normals() {
    let gpu = gpu_or_skip!(gpu());
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    // Smaller than the allocation, as a window almost always is: a depth
    // texture can only be copied whole, and a region-sized copy of it was
    // refused — a crash the first time an EXR was rendered for real.
    let (rw, rh) = (SIZE - 14, SIZE - 22);
    viewport.set_render_size(rw, rh);
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.set_display(&gpu.queue, FRAMED);
    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
    one_material(&gpu, &mut viewport, grey());
    let mut draws = vec![Draw::new(&cube, glam::DAffine3::IDENTITY)];
    let camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };
    viewport.set_lights(
        &gpu.device,
        &gpu.queue,
        camera.eye(),
        &[key_light(20_000.0)],
    );
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    let _ = read_frame(&gpu, &mut viewport, &draws);
    let forward = (camera.pivot - camera.eye()).normalize().as_vec3();
    let passes = viewport
        .read_passes(&gpu.device, &gpu.queue, FRAMED.exposure, forward)
        .expect("passes read back");
    assert_eq!((passes.width, passes.height), (rw, rh));
    let centre = (rh / 2 * rw + rw / 2) as usize;
    let d = passes.depth[centre];
    assert!(
        d > 3.5 && d < 5.5,
        "the cube's front is {d} m away along the view"
    );
    assert_eq!(passes.depth[0], 1e10, "the empty corner has no depth");
    let n = glam::Vec3::from(passes.normal[centre]);
    assert!(
        n.length() > 0.9 && n.dot(-forward) > 0.3,
        "the normal {n} does not face the camera"
    );
    assert!(
        passes.combined[centre][0] > 0.01,
        "the picture is black where the lit cube is"
    );
}

/// A small glowing panel is the only light. The path tracer aims at it as well
/// as finding it by bouncing, and weighs the two: the floor beside it comes out
/// as bright as the panel's radiance, size and angle say — not twice, which
/// counting both ways would give — and smooth after a few samples, where
/// finding it only by bouncing is speckle.
#[test]
fn a_glowing_panel_lights_the_floor_as_much_as_it_should_and_quietly() {
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        pivot: glam::DVec3::new(0.3, 0.0, 0.0),
        distance: 3.0,
        rotation: glam::DQuat::from_rotation_z(0.4) * glam::DQuat::from_rotation_x(0.5),
        ..Camera::default()
    };
    let patch = glowing_panel_patch(&gpu, 16, 0.2, camera, glam::DVec3::new(0.3, 0.0, 0.0));
    let mean = patch.iter().sum::<f32>() / patch.len() as f32;
    // The floor point is 0.3 m along and 1 m below the panel: irradiance is
    // L·A·cosθ²/d², and a matte floor of albedo 0.8 sends back 0.8/π of it.
    let (l, a, d2) = (20_000.0f32, 0.04f32, 0.3f32 * 0.3 + 1.0);
    let expected = 0.8 / std::f32::consts::PI * l * a / (d2 * d2) * FRAMED.exposure.exp2();
    assert!(
        (mean - expected).abs() < expected * 0.15,
        "the floor beside the panel is {mean}, the panel's light says {expected}"
    );
    let spread =
        (patch.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / patch.len() as f32).sqrt();
    assert!(
        spread < mean * 0.25,
        "sixteen samples are still speckle: spread {spread} about {mean}"
    );
}

/// The floor under a glowing panel (only light, 20 000 nits, `size` metres
/// square, 1 m up), path traced for `frames` samples, read back linear: the
/// green of a 5x5 patch around where `point` lands.
fn glowing_panel_patch(
    gpu: &Gpu,
    frames: u32,
    size: f32,
    camera: Camera,
    point: glam::DVec3,
) -> Vec<f32> {
    glowing_panel_patch_sided(gpu, frames, size, camera, point, false)
}

/// The same, the panel glowing only the way its normals face (up) when
/// `one_sided`.
fn glowing_panel_patch_sided(
    gpu: &Gpu,
    frames: u32,
    size: f32,
    camera: Camera,
    point: glam::DVec3,
    one_sided: bool,
) -> Vec<f32> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.path_trace = true;
    viewport.path_trace_denoise = false;
    viewport.show_bloom = false;
    viewport.show_reflections = false;
    viewport.ambient_occlusion = false;
    viewport.contact_shadows = false;
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.set_display(&gpu.queue, FRAMED);
    viewport.set_world([0.0; 3]);
    let floor_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(12.0, 2, 2));
    let panel_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(size, 1, 1));
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_cpu, glam::Vec4::ONE);
    let panel =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &panel_cpu, glam::Vec4::ONE);
    one_material(gpu, &mut viewport, grey());
    let lift = glam::DAffine3::from_translation(glam::DVec3::new(0.0, 0.0, 1.0));
    let parts = [
        (&floor_cpu, glam::DAffine3::IDENTITY, 0u32),
        (&panel_cpu, lift, 1u32),
    ];
    let data: Vec<(Vec<[f32; 3]>, Vec<glam::Vec3>)> = parts
        .iter()
        .map(|(m, _, _)| {
            (
                m.positions.iter().map(|p| p.0).collect(),
                m.normals
                    .iter()
                    .map(|n| kreate_render::mesh_gpu::oct_decode(*n))
                    .collect(),
            )
        })
        .collect();
    let meshes: Vec<_> = parts
        .iter()
        .zip(&data)
        .map(
            |((m, t, surface), (p, n))| kreate_render::pathtrace::PtMesh {
                positions: p,
                uvs: &m.uvs,
                normals: n,
                indices: &m.indices,
                transform: *t,
                surface: *surface,
                runs: &[],
            },
        )
        .collect();
    let floor_surface = kreate_render::pathtrace::PtSurface {
        base_color: [0.8, 0.8, 0.8],
        roughness: 0.9,
        ..Default::default()
    };
    let glow = kreate_render::pathtrace::PtSurface {
        base_color: [0.0; 3],
        emission: [20_000.0; 3],
        one_sided,
        ..Default::default()
    };
    viewport.set_path_trace_scene(
        &gpu.device,
        &gpu.queue,
        &meshes,
        &[floor_surface, glow],
        &[],
        &Default::default(),
    );
    let mut draws = vec![
        Draw::new(&floor, glam::DAffine3::IDENTITY),
        Draw::new(&panel, lift),
    ];
    for _ in 0..frames {
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[]);
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        let _ = read_frame(gpu, &mut viewport, &draws);
    }
    let forward = (camera.pivot - camera.eye()).normalize().as_vec3();
    let passes = viewport
        .read_passes(&gpu.device, &gpu.queue, FRAMED.exposure, forward)
        .expect("passes");
    // The pixel where the floor point lands.
    let vp = camera.view_projection(1.0);
    let at = (point - camera.eye()).as_vec3();
    let clip = vp * at.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * passes.width as f32) as i32;
    let y = ((0.5 - ndc.y * 0.5) * passes.height as f32) as i32;
    // A 5x5 patch around it.
    let mut values = Vec::new();
    for dy in -2..=2 {
        for dx in -2..=2 {
            let i = ((y + dy) as u32 * passes.width + (x + dx) as u32) as usize;
            values.push(passes.combined[i][1]);
        }
    }
    values
}

/// Under a big panel — 2 m square, 1 m up — paths bounce into the glow as
/// often as they aim at it, so counting both whole would show: the floor
/// below its middle is exactly what the rectangle's form factor says.
#[test]
fn a_big_glowing_panel_is_counted_once() {
    let gpu = gpu_or_skip!(gpu());
    // Low, looking in under the panel's edge at the floor below its middle.
    let camera = Camera {
        pivot: glam::DVec3::ZERO,
        distance: 4.0,
        rotation: glam::DQuat::from_rotation_x(1.35),
        ..Camera::default()
    };
    let patch = glowing_panel_patch(&gpu, 64, 2.0, camera, glam::DVec3::ZERO);
    let mean = patch.iter().sum::<f32>() / patch.len() as f32;
    // A point below the corner of a w×l rectangle at height h, both sides
    // parallel, sees it with form factor
    // (1/2π)[X/√(1+X²)·atan(Y/√(1+X²)) + Y/√(1+Y²)·atan(X/√(1+Y²))], X=w/h,
    // Y=l/h. Four 1 m quarters at 1 m: X = Y = 1.
    let corner =
        (2.0f32 * (1.0 / 2f32.sqrt()) * (1.0 / 2f32.sqrt()).atan()) / (2.0 * std::f32::consts::PI);
    let form = 4.0 * corner;
    // Radiance L·F irradiates πL·F; a matte floor of 0.8 returns 0.8/π of it.
    let expected = 0.8 * 20_000.0 * form * FRAMED.exposure.exp2();
    assert!(
        (mean - expected).abs() < expected * 0.1,
        "the floor under the panel is {mean}, the form factor says {expected}"
    );
}

/// An equirectangular sky, `Rgba16Float`: `radiance` in the top `rows` of
/// `height`, black below.
fn cap_sky(width: u32, height: u32, rows: u32, radiance: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 8) as usize);
    for y in 0..height {
        let v = if y < rows { radiance } else { 0.0 };
        for _ in 0..width {
            for c in [v, v, v, 1.0] {
                out.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
        }
    }
    out
}

/// A matte floor under `sky`, path traced for `frames` samples with no lamps
/// and nothing glowing: the green of a 5x5 patch at the origin, linear.
fn sky_patch(gpu: &Gpu, frames: u32, sky: &[u8], (w, h): (u32, u32)) -> Vec<f32> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.path_trace = true;
    viewport.path_trace_denoise = false;
    viewport.show_bloom = false;
    viewport.show_reflections = false;
    viewport.ambient_occlusion = false;
    viewport.contact_shadows = false;
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.set_display(&gpu.queue, FRAMED);
    viewport.set_environment(&gpu.device, &gpu.queue, w, h, sky);
    viewport.set_world_shape(&gpu.queue, 0.0, 1.0);
    let floor_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(40.0, 2, 2));
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_cpu, glam::Vec4::ONE);
    one_material(gpu, &mut viewport, grey());
    let positions: Vec<[f32; 3]> = floor_cpu.positions.iter().map(|p| p.0).collect();
    let normals: Vec<glam::Vec3> = floor_cpu
        .normals
        .iter()
        .map(|n| kreate_render::mesh_gpu::oct_decode(*n))
        .collect();
    let meshes = [kreate_render::pathtrace::PtMesh {
        positions: &positions,
        uvs: &floor_cpu.uvs,
        normals: &normals,
        indices: &floor_cpu.indices,
        transform: glam::DAffine3::IDENTITY,
        surface: 0,
        runs: &[],
    }];
    let matte = kreate_render::pathtrace::PtSurface {
        base_color: [0.8; 3],
        roughness: 0.9,
        ..Default::default()
    };
    viewport.set_path_trace_scene(
        &gpu.device,
        &gpu.queue,
        &meshes,
        &[matte],
        &[],
        &Default::default(),
    );
    let camera = Camera {
        pivot: glam::DVec3::ZERO,
        distance: 3.0,
        rotation: glam::DQuat::from_rotation_x(0.6),
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&floor, glam::DAffine3::IDENTITY)];
    for _ in 0..frames {
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[]);
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        let _ = read_frame(gpu, &mut viewport, &draws);
    }
    let forward = (camera.pivot - camera.eye()).normalize().as_vec3();
    let passes = viewport
        .read_passes(&gpu.device, &gpu.queue, FRAMED.exposure, forward)
        .expect("passes");
    let (x, y) = (passes.width as i32 / 2, passes.height as i32 / 2);
    let mut values = Vec::new();
    for dy in -2..=2 {
        for dx in -2..=2 {
            values.push(
                passes.combined[((y + dy) as u32 * passes.width + (x + dx) as u32) as usize][1],
            );
        }
    }
    values
}

/// The path tracer aims at the environment where its light is, and weighs
/// that against bouncing into it: a floor under a bright cap of sky is as
/// bright as the cap's radiance and size say, and under a small sun it is
/// smooth after a few samples.
#[test]
fn an_image_sky_lights_the_floor_as_much_as_it_should_and_quietly() {
    let gpu = gpu_or_skip!(gpu());
    let (w, h) = (256, 128);
    let mean = |p: &[f32]| p.iter().sum::<f32>() / p.len() as f32;
    // A cap of radiance L to θ₀ from straight up irradiates a level floor
    // with πL·sin²θ₀; a matte floor of 0.8 returns 0.8·L·sin²θ₀ of it.
    let rows = 16;
    let theta = rows as f32 / h as f32 * std::f32::consts::PI;
    let cap = mean(&sky_patch(&gpu, 32, &cap_sky(w, h, rows, 1000.0), (w, h)));
    let expected = 0.8 * 1000.0 * theta.sin().powi(2) * FRAMED.exposure.exp2();
    assert!(
        (cap - expected).abs() < expected * 0.12,
        "under the cap the floor is {cap}, its size says {expected}"
    );
    // A sun: two rows, far brighter — near the most a half float holds.
    // Smooth at sixteen samples.
    let sun = sky_patch(&gpu, 16, &cap_sky(w, h, 2, 60_000.0), (w, h));
    let m = mean(&sun);
    let spread = (sun.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / sun.len() as f32).sqrt();
    assert!(
        m > 0.0 && spread < m * 0.25,
        "under a small sun sixteen samples are speckle: {spread} about {m}"
    );
}

/// An area light's panel glows from its front only: the grid's normals face
/// up, so a one-sided panel above the floor lights nothing under it, by aiming
/// or by bouncing, where the same panel glowing both ways does.
#[test]
fn a_one_sided_panel_lights_only_what_it_faces() {
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        pivot: glam::DVec3::new(0.3, 0.0, 0.0),
        distance: 3.0,
        rotation: glam::DQuat::from_rotation_z(0.4) * glam::DQuat::from_rotation_x(0.5),
        ..Camera::default()
    };
    let at = glam::DVec3::new(0.3, 0.0, 0.0);
    let mean = |p: Vec<f32>| p.iter().sum::<f32>() / p.len() as f32;
    let both = mean(glowing_panel_patch_sided(&gpu, 16, 0.2, camera, at, false));
    let facing_away = mean(glowing_panel_patch_sided(&gpu, 16, 0.2, camera, at, true));
    assert!(both > 0.01, "the two-sided panel lit nothing: {both}");
    assert!(
        facing_away < both * 0.02,
        "a panel facing away lit the floor: {facing_away} against {both}"
    );
}

/// A floor wearing `graph`, path traced under a low sun across its diagonal:
/// the compiled material run by the tracer, the way the application hands it
/// over. Low, so that a surface tilted toward it is lit more and one tilted
/// away less — overhead, every tilt would only darken.
fn traced_floor(gpu: &Gpu, graph: &kreate_scene::shader::ShaderGraph) -> Vec<[f32; 4]> {
    floor_shot(gpu, graph, true)
}

/// The same floor, traced or rasterised.
fn floor_shot(gpu: &Gpu, graph: &kreate_scene::shader::ShaderGraph, traced: bool) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.path_trace = traced;
    viewport.path_trace_denoise = false;
    viewport.show_bloom = false;
    viewport.show_reflections = false;
    viewport.ambient_occlusion = false;
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.set_display(&gpu.queue, FRAMED);
    graph_material(gpu, &mut viewport, graph);
    let floor_cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::grid(4.0, 1, 1));
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_cpu, glam::Vec4::ONE);
    let compiled = kreate_shader::compile(graph).expect("compiles");
    let code = format!(
        "{}{}",
        kreate_shader::tracer_declarations(),
        kreate_shader::for_tracer(&compiled, 0, 0, 0, 0)
    );
    let positions: Vec<[f32; 3]> = floor_cpu.positions.iter().map(|p| p.0).collect();
    let normals: Vec<glam::Vec3> = floor_cpu
        .normals
        .iter()
        .map(|n| kreate_render::mesh_gpu::oct_decode(*n))
        .collect();
    let meshes = [kreate_render::pathtrace::PtMesh {
        positions: &positions,
        uvs: &floor_cpu.uvs,
        normals: &normals,
        indices: &floor_cpu.indices,
        transform: glam::DAffine3::IDENTITY,
        surface: 0,
        runs: &[],
    }];
    let surface = kreate_render::pathtrace::PtSurface {
        base_color: [0.8, 0.8, 0.8],
        roughness: 0.5,
        material: Some(0),
        ..Default::default()
    };
    viewport.set_path_trace_scene(
        &gpu.device,
        &gpu.queue,
        &meshes,
        &[surface],
        &[],
        &kreate_render::pathtrace::PtMaterials {
            code: &code,
            count: 1,
            values: &compiled.constants,
        },
    );
    let camera = Camera {
        pivot: glam::DVec3::ZERO,
        distance: 3.0,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&floor, glam::DAffine3::IDENTITY)];
    let mut out = Vec::new();
    for _ in 0..32 {
        let low = PlacedLight {
            direction: glam::Vec3::new(-0.6, -0.6, -0.5).normalize(),
            ..key_light(20_000.0)
        };
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[low]);
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        out = read_frame(gpu, &mut viewport, &draws);
    }
    out
}

/// A grey surface with `kind` wired from its output `from` into socket `to`.
fn grey_with(
    kind: kreate_scene::shader::NodeKind,
    from: usize,
    to: usize,
) -> kreate_scene::shader::ShaderGraph {
    let mut graph =
        kreate_scene::shader::ShaderGraph::standard("test", grey().base_color, 0.0, 0.5);
    let node = graph
        .add_node(kreate_scene::shader::Node::new("node", kind))
        .expect("added");
    graph
        .link(kreate_scene::shader::Link {
            from_node: node,
            from_socket: from,
            to_node: 0,
            to_socket: to,
        })
        .expect("the sockets agree");
    graph
}

/// The tracer runs the material's own code: a checker reaches its picture
/// in both colours, read where each ray lands on the object.
#[test]
fn the_path_tracer_paints_with_the_materials_own_code() {
    let gpu = gpu_or_skip!(gpu());
    let frame = traced_floor(
        &gpu,
        &grey_with(
            kreate_scene::shader::NodeKind::Checker {
                scale: 1.0,
                color1: kreate_math::SceneRgb::new(0.8, 0.02, 0.02),
                color2: kreate_math::SceneRgb::new(0.02, 0.02, 0.8),
            },
            0,
            0,
        ),
    );
    let reddish = frame.iter().filter(|p| p[0] > p[2] + 0.05).count();
    let bluish = frame.iter().filter(|p| p[2] > p[0] + 0.05).count();
    assert!(
        reddish > 200 && bluish > 200,
        "the traced floor has {reddish} red pixels and {bluish} blue ones"
    );
}

/// And its bump: a wave's bands tilt the traced floor's light both ways
/// against the flat floor's.
///
/// Bands a metre apart: the tracer averages over each pixel, and bands finer
/// than one would average to a floor that is only darker.
#[test]
fn the_path_tracer_tilts_the_light_by_a_bump() {
    let gpu = gpu_or_skip!(gpu());
    let flat = traced_floor(
        &gpu,
        &kreate_scene::shader::ShaderGraph::standard("test", grey().base_color, 0.0, 0.5),
    );
    let mut graph = grey_with(
        kreate_scene::shader::NodeKind::Bump {
            strength: 1.0,
            distance: 0.3,
        },
        0,
        3,
    );
    let wave = graph
        .add_node(kreate_scene::shader::Node::new(
            "wave",
            kreate_scene::shader::NodeKind::Wave {
                kind: kreate_scene::shader::WaveKind::Bands,
                scale: 0.5,
                distortion: 0.0,
                detail: 2.0,
                detail_scale: 1.0,
            },
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
    let bumped = traced_floor(&gpu, &graph);
    let lum = |p: &[f32; 4]| p[0] + p[1] + p[2];
    // Lit where the viewport lights it: the tilt goes the same way in both.
    // A sign wrong in either would light each band's other side.
    let raster_flat = floor_shot(
        &gpu,
        &kreate_scene::shader::ShaderGraph::standard("test", grey().base_color, 0.0, 0.5),
        false,
    );
    let raster_bumped = floor_shot(&gpu, &graph, false);
    // Over the floor only — the sky above it is the same in all four — and
    // each about its own mean there: a bump darkens overall, in both, and
    // that would agree whichever way the bands were lit.
    let sky = raster_flat[0];
    let floor: Vec<usize> = (0..flat.len()).filter(|i| raster_flat[*i] != sky).collect();
    assert!(floor.len() > 1000, "the floor is not in the picture");
    let centred = |after: &[[f32; 4]], before: &[[f32; 4]]| {
        let d: Vec<f32> = floor
            .iter()
            .map(|i| lum(&after[*i]) - lum(&before[*i]))
            .collect();
        let mean = d.iter().sum::<f32>() / d.len() as f32;
        d.into_iter().map(|x| x - mean).collect::<Vec<f32>>()
    };
    let traced_change = centred(&bumped, &flat);
    let raster_change = centred(&raster_bumped, &raster_flat);
    // Correlated, not merely alike: a tilt either way darkens a little by
    // itself, so even bands lit from the wrong side would agree some.
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let agreement = dot(&traced_change, &raster_change)
        / (dot(&traced_change, &traced_change) * dot(&raster_change, &raster_change)).sqrt();
    assert!(
        agreement > 0.6,
        "the traced bump and the viewport's correlate by only {agreement}"
    );
    let brighter = flat
        .iter()
        .zip(&bumped)
        .filter(|(a, b)| lum(b) > lum(a) + 0.1)
        .count();
    let darker = flat
        .iter()
        .zip(&bumped)
        .filter(|(a, b)| lum(b) + 0.1 < lum(a))
        .count();
    assert!(
        brighter > 100 && darker > 100,
        "against the flat floor, {brighter} traced pixels are lighter and {darker} darker"
    );
}
