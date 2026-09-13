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
}

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

    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY).with_surface(surface)];
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
    if scene.idle {
        lights.push(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.0, -0.2, -0.98).normalize(),
            amount: [2_000.0; 3],
            cast_shadow: scene.idle_casts,
            ..PlacedLight::default()
        });
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
    if scene.crowd {
        for i in 0..9 {
            let a = i as f32 * 0.7;
            lights.push(PlacedLight {
                falloff: Falloff::Parallel,
                direction: glam::Vec3::new(a.cos() * 0.5, a.sin() * 0.5, -0.85).normalize(),
                amount: [500.0; 3],
                cast_shadow: true,
                ..PlacedLight::default()
            });
        }
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
    viewport.shading = ShadingMode::Rendered;
    viewport.set_display(&gpu.queue, FRAMED);
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let floor_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::plane(16.0));
    let box_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(1.6));
    // A sphere for the angles a box does not have. Every incidence from
    // head-on to grazing appears on one, and the band where it turns away from
    // the light is where acne lives in every renderer that has it.
    let ball_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::uv_sphere(0.9, 24, 16));
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_mesh, glam::Vec4::ONE);
    let boxes =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &box_mesh, glam::Vec4::ONE);
    let balls =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &ball_mesh, glam::Vec4::ONE);

    let camera = Camera {
        distance: 22.0,
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
    let mut draws = vec![Draw::new(&floor, glam::DAffine3::IDENTITY).with_surface(grey())];
    for row in 0..3 {
        for col in 0..3 {
            let n = row * 3 + col;
            let x = -4.5 + col as f64 * 4.5;
            let y = -4.5 + row as f64 * 4.5;
            let at = glam::DVec3::new(x, y, 0.8);
            let draw = match n % 3 {
                // Standing on the floor, not hovering over it: half the cube's
                // height up puts its underside at zero.
                0 => Draw::new(&boxes, glam::DAffine3::from_translation(at)),
                1 => Draw::new(
                    &boxes,
                    glam::DAffine3::from_rotation_translation(
                        glam::DQuat::from_euler(
                            glam::EulerRot::XYZ,
                            0.6,
                            0.4,
                            n as f64 * 0.5,
                        ),
                        at,
                    ),
                ),
                _ => Draw::new(&balls, glam::DAffine3::from_translation(at)),
            };
            draws.push(draw.with_surface(grey()));
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
    assert_eq!(both.passes, 2, "two casting lights, {} maps", both.passes);
    assert_eq!(key_only.passes, 1);

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
    assert_eq!(one.passes, 1, "{} maps for one caster", one.passes);
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
            ..ShadowScene::default()
        },
    );
    assert_eq!(
        crowd.passes,
        kreate_render::shadow::MAX_SHADOWS,
        "{} maps for eleven casters",
        crowd.passes
    );
    // And the scene is still lit by all of them: a light denied a map is not a
    // light denied.
    let brightest = crowd
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p[0]));
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
    assert_eq!(frames[1].passes, 0, "a lamp aimed at nothing rendered a map");

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
fn a_point_light_overhead_casts_nothing_and_blacks_nothing_out() {
    // Two things at once, and the first is a limitation stated at the level a
    // user meets it.
    //
    // A bare point light renders one of its six faces — face zero, which looks
    // along +X. A lamp hanging *above* a scene therefore points its one map
    // sideways and catches nothing at all, so the pass is drawn and the picture
    // is identical. `shadow::shadow_view` says a point light shadows a sixth of
    // the scene; this is what that sixth is worth when it is the wrong sixth.
    //
    // The second is the decision the sampler makes outside a frustum, and this
    // is the only arrangement in the file that can reach it: a sun's box is
    // fitted to what is drawn and a spot's square frustum contains its own cone,
    // so under either of those nothing lit is ever outside. Here almost
    // everything lit is. Treating outside as shadowed would put a hard black
    // edge at the frustum's border, and here that means blacking out the scene
    // — so an unchanged picture is the whole assertion.
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
    // A map is rendered, and it is wasted. Recorded rather than hidden: it is
    // the cost of the missing five faces.
    assert_eq!(shadowed.passes, 1);
    let (darker, brighter) = difference(&lit, &shadowed);
    assert_eq!(
        (darker, brighter),
        (0, 0),
        "the overhead point light changed {darker} + {brighter} pixels"
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
    assert_eq!(moved, 0, "{moved} pixels moved when the lights were given ends");

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
    let studio = summary(&shoot(
        &gpu,
        ShadingMode::Studio,
        grey(),
        &light,
        FRAMED,
    ));
    let flat = summary(&shoot(
        &gpu,
        ShadingMode::Flat,
        grey(),
        &light,
        FRAMED,
    ));
    let rendered = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &light,
        FRAMED,
    ));

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
    let shot = shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &many,
        FRAMED,
    );
    let shot = summary(&shot);
    assert!(
        shot.brightest > 0.05,
        "the grown buffer lit nothing: {}",
        shot.brightest
    );
    assert!(shot.lit > 400, "only {} pixels were lit", shot.lit);
}
