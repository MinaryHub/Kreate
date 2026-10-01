//! Visual regression: every corpus specimen, five fixed views, against golden
//! images.
//!
//! # Why this exists when the numeric tests already pass
//!
//! The round-trip tests in `kreate-io` compare positions, faces and attributes,
//! and they are thorough. They are also blind to an entire class of bug, because
//! the things that go wrong in a renderer are not values but *conventions*:
//!
//! - a normal pointing inwards — same vector, wrong sign;
//! - a face wound the other way — same three vertices;
//! - a UV flipped in V — same two floats;
//! - a smoothing group lost — same geometry, different shading;
//! - a depth-offset layer in the wrong order — same overlay, hidden.
//!
//! Every one of those leaves the numbers intact and the picture wrong, and every
//! one is instantly obvious in an image. That is the whole argument for a golden
//! image harness, and it is why the plan calls it the most valuable single piece
//! of the interoperability test system.
//!
//! # The threshold, and why it is not zero
//!
//! Goldens are compared with a tolerance, because a byte-exact comparison across
//! GPUs is not a thing that works. Rasterisation rules, floating point contract
//! ion in shaders, and texture filtering all differ between an Intel iGPU and an
//! NVIDIA discrete part — both of which are in this development machine, which
//! is exactly why the problem cannot be ignored until later.
//!
//! What makes a tolerance workable rather than a fudge is that the two kinds of
//! difference do not look alike:
//!
//! - **driver differences** are a fraction of a channel, on silhouette and
//!   shading-gradient pixels, scattered;
//! - **the bugs this catches** are large, on solid interior regions, contiguous.
//!
//! So the metric counts pixels that differ by more than [`CHANNEL_TOLERANCE`]
//! and fails when they exceed [`PIXEL_BUDGET`] of the frame. A flipped normal
//! changes most of a shaded face; antialiasing on the outline of that face is a
//! few hundred pixels at most. The falsification tests at the bottom of this
//! file assert that separation directly rather than trusting it.
//!
//! # Regenerating
//!
//! `UPDATE_GOLDENS=1 cargo test -p kreate-render --test visual_regression`
//!
//! The images are small and are meant to be *looked at* in the diff. A golden
//! update that a reviewer waves through is a golden update that has silently
//! accepted a rendering regression, so the review is the point of the workflow,
//! not an obstacle to it.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ShadingMode, Viewport};
use kreate_testkit::corpus;
use std::path::PathBuf;

/// Small on purpose. A golden is read by a person in a diff, and the bugs this
/// catches are whole-face, not sub-pixel. Nine specimens at five views each is
/// forty-five images, and they should not add a megabyte to the repository.
const SIZE: u32 = 192;
const BYTES_PER_ROW: u32 = SIZE * 4;

/// How far one channel may drift before the pixel counts as different.
///
/// About 5%. Comfortably above what a different rasteriser produces on a
/// gradient, comfortably below the change from shading a face the wrong way.
const CHANNEL_TOLERANCE: i32 = 12;

/// How much of the frame may differ before the comparison fails.
///
/// A 192x192 frame is 36,864 pixels, so this is roughly 550 of them — enough for
/// the antialiased outline of a model that fills the view, nowhere near enough
/// for a face.
const PIXEL_BUDGET: f32 = 0.015;

/// One view: a name, a yaw and pitch orbit from the default camera, and whether
/// the wireframe overlay is drawn.
#[derive(Clone, Copy)]
struct View {
    name: &'static str,
    yaw: f64,
    pitch: f64,
    /// The wireframe is the single clearest visual signal for *topology*: a
    /// quad that came back as two triangles grows a diagonal, and no amount of
    /// shading shows that. It also covers the surface in lines, which is why one
    /// view goes without it.
    wireframe: bool,
}

/// Several views rather than one, because a normal or winding error is invisible
/// from whichever side happens to look right. Front and back in particular are
/// the pair that catches an inside-out model: one of them goes dark.
///
/// The last view repeats the three-quarter angle with the overlay off. That is
/// the *shading* view, and it exists because the wireframe hides exactly what
/// the smoothing specimen is about — the first read of these goldens mistook a
/// wireframe for a faceted wall, which is a mistake the harness should not make
/// available.
const VIEWS: [View; 5] = [
    View {
        name: "front",
        yaw: 0.0,
        pitch: 0.0,
        wireframe: true,
    },
    View {
        name: "back",
        yaw: std::f64::consts::PI,
        pitch: 0.0,
        wireframe: true,
    },
    View {
        name: "three_quarter",
        yaw: 0.9,
        pitch: 0.5,
        wireframe: true,
    },
    View {
        name: "below",
        yaw: 0.4,
        pitch: -1.1,
        wireframe: true,
    },
    View {
        name: "shaded",
        yaw: 0.9,
        pitch: 0.5,
        wireframe: false,
    },
];

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
}

fn updating() -> bool {
    std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v != "0")
}

struct Frame {
    pixels: Vec<[u8; 4]>,
}

impl Frame {
    fn to_rgba(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| *p).collect()
    }

    /// Fraction of pixels that are not the viewport background.
    ///
    /// The background is passed in rather than sampled, because a model that
    /// happens to be the same grey as the background would otherwise be counted
    /// as empty.
    fn coverage(&self, background: [u8; 4]) -> f32 {
        let differs = |p: &&[u8; 4]| {
            (0..3).any(|i| (p[i] as i32 - background[i] as i32).abs() > CHANNEL_TOLERANCE)
        };
        self.pixels.iter().filter(differs).count() as f32 / self.pixels.len().max(1) as f32
    }

    /// Fraction of pixels differing from `other` by more than the tolerance.
    fn difference(&self, other: &Frame) -> f32 {
        let differing = self
            .pixels
            .iter()
            .zip(&other.pixels)
            .filter(|(a, b)| {
                // Alpha is excluded: the target is opaque, so a difference there
                // would be a bug in the readback rather than in the render, and
                // including it only adds noise.
                (0..3).any(|i| (a[i] as i32 - b[i] as i32).abs() > CHANNEL_TOLERANCE)
            })
            .count();
        differing as f32 / self.pixels.len().max(1) as f32
    }
}

/// One device, one target, one viewport, reused for every frame of the run.
///
/// Built once because building it is the expensive part: creating a `Device` and
/// compiling the pipeline set costs about a second, and doing that thirty-six
/// times made this the slowest test in the workspace by an order of magnitude,
/// for no signal at all. The renderer is stateless between frames here — the
/// viewport is reconfigured per frame from the camera — so sharing it changes
/// nothing about what is measured.
struct Harness {
    gpu: kreate_render::Gpu,
    target: wgpu::Texture,
    texture_view: wgpu::TextureView,
    viewport: Viewport,
    readback: wgpu::Buffer,
}

impl Harness {
    /// `None` when there is no usable adapter.
    fn new() -> Option<Self> {
        let opts = GpuOptions::default();
        let instance = kreate_render::Gpu::create_instance(&opts);
        let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visual regression target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
        // Studio shading rather than matcap: matcap is a texture lookup on the
        // view space normal, which makes a flipped normal look merely different.
        // Directional lighting makes it look *wrong*, which is what a regression
        // test wants.
        viewport.shading = ShadingMode::Studio;
        // The grid would be most of every image and would change with the
        // camera, which is a lot of pixels of noise around the thing under test.
        viewport.show_grid = false;

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visual regression readback"),
            size: (BYTES_PER_ROW * SIZE) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Some(Self {
            gpu,
            target,
            texture_view,
            viewport,
            readback,
        })
    }

    /// The clear colour, encoded the way the sRGB target stores it, so a frame
    /// can be asked how much of it is background.
    fn background(&self) -> [u8; 4] {
        let c = self.viewport.background;
        let to_u8 = |v: f64| {
            let s = if v <= 0.0031308 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (s.clamp(0.0, 1.0) * 255.0).round() as u8
        };
        [to_u8(c.r), to_u8(c.g), to_u8(c.b), 255]
    }

    fn render(&mut self, mesh: &kreate_mesh::MeshData, view: View) -> Frame {
        let gpu = &self.gpu;
        let viewport = &mut self.viewport;
        viewport.show_wireframe = view.wireframe;
        let cpu = RenderMeshCpu::extract(mesh);
        let gpu_mesh = GpuMesh::upload(
            &gpu.device,
            &viewport.layouts,
            &cpu,
            Vec4::new(0.75, 0.72, 0.68, 1.0),
        );
        let mut draws = vec![Draw::new(&gpu_mesh, DAffine3::IDENTITY)];

        // Framed from the mesh's own bounds, so a specimen spanning nine orders
        // of magnitude is as visible as a unit cube and the goldens do not have
        // to be hand-tuned per specimen.
        let bounds = mesh.bounds();
        let center = (bounds.min + bounds.max) * 0.5;
        let radius = ((bounds.max - bounds.min).length() * 0.5).max(1e-6);
        let mut camera = Camera::default();
        // Barely any padding. Every pixel the model does not cover is a pixel
        // that cannot disagree with the golden, and a first pass at 1.4x left the
        // specimens filling about a sixth of the frame — a harness with five
        // times less signal than it could have for the same thirty kilobytes.
        camera.frame_sphere(
            DVec3::new(center.x as f64, center.y as f64, center.z as f64),
            radius as f64 * 1.05,
            1.0,
        );
        camera.orbit(view.yaw, view.pitch);

        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visual regression"),
            });
        viewport.render(&mut encoder, &self.texture_view, &draws);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(BYTES_PER_ROW),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().expect("map callback").expect("map");
        let data = self.readback.slice(..).get_mapped_range();
        let pixels = data.as_chunks::<4>().0.to_vec();
        drop(data);
        self.readback.unmap();

        Frame { pixels }
    }
}

fn load_golden(path: &std::path::Path) -> Option<Frame> {
    let decoded = image::open(path).ok()?.to_rgba8();
    if decoded.width() != SIZE || decoded.height() != SIZE {
        return None;
    }
    Some(Frame {
        pixels: decoded.as_raw().as_chunks::<4>().0.to_vec(),
    })
}

fn save_golden(path: &std::path::Path, frame: &Frame) {
    std::fs::create_dir_all(path.parent().expect("a parent")).expect("create goldens dir");
    image::save_buffer(path, &frame.to_rgba(), SIZE, SIZE, image::ColorType::Rgba8)
        .expect("write golden");
}

#[test]
fn every_specimen_matches_its_golden() {
    let Some(mut harness) = Harness::new() else {
        // Skipped rather than failed: a CI runner without a GPU reporting a
        // rendering regression it never rendered is worse than no signal.
        eprintln!("no usable GPU adapter; skipping visual regression");
        return;
    };

    let dir = goldens_dir();
    let mut missing = Vec::new();
    let mut regressed = Vec::new();

    for specimen in corpus() {
        for view in VIEWS {
            let name = format!("{}_{}.png", specimen.name, view.name);
            let path = dir.join(&name);
            let frame = harness.render(&specimen.mesh, view);

            if updating() {
                save_golden(&path, &frame);
                continue;
            }
            let Some(golden) = load_golden(&path) else {
                missing.push(name);
                continue;
            };
            let difference = frame.difference(&golden);
            if difference > PIXEL_BUDGET {
                // The actual image is written next to the golden so the failure
                // can be looked at, which is the only way to act on it.
                save_golden(
                    &dir.join(format!("{}_{}.actual.png", specimen.name, view.name)),
                    &frame,
                );
                regressed.push(format!(
                    "{name}: {:.2}% of pixels differ (budget {:.2}%) — {}",
                    difference * 100.0,
                    PIXEL_BUDGET * 100.0,
                    specimen.hazard
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "no golden for {missing:?}\n\
         run with UPDATE_GOLDENS=1 to create them, then look at what you committed"
    );
    assert!(
        regressed.is_empty(),
        "visual regressions:\n  {}",
        regressed.join("\n  ")
    );
}

#[test]
fn every_golden_shows_enough_of_the_model() {
    // A golden that is mostly background is a golden that cannot fail. This is
    // the precondition the whole harness rests on, and it is not self-evident:
    // the camera frames each specimen from its own bounds, so a change to the
    // framing, to `frame_sphere`, or to a specimen's proportions can quietly
    // shrink a model to a speck while every comparison keeps passing.
    let Some(mut harness) = Harness::new() else {
        eprintln!("no usable GPU adapter; skipping");
        return;
    };
    let background = harness.background();
    let mut thin = Vec::new();
    for specimen in corpus() {
        for view in VIEWS {
            let coverage = harness.render(&specimen.mesh, view).coverage(background);
            // Flat specimens seen edge-on are legitimately thin, so this is a
            // floor on the *best* view rather than on every one.
            if coverage > MINIMUM_COVERAGE {
                continue;
            }
            thin.push(format!(
                "{}_{}: {:.1}%",
                specimen.name,
                view.name,
                coverage * 100.0
            ));
        }
        let covered = VIEWS
            .iter()
            .map(|v| harness.render(&specimen.mesh, *v).coverage(background))
            .fold(0.0f32, f32::max);
        assert!(
            covered > MINIMUM_COVERAGE,
            "{} covers at most {:.1}% of the frame in any view; \
             its goldens cannot catch anything",
            specimen.name,
            covered * 100.0
        );
    }
    // Not an assertion: reported so that a specimen drifting towards invisible
    // is noticeable before it gets there.
    if !thin.is_empty() {
        eprintln!("thin views (informational): {}", thin.join(", "));
    }
}

/// How much of the frame a specimen must cover in its best view.
const MINIMUM_COVERAGE: f32 = 0.10;

#[test]
fn the_views_disagree_with_each_other() {
    // The harness's own precondition. Four views that all render the same pixels
    // are one view, and a regression visible only from behind would slip through
    // every one of them. Checked on the cube, which is symmetric enough that a
    // broken camera would plausibly produce four identical frames.
    //
    // Not on the cube any more: the studio lights turn with the camera, as
    // Blender's do, so a cube seen square from the front and from the back is
    // the same picture — correctly. A concave outline is not.
    let specimens = corpus();
    let cube = &specimens
        .iter()
        .find(|s| s.name == "concave_ngon")
        .expect("the concave specimen")
        .mesh;
    let Some(mut harness) = Harness::new() else {
        eprintln!("no usable GPU adapter; skipping");
        return;
    };
    let first = harness.render(cube, VIEWS[0]);
    for view in &VIEWS[1..] {
        let other = harness.render(cube, *view);
        let difference = first.difference(&other);
        assert!(
            difference > PIXEL_BUDGET,
            "the {} view is indistinguishable from {} ({:.3}%)",
            view.name,
            VIEWS[0].name,
            difference * 100.0
        );
    }
}

#[test]
fn the_metric_separates_a_real_regression_from_rendering_noise() {
    // The claim the tolerance rests on: the differences this harness must ignore
    // are far smaller than the ones it must catch. Asserted with two synthetic
    // frames rather than by rendering, so the numbers are exact and the test does
    // not need a GPU.
    let base = Frame {
        pixels: (0..(SIZE * SIZE))
            .map(|i| {
                let v = (i % 200) as u8 + 30;
                [v, v, v, 255]
            })
            .collect(),
    };

    // Noise of the kind a different rasteriser produces: every pixel off by a
    // little, plus a band of genuinely different pixels along one edge standing
    // in for antialiasing on a silhouette.
    let noisy = Frame {
        pixels: base
            .pixels
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let edge = (i as u32).is_multiple_of(SIZE);
                let d = if edge { 90 } else { CHANNEL_TOLERANCE as u8 };
                [
                    p[0].saturating_add(d),
                    p[1].saturating_add(d),
                    p[2].saturating_add(d),
                    255,
                ]
            })
            .collect(),
    };
    let noise = base.difference(&noisy);
    assert!(
        noise <= PIXEL_BUDGET,
        "the tolerance rejects ordinary driver noise: {:.3}% > {:.3}%",
        noise * 100.0,
        PIXEL_BUDGET * 100.0
    );

    // A regression of the kind this exists to catch: a face shaded the wrong
    // way. Even a tenth of the frame is far past the budget.
    let regressed = Frame {
        pixels: base
            .pixels
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i < (SIZE * SIZE / 10) as usize {
                    [255 - p[0], 255 - p[1], 255 - p[2], 255]
                } else {
                    *p
                }
            })
            .collect(),
    };
    let regression = base.difference(&regressed);
    assert!(
        regression > PIXEL_BUDGET * 4.0,
        "a tenth of the frame inverted scored only {:.3}%",
        regression * 100.0
    );
}
