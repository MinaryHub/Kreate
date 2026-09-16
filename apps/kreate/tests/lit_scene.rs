//! What a user gets by pressing the buttons.
//!
//! Everything else about lighting is checked a layer at a time: the datablock,
//! the join, the packing, the shading, the display transform. All of it passed
//! while the shipped program drew a black screen, because nothing asked the one
//! question a user asks — *add a cube, add a lamp, switch to rendered: is there
//! a picture?*
//!
//! So this builds the document through the operators the menu runs, with the
//! parameters the menu sends, and renders it the way the frame loop does. It is
//! the only test here that would notice a default nobody can see by.

use kreate::scenelights;
use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_render::viewport::{Draw, ShadingMode, Viewport};
use kreate_render::{Camera, GpuOptions, RenderMeshCpu};
use kreate_scene::Document;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
    Some(Gpu {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
    })
}

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Runs an operator the way the header menu does.
fn menu(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    op: &str,
    pairs: &[(&'static str, PropertyValue)],
) {
    let mut params = PropertyMap::new();
    for (k, v) in pairs {
        params.set(k, v.clone());
    }
    history
        .dispatch(registry, doc, &Context { mode: Mode::Object }, op, &params)
        .expect("dispatch");
}

/// A cube and a lamp, added exactly as the menu adds them.
fn a_scene_a_user_would_build(kind: u32) -> Document {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();

    // The header's Add menu sends `shape` for a cube and `kind` for a light,
    // and nothing else: every other parameter is the operator's default. That
    // is the point — the defaults are what is on trial.
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_primitive",
        &[("shape", PropertyValue::Enum(1))],
    );
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_light",
        &[("kind", PropertyValue::Enum(kind))],
    );
    doc
}

/// Renders the document's meshes under its lights, as the frame loop does.
fn shoot(gpu: &Gpu, doc: &Document) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let camera = Camera::default();

    // The same three calls `main` makes, in the same order, from the same
    // functions. A test that assembled the lights itself would be checking its
    // own arithmetic.
    viewport.set_world(scenelights::world_radiance(doc));
    viewport.set_display(
        &gpu.queue,
        kreate_render::display::Display {
            view: kreate_math::ViewTransform::AgX,
            exposure: scenelights::shot_exposure(doc, 24.0, camera.pivot) as f32,
        },
    );
    let lights = scenelights::placed_lights(doc);
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &lights);

    // The environment, through the same two calls the frame loop makes. A
    // no-op for a flat world, which is most of the tests here — and present in
    // all of them so that "the frame loop does this" stays true of the harness.
    if let Some(image) = kreate::environment::background_image(doc) {
        viewport.set_environment(
            &gpu.device,
            &gpu.queue,
            image.width(),
            image.height(),
            &kreate::environment::pixels_of(image),
        );
    }
    let (rotation, strength) = doc
        .active_world()
        .and_then(|id| doc.world(id))
        .map(|w| (w.rotation as f32, w.strength as f32))
        .unwrap_or((0.0, 1.0));
    viewport.set_world_shape(&gpu.queue, rotation, strength);

    // Through `world_transform`, like the frame loop: an object's own
    // transform is its parent's space and a lamp under a rig would be placed
    // wrongly by anything else.
    let meshes: Vec<_> = doc
        .objects()
        .filter_map(|(id, o)| match o.data {
            kreate_scene::document::ObjectData::Mesh(mesh) => {
                Some((doc.mesh(mesh)?, doc.world_transform(id)))
            }
            _ => None,
        })
        .collect();
    let uploaded: Vec<_> = meshes
        .iter()
        .map(|(mesh, transform)| {
            let cpu = RenderMeshCpu::extract(mesh);
            (
                kreate_render::GpuMesh::upload(
                    &gpu.device,
                    &viewport.layouts,
                    &cpu,
                    glam::Vec4::ONE,
                ),
                *transform,
            )
        })
        .collect();
    let mut draws: Vec<Draw> = uploaded
        .iter()
        .map(|(mesh, transform)| Draw::new(mesh, *transform))
        .collect();

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

#[test]
fn a_new_document_renders_a_picture_with_nothing_added() {
    // The end of a long thread, as one assertion: open the application, press
    // `Z`, see something.
    //
    // A new document used to contain a cube and nothing else, so the rendered
    // mode was correctly and uselessly black — nothing was lighting anything,
    // and until the Add menu shipped there was no way to change that. The fix
    // is not for the renderer to invent light, which §2.1 is emphatic about,
    // but for a new document to be a scene with some in it.
    let gpu = gpu_or_skip!(gpu());
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);

    let pixels = shoot(&gpu, &doc);
    let brightest = pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(
        brightest > 0.1,
        "a new document rendered black: brightest {brightest}"
    );
    let saturated = pixels.iter().filter(|p| p[0] >= 0.99).count();
    let lit = pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        saturated * 2 < lit.max(1),
        "a new document blew out: {saturated} of {lit} lit pixels clipped"
    );

    // And every panel the properties editor offers has something behind it. A
    // world nobody can reach is a tab that never appears; a camera nobody can
    // reach is an exposure nobody can set.
    assert!(doc.active_world().is_some(), "a new document has no world");
    assert!(doc.cameras().next().is_some(), "a new document has no camera");
    assert!(
        doc.objects()
            .any(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Light(_))),
        "a new document has no lamp"
    );
    // The cube is what is selected, because it is what somebody opening the
    // application is going to do something to.
    let active = doc.selection.active.and_then(|id| doc.object(id));
    assert!(
        matches!(
            active.map(|o| &o.data),
            Some(kreate_scene::document::ObjectData::Mesh(_))
        ),
        "a new document does not start on the cube"
    );
}

#[test]
fn adding_a_lamp_from_the_menu_lights_the_scene() {
    // The whole complaint in one test. Every kind, because the four have
    // different units and different defaults, and "1000" is a bright room in
    // candela and a dull afternoon in lux — a single default shared between
    // them would leave three of the four invisible.
    let gpu = gpu_or_skip!(gpu());
    for (kind, name) in [(0u32, "point"), (1, "sun"), (2, "spot"), (3, "area")] {
        let doc = a_scene_a_user_would_build(kind);
        let pixels = shoot(&gpu, &doc);

        let brightest = pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
        assert!(
            brightest > 0.1,
            "a {name} added from the menu left the picture black: brightest {brightest}"
        );

        // And not blown out either, which is the other way a default can be
        // useless: a picture where everything is white shows no shape at all.
        let saturated = pixels.iter().filter(|p| p[0] >= 0.99).count();
        let lit = pixels.iter().filter(|p| p[0] > 0.01).count();
        assert!(
            saturated * 2 < lit.max(1),
            "a {name} blew the picture out: {saturated} of {lit} lit pixels clipped"
        );
    }
}

#[test]
fn a_scene_with_no_lamp_is_black_and_that_is_why_the_menu_matters() {
    // The control, and the reason the test above is worth its runtime: a cube
    // on its own in rendered mode is black, correctly — there is nothing
    // lighting it. That was the shipped experience, because the menu had no way
    // to add the thing that fixes it.
    let gpu = gpu_or_skip!(gpu());
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_primitive",
        &[("shape", PropertyValue::Enum(1))],
    );

    let pixels = shoot(&gpu, &doc);
    let brightest = pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(brightest < 0.02, "an unlit scene was not black: {brightest}");
}

#[test]
fn turning_a_lamp_down_moves_the_meter_and_not_the_picture() {
    // The panel's end of it, and a result worth stating plainly because it
    // looks like a bug: with one lamp in the scene, turning its power down
    // changes nothing on screen. The meter follows it and the exposure opens by
    // exactly as much.
    //
    // That is what exposure *means*. A correctly exposed photograph of a room
    // does not say whether the room was lit by a bright lamp or a dim one; what
    // it says is how the lamps compare with each other and how the light falls
    // off. So the thing to check is that the meter moved, not that the picture
    // did — and the picture not moving is the assertion, not the omission.
    let gpu = gpu_or_skip!(gpu());
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = a_scene_a_user_would_build(0);

    let before_lux = scenelights::metered_illuminance(&doc, kreate_math::DVec3::ZERO)
        .expect("a lit scene");
    let bright = shoot(&gpu, &doc);
    menu(
        &registry,
        &mut history,
        &mut doc,
        "light.set_properties",
        &[
            ("kind", PropertyValue::Enum(0)),
            ("intensity", PropertyValue::Float(100.0)),
        ],
    );
    let after_lux = scenelights::metered_illuminance(&doc, kreate_math::DVec3::ZERO)
        .expect("still lit");
    let dim = shoot(&gpu, &doc);

    // A tenth of the power is a tenth of the reading.
    assert!(
        (before_lux / after_lux - 10.0).abs() < 0.1,
        "{before_lux} then {after_lux}"
    );
    let a = bright.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    let b = dim.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(
        (a - b).abs() < 0.02,
        "the meter did not keep up: {a} then {b}"
    );
}

#[test]
fn with_two_lamps_turning_one_down_changes_the_balance() {
    // What *is* visible, and the reason the panel is worth having. The meter
    // holds the overall level, so what an edit shows is the difference between
    // the lamps — which is what lighting a shot consists of.
    let gpu = gpu_or_skip!(gpu());
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = a_scene_a_user_would_build(0);

    // A second lamp on the other side, so the two light different faces.
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_light",
        &[
            ("kind", PropertyValue::Enum(0)),
            ("location", PropertyValue::Vec3([-4.0, 0.0, 1.0])),
        ],
    );
    let even = shoot(&gpu, &doc);

    // The second lamp is the active one, so the panel edits that one.
    menu(
        &registry,
        &mut history,
        &mut doc,
        "light.set_properties",
        &[
            ("kind", PropertyValue::Enum(0)),
            ("intensity", PropertyValue::Float(20.0)),
        ],
    );
    let lopsided = shoot(&gpu, &doc);

    let moved = even
        .iter()
        .zip(&lopsided)
        .filter(|(a, b)| (a[0] - b[0]).abs() > 0.05)
        .count();
    assert!(
        moved > 50,
        "turning one of two lamps down changed {moved} pixels"
    );
}

// ---------------------------------------------------------------------------
// The environment.
//
// §7.7's gate, applied to the pillar's last piece: an HDRI that can be loaded,
// saved, prefiltered and shaded is a feature the user does not have until
// picking one puts a sky on the screen. Everything below the interface has its
// own tests and all of them passed while there was no way to choose an image.

/// An equirectangular sky: bright on one side of the horizon, dark elsewhere.
///
/// Not uniform, deliberately. A uniform sky would make "the environment is
/// lighting the scene" and "the flat colour is" the same picture, which is the
/// identity the renderer's tests are *for* — here the point is that a direction
/// survives all the way to the screen.
fn a_sky(width: u32, height: u32) -> kreate_scene::Image {
    let mut image = kreate_scene::Image::new(
        "Sky",
        width,
        height,
        kreate_scene::PixelFormat::Rgba16F,
    );
    image.color_space = kreate_scene::ColorSpace::Raw;
    let (across, down) = image.tile_counts();
    for ty in 0..down {
        for tx in 0..across {
            let tile = image.tile_mut(tx, ty).expect("in range");
            for row in 0..kreate_scene::TILE {
                for column in 0..kreate_scene::TILE {
                    let x = tx * kreate_scene::TILE + column;
                    let y = ty * kreate_scene::TILE + row;
                    if x >= width || y >= height {
                        continue;
                    }
                    let u = (x as f32 + 0.5) / width as f32;
                    // Bright over one half of the sphere and dark over the
                    // other. Split by longitude rather than by height, so that
                    // *which way the camera looks* is not part of the claim —
                    // the first version of this asked whether the top of the
                    // frame was brighter than the bottom, and the answer
                    // depended on where `Camera::default` happens to aim.
                    //
                    // Neither half is the flat grey the comparison below uses.
                    // The first version made the dark half 0.05, which *is*
                    // that grey — so half the frame matched by construction and
                    // the test failed for a reason that had nothing to do with
                    // the renderer.
                    let value = if u < 0.5 { 8.0 } else { 0.8 };
                    let bits = half::f16::from_f32(value).to_le_bytes();
                    let at = (row * kreate_scene::TILE + column) as usize * 8;
                    for c in 0..3 {
                        tile.bytes_mut()[at + c * 2..at + c * 2 + 2].copy_from_slice(&bits);
                    }
                    tile.bytes_mut()[at + 6..at + 8]
                        .copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                }
            }
        }
    }
    image
}

/// A new document with a sky chosen, through the operator the panel runs.
fn a_scene_with_a_sky() -> Document {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    // The starting scene, because that is what a user has when they open the
    // world tab: a cube, a lamp, a camera and a world to point at an image.
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);

    // The application adds the datablock and the operator makes the link —
    // which is the split `pick_world_image` makes, for the reason an operator
    // cannot open a file.
    let id = doc.add_image(a_sky(128, 64));
    let uid = doc.image_uid(id).expect("assigned on add").raw();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "world.set_image",
        &[("uid", PropertyValue::Text(uid.to_string()))],
    );
    doc
}

#[test]
fn choosing_a_sky_puts_a_sky_on_the_screen() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let doc = a_scene_with_a_sky();
    assert!(
        matches!(
            doc.world(doc.active_world().expect("a world"))
                .expect("it is there")
                .background,
            kreate_scene::Background::Image(_)
        ),
        "the operator did not make the link"
    );

    let sky = shoot(&gpu, &doc);

    // It is the sky and not the clear colour: the same document with a flat
    // background renders a different picture.
    let mut flat = a_scene_with_a_sky();
    let world = flat.active_world().expect("a world");
    flat.world_mut(world).expect("it is there").background =
        kreate_scene::Background::Flat(kreate_math::SceneRgb::new(0.05, 0.05, 0.06));
    let grey = shoot(&gpu, &flat);
    assert!(
        differing(&sky, &grey) > (SIZE * SIZE / 4) as usize,
        "only {} pixels of {} differ between a sky and a grey world",
        differing(&sky, &grey),
        SIZE * SIZE
    );

    // And it is *oriented*: turning the world half a turn swaps the bright half
    // of the sphere for the dark one, so the picture has to change.
    //
    // Asked this way rather than "the top of the frame is brighter than the
    // bottom", which is what this was first — that answer depends on where
    // `Camera::default` happens to aim, and a test that depends on a default
    // nobody chose for it is a test that breaks for an unrelated reason.
    let mut turned = a_scene_with_a_sky();
    let world = turned.active_world().expect("a world");
    turned.world_mut(world).expect("it is there").rotation = std::f64::consts::PI;
    let other_way = shoot(&gpu, &turned);
    assert!(
        differing(&sky, &other_way) > (SIZE * SIZE / 8) as usize,
        "turning the world changed only {} pixels of {}",
        differing(&sky, &other_way),
        SIZE * SIZE
    );
}

/// How many pixels of two shots differ enough to see.
fn differing(a: &[[f32; 4]], b: &[[f32; 4]]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(p, q)| (p[0] - q[0]).abs() > 0.02)
        .count()
}

#[test]
fn turning_the_sky_off_again_goes_back_to_a_colour() {
    // The other half of the control, and the one that is easy to leave out: a
    // user who picks a sky to look at it has to be able to stop.
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = a_scene_with_a_sky();

    menu(
        &registry,
        &mut history,
        &mut doc,
        "world.set_image",
        &[("uid", PropertyValue::Text(String::new()))],
    );
    assert!(matches!(
        doc.world(doc.active_world().expect("a world"))
            .expect("it is there")
            .background,
        kreate_scene::Background::Flat(_)
    ));
    // The image stays in the document with nothing pointing at it, which is
    // what an unused datablock is — and what makes undoing the link safe.
    assert_eq!(doc.image_count(), 1);
}


