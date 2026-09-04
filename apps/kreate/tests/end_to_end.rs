//! End-to-end tests across every layer.
//!
//! These live in the application crate because it is the only one that depends
//! on all of `kreate-mesh`, `kreate-scene`, `kreate-ops` and `kreate-render` —
//! which is itself the dependency direction working as intended. The point is to
//! catch things that each layer's own tests cannot: that an operator's effect
//! reaches the framebuffer, and that undoing it puts the *pixels* back.

use glam::Vec4;
use kreate_ops::{Context, History, OpResult, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, Viewport};
use kreate_scene::Document;

const SIZE: u32 = 192;
const BYTES_PER_ROW: u32 = SIZE * 4;

fn registry() -> OperatorRegistry {
    let mut r = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut r);
    r
}

fn params(pairs: &[(&'static str, PropertyValue)]) -> PropertyMap {
    let mut p = PropertyMap::new();
    for (k, v) in pairs {
        p.set(k, v.clone());
    }
    p
}

/// Renders a document exactly the way the application does: one draw per visible
/// object, GPU geometry shared by mesh id, world transforms from the hierarchy.
///
/// Returns `None` when there is no GPU, so a runner without one skips rather
/// than reporting a failure it cannot act on.
fn render(doc: &Document, camera: &Camera) -> Option<Vec<[u8; 4]>> {
    render_as(doc, camera, false)
}

/// Renders the way the application does *in edit mode* when `editing` is set:
/// element overlays on, and the object-level selection highlight deliberately
/// off, because a whole orange wireframe would drown out the element selection
/// the user is actually looking at.
fn render_in_edit_mode(doc: &Document, camera: &Camera) -> Option<Vec<[u8; 4]>> {
    render_as(doc, camera, true)
}

fn render_as(doc: &Document, camera: &Camera, edit_mode: bool) -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("e2e target"),
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
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
    // The grid is anchored to the world, which would let it mask a change in the
    // geometry; these tests are about the objects.
    viewport.show_grid = false;
    viewport.show_elements = edit_mode;

    let mut uploaded: Vec<(kreate_scene::document::MeshId, GpuMesh)> = Vec::new();
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
        let cpu = RenderMeshCpu::extract(mesh);
        uploaded.push((
            mesh_id,
            GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &cpu,
                Vec4::new(0.75, 0.72, 0.68, 1.0),
            ),
        ));
    }

    let mut draws: Vec<Draw> = Vec::new();
    for (id, object) in doc.objects() {
        if !object.visible {
            continue;
        }
        let Some(mesh_id) = object.mesh() else {
            continue;
        };
        if let Some((_, m)) = uploaded.iter().find(|(mid, _)| *mid == mesh_id) {
            draws.push(
                Draw::new(m, doc.world_transform(id))
                    .selected(!edit_mode && doc.selection.contains(id))
                    .editing(edit_mode && doc.selection.contains(id)),
            );
        }
    }

    viewport.prepare(&gpu.device, &gpu.queue, camera, &mut draws);

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("e2e") });
    viewport.render(&mut encoder, &view, &draws);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
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
    Some(pixels)
}

fn differing(a: &[[u8; 4]], b: &[[u8; 4]]) -> f32 {
    let n = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (0..3).any(|i| (x[i] as i32 - y[i] as i32).abs() > 6))
        .count();
    n as f32 / a.len() as f32
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

/// Writes a frame to `target/e2e-<label>.png`.
///
/// Always, not behind a flag. Twice now a rendering defect has been diagnosed
/// wrongly by reading the code and correctly by looking at the picture, so the
/// picture is written every run.
fn dump(pixels: &[[u8; 4]], label: &str) {
    let mut bytes = Vec::with_capacity(pixels.len() * 4);
    for p in pixels {
        bytes.extend_from_slice(p);
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("e2e-{label}.png"));
    if let Some(img) = image::RgbaImage::from_raw(SIZE, SIZE, bytes) {
        let _ = img.save(&path);
        eprintln!("wrote {}", path.display());
    }
}

/// A fixed camera, so two renders of the same document are comparable.
fn camera() -> Camera {
    Camera {
        pivot: glam::DVec3::ZERO,
        distance: 12.0,
        ..Camera::default()
    }
}

// --- document-level chain ---------------------------------------------------

#[test]
fn a_session_of_operators_builds_a_scene_and_undoes_cleanly() {
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();

    let steps: &[(&str, &[(&'static str, PropertyValue)])] = &[
        ("object.add_primitive", &[("shape", PropertyValue::Enum(1))]),
        ("object.duplicate", &[("linked", PropertyValue::Bool(true))]),
        (
            "object.translate",
            &[("translation", PropertyValue::Vec3([0.0, 3.0, 0.0]))],
        ),
        (
            "object.add_primitive",
            &[
                ("shape", PropertyValue::Enum(5)),
                ("location", PropertyValue::Vec3([4.0, 0.0, 0.0])),
            ],
        ),
    ];
    for (id, p) in steps {
        assert_eq!(
            h.dispatch(&r, &mut doc, &ctx, id, &params(p)).expect(id),
            OpResult::Finished,
            "{id}"
        );
    }

    assert_eq!(doc.object_count(), 3);
    assert_eq!(doc.mesh_count(), 2, "the linked duplicate shares geometry");

    // Undo everything, one step at a time.
    while h.can_undo() {
        h.undo(&mut doc);
    }
    assert_eq!(doc.object_count(), 0);
    assert!(doc.selection.is_empty());

    // And redo everything back.
    while h.can_redo() {
        h.redo(&r, &mut doc, &ctx).expect("redo");
    }
    assert_eq!(doc.object_count(), 3);
    assert_eq!(doc.mesh_count(), 2);
}

#[test]
fn the_session_reads_back_as_a_runnable_script() {
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[("shape", PropertyValue::Enum(5))]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.scale",
        &params(&[("scale", PropertyValue::Vec3([2.0, 2.0, 2.0]))]),
    )
    .unwrap();

    let script = h.to_script();
    assert_eq!(script.lines().count(), 2);
    // Every line must name an operator that actually exists, or "copy as script"
    // produces something that cannot be run.
    for line in script.lines() {
        let id = line
            .trim_start_matches("kreate.ops.")
            .split('(')
            .next()
            .unwrap();
        assert!(r.get(id).is_some(), "script names unknown operator `{id}`");
    }
}

// --- through to pixels ------------------------------------------------------

#[test]
fn an_operator_changes_what_is_drawn() {
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[("shape", PropertyValue::Enum(1))]),
    )
    .unwrap();

    let before = gpu_or_skip!(render(&doc, &camera()));

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.scale",
        &params(&[("scale", PropertyValue::Vec3([2.0, 2.0, 2.0]))]),
    )
    .unwrap();
    let after = gpu_or_skip!(render(&doc, &camera()));

    let changed = differing(&before, &after);
    assert!(
        changed > 0.02,
        "scaling the cube changed only {:.2}% of the frame; the transform is not \
         reaching the renderer",
        changed * 100.0
    );
}

#[test]
fn undo_restores_the_rendered_image() {
    // The strongest end-to-end property available without a human looking at the
    // screen: an operator, then undo, must return the *pixels* to their original
    // state. It exercises the operator, the undo payload, the document, the mesh
    // kernel, the GPU extraction and the renderer in one assertion.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[("shape", PropertyValue::Enum(1))]),
    )
    .unwrap();

    let original = gpu_or_skip!(render(&doc, &camera()));

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.rotate",
        &params(&[
            ("angle", PropertyValue::Float(0.7)),
            ("axis", PropertyValue::Enum(1)),
        ]),
    )
    .unwrap();
    let rotated = gpu_or_skip!(render(&doc, &camera()));
    assert!(
        differing(&original, &rotated) > 0.02,
        "the rotation should have been visible in the first place"
    );

    assert!(h.undo(&mut doc));
    let restored = gpu_or_skip!(render(&doc, &camera()));
    let residual = differing(&original, &restored);
    assert!(
        residual < 0.002,
        "{:.3}% of pixels differ after undo; the document was not fully restored",
        residual * 100.0
    );
}

#[test]
fn adjusting_the_last_operation_is_visible_and_does_not_compound() {
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[("shape", PropertyValue::Enum(1))]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.scale",
        &params(&[("scale", PropertyValue::Vec3([2.0, 2.0, 2.0]))]),
    )
    .unwrap();
    let at_two = gpu_or_skip!(render(&doc, &camera()));

    // Adjust to 3x, then back to 2x. The image must return, which it cannot do
    // if adjusting compounds onto the previous result.
    for scale in [3.0, 2.0] {
        h.adjust_last(
            &r,
            &mut doc,
            &ctx,
            &params(&[("scale", PropertyValue::Vec3([scale, scale, scale]))]),
        )
        .unwrap();
    }
    let back_at_two = gpu_or_skip!(render(&doc, &camera()));

    assert_eq!(
        h.entries().len(),
        2,
        "adjusting must not add history entries"
    );
    let residual = differing(&at_two, &back_at_two);
    assert!(
        residual < 0.002,
        "{:.3}% of pixels differ; adjusting compounded instead of replacing",
        residual * 100.0
    );
}

#[test]
fn linked_duplicates_draw_twice_from_one_upload() {
    // Datablock indirection, verified at the framebuffer: two objects sharing one
    // mesh must produce two visible objects.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[("shape", PropertyValue::Enum(1))]),
    )
    .unwrap();
    let one = gpu_or_skip!(render(&doc, &camera()));

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.duplicate",
        &params(&[("linked", PropertyValue::Bool(true))]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.translate",
        &params(&[("translation", PropertyValue::Vec3([4.0, 0.0, 0.0]))]),
    )
    .unwrap();
    let two = gpu_or_skip!(render(&doc, &camera()));

    assert_eq!(doc.mesh_count(), 1, "only one mesh datablock");
    assert!(
        differing(&one, &two) > 0.02,
        "the second instance should be visible"
    );
}

#[test]
fn two_linked_duplicates_appear_at_two_different_places() {
    // Sharing one mesh datablock between two objects must still draw two
    // objects. The weaker check — "the frame changed" — passes even if both
    // draws land on top of each other, so this counts the separated regions.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[
            ("shape", PropertyValue::Enum(1)),
            ("size", PropertyValue::Float(1.5)),
            ("location", PropertyValue::Vec3([-3.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.duplicate",
        &params(&[("linked", PropertyValue::Bool(true))]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.translate",
        &params(&[("translation", PropertyValue::Vec3([6.0, 0.0, 0.0]))]),
    )
    .unwrap();
    assert_eq!(doc.mesh_count(), 1, "one datablock");
    assert_eq!(doc.object_count(), 2, "two objects");

    let pixels = gpu_or_skip!(render(&doc, &camera()));
    dump(&pixels, "two-linked-duplicates");

    // Count columns that contain geometry. Two cubes six metres apart, viewed
    // from the default three-quarter angle, occupy two separated bands with a
    // gap between them; one cube drawn twice occupies a single band.
    //
    // "Contains geometry" is measured against the corner pixel rather than an
    // absolute threshold: the clear colour is dark in linear space but lands
    // around 66 once the sRGB target has encoded it, so a plausible-looking
    // constant marks the whole background as occupied and every frame as one
    // band — including the ones that are correct.
    let background = pixels[0];
    let occupied: Vec<bool> = (0..SIZE)
        .map(|x| {
            (0..SIZE).any(|y| {
                let p = pixels[(y * SIZE + x) as usize];
                (0..3).any(|c| p[c].abs_diff(background[c]) > 30)
            })
        })
        .collect();
    let bands = occupied.windows(2).filter(|w| !w[0] && w[1]).count() + usize::from(occupied[0]);
    assert_eq!(
        bands, 2,
        "expected two separated objects, found {bands} band(s); the second \n         draw is probably overwriting the first one's transform"
    );
}

#[test]
fn selecting_one_of_two_linked_duplicates_highlights_only_that_one() {
    // Object-mode selection is per *draw*. Both objects share one mesh
    // datablock, so the highlight cannot travel in the mesh's own flags buffer
    // without lighting up both — which is the mistake this pins down.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[
            ("shape", PropertyValue::Enum(1)),
            ("size", PropertyValue::Float(1.5)),
            ("location", PropertyValue::Vec3([-3.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.duplicate",
        &params(&[("linked", PropertyValue::Bool(true))]),
    )
    .unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.translate",
        &params(&[("translation", PropertyValue::Vec3([6.0, 0.0, 0.0]))]),
    )
    .unwrap();
    // Duplicating selects the copy and deselects the original, which is what
    // makes this a one-of-two test rather than a none-or-both one.
    assert_eq!(doc.selection.objects.len(), 1, "the copy alone is selected");

    let pixels = gpu_or_skip!(render(&doc, &camera()));
    dump(&pixels, "selection-highlight");

    // The selection colour is a saturated orange; every other colour in the
    // frame — background, surface shading, ordinary wireframe — is neutral, so
    // "red much greater than blue" isolates it without a golden image.
    let mut left = 0usize;
    let mut right = 0usize;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let p = pixels[(y * SIZE + x) as usize];
            if i32::from(p[0]) - i32::from(p[2]) > 60 {
                if x < SIZE / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }

    assert!(
        right > 40,
        "the selected copy is on the right and should be outlined in orange, \
         found {right} orange pixels there"
    );
    assert_eq!(
        left, 0,
        "the unselected original must stay neutral; {left} orange pixels leaked \
         onto it, so the highlight is travelling with the shared datablock"
    );
}

#[test]
fn a_parented_child_follows_its_parent_on_screen() {
    // The hierarchy reaching the renderer through world_transform.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[("shape", PropertyValue::Enum(1))]),
    )
    .unwrap();
    let parent = doc.selection.active.unwrap();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[
            ("shape", PropertyValue::Enum(1)),
            ("location", PropertyValue::Vec3([3.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    let child = doc.selection.active.unwrap();
    doc.set_parent(child, Some(parent)).unwrap();

    let before = gpu_or_skip!(render(&doc, &camera()));

    // Move the parent only; the child must move with it.
    doc.selection.clear();
    doc.selection.add(parent);
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.translate",
        &params(&[("translation", PropertyValue::Vec3([0.0, 0.0, 3.0]))]),
    )
    .unwrap();

    let child_world = doc.world_transform(child).translation;
    assert!(
        (child_world.z - 3.0).abs() < 1e-9,
        "child should have followed the parent, world z = {}",
        child_world.z
    );

    let after = gpu_or_skip!(render(&doc, &camera()));
    assert!(
        differing(&before, &after) > 0.02,
        "the move should be visible"
    );
}

#[test]
fn a_gpu_cache_keyed_on_the_document_version_sees_every_change() {
    // The defect a user found by dragging a value in the properties panel and
    // watching nothing happen.
    //
    // Adjusting restores a snapshot and re-runs the operator. If the snapshot
    // carries its old version number back with it, the counter lands on exactly
    // the value it had before, and a cache comparing versions concludes nothing
    // changed — so the document is different and the screen is not.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();

    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[
            ("shape", PropertyValue::Enum(1)),
            ("size", PropertyValue::Float(2.0)),
        ]),
    )
    .unwrap();

    let mut versions = std::collections::HashSet::new();
    versions.insert(doc.version());

    for size in [4.0, 1.0, 6.0, 2.5] {
        h.adjust_last(
            &r,
            &mut doc,
            &ctx,
            &params(&[
                ("shape", PropertyValue::Enum(1)),
                ("size", PropertyValue::Float(size)),
            ]),
        )
        .unwrap();
        assert!(
            versions.insert(doc.version()),
            "version {} was seen before; a cache would skip this change",
            doc.version()
        );
    }

    // And the document really did change each time.
    let mesh = doc
        .objects()
        .find_map(|(_, o)| o.mesh())
        .expect("an object");
    let extent = doc.mesh(mesh).unwrap().bounds().size().x;
    assert!(
        (extent - 2.5).abs() < 1e-4,
        "final size is {extent}, expected 2.5"
    );
}

#[test]
fn adjusting_a_size_is_visible_on_screen() {
    // The same defect, checked where the user sees it. Two renders that differ
    // only by an adjusted parameter must not produce the same picture.
    let r = registry();
    let mut h = History::new();
    let mut doc = Document::new();
    let ctx = Context::default();
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &params(&[
            ("shape", PropertyValue::Enum(1)),
            ("size", PropertyValue::Float(1.0)),
        ]),
    )
    .unwrap();
    let small = gpu_or_skip!(render(&doc, &camera()));

    h.adjust_last(
        &r,
        &mut doc,
        &ctx,
        &params(&[
            ("shape", PropertyValue::Enum(1)),
            ("size", PropertyValue::Float(4.0)),
        ]),
    )
    .unwrap();
    let large = gpu_or_skip!(render(&doc, &camera()));

    let changed = differing(&small, &large);
    assert!(
        changed > 0.05,
        "adjusting the size changed only {:.2}% of the frame",
        changed * 100.0
    );
}

#[test]
fn an_empty_document_renders_without_panicking() {
    let doc = Document::new();
    let pixels = gpu_or_skip!(render(&doc, &camera()));
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
}

/// Pixels that are not the viewport background.
fn lit_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 90 || p[1] > 90 || p[2] > 90)
        .count()
}

/// The GPU mesh cache, driven the way the application drives it.
///
/// The distinction that matters: the cache is keyed on the *identity* of the
/// evaluated mesh handle, and slots persist across frames. A test that re-uploads
/// everything each frame — which is what the plain `render` helper above does —
/// cannot see a cache that fails to adopt a new handle, because it has no cache.
struct MeshCache {
    slots: Vec<(std::sync::Arc<kreate_mesh::MeshData>, GpuMesh)>,
}

impl MeshCache {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Mirrors `App::sync_gpu`, including the re-flagging path.
    fn sync(
        &mut self,
        gpu: &kreate_render::Gpu,
        layouts: &kreate_render::MeshLayouts,
        scene: &[(
            kreate_scene::ObjectId,
            std::sync::Arc<kreate_mesh::MeshData>,
        )],
    ) {
        let held: Vec<_> = self.slots.iter().map(|(a, _)| a.clone()).collect();
        let plan = kreate::gpu_sync::plan_sync(&held, scene);
        if !plan.needs_work(held.len()) {
            return;
        }
        let mut old: Vec<Option<(std::sync::Arc<kreate_mesh::MeshData>, GpuMesh)>> =
            self.slots.drain(..).map(Some).collect();
        for (slot, mesh) in &plan.reflags {
            let Some(Some((handle, gpu_mesh))) = old.get_mut(*slot) else {
                continue;
            };
            let cpu = RenderMeshCpu::extract(mesh);
            if gpu_mesh.update_flags(&gpu.queue, &cpu) {
                *handle = mesh.clone();
            } else {
                old[*slot] = None;
            }
        }
        let mut next = Vec::with_capacity(plan.total());
        for slot in &plan.keep {
            if let Some(entry) = old.get_mut(*slot).and_then(Option::take) {
                next.push(entry);
            }
        }
        for arc in &plan.uploads {
            let cpu = RenderMeshCpu::extract(arc);
            let uploaded =
                GpuMesh::upload(&gpu.device, layouts, &cpu, Vec4::new(0.75, 0.72, 0.68, 1.0));
            next.push((arc.clone(), uploaded));
        }
        self.slots = next;
    }

    /// How the draw loop finds a slot: by handle identity, and by nothing else.
    fn find(&self, mesh: &std::sync::Arc<kreate_mesh::MeshData>) -> Option<&GpuMesh> {
        self.slots
            .iter()
            .find(|(a, _)| std::sync::Arc::ptr_eq(a, mesh))
            .map(|(_, m)| m)
    }
}

#[test]
fn a_selection_only_change_leaves_the_object_findable_in_the_cache() {
    // The reported bug, reduced to the line that caused it: select a cube, press
    // Tab, and the cube disappears.
    //
    // Tab selects every element, which changes the mesh handle without changing
    // any geometry, so the sync plan says to keep the slot and rewrite its flag
    // buffers. The application produced that plan and then ignored the
    // re-flagging half of it — so the slot kept the *previous* handle, the draw
    // loop's `Arc::ptr_eq` matched nothing, and the object was quietly left out
    // of the frame.
    //
    // Asserted against the cache rather than against pixels, because that is
    // where the defect is: a pixel test that re-uploaded every frame passed
    // while the application was broken.
    let mut doc = Document::new();
    let object = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
    doc.selection.add(object);

    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let viewport = Viewport::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);

    let mut cache = MeshCache::new();
    let mut modifiers = kreate_scene::ModifierCache::new();

    let scene = modifiers.evaluate_scene(&doc);
    cache.sync(&gpu, &viewport.layouts, &scene);
    assert!(
        cache.find(&scene[0].1).is_some(),
        "the object was not in the cache before edit mode"
    );

    // What Tab does first: select every element.
    let r = registry();
    let mut h = History::new();
    let ctx = Context {
        mode: kreate_ops::Mode::EditMesh,
    };
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "mesh.select",
        &params(&[("action", PropertyValue::Enum(0))]),
    )
    .expect("select all elements");

    let scene = modifiers.evaluate_scene(&doc);
    cache.sync(&gpu, &viewport.layouts, &scene);
    assert!(
        cache.find(&scene[0].1).is_some(),
        "after a selection-only change the cache no longer matches the evaluated \
         mesh, so the object would not be drawn"
    );
    assert_eq!(
        cache.slots.len(),
        1,
        "the mesh was re-uploaded rather than re-flagged"
    );
}

#[test]
fn entering_edit_mode_does_not_make_the_object_disappear() {
    // Reported from the running application: select the cube, press Tab, and the
    // cube vanishes. Tab does two things — it turns on the element overlays and
    // it selects every element — so this reproduces both and checks the frame
    // still has geometry in it.
    //
    // Written against the pixels rather than against the draw list, because
    // "the object is in the draw list" is exactly what would still be true if
    // the overlays were painting over it or the cull were rejecting it.
    let mut doc = Document::new();
    let object = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
    doc.selection.add(object);

    let camera = camera();
    let Some(before) = render(&doc, &camera) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let lit_before = lit_pixels(&before);
    assert!(lit_before > 500, "the object was not visible to begin with");

    // What Tab does to the document: select every element of the target mesh.
    let r = registry();
    let mut h = History::new();
    // Edit mode, because that is the mode Tab puts the application into and
    // `mesh.select` polls false outside it.
    let ctx = Context {
        mode: kreate_ops::Mode::EditMesh,
    };
    h.dispatch(
        &r,
        &mut doc,
        &ctx,
        "mesh.select",
        &params(&[("action", PropertyValue::Enum(0))]),
    )
    .expect("select all elements");

    let after = render_in_edit_mode(&doc, &camera).expect("adapter worked a moment ago");
    let lit_after = lit_pixels(&after);
    assert!(
        lit_after > 500,
        "entering edit mode left {lit_after} lit pixels against {lit_before} before it"
    );
}
