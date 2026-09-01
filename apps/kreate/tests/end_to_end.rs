//! End-to-end tests across every layer.
//!
//! These live in the application crate because it is the only one that depends
//! on all of `kreate-mesh`, `kreate-scene`, `kreate-ops` and `kreate-render` —
//! which is itself the dependency direction working as intended. The point is to
//! catch things that each layer's own tests cannot: that an operator's effect
//! reaches the framebuffer, and that undoing it puts the *pixels* back.

use glam::{DAffine3, Vec4};
use kreate_ops::{Context, History, OpResult, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_render::{Camera, GpuMesh, GpuOptions, RenderMeshCpu, Viewport};
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
                &viewport.object_layout,
                &cpu,
                Vec4::new(0.75, 0.72, 0.68, 1.0),
            ),
        ));
    }

    let mut draws: Vec<(&GpuMesh, DAffine3)> = Vec::new();
    for (id, object) in doc.objects() {
        if !object.visible {
            continue;
        }
        let Some(mesh_id) = object.mesh() else {
            continue;
        };
        if let Some((_, m)) = uploaded.iter().find(|(mid, _)| *mid == mesh_id) {
            draws.push((m, doc.world_transform(id)));
        }
    }

    viewport.prepare(&gpu.queue, camera, &draws);

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
fn an_empty_document_renders_without_panicking() {
    let doc = Document::new();
    let pixels = gpu_or_skip!(render(&doc, &camera()));
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
}
