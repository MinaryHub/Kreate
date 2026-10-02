//! Selected faces and vertex points, against a real GPU.
//!
//! Two claims, and the second is the one that matters for performance:
//!
//! - a selected element is drawn differently from an unselected one, in every
//!   domain, or the user cannot see what they have selected;
//! - **changing a selection costs three buffer writes and no geometry upload.**
//!   A renderer that rebuilds a mesh when the selection changes turns one click
//!   on a dense mesh into a visible stall and rubber-band selection into a
//!   slideshow. It is the classic mistake in this part of a modelling tool, and
//!   it is invisible until the mesh is big enough that it is too late.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3, Vec4};
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::MeshData;
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, Viewport};
use std::sync::Arc;

const SIZE: u32 = 256;
const BYTES_PER_ROW: u32 = SIZE * 4;

/// What to select before rendering.
#[derive(Clone, Copy, Default)]
struct Selection {
    faces: u32,
    verts: u32,
}

/// A cube with the first `n` elements of each domain selected.
fn cube_with(selection: Selection) -> MeshData {
    let mut mesh = kreate_mesh::primitives::cube(2.0);
    let faces = mesh.num_faces() as usize;
    let verts = mesh.num_verts() as usize;
    mesh.attributes_mut().insert(
        names::SELECT_FACE,
        Domain::Face,
        AttrData::Bool(Arc::new(
            (0..faces).map(|i| i < selection.faces as usize).collect(),
        )),
    );
    mesh.attributes_mut().insert(
        names::SELECT_VERT,
        Domain::Point,
        AttrData::Bool(Arc::new(
            (0..verts).map(|i| i < selection.verts as usize).collect(),
        )),
    );
    mesh
}

/// Pixels that read as the selection colour.
///
/// Measured as *warmth* — red well above blue — rather than against absolute
/// channel values. The target is sRGB, so the linear 0.16 blue of the selection
/// colour arrives as 116 and any "blue is low" threshold rejects the very
/// colour it is looking for. A relative test also survives the face fill being
/// blended over the surface at 28% and the dots being drawn opaque.
///
/// The neutral surface and the background are both achromatic, so both come out
/// at a warmth near zero.
fn selected_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 140 && (p[0] as i32 - p[2] as i32) > 30)
        .count()
}

/// Pixels that are neither the background nor the neutral surface.
fn pale_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 130 && p[1] > 130 && p[2] > 130)
        .count()
}

/// Renders one frame with the element overlays on.
///
/// `after_upload` runs with the uploaded mesh in hand, so a test can rewrite the
/// flag buffers *without* re-uploading anything — which is the whole point of
/// keeping them separate.
fn render(
    mesh: &MeshData,
    show_elements: bool,
    after_upload: impl FnOnce(&wgpu::Queue, &GpuMesh),
) -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless target"),
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
    viewport.show_elements = show_elements;
    // Off: the grid's X axis is red, and counting it as a selection highlight is
    // exactly the sort of thing that makes a pixel test pass for the wrong
    // reason.
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let cpu = RenderMeshCpu::extract(mesh);
    let uploaded = GpuMesh::upload(
        &gpu.device,
        &viewport.layouts,
        &cpu,
        Vec4::new(0.45, 0.45, 0.47, 1.0),
    );
    after_upload(&gpu.queue, &uploaded);

    let mut draws = if PAIR.with(|p| p.get()) {
        // Two copies: the left being edited, the right only standing there.
        vec![
            Draw::new(&uploaded, DAffine3::from_translation(DVec3::new(-1.2, 0.0, 0.0))).editing(true),
            Draw::new(&uploaded, DAffine3::from_translation(DVec3::new(1.2, 0.0, 0.0))),
        ]
    } else {
        vec![Draw::new(&uploaded, DAffine3::IDENTITY)]
    };
    let mut camera = Camera::default();
    camera = camera.looking_along(kreate_render::ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = 5.0;

    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless"),
        });
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
    let pixels: Vec<[u8; 4]> = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();
    Some(pixels)
}

/// Whether a previewed drag is adding or removing.
#[derive(Clone, Copy)]
enum PreviewKind {
    Add,
    Remove,
}

/// Renders with a rubber-band preview over the given elements.
fn render_with_preview(
    mesh: &MeshData,
    preview: Option<(PreviewKind, &[u32])>,
) -> Option<Vec<[u8; 4]>> {
    render_with_preview_applied(mesh, preview, true)
}

/// The same, with control over whether the preview applies to the draw.
fn render_with_preview_applied(
    mesh: &MeshData,
    preview: Option<(PreviewKind, &[u32])>,
    applies: bool,
) -> Option<Vec<[u8; 4]>> {
    render_preview_full(mesh, preview, applies, None)
}

/// The full harness, with the element count the preview declares.
///
/// The count is normally the mesh's own, but a bitset outlives an edit by a
/// frame or two — so it has to be possible to declare fewer than the mesh has
/// and check that the surplus bits are ignored.
fn render_preview_full(
    mesh: &MeshData,
    preview: Option<(PreviewKind, &[u32])>,
    applies: bool,
    elements: Option<u32>,
) -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless target"),
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
    viewport.show_elements = true;
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let cpu = RenderMeshCpu::extract(mesh);
    let uploaded = GpuMesh::upload(
        &gpu.device,
        &viewport.layouts,
        &cpu,
        Vec4::new(0.45, 0.45, 0.47, 1.0),
    );

    // A bitset written by hand rather than resolved by a dispatch: what is being
    // tested is that the overlays read it, not that the reduction fills it —
    // which `region_select` covers against a real drag.
    let words = 1024usize;
    let mut bits = vec![0u32; words];
    if let Some((_, elements)) = &preview {
        for e in *elements {
            bits[(*e as usize) >> 5] |= 1 << (*e & 31);
        }
    }
    let bits_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test preview bits"),
        size: (words * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&bits_buffer, 0, bytemuck::cast_slice(&bits));

    viewport.prepare_preview(
        &gpu.device,
        &gpu.queue,
        &bits_buffer,
        1,
        preview
            .as_ref()
            .map(|(kind, _)| kreate_render::PreviewSelection {
                domain: kreate_render::PickDomain::Face,
                deselect: matches!(kind, PreviewKind::Remove),
                elements: elements.unwrap_or_else(|| mesh.num_faces()),
            }),
    );

    let mut draws = vec![Draw::new(&uploaded, DAffine3::IDENTITY).preview(applies)];
    let mut camera = Camera::default();
    camera = camera.looking_along(kreate_render::ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = 5.0;
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless"),
        });
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
    let pixels: Vec<[u8; 4]> = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();
    Some(pixels)
}

#[test]
fn a_selected_face_is_filled() {
    let Some(none) = render(&cube_with(Selection::default()), true, |_, _| {}) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let all = render(
        &cube_with(Selection { faces: 6, verts: 0 }),
        true,
        |_, _| {},
    )
    .expect("the adapter was there a moment ago");

    let before = selected_pixels(&none);
    let after = selected_pixels(&all);
    assert!(
        after > before + 500,
        "selecting every face changed almost nothing: {before} then {after}"
    );
}

#[test]
fn an_unselected_face_is_not_filled() {
    // The companion assertion, and the one that catches a fill that ignores the
    // flags: a mesh with nothing selected must have no highlight at all.
    let Some(pixels) = render(&cube_with(Selection::default()), true, |_, _| {}) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let highlighted = selected_pixels(&pixels);
    assert!(
        highlighted < 200,
        "nothing is selected but {highlighted} pixels are highlighted"
    );
}

#[test]
fn vertex_points_are_drawn_as_handles() {
    // A vertex you cannot see is a vertex you cannot aim at, so unselected
    // vertices are drawn too — pale rather than invisible.
    let Some(without) = render(&cube_with(Selection::default()), false, |_, _| {}) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let with = render(&cube_with(Selection::default()), true, |_, _| {})
        .expect("the adapter was there a moment ago");

    let before = pale_pixels(&without);
    let after = pale_pixels(&with);
    assert!(
        after > before + 30,
        "no vertex dots appeared: {before} then {after}"
    );
}

#[test]
fn a_selected_vertex_is_drawn_in_the_selection_colour() {
    let Some(none) = render(&cube_with(Selection::default()), true, |_, _| {}) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let some = render(
        &cube_with(Selection { faces: 0, verts: 8 }),
        true,
        |_, _| {},
    )
    .expect("the adapter was there a moment ago");

    let before = selected_pixels(&none);
    let after = selected_pixels(&some);
    assert!(
        after > before + 40,
        "selecting every vertex changed nothing: {before} then {after}"
    );
}

#[test]
fn changing_a_selection_needs_no_geometry_upload() {
    // The load-bearing claim. The mesh is uploaded with nothing selected, then
    // only the flag buffers are rewritten — no positions, no normals, no
    // indices — and the highlight has to appear anyway.
    //
    // If this ever needs a re-upload to work, selection on a million-vertex mesh
    // becomes a stall on every click, and the buffer split that the whole GPU
    // mesh design rests on has stopped paying for itself.
    let selected = cube_with(Selection { faces: 6, verts: 8 });
    let Some(pixels) = render(&cube_with(Selection::default()), true, |queue, mesh| {
        // The flags alone, which is the path the application takes: a click
        // must not pay to extract a render mesh it does not use.
        let flags = kreate_render::ElementFlags::extract(&selected);
        assert!(
            mesh.update_flags(queue, &flags),
            "a selection change must not be rejected as a topology change"
        );
    }) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let highlighted = selected_pixels(&pixels);
    assert!(
        highlighted > 500,
        "the flags-only update did not reach the screen: {highlighted} pixels"
    );
}

#[test]
fn a_drag_in_progress_is_previewed_without_touching_the_selection() {
    // The payoff of the bitset being a bitset: the highlight is drawn straight
    // out of what the compute reduction wrote, so it follows the pointer with no
    // readback and no upload — and what the user sees during the drag is exactly
    // what the commit will apply.
    //
    // Nothing here is selected in the document. Everything orange on screen came
    // from the preview alone.
    let Some(without) = render_with_preview(&cube_with(Selection::default()), None) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let with = render_with_preview(
        &cube_with(Selection::default()),
        Some((PreviewKind::Add, &[0, 1, 2, 3, 4, 5])),
    )
    .expect("the adapter was there a moment ago");

    let before = selected_pixels(&without);
    let after = selected_pixels(&with);
    assert!(
        after > before + 500,
        "the preview never reached the screen: {before} then {after}"
    );
}

#[test]
fn a_removing_drag_previews_the_removal() {
    // Ctrl+drag takes elements out. Previewing only additions would make it look
    // like nothing was happening until the moment it took effect.
    let selected = cube_with(Selection { faces: 6, verts: 8 });
    let Some(untouched) = render_with_preview(&selected, None) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let removing = render_with_preview(&selected, Some((PreviewKind::Remove, &[0, 1, 2, 3, 4, 5])))
        .expect("the adapter was there a moment ago");

    let before = selected_pixels(&untouched);
    let after = selected_pixels(&removing);
    assert!(
        after < before / 2,
        "a removing drag still looked selected: {before} then {after}"
    );
}

#[test]
fn a_preview_that_does_not_apply_to_a_draw_is_ignored() {
    // A drag resolves elements of the one object being edited, and the bitset is
    // indexed by that mesh's numbering. Applied to a neighbour it would light up
    // whatever elements happened to share those indices — a highlight appearing
    // on an object the user is not even editing.
    let Some(elsewhere) = render_with_preview_applied(
        &cube_with(Selection::default()),
        Some((PreviewKind::Add, &[0, 1, 2, 3, 4, 5])),
        false,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let highlighted = selected_pixels(&elsewhere);
    assert!(
        highlighted < 200,
        "a preview meant for another object was drawn here: {highlighted} pixels"
    );
}

#[test]
fn bits_past_the_declared_element_count_are_ignored() {
    // The bitset outlives an edit by a frame or two, so it can carry bits for
    // elements the mesh no longer has. The declared count is what bounds it, and
    // an off-by-one there would highlight a face that is gone.
    //
    // The face a camera happens to see is not something to assume — `cube`'s
    // face order is its own business — so the visible one is *found* first, and
    // the bound is then checked either side of it. Testing a fixed index would
    // pass or fail on which face the primitive generator emitted first.
    let cube = cube_with(Selection::default());
    let probe = |elements: u32, bits: &[u32]| {
        render_preview_full(&cube, Some((PreviewKind::Add, bits)), true, Some(elements))
            .map(|pixels| selected_pixels(&pixels))
    };

    let Some(_) = probe(6, &[]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    // Which face fills the middle of the frame, whichever index that is.
    let visible = (0..6u32)
        .find(|f| probe(6, &[*f]).unwrap() > 500)
        .expect("looking straight down at a cube, one face fills the frame");

    // Declaring exactly enough includes it; one fewer excludes it.
    let included = probe(visible + 1, &[visible]).unwrap();
    let excluded = probe(visible, &[visible]).unwrap();
    assert!(
        included > 500,
        "face {visible} should be previewed when it is within the count: {included}"
    );
    assert!(
        excluded < 200,
        "face {visible} was previewed past the declared count: {excluded}"
    );
}

#[test]
fn a_flags_update_is_refused_when_the_topology_changed() {
    // The buffers are sized from the element counts, and a selection cannot
    // change those. A caller reaching this with a differently shaped mesh has a
    // topology change on its hands and needs a full upload — writing past the
    // end of the buffers instead would be a silent corruption.
    let rendered = render(&cube_with(Selection::default()), true, |queue, mesh| {
        let other =
            kreate_render::ElementFlags::extract(&kreate_mesh::primitives::uv_sphere(1.0, 8, 4));
        assert!(
            !mesh.update_flags(queue, &other),
            "a differently shaped mesh must be refused"
        );
    });
    if rendered.is_none() {
        eprintln!("no GPU adapter; skipping");
    }
}

thread_local! {
    /// Whether `render` draws two copies, only the first of them edited.
    static PAIR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// In edit mode only the mesh being edited shows its vertices and faces. The
/// object beside it is scenery: its elements cannot be picked, and drawn as
/// handles they were handles that did nothing.
#[test]
fn only_the_edited_mesh_shows_its_elements() {
    let mesh = kreate_mesh::primitives::cube(1.0);
    PAIR.with(|p| p.set(true));
    let plain = render(&mesh, false, |_, _| {});
    let shown = render(&mesh, true, |_, _| {});
    PAIR.with(|p| p.set(false));
    let (Some(plain), Some(shown)) = (plain, shown) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let half = |pixels: &[[u8; 4]], right: bool| -> Vec<[u8; 4]> {
        pixels
            .iter()
            .enumerate()
            .filter(|(i, _)| ((*i as u32 % SIZE) >= SIZE / 2) == right)
            .map(|(_, p)| *p)
            .collect()
    };
    let changed = |right: bool| {
        half(&plain, right).iter().zip(&half(&shown, right)).filter(|(a, b)| a != b).count()
    };
    assert!(changed(false) > 20, "the edited mesh shows no elements: {}", changed(false));
    assert_eq!(changed(true), 0, "the mesh not being edited shows its elements");
}
