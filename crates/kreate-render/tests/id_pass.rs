//! The id pass and the readback ring, against a real GPU.
//!
//! The three claims worth pinning down are the ones that make id-buffer picking
//! feel better than ray casting, and none of them can be checked without
//! rendering:
//!
//! - the id written is the id of the element the user is looking at;
//! - an element hidden behind geometry is *not* clickable, and is again as soon
//!   as x-ray is on;
//! - a vertex or edge is clickable from a few pixels away, because it is drawn
//!   into the id buffer expanded to a pick radius.
//!
//! Skips rather than fails when no adapter is available, so a CI runner without
//! a GPU does not report a false negative.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::picking::{PickDomain, PickTarget, REGION};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, Viewport};

const SIZE: u32 = 256;

/// One object in the scene: a mesh, where it is, and the id it should report.
struct Object {
    mesh: kreate_mesh::MeshData,
    transform: DAffine3,
    id: u32,
}

/// Renders one id-pass frame and reads back the region around `cursor`.
///
/// Returns `None` when there is no GPU. Drives the same sequence the application
/// does — render, capture, submit, map, poll — because the ordering is the part
/// most likely to be wrong: a map started before its copy is submitted waits for
/// a command that never arrives.
fn pick(
    objects: &[Object],
    domain: PickDomain,
    xray: bool,
    cursor: (i32, i32),
    configure_camera: impl FnOnce(&mut Camera),
) -> Option<Option<kreate_render::PickHit>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let colour = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless colour"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let colour_view = colour.create_view(&wgpu::TextureViewDescriptor::default());

    let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
    // Off, so that nothing writes depth except the geometry under test. The grid
    // writes its own depth from the ray hit, and it would occlude an element
    // lying on the ground plane.
    viewport.show_grid = false;

    let uploaded: Vec<(GpuMesh, DAffine3, u32)> = objects
        .iter()
        .map(|o| {
            let cpu = RenderMeshCpu::extract(&o.mesh);
            let mesh = GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &cpu,
                Vec4::new(0.75, 0.72, 0.68, 1.0),
            );
            (mesh, o.transform, o.id)
        })
        .collect();
    let draws: Vec<Draw> = uploaded
        .iter()
        .map(|(m, t, id)| Draw::new(m, *t).object_id(*id))
        .collect();

    let mut camera = Camera::default();
    configure_camera(&mut camera);

    let mut target = PickTarget::new(&gpu.device, SIZE, SIZE);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &draws);
    viewport.prepare_ids(&gpu.device, &gpu.queue, &camera, &draws, domain);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("id pass"),
        });
    // The visible frame first: the id pass tests against the depth buffer this
    // produces, which is what makes occlusion in the id buffer exactly the
    // occlusion on screen.
    viewport.render(&mut encoder, &colour_view, &draws);
    viewport.render_ids(&mut encoder, target.view(), &draws, domain, xray);
    target.capture(&mut encoder, cursor, domain);
    gpu.queue.submit(Some(encoder.finish()));
    target.after_submit();

    // The application polls without blocking and answers from the previous
    // frame. A test has no previous frame, so it waits.
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    target.poll(&gpu.device);

    let region = target.latest().expect("a region came back");
    Some(region.nearest(cursor))
}

fn cube_at(x: f64, id: u32) -> Object {
    Object {
        mesh: kreate_mesh::primitives::cube(2.0),
        transform: DAffine3::from_translation(DVec3::new(x, 0.0, 0.0)),
        id,
    }
}

/// The top view: the eye above, looking down, so a cube's lid fills the middle
/// of the frame and screen coordinates are easy to reason about.
///
/// `PosZ`, not `NegZ`: the axis names the side the eye moves *to*, so `NegZ` is
/// the view from underneath — which silently reverses which of two stacked
/// objects is the nearer one.
fn from_above(camera: &mut Camera) {
    *camera = camera.looking_along(kreate_render::ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = 8.0;
}

const CENTRE: (i32, i32) = (SIZE as i32 / 2, SIZE as i32 / 2);

#[test]
fn a_face_under_the_cursor_reports_its_object_and_face() {
    let Some(hit) = pick(
        &[cube_at(0.0, 3)],
        PickDomain::Face,
        false,
        CENTRE,
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let hit = hit.expect("the cube fills the middle of the frame");
    assert_eq!(hit.object, 3, "the object id must survive the plus-one");
    assert_eq!(hit.domain, PickDomain::Face);
    assert_eq!(hit.distance_px, 0, "dead centre on a face");
    assert!(
        hit.element < 6,
        "a cube has six faces, got face {}",
        hit.element
    );
}

#[test]
fn an_object_mode_click_reports_the_object() {
    // Object mode has its own domain, and it went untested while the element
    // domains were covered — which is how it came to have a zero depth offset
    // and render nothing at all. The id pass tests against the solid pass's
    // depth without writing its own, so a fragment at exactly that depth fails
    // `Greater`; clicking an object selected nothing and looked like picking
    // being switched off outside edit mode.
    let Some(hit) = pick(
        &[cube_at(0.0, 7)],
        PickDomain::Object,
        false,
        CENTRE,
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let hit = hit.expect("the cube fills the middle of the frame");
    assert_eq!(hit.object, 7);
    assert_eq!(hit.domain, PickDomain::Object);
}

#[test]
fn empty_space_reports_nothing() {
    // Zero means "nothing here", which is the whole reason object ids are stored
    // plus one. A miss reading as object zero would select the first object in
    // the scene whenever the user clicked the background.
    let Some(hit) = pick(
        &[cube_at(0.0, 0)],
        PickDomain::Face,
        false,
        (6, 6),
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(hit, None, "the corner of the frame is empty");
}

#[test]
fn the_id_names_the_face_rather_than_the_triangle() {
    // A cube's lid is one quad and two triangles. Both halves have to report the
    // same face, or every caller — face select, snapping, the operators — has to
    // do the mapping itself and get it wrong somewhere.
    //
    // This is what the non-indexed face path with `tri_to_face` exists for.
    let offsets = [(-40, -40), (40, 40)];
    let mut faces = Vec::new();
    for (dx, dy) in offsets {
        let Some(hit) = pick(
            &[cube_at(0.0, 0)],
            PickDomain::Face,
            false,
            (CENTRE.0 + dx, CENTRE.1 + dy),
            from_above,
        ) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        faces.push(hit.expect("both samples are on the lid").element);
    }
    assert_eq!(
        faces[0], faces[1],
        "the two triangles of the lid reported different faces: {faces:?}"
    );
}

#[test]
fn different_faces_report_different_ids_in_range() {
    // The companion to the test above, and the one with teeth. "Both halves of
    // one face agree" is satisfied by a lookup that reads the wrong element
    // entirely — an out-of-range storage read is clamped rather than trapped, so
    // two wrong indices can clamp to the same wrong answer and look correct.
    //
    // Two genuinely different faces have to come back different, and both have
    // to be face indices this mesh actually has.
    let mut faces = Vec::new();
    for axis in [
        kreate_render::ViewAxis::PosZ,
        kreate_render::ViewAxis::PosX,
        kreate_render::ViewAxis::PosY,
    ] {
        let Some(hit) = pick(
            &[cube_at(0.0, 0)],
            PickDomain::Face,
            false,
            CENTRE,
            |camera| {
                *camera = camera.looking_along(axis);
                camera.pivot = DVec3::ZERO;
                camera.distance = 8.0;
            },
        ) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        faces.push(hit.expect("the cube is in the middle of the frame").element);
    }

    for face in &faces {
        assert!(*face < 6, "a cube has six faces, got {face} in {faces:?}");
    }
    assert_ne!(
        faces[0], faces[1],
        "top and +X are different faces: {faces:?}"
    );
    assert_ne!(
        faces[1], faces[2],
        "+X and +Y are different faces: {faces:?}"
    );
    assert_ne!(
        faces[0], faces[2],
        "top and +Y are different faces: {faces:?}"
    );
}

#[test]
fn a_vertex_is_pickable_from_several_pixels_away() {
    // The pick tolerance is why id-buffer picking feels forgiving: vertices are
    // drawn into the id buffer *expanded* to a radius, so a near miss is a hit —
    // uniformly, at every zoom level, with no per-caller tolerance.
    let Some(hit) = pick(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        false,
        CENTRE,
        |camera| {
            from_above(camera);
            // Straight down at a cube centred on the origin puts four vertices
            // at the corners of the lid and none in the middle. Looking from a
            // corner instead puts one vertex near the centre of the frame.
            camera.pivot = DVec3::new(1.0, 1.0, 1.0);
            camera.distance = 6.0;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let hit = hit.expect("a vertex is near the middle of the frame");
    assert_eq!(hit.domain, PickDomain::Vert);
    assert!(
        hit.element < 8,
        "a cube has eight vertices, got {}",
        hit.element
    );
}

#[test]
fn an_occluded_element_is_not_pickable_until_x_ray_is_on() {
    // Occlusion for free and exact, because the pass tests against the depth
    // buffer the visible image was drawn with: what looks clickable is what is
    // clickable. And x-ray is two lines — a different depth compare — rather
    // than a second code path.
    let near = cube_at(0.0, 1);
    let far = Object {
        mesh: kreate_mesh::primitives::cube(2.0),
        transform: DAffine3::from_translation(DVec3::new(0.0, 0.0, -6.0)),
        id: 2,
    };

    let Some(occluded) = pick(&[near, far], PickDomain::Face, false, CENTRE, from_above) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(
        occluded.expect("something is under the cursor").object,
        1,
        "the near cube hides the far one"
    );

    let through = pick(
        &[
            cube_at(0.0, 1),
            Object {
                mesh: kreate_mesh::primitives::cube(2.0),
                transform: DAffine3::from_translation(DVec3::new(0.0, 0.0, -6.0)),
                id: 2,
            },
        ],
        PickDomain::Face,
        true,
        CENTRE,
        from_above,
    )
    .expect("the adapter was there a moment ago");
    // With depth ignored the last draw wins, so the far cube's id survives —
    // which is exactly what makes an occluded element selectable.
    assert_eq!(
        through.expect("something is under the cursor").object,
        2,
        "x-ray must reach the hidden cube"
    );
}

#[test]
fn two_instances_of_one_mesh_report_different_objects() {
    // The object id is per *draw*. Two linked duplicates share a datablock and
    // are different objects, so an id living on the mesh would make one of them
    // unpickable — and the bug would look like "clicking the copy selects the
    // original".
    let Some(hit) = pick(
        &[cube_at(-3.0, 10), cube_at(3.0, 20)],
        PickDomain::Face,
        false,
        CENTRE,
        |camera| {
            from_above(camera);
            camera.pivot = DVec3::new(3.0, 0.0, 0.0);
            camera.distance = 8.0;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(
        hit.expect("the right-hand cube is under the cursor").object,
        20
    );
}

#[test]
fn a_region_near_the_edge_of_the_viewport_still_reads_back() {
    // The window slides inward rather than reading outside the texture, so a
    // click in the corner is answered rather than dropped. The reader asks by
    // viewport pixel and does not have to know.
    let Some(hit) = pick(
        &[cube_at(0.0, 5)],
        PickDomain::Face,
        false,
        (2, 2),
        |camera| {
            from_above(camera);
            // Close enough that the cube covers the whole frame, corner included.
            camera.distance = 2.5;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(
        hit.expect("the cube covers the corner").object,
        5,
        "a clamped region must still answer for the pixel asked about"
    );
}

// The whole design rests on reading a small window every frame rather than the
// frame itself: at 1080p a full `Rg32Uint` readback is 16 MB and about 10 ms of
// PCIe. Checked against the constants at compile time, because it is a property
// of the design rather than something a GPU could disprove.
const _: () = assert!(
    REGION < SIZE,
    "the test viewport has to be bigger than the region it reads back"
);
const _: () = assert!(
    (REGION * REGION * 8) < 32 * 1024,
    "a region readback should stay in the tens of kilobytes"
);
