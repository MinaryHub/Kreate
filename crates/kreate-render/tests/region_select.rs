//! Box, circle and lasso selection against a real GPU.
//!
//! The claims that matter:
//!
//! - a box collects every element it covers and nothing it does not;
//! - occlusion still applies, so a box does not select through a wall unless
//!   x-ray is on;
//! - elements belonging to *other* objects are not folded in, however far the
//!   box strays;
//! - the shapes agree with what the user was shown — a circle is round, and a
//!   lasso follows its outline rather than its bounding box.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::picking::{PickDomain, PickTarget};
use kreate_render::region_select::{Collect, RegionSelect, SelectShape};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, Viewport};

const SIZE: u32 = 256;

struct Object {
    mesh: kreate_mesh::MeshData,
    transform: DAffine3,
    id: u32,
}

/// Renders the id pass and runs the reduction, returning the resolved elements.
///
/// Drives the same sequence the application does: render, dispatch, submit, map,
/// poll. The ordering is the part most likely to be wrong.
fn select(
    objects: &[Object],
    domain: PickDomain,
    xray: bool,
    shape: &SelectShape,
    collect: Collect,
    configure_camera: impl FnOnce(&mut Camera),
) -> Option<Vec<u32>> {
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
    // The grid writes its own depth and would occlude geometry on the ground
    // plane, which has nothing to do with what is being tested here.
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

    let ids = PickTarget::new(&gpu.device, SIZE, SIZE);
    let mut region = RegionSelect::new(&gpu.device);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &draws);
    viewport.prepare_ids(&gpu.device, &gpu.queue, &camera, &draws, domain);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("region select"),
        });
    viewport.render(&mut encoder, &colour_view, &draws);
    viewport.render_ids(&mut encoder, ids.view(), &draws, domain, xray);
    region.dispatch(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        ids.view(),
        (SIZE, SIZE),
        shape,
        collect,
    );
    gpu.queue.submit(Some(encoder.finish()));
    region.after_submit();

    // The application never blocks and answers from the previous frame. A test
    // has no previous frame, so it waits.
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    region.poll(&gpu.device);
    Some(region.resolved().expect("a result came back").to_vec())
}

fn cube_at(x: f64, id: u32) -> Object {
    Object {
        mesh: kreate_mesh::primitives::cube(2.0),
        transform: DAffine3::from_translation(DVec3::new(x, 0.0, 0.0)),
        id,
    }
}

/// The top view: the eye above, looking down. `PosZ`, because the axis names the
/// side the eye moves *to*.
fn from_above(camera: &mut Camera) {
    *camera = camera.looking_along(kreate_render::ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = 8.0;
}

/// The whole viewport.
fn everything() -> SelectShape {
    SelectShape::Box {
        from: (0.0, 0.0),
        to: (SIZE as f32, SIZE as f32),
    }
}

#[test]
fn a_box_over_the_whole_frame_collects_the_visible_faces() {
    let Some(selected) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Face,
        false,
        &everything(),
        Collect::Elements {
            object: 0,
            count: 6,
        },
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // Looking straight down at a cube: the lid, and nothing else, because the
    // sides are edge-on and the rest is behind it.
    assert_eq!(selected.len(), 1, "expected just the lid, got {selected:?}");
    assert!(selected[0] < 6);
}

#[test]
fn a_box_over_empty_space_collects_nothing() {
    let Some(selected) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Face,
        false,
        &SelectShape::Box {
            from: (0.0, 0.0),
            to: (8.0, 8.0),
        },
        Collect::Elements {
            object: 0,
            count: 6,
        },
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert!(selected.is_empty(), "got {selected:?}");
}

#[test]
fn a_box_collects_many_vertices_at_once() {
    // The point of the whole mechanism: one dispatch, a bitset back, and the
    // count is what the box covered rather than what a per-element loop managed.
    let Some(selected) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        true,
        &everything(),
        Collect::Elements {
            object: 0,
            count: 8,
        },
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(selected.len(), 8, "all eight, got {selected:?}");
    assert_eq!(selected, vec![0, 1, 2, 3, 4, 5, 6, 7], "and in order");
}

#[test]
fn occlusion_applies_unless_x_ray_is_on() {
    // A box must not select through a wall. Looking down at a cube, the four
    // bottom vertices are hidden behind the lid.
    let Some(front) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        false,
        &everything(),
        Collect::Elements {
            object: 0,
            count: 8,
        },
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let through = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        true,
        &everything(),
        Collect::Elements {
            object: 0,
            count: 8,
        },
        from_above,
    )
    .expect("the adapter was there a moment ago");

    assert!(
        front.len() < through.len(),
        "occlusion changed nothing: {} against {}",
        front.len(),
        through.len()
    );
    assert_eq!(through.len(), 8, "x-ray reaches all of them");
}

#[test]
fn another_objects_elements_are_not_folded_in() {
    // Two objects share an element numbering — face 0 exists in both — so a box
    // over a neighbour would select unrelated geometry in the object being
    // edited. The failure is silent and looks like a selection picking up faces
    // at random.
    //
    // Stated as directly as it can be: the box covers *only* the left-hand cube,
    // and the elements being collected belong to the right-hand one. The answer
    // has to be empty. Comparing counts instead would prove nothing, because two
    // cubes have the same six face indices.
    let scene = || [cube_at(-3.0, 1), cube_at(3.0, 2)];
    let left_half = SelectShape::Box {
        from: (0.0, 0.0),
        to: (SIZE as f32 / 2.0 - 8.0, SIZE as f32),
    };
    let view = |camera: &mut Camera| {
        from_above(camera);
        camera.distance = 14.0;
    };

    // First: the box does cover the left-hand cube, or the test proves nothing.
    let Some(theirs) = select(
        &scene(),
        PickDomain::Face,
        false,
        &left_half,
        Collect::Elements {
            object: 1,
            count: 6,
        },
        view,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert!(
        !theirs.is_empty(),
        "the box should cover the left-hand cube for this to mean anything"
    );

    let ours = select(
        &scene(),
        PickDomain::Face,
        false,
        &left_half,
        Collect::Elements {
            object: 2,
            count: 6,
        },
        view,
    )
    .expect("the adapter was there a moment ago");
    assert!(
        ours.is_empty(),
        "the other object's faces were folded in: {ours:?}"
    );
}

#[test]
fn an_object_mode_box_collects_the_objects_it_covers() {
    // The same reduction, gathering from the channel that names objects rather
    // than the one that names elements. Object mode has the same shapes, the
    // same occlusion and the same cost, so a second pipeline for it would be
    // this one with a different index.
    let Some(objects) = select(
        &[cube_at(-3.0, 1), cube_at(3.0, 2)],
        // Faces, because an object is picked by the surface the user clicked.
        PickDomain::Object,
        false,
        &everything(),
        Collect::Objects { count: 8 },
        |camera| {
            from_above(camera);
            camera.distance = 14.0;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(objects, vec![1, 2], "both cubes, decoded from the plus-one");
}

#[test]
fn an_object_mode_box_collects_only_what_it_covers() {
    let left_half = SelectShape::Box {
        from: (0.0, 0.0),
        to: (SIZE as f32 / 2.0 - 8.0, SIZE as f32),
    };
    let Some(objects) = select(
        &[cube_at(-3.0, 1), cube_at(3.0, 2)],
        PickDomain::Object,
        false,
        &left_half,
        Collect::Objects { count: 8 },
        |camera| {
            from_above(camera);
            camera.distance = 14.0;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(objects, vec![1], "only the left-hand cube: {objects:?}");
}

#[test]
fn an_object_mode_box_over_empty_space_collects_nothing() {
    // Zero means "nothing here", and object ids are stored plus one precisely so
    // that an untouched pixel is a miss rather than object zero. Collecting
    // objects is where that matters most: a miss reading as object zero would
    // select the first object in the scene whenever a box missed.
    let Some(objects) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Object,
        false,
        &SelectShape::Box {
            from: (0.0, 0.0),
            to: (8.0, 8.0),
        },
        Collect::Objects { count: 8 },
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert!(objects.is_empty(), "got {objects:?}");
}

#[test]
fn an_object_hidden_behind_another_is_not_collected() {
    // Occlusion, from the same depth buffer the visible frame was drawn with:
    // an object-mode box must not reach through a wall either.
    let hidden = Object {
        mesh: kreate_mesh::primitives::cube(2.0),
        transform: DAffine3::from_translation(DVec3::new(0.0, 0.0, -6.0)),
        id: 2,
    };
    let Some(objects) = select(
        &[cube_at(0.0, 1), hidden],
        PickDomain::Object,
        false,
        &everything(),
        Collect::Objects { count: 8 },
        from_above,
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(objects, vec![1], "the near cube hides the far one");
}

#[test]
fn a_circle_is_round_rather_than_square() {
    // A circle that behaved like its bounding box would reach into the corners.
    // Placed so the cube's lid fills the frame, a small circle in the centre
    // hits it and one in the corner of the same bounding box does not.
    let radius = 60.0;
    let centre = (SIZE as f32 / 2.0, SIZE as f32 / 2.0);
    let Some(inside) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        true,
        &SelectShape::Circle { centre, radius },
        Collect::Elements {
            object: 0,
            count: 8,
        },
        |camera| {
            from_above(camera);
            camera.distance = 4.0;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    // The same bounding box, but only its corner: outside the circle.
    let corner = SelectShape::Box {
        from: (centre.0 - radius, centre.1 - radius),
        to: (centre.0 - radius * 0.75, centre.1 - radius * 0.75),
    };
    let in_corner = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        true,
        &corner,
        Collect::Elements {
            object: 0,
            count: 8,
        },
        |camera| {
            from_above(camera);
            camera.distance = 4.0;
        },
    )
    .expect("the adapter was there a moment ago");

    assert!(
        !in_corner.is_empty(),
        "the corner of the bounding box should contain a vertex to distinguish"
    );
    assert!(
        !in_corner.iter().all(|v| inside.contains(v)),
        "the circle reached into its bounding box's corner: {inside:?} against {in_corner:?}"
    );
}

#[test]
fn a_lasso_follows_its_outline_rather_than_its_bounding_box() {
    // A triangle covering the lower-left half of the frame. Its bounding box is
    // the whole frame, so a lasso that tested the box would select everything.
    let s = SIZE as f32;
    let triangle = SelectShape::Lasso(vec![(0.0, 0.0), (s, 0.0), (0.0, s)]);
    let Some(in_triangle) = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        true,
        &triangle,
        Collect::Elements {
            object: 0,
            count: 8,
        },
        |camera| {
            from_above(camera);
            camera.distance = 4.0;
        },
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let in_box = select(
        &[cube_at(0.0, 0)],
        PickDomain::Vert,
        true,
        &everything(),
        Collect::Elements {
            object: 0,
            count: 8,
        },
        |camera| {
            from_above(camera);
            camera.distance = 4.0;
        },
    )
    .expect("the adapter was there a moment ago");

    assert!(!in_triangle.is_empty(), "the half it covers has vertices");
    assert!(
        in_triangle.len() < in_box.len(),
        "the lasso behaved like its bounding box: {in_triangle:?} against {in_box:?}"
    );
    for v in &in_triangle {
        assert!(
            in_box.contains(v),
            "vertex {v} is outside the frame entirely"
        );
    }
}

#[test]
fn a_degenerate_lasso_selects_nothing() {
    // A stray click registers as a one- or two-point gesture. Selecting
    // everything would be a destructive surprise from a slip of the hand.
    for points in [vec![], vec![(10.0, 10.0)], vec![(10.0, 10.0), (20.0, 20.0)]] {
        let Some(selected) = select(
            &[cube_at(0.0, 0)],
            PickDomain::Vert,
            true,
            &SelectShape::Lasso(points.clone()),
            Collect::Elements {
                object: 0,
                count: 8,
            },
            from_above,
        ) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(
            selected.is_empty(),
            "a {}-point lasso selected {selected:?}",
            points.len()
        );
    }
}

#[test]
fn a_second_drag_does_not_inherit_the_first() {
    // The reduction only ever *sets* bits, so the buffer has to be cleared each
    // dispatch. A stale bit would make every drag select everything the previous
    // ones did — a selection that only ever grows.
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
    viewport.show_grid = false;
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

    let cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let mesh = GpuMesh::upload(
        &gpu.device,
        &viewport.layouts,
        &cpu,
        Vec4::new(0.75, 0.72, 0.68, 1.0),
    );
    let draws = vec![Draw::new(&mesh, DAffine3::IDENTITY).object_id(0)];
    let mut camera = Camera::default();
    from_above(&mut camera);
    camera.distance = 4.0;

    let ids = PickTarget::new(&gpu.device, SIZE, SIZE);
    let mut region = RegionSelect::new(&gpu.device);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &draws);
    viewport.prepare_ids(&gpu.device, &gpu.queue, &camera, &draws, PickDomain::Vert);

    let run = |region: &mut RegionSelect, shape: &SelectShape| -> Vec<u32> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.render(&mut encoder, &colour_view, &draws);
        viewport.render_ids(&mut encoder, ids.view(), &draws, PickDomain::Vert, true);
        region.dispatch(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            ids.view(),
            (SIZE, SIZE),
            shape,
            Collect::Elements {
                object: 0,
                count: 8,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        region.after_submit();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        region.poll(&gpu.device);
        region.resolved().expect("a result").to_vec()
    };

    let wide = run(&mut region, &everything());
    assert!(!wide.is_empty(), "the first drag found something");

    let empty = run(
        &mut region,
        &SelectShape::Box {
            from: (0.0, 0.0),
            to: (4.0, 4.0),
        },
    );
    assert!(
        empty.is_empty(),
        "the second drag inherited the first: {empty:?}"
    );
}
