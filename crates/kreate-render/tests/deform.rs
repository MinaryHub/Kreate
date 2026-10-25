//! The GPU's answer to a drag has to be the CPU's answer.
//!
//! This is the test the deform path rests on. A preview that is *nearly* right
//! is worse than a slow one: the user shapes the mesh against what they see,
//! the pointer comes up, the CPU recomputes, and the result moves. So the claim
//! is equality with the path that has always been authoritative — to float
//! precision, for every kind of gesture, including a weighted one.
//!
//! Read back rather than looked at. A pixel test would confirm that *something*
//! moved; only the buffers say whether it moved to the right place.
//!
//! # The one thing a drag freezes, and why these tests have to know
//!
//! `RenderMeshCpu::extract` splits a mesh vertex into one render vertex per
//! distinct corner normal — and it deduplicates by the *encoded normal value*,
//! not by smooth group. So the render mesh's vertex set is a function of the
//! geometry, not of the topology alone, and a deformation can in principle
//! change it. A drag cannot: the split is frozen when it begins, because
//! following it would mean re-uploading the index buffer every frame, which is
//! the cost this whole path exists to avoid.
//!
//! That makes "the GPU agrees with the CPU" a claim with a precondition, so
//! every comparison below asserts the precondition rather than assuming it:
//! `assert_split_holds` fails loudly if the fixture's gesture moved the mesh
//! far enough to change the split. And `a_crease_sharp_enough_to_split_is_the_
//! known_limit` pins down what happens when it does — the limit is real, and
//! belongs in a test rather than in a comment.

use kreate_render::mesh_gpu::oct_decode;
use kreate_render::{Deform, Deformer, Gesture, GpuOptions, Placement, RenderMeshCpu};

/// The object transform a fixture's mesh sits under.
///
/// Almost every fixture here leaves it at the identity, which is deliberate but
/// not sufficient: local and world space are the same thing there, so a shader
/// applying a world gesture straight to local positions passes. `OBJECT` is a
/// real one — moved, turned and non-uniformly scaled — and the tests that use
/// it are the ones that can tell the difference.
fn identity() -> glam::DAffine3 {
    glam::DAffine3::IDENTITY
}

fn awkward_object() -> glam::DAffine3 {
    glam::DAffine3::from_scale_rotation_translation(
        glam::DVec3::new(3.0, 0.4, 1.75),
        glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -0.9, 1.3),
        glam::DVec3::new(-3.0, 7.0, 1.0),
    )
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// A device on the CPU — Mesa's lavapipe, on a runner with no GPU —
    /// whose arithmetic lands further from the CPU reference than any
    /// real GPU's: its sines and its specular lobes are a few parts in a
    /// thousand off, where the tests ask for one. Tolerated on such a
    /// device alone.
    software: bool,
}

/// The one device this test binary uses.
///
/// Shared rather than created per test, and a falsification sweep is what
/// forced the question: eighteen repeated runs of this binary created several
/// hundred devices between them and the driver eventually wedged, leaving
/// `poll(wait_indefinitely)` waiting forever — fifteen minutes elapsed against
/// two seconds of CPU.
///
/// There was never a reason for one each. No test here mutates device state,
/// so per-test creation buys no isolation and costs the driver eight times the
/// work. `wgpu::Device` and `Queue` are `Arc` handles, so cloning them is a
/// refcount bump and every call site stays as it was.
fn gpu() -> Option<Gpu> {
    static SHARED: std::sync::OnceLock<Option<(wgpu::Device, wgpu::Queue, bool)>> =
        std::sync::OnceLock::new();
    let (device, queue, software) = SHARED
        .get_or_init(|| {
            let opts = GpuOptions::default();
            let instance = kreate_render::Gpu::create_instance(&opts);
            let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
            let software = gpu.info.device_type == "Cpu";
            Some((gpu.device.clone(), gpu.queue.clone(), software))
        })
        .clone()?;
    Some(Gpu {
        device,
        queue,
        software,
    })
}

/// Skips when there is no adapter, which is the state of a headless runner.
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

fn read_back(gpu: &Gpu, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deform readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    data
}

/// Positions and packed normals, as the two paths both produce them.
struct Shaded {
    positions: Vec<glam::Vec3>,
    normals: Vec<u32>,
}

/// Runs a drag on the GPU, one frame per gesture, and reads back what the
/// rasteriser would draw at the end of it.
///
/// A *sequence* rather than one gesture, because §2.2's rule — every frame
/// recomputes from the rest state, never from the last frame — is only a claim
/// once there has been a previous frame. Each gesture gets its own encoder and
/// its own submit, so the passes really do run in order and the second frame
/// really does see what the first one left behind.
fn on_gpu(gpu: &Gpu, mesh: &kreate_mesh::MeshData, weights: &[f32], drags: &[Drag]) -> Shaded {
    let cpu = RenderMeshCpu::extract(mesh);
    // `MeshLayouts` is owned by a viewport rather than free-standing, and this
    // is the cheapest honest way to get one: nothing is drawn, but the layouts
    // that come back are the ones a real draw would bind.
    let viewport =
        kreate_render::Viewport::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&gpu.device);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("deform"),
        });
    let deform = Deform::begin(
        &gpu.device,
        &mut encoder,
        &deformer,
        &gpu_mesh,
        &cpu,
        weights,
    )
    .expect("a mesh with vertices and one weight each");
    gpu.queue.submit([encoder.finish()]);

    for drag in drags {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deform frame"),
            });
        deform.set(&gpu.queue, &drag.gesture);
        deform.dispatch(&mut encoder, &deformer);
        gpu.queue.submit([encoder.finish()]);
    }

    let verts = cpu.positions.len();
    Shaded {
        positions: to_positions(&read_back(gpu, gpu_mesh.positions(), (verts * 12) as u64)),
        normals: to_normals(&read_back(gpu, gpu_mesh.normals(), (verts * 4) as u64)),
    }
}

fn to_positions(bytes: &[u8]) -> Vec<glam::Vec3> {
    bytes
        .as_chunks::<12>()
        .0
        .iter()
        .map(|b| {
            let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            glam::Vec3::new(f(0), f(4), f(8))
        })
        .collect()
}

fn to_normals(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

/// What a drag is: a gesture, the object it lands on, and where it turns.
///
/// Carried together because the CPU mirror below needs all three and a
/// `Gesture` deliberately does not expose the placement it packed — the shader
/// reads it and nothing else should have to unpack matrices to agree with it.
#[derive(Clone, Copy)]
struct Drag {
    gesture: Gesture,
    to_world: glam::DAffine3,
    pivot: glam::DVec3,
}

impl Drag {
    fn moved(offset: glam::Vec3, to_world: glam::DAffine3, pivot: glam::DVec3) -> Self {
        Self {
            gesture: Gesture::moved(offset, place(to_world, pivot)),
            to_world,
            pivot,
        }
    }
    fn turned(rotation: glam::Quat, to_world: glam::DAffine3, pivot: glam::DVec3) -> Self {
        Self {
            gesture: Gesture::turned(rotation, place(to_world, pivot)),
            to_world,
            pivot,
        }
    }
    fn scaled(factor: glam::Vec3, to_world: glam::DAffine3, pivot: glam::DVec3) -> Self {
        Self {
            gesture: Gesture::scaled(factor, place(to_world, pivot)),
            to_world,
            pivot,
        }
    }
    fn nothing() -> Self {
        Self {
            gesture: Gesture::IDENTITY,
            to_world: glam::DAffine3::IDENTITY,
            pivot: glam::DVec3::ZERO,
        }
    }
}

fn place(to_world: glam::DAffine3, pivot: glam::DVec3) -> Placement {
    Placement::new(to_world, pivot).expect("the fixture's object is invertible")
}

/// The gesture applied to the document mesh, the way `ops::mesh` applies it.
fn deformed(
    mesh: &kreate_mesh::MeshData,
    weight_of: &impl Fn(u32) -> f32,
    drag: &Drag,
) -> kreate_mesh::MeshData {
    let mut moved = mesh.clone();
    let g = &drag.gesture;
    let offset = glam::DVec3::new(
        g.translation[0] as f64,
        g.translation[1] as f64,
        g.translation[2] as f64,
    );
    let scale = glam::DVec3::new(g.scale[0] as f64, g.scale[1] as f64, g.scale[2] as f64);
    let rotation = glam::DQuat::from_xyzw(
        g.rotation[0] as f64,
        g.rotation[1] as f64,
        g.rotation[2] as f64,
        g.rotation[3] as f64,
    );
    let to_local = drag.to_world.inverse();
    for (v, p) in moved.positions_mut().iter_mut().enumerate() {
        let w = weight_of(v as u32) as f64;
        if w == 0.0 {
            continue;
        }
        // World in, world out — the arrangement `run_transform` uses, and for
        // the same reason: the gesture is in world units and the mesh is in the
        // object's own.
        let world = drag.to_world.transform_point3(p.as_dvec3());
        // The rule §2.2 states and the shader mirrors: the *transform* is
        // blended towards the identity, not the result towards the rest
        // position. For a rotation those differ, and the difference is the whole
        // reason the gesture travels decomposed.
        let s = glam::DVec3::ONE.lerp(scale, w);
        let q = glam::DQuat::IDENTITY.slerp(rotation, w);
        let placed = drag.pivot + q * ((world - drag.pivot) * s) + offset * w;
        *p = to_local.transform_point3(placed).as_vec3();
    }
    moved
}

/// The CPU's answer: deform the document, then extract as always.
fn on_cpu(mesh: &kreate_mesh::MeshData, weight_of: &impl Fn(u32) -> f32, drag: &Drag) -> Shaded {
    let cpu = RenderMeshCpu::extract(&deformed(mesh, weight_of, drag));
    Shaded {
        positions: cpu
            .positions
            .iter()
            .map(|p| glam::Vec3::from_array(p.0))
            .collect(),
        normals: cpu.normals.clone(),
    }
}

/// The precondition, checked rather than hoped for.
///
/// A drag holds the vertex split it began with. Where the gesture would have
/// changed it the two paths are answering different questions, and comparing
/// them index by index would be meaningless — so say so, and say it about the
/// fixture rather than about the shader.
fn assert_split_holds(
    label: &str,
    mesh: &kreate_mesh::MeshData,
    weight_of: &impl Fn(u32) -> f32,
    drag: &Drag,
) {
    let rest = RenderMeshCpu::extract(mesh);
    let after = RenderMeshCpu::extract(&deformed(mesh, weight_of, drag));
    assert_eq!(
        rest.render_to_vert,
        after.render_to_vert,
        "{label}: this gesture changes the vertex split ({} render vertices at \
         rest, {} after), so the fixture is testing the frozen-split limit \
         rather than the shader. Use a gentler gesture, or \
         `a_crease_sharp_enough_to_split_is_the_known_limit`.",
        rest.positions.len(),
        after.positions.len()
    );
}

/// How far a GPU vertex may sit from the CPU's: a tenth of a millimetre
/// on a metre, ten times that on a software device (see `Gpu::software`).
fn position_tolerance(software: bool) -> f32 {
    if software {
        1e-3
    } else {
        1e-4
    }
}

fn compare(label: &str, gpu: &Shaded, cpu: &Shaded, software: bool) {
    let tolerance = position_tolerance(software);
    assert_eq!(
        gpu.positions.len(),
        cpu.positions.len(),
        "{label}: vertex count"
    );
    assert_eq!(
        gpu.normals.len(),
        cpu.normals.len(),
        "{label}: normal count"
    );

    let mut worst_position = 0.0f32;
    for (i, (a, b)) in gpu.positions.iter().zip(&cpu.positions).enumerate() {
        let d = (*a - *b).length();
        worst_position = worst_position.max(d);
        assert!(
            d < tolerance,
            "{label}: vertex {i} is at {a:?} on the GPU and {b:?} on the CPU"
        );
    }

    let mut worst_normal = 0.0f32;
    for (i, (a, b)) in gpu.normals.iter().zip(&cpu.normals).enumerate() {
        let (na, nb) = (oct_decode(*a), oct_decode(*b));
        let degrees = na.dot(nb).clamp(-1.0, 1.0).acos().to_degrees();
        worst_normal = worst_normal.max(degrees);
        assert!(
            degrees < 1.0,
            "{label}: normal {i} is {na:?} on the GPU and {nb:?} on the CPU, \
             {degrees} degrees apart"
        );
    }
    eprintln!(
        "{label}: {} vertices, worst position {worst_position:e}, \
         worst normal {worst_normal:.4} degrees",
        gpu.positions.len()
    );
}

/// One weight per *render* vertex, which is what the buffer is indexed by.
fn per_render_vertex(mesh: &kreate_mesh::MeshData, weight_of: &impl Fn(u32) -> f32) -> Vec<f32> {
    RenderMeshCpu::extract(mesh)
        .render_to_vert
        .iter()
        .map(|v| weight_of(*v))
        .collect()
}

/// Runs one gesture both ways and asserts they agree.
fn both_ways(
    gpu: &Gpu,
    label: &str,
    mesh: &kreate_mesh::MeshData,
    weight_of: impl Fn(u32) -> f32,
    drag: Drag,
) {
    frames_then(gpu, label, mesh, weight_of, &[drag]);
}

/// Runs a whole drag, and asserts the *last* frame is the CPU's answer to the
/// last gesture — with no trace of the ones before it.
fn frames_then(
    gpu: &Gpu,
    label: &str,
    mesh: &kreate_mesh::MeshData,
    weight_of: impl Fn(u32) -> f32,
    drags: &[Drag],
) {
    let last = drags.last().expect("a drag has at least one frame");
    assert_split_holds(label, mesh, &weight_of, last);
    let weights = per_render_vertex(mesh, &weight_of);
    let from_gpu = on_gpu(gpu, mesh, &weights, drags);
    let from_cpu = on_cpu(mesh, &weight_of, last);
    compare(label, &from_gpu, &from_cpu, gpu.software);
}

/// A smooth falloff about the origin, which is what proportional editing hands
/// the shader: a weight that is neither zero nor one across most of the mesh.
fn falloff(mesh: &kreate_mesh::MeshData, reach: f32) -> impl Fn(u32) -> f32 {
    let positions = mesh.positions().to_vec();
    move |v| (1.0 - positions[v as usize].length() / reach).clamp(0.0, 1.0)
}

/// Fails unless the weights really are graded.
///
/// Every one of these fixtures is worthless if its weights collapse to zero and
/// one: the `w == 0.0` early-out then hides a shader that ignores the weight
/// entirely, which is a break the first sweep of this file did not catch.
fn assert_graded(label: &str, mesh: &kreate_mesh::MeshData, weight_of: &impl Fn(u32) -> f32) {
    let weights = per_render_vertex(mesh, weight_of);
    let graded = weights.iter().filter(|w| **w > 0.05 && **w < 0.95).count();
    assert!(
        graded > 8,
        "{label}: only {graded} vertices are partly weighted; the falloff is untested"
    );
}

/// The interior of a grid, which is the shape a drag has: some vertices moved,
/// some not, and a boundary between them where a smooth group spans both.
fn interior(divisions: u32) -> impl Fn(u32) -> f32 {
    let side = divisions + 1;
    move |v| {
        let (row, col) = (v / side, v % side);
        if row > 0 && col > 0 && row < divisions && col < divisions {
            1.0
        } else {
            0.0
        }
    }
}

#[test]
fn a_move_deforms_the_same_way_on_both() {
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 6, 6);
    // Out of plane, so the normals genuinely change and this is not a test of
    // positions wearing a normal check. Small enough that the crease it makes
    // stays under the auto-smooth angle — `assert_split_holds` is what says so.
    both_ways(
        &g,
        "move",
        &mesh,
        interior(6),
        Drag::moved(
            glam::Vec3::new(0.1, -0.05, 0.12),
            identity(),
            glam::DVec3::ZERO,
        ),
    );
}

#[test]
fn a_rotation_deforms_the_same_way_on_both() {
    let g = gpu_or_skip!(gpu());
    // A cube, because its six faces have six distinct normals and a rotation
    // keeps them distinct: the split cannot move, so the gesture can be a large
    // one. Every render vertex's group is a single face here, which is the
    // degenerate end of the normal pass.
    let mesh = kreate_mesh::primitives::cube(2.0);
    both_ways(
        &g,
        "rotate a whole cube",
        &mesh,
        |_| 1.0,
        Drag::turned(
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.7, -0.4, 1.1),
            identity(),
            glam::DVec3::new(0.3, -0.2, 0.5),
        ),
    );
}

#[test]
fn a_non_uniform_scale_deforms_the_same_way_on_both() {
    let g = gpu_or_skip!(gpu());
    // Different in every axis. A shader that used one component for all three,
    // or that transformed the normals by the scale rather than recomputing
    // them, is wrong here and right for a uniform one — a scale needs the
    // inverse transpose, which is not the scale.
    let mesh = kreate_mesh::primitives::cube(2.0);
    both_ways(
        &g,
        "scale a whole cube",
        &mesh,
        |_| 1.0,
        Drag::scaled(
            glam::Vec3::new(1.7, 0.4, 2.5),
            identity(),
            glam::DVec3::new(-0.3, 0.4, 0.0),
        ),
    );
}

#[test]
fn a_partial_scale_deforms_the_same_way_on_both() {
    let g = gpu_or_skip!(gpu());
    // The same arithmetic, but now with a boundary in it: the ring stays put
    // while the interior shrinks, so the faces that straddle the boundary tilt
    // and their vertices' groups span moved and unmoved geometry.
    let mesh = kreate_mesh::primitives::grid(4.0, 6, 6);
    both_ways(
        &g,
        "scale a grid's interior",
        &mesh,
        interior(6),
        Drag::scaled(
            glam::Vec3::new(0.85, 0.9, 1.0),
            identity(),
            glam::DVec3::ZERO,
        ),
    );
}

#[test]
fn a_weighted_rotation_deforms_the_same_way_on_both() {
    let g = gpu_or_skip!(gpu());
    // Proportional editing, which is where transforming the normals instead of
    // recomputing them would have been visibly wrong: the surface genuinely
    // bends, so its normals are not the rest normals turned by anything.
    //
    // The weight has to be graded — neither zero nor one across a useful part
    // of the mesh — because in between is the only place a normalised lerp
    // would differ from a slerp, and in between is where this tool lives.
    let mesh = kreate_mesh::primitives::grid(4.0, 8, 8);
    let weight_of = falloff(&mesh, 2.5);
    assert_graded("weighted rotate", &mesh, &weight_of);

    both_ways(
        &g,
        "weighted rotate",
        &mesh,
        weight_of,
        Drag::turned(
            glam::Quat::from_rotation_x(0.5),
            identity(),
            glam::DVec3::ZERO,
        ),
    );
}

#[test]
fn a_weighted_move_deforms_the_same_way_on_both() {
    // The gap the first falsification sweep found: every move here used weights
    // of exactly zero or one, and a shader that ignored the weight altogether
    // passed all of them. A vertex at weight zero returns early, and one at
    // weight one is unaffected by the multiply — so the bug had nowhere to show.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 8, 8);
    let weight_of = falloff(&mesh, 2.5);
    assert_graded("weighted move", &mesh, &weight_of);
    both_ways(
        &g,
        "weighted move",
        &mesh,
        weight_of,
        Drag::moved(
            glam::Vec3::new(0.1, 0.05, 0.3),
            identity(),
            glam::DVec3::ZERO,
        ),
    );
}

#[test]
fn a_weighted_scale_deforms_the_same_way_on_both() {
    // The same gap, for the scale. And the pivot is below the plane so that the
    // z factor does something — a flat grid scaled about its own plane stays
    // flat, and the normals would never move.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 8, 8);
    let weight_of = falloff(&mesh, 2.5);
    assert_graded("weighted scale", &mesh, &weight_of);
    both_ways(
        &g,
        "weighted scale",
        &mesh,
        weight_of,
        Drag::scaled(
            glam::Vec3::new(1.2, 1.15, 1.6),
            identity(),
            glam::DVec3::new(0.0, 0.0, -1.0),
        ),
    );
}

#[test]
fn a_rotation_past_a_half_turn_takes_the_shorter_arc() {
    // 3.6 radians is more than pi, so the gesture's quaternion has a negative
    // w and `dot(identity, q)` is negative. A slerp that does not flip the sign
    // first goes the long way round: at half weight it lands almost a full turn
    // from where it should, in the opposite direction.
    //
    // The weight is a uniform half rather than a falloff, and that is the trick
    // that makes this testable at all: it is a rigid rotation of the whole mesh,
    // so no dihedral changes and the vertex split cannot move — but the slerp is
    // still evaluated strictly between its endpoints, which is the only place
    // the sign matters.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::cube(2.0);
    let quarter_turn_past_pi = glam::Quat::from_rotation_x(3.6);
    assert!(
        quarter_turn_past_pi.w < 0.0,
        "this fixture needs a quaternion on the far side of the identity"
    );
    both_ways(
        &g,
        "past a half turn",
        &mesh,
        |_| 0.5,
        Drag::turned(
            quarter_turn_past_pi,
            identity(),
            glam::DVec3::new(0.2, -0.1, 0.3),
        ),
    );
}

#[test]
fn a_composed_gesture_agrees_on_the_order() {
    // Nothing in the program sends one of these: a gesture is one of move,
    // rotate or scale, so two of the three are always the identity and the
    // order they compose in is unobservable. That is exactly why it is worth
    // pinning — the CPU mirror in this file and the shader agree on scale, then
    // rotate about the pivot, then translate, and if a combined gesture is ever
    // wanted the agreement is already tested rather than assumed.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::cube(2.0);
    both_ways(
        &g,
        "composed",
        &mesh,
        |_| 1.0,
        Drag {
            gesture: Gesture {
                translation: [0.4, -0.2, 0.7, 0.0],
                rotation: glam::Quat::from_euler(glam::EulerRot::XYZ, 0.6, 0.3, -0.9).to_array(),
                scale: [1.4, 0.6, 2.1, 0.0],
                place: place(identity(), glam::DVec3::new(0.1, 0.5, -0.3)),
            },
            to_world: identity(),
            pivot: glam::DVec3::new(0.1, 0.5, -0.3),
        },
    );
}

#[test]
fn every_frame_starts_from_the_rest_state() {
    // §2.2's rule, and the reason it is a rule: a long drag that applied deltas
    // would accumulate float error, and a cancel would have nothing exact to go
    // back to. So the tenth frame of a drag has to be identical to the first
    // frame of a drag that went straight there.
    //
    // This is also the only test that can catch a shader which *skips* writing
    // an unweighted vertex instead of writing the rest position — on a first
    // frame the live buffer still holds the rest data, so the bug is invisible.
    // The first sweep of this file missed it for exactly that reason.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 8, 8);
    let weight_of = falloff(&mesh, 2.5);
    assert_graded("a whole drag", &mesh, &weight_of);

    // A pointer travelling: several frames of a move, then a turn, then the
    // frame that actually gets compared. Every one of them is a different
    // gesture, so a shader accumulating deltas ends up somewhere else entirely.
    let at = |p: glam::Vec3| Drag::moved(p, identity(), glam::DVec3::ZERO);
    let target = at(glam::Vec3::new(0.1, 0.05, 0.25));
    frames_then(
        &g,
        "a whole drag",
        &mesh,
        weight_of,
        &[
            Drag::nothing(),
            at(glam::Vec3::new(0.5, 0.0, -0.4)),
            at(glam::Vec3::new(-0.3, 0.8, 0.9)),
            Drag::turned(
                glam::Quat::from_rotation_y(0.4),
                identity(),
                glam::DVec3::ZERO,
            ),
            Drag::scaled(glam::Vec3::splat(1.6), identity(), glam::DVec3::ZERO),
            target,
        ],
    );
}

#[test]
fn shrinking_the_proportional_radius_mid_drag_releases_what_it_let_go() {
    // The break that survived the first sweep of this file, and the reason it
    // survived: a shader that *skips* an unweighted vertex rather than writing
    // the rest position is indistinguishable from a correct one for as long as
    // the falloff never changes, because the live buffer still holds the rest
    // data at every vertex the drag has not touched.
    //
    // The scroll wheel changes that. It adjusts the proportional radius while
    // the pointer is still down, so a vertex the falloff used to reach can stop
    // being reached — and it has last frame's displacement sitting in the live
    // buffer with nothing else to put it back. Which makes the write on the
    // zero-weight path load-bearing, and makes `Deform::set_weights` a feature
    // this path needs rather than one it might like.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 8, 8);
    let drag = Drag::moved(
        glam::Vec3::new(0.1, 0.05, 0.25),
        identity(),
        glam::DVec3::ZERO,
    );

    // Wide first, then narrow — the wheel turned the wrong way and the user
    // pulled it back.
    let wide = falloff(&mesh, 3.5);
    let narrow = falloff(&mesh, 1.2);
    assert_graded("a shrinking radius", &mesh, &narrow);
    // The fixture only means anything if the wide radius genuinely reached
    // vertices the narrow one does not.
    let released = (0..mesh.num_verts())
        .filter(|v| wide(*v) > 0.0 && narrow(*v) == 0.0)
        .count();
    assert!(
        released > 8,
        "only {released} vertices were let go; the radius barely changed"
    );

    assert_split_holds("a shrinking radius", &mesh, &narrow, &drag);
    let cpu = RenderMeshCpu::extract(&mesh);
    let viewport =
        kreate_render::Viewport::new(&g.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&g.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&g.device);
    let mut encoder = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shrinking radius"),
        });
    let deform = Deform::begin(
        &g.device,
        &mut encoder,
        &deformer,
        &gpu_mesh,
        &cpu,
        &per_render_vertex(&mesh, &wide),
    )
    .expect("a mesh with vertices and one weight each");
    deform.set(&g.queue, &drag.gesture);
    deform.dispatch(&mut encoder, &deformer);
    g.queue.submit([encoder.finish()]);

    // The wheel turns. Same gesture, narrower falloff.
    assert!(
        deform.set_weights(&g.queue, &per_render_vertex(&mesh, &narrow)),
        "the weights were the right length and should have been accepted"
    );
    let mut encoder = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shrinking radius, second frame"),
        });
    deform.dispatch(&mut encoder, &deformer);
    g.queue.submit([encoder.finish()]);

    let verts = cpu.positions.len();
    let from_gpu = Shaded {
        positions: to_positions(&read_back(&g, gpu_mesh.positions(), (verts * 12) as u64)),
        normals: to_normals(&read_back(&g, gpu_mesh.normals(), (verts * 4) as u64)),
    };
    compare(
        "a shrinking radius",
        &from_gpu,
        &on_cpu(&mesh, &narrow, &drag),
        g.software,
    );
}

#[test]
fn a_falloff_of_the_wrong_length_is_refused() {
    // A short write would leave the tail of the falloff as whatever the last
    // one said, which is a drag that behaves differently at the far edge of the
    // mesh than at the near one — the kind of thing that is noticed once and
    // never reproduced.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 4, 4);
    let cpu = RenderMeshCpu::extract(&mesh);
    let viewport =
        kreate_render::Viewport::new(&g.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&g.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&g.device);
    let mut encoder = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let deform = Deform::begin(
        &g.device,
        &mut encoder,
        &deformer,
        &gpu_mesh,
        &cpu,
        &vec![1.0; cpu.positions.len()],
    )
    .expect("a mesh with vertices and one weight each");
    g.queue.submit([encoder.finish()]);

    assert!(
        !deform.set_weights(&g.queue, &vec![1.0; cpu.positions.len() - 1]),
        "a short falloff was accepted"
    );
    assert!(
        !deform.set_weights(&g.queue, &vec![1.0; cpu.positions.len() + 1]),
        "a long falloff was accepted"
    );
    assert!(
        deform.set_weights(&g.queue, &vec![0.5; cpu.positions.len()]),
        "the right length was refused"
    );

    // One per *mesh* vertex is the mistake a caller actually makes, and on a
    // mesh that splits it is the wrong length — which is what makes the check
    // worth having rather than pedantry.
    let cube = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    assert_ne!(
        cube.positions.len(),
        kreate_mesh::primitives::cube(2.0).num_verts() as usize,
        "a cube was supposed to split, or this check proves nothing"
    );
}

#[test]
fn a_drag_on_a_moved_and_scaled_object_deforms_the_same_way_on_both() {
    // The case every other fixture here cannot see. Until this test the mesh
    // was always already in world space, where local and world are the same
    // thing — so a shader that wrote a world offset straight into local
    // positions passed all of them, and the CPU path did exactly that for
    // months.
    //
    // A move, because it is the one gesture where the error is a pure
    // direction change: on this object a drag along world X comes out along
    // some other axis entirely, scaled by a factor of three.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 6, 6);
    both_ways(
        &g,
        "move a placed object",
        &mesh,
        interior(6),
        Drag::moved(
            glam::Vec3::new(0.12, -0.06, 0.09),
            awkward_object(),
            glam::DVec3::ZERO,
        ),
    );
}

#[test]
fn a_weighted_rotation_on_a_sheared_object_deforms_the_same_way_on_both() {
    // The case that rules out conjugating the gesture into the object's space
    // instead of going out to the world and back. This object is non-uniformly
    // scaled *and* rotated, so a world-axis rotation seen from its space is a
    // shear — and no quaternion can express one. The round trip can, exactly.
    //
    // Weighted as well, so the slerp is evaluated between its endpoints under
    // the conjugation rather than only at them.
    //
    // A small angle, and the object is why: it scales x by three and y by
    // four tenths, so a gentle world rotation arrives in the mesh's own space
    // as a much steeper bend. Anything larger crosses the auto-smooth angle and
    // `assert_split_holds` stops the test — correctly, since it would then be
    // measuring the frozen split rather than the arithmetic.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::grid(4.0, 8, 8);
    let to_world = awkward_object();
    let weight_of = falloff(&mesh, 2.5);
    assert_graded("rotate a sheared object", &mesh, &weight_of);

    // The pivot in world space, where the selection actually is — the median of
    // the whole mesh here, which is what a select-all drag would give.
    let positions = mesh.positions().to_vec();
    let mut total = glam::DVec3::ZERO;
    for p in &positions {
        total += to_world.transform_point3(p.as_dvec3());
    }
    let pivot = total / positions.len() as f64;

    both_ways(
        &g,
        "rotate a sheared object",
        &mesh,
        weight_of,
        Drag::turned(glam::Quat::from_rotation_x(0.08), to_world, pivot),
    );
}

#[test]
fn the_identity_gesture_leaves_the_mesh_exactly_where_it_was() {
    // What the first frame of a drag uploads. Pressing `G` alone must change
    // nothing at all: a shader that wrote something slightly different would
    // make every gesture in the program start with a jump.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 12, 6);
    let cpu = RenderMeshCpu::extract(&mesh);
    let weights = vec![1.0f32; cpu.positions.len()];

    let after = on_gpu(&g, &mesh, &weights, &[Drag::nothing()]);
    for (i, (a, b)) in after.positions.iter().zip(&cpu.positions).enumerate() {
        let b = glam::Vec3::from_array(b.0);
        assert!((*a - b).length() < 1e-6, "vertex {i} moved: {a:?} vs {b:?}");
    }
    for (i, (a, b)) in after.normals.iter().zip(&cpu.normals).enumerate() {
        let (na, nb) = (oct_decode(*a), oct_decode(*b));
        assert!(na.dot(nb) > 0.9999, "normal {i} turned: {na:?} vs {nb:?}");
    }
}

#[test]
fn the_shader_encodes_a_normal_the_way_the_cpu_does() {
    // Two implementations of the octahedral encoding, in two languages. A
    // mismatch shows as shading that is subtly wrong everywhere and traceable
    // to nothing, so it gets its own test rather than being left to the ones
    // above to notice.
    //
    // A sphere, whose normals point in every direction — including the lower
    // hemisphere, which is the half of the encoding with a fold in it.
    let g = gpu_or_skip!(gpu());
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 16, 8);
    let cpu = RenderMeshCpu::extract(&mesh);
    // Nothing moves, so the GPU recomputes the normals of the mesh as it is:
    // the CPU's own answer, run through the GPU's encoder.
    let weights = vec![0.0f32; cpu.positions.len()];
    let after = on_gpu(&g, &mesh, &weights, &[Drag::nothing()]);

    let mut folded = 0;
    let mut worst = 0.0f32;
    for (i, (a, b)) in after.normals.iter().zip(&cpu.normals).enumerate() {
        let nb = oct_decode(*b);
        if nb.z < 0.0 {
            folded += 1;
        }
        let na = oct_decode(*a);
        let degrees = na.dot(nb).clamp(-1.0, 1.0).acos().to_degrees();
        worst = worst.max(degrees);
        assert!(
            degrees < 0.2,
            "normal {i}: GPU {na:?}, CPU {nb:?}, {degrees} degrees apart"
        );
    }
    assert!(
        folded > 20,
        "only {folded} normals face away; the folded half of the encoding is untested"
    );
    eprintln!(
        "encoding agrees to {worst:.4} degrees over {} normals",
        after.normals.len()
    );
}

#[test]
fn a_slerped_rotation_is_not_a_lerped_one() {
    // Guards the fixture above rather than the shader: if the gesture and the
    // falloff were such that a normalised lerp gave the same answer as a slerp,
    // `a_weighted_rotation_deforms_the_same_way_on_both` would pass against a
    // shader that used the cheaper one — and the difference between them is
    // exactly what the decomposed gesture is for.
    //
    // Pure CPU: it is a claim about the arithmetic, not about the device.
    let q = glam::Quat::from_rotation_x(0.5);
    let p = glam::Vec3::new(1.0, 0.7, -0.3);
    let mut worst = 0.0f32;
    for i in 1..10 {
        let w = i as f32 / 10.0;
        let slerped = glam::Quat::IDENTITY.slerp(q, w);
        let lerped = glam::Quat::from_vec4(
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0).lerp(glam::Vec4::from(q.to_array()), w),
        )
        .normalize();
        worst = worst.max((slerped * p - lerped * p).length());
    }
    // `compare` tolerates 1e-4, so a difference above that means the fixture
    // would catch the substitution.
    assert!(
        worst > 1e-4,
        "slerp and normalised lerp differ by only {worst:e} at this angle; \
         the weighted test cannot tell them apart"
    );
    eprintln!("slerp against normalised lerp: {worst:e}");
}

#[test]
fn a_crease_sharp_enough_to_split_is_the_known_limit() {
    // The precondition every other test asserts, stated as a fact about the
    // system rather than left as a comment.
    //
    // `extract` deduplicates render vertices by encoded normal, so a drag that
    // bends a smooth surface past the auto-smooth angle changes what the render
    // mesh *should* be — and a drag cannot follow it, because the index buffer
    // is uploaded once. So the preview shows the new crease smooth-shaded, and
    // the extraction on mouse-up is what puts the hard edge in.
    //
    // This is a real limit and it is the right trade: the alternative is
    // re-extracting the mesh every frame, which is the 83 ms this path exists
    // to avoid. It is bounded, too — positions are always exact, and only the
    // shading of a newly-made crease lags, until the pointer comes up.
    let mesh = kreate_mesh::primitives::grid(4.0, 6, 6);
    let weight_of = interior(6);
    // Far enough out of plane that the boundary faces meet at well over the
    // thirty degrees `DEFAULT_AUTO_SMOOTH` allows.
    let drag = Drag::moved(
        glam::Vec3::new(0.0, 0.0, 1.5),
        identity(),
        glam::DVec3::ZERO,
    );

    let rest = RenderMeshCpu::extract(&mesh);
    let after = RenderMeshCpu::extract(&deformed(&mesh, &weight_of, &drag));
    assert!(
        after.positions.len() > rest.positions.len(),
        "this gesture was supposed to split vertices: {} at rest, {} after. If \
         it no longer does, the auto-smooth angle changed and the limit this \
         test documents needs restating.",
        rest.positions.len(),
        after.positions.len()
    );

    // And the flat grid really was one render vertex per mesh vertex, so the
    // growth above is a split and not an artefact of the fixture.
    assert_eq!(rest.positions.len(), mesh.num_verts() as usize);
    eprintln!(
        "a 1.5-unit crease splits {} render vertices into {}",
        rest.positions.len(),
        after.positions.len()
    );
}
