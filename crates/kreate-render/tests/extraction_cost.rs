//! What it costs to get a mesh ready for the GPU.
//!
//! The buffers are split by update frequency so that a click writes three small
//! ones and a drag writes two — that is `mesh_gpu`'s founding decision. What was
//! never split was the *extraction* that fills them, so every caller paid for a
//! whole render mesh: triangulation, corner normals and the vertex split, on
//! every click and every pointer position.
//!
//! These are absolute budgets in the manner of `kreate-ops`' `large_mesh`:
//! deliberately loose, there to catch a change of shape rather than to police
//! milliseconds. Each says what the number was before the change that made it
//! and what it is now.
//!
//! Release only. Unoptimised mesh arithmetic is roughly fifty times slower and
//! the numbers say nothing about the design.

use std::time::{Duration, Instant};

/// A million faces and a million vertices.
const DIVISIONS: u32 = 1000;

/// Serialised, so two benchmarks do not measure each other.
static BENCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serially() -> std::sync::MutexGuard<'static, ()> {
    BENCH.lock().unwrap_or_else(|e| e.into_inner())
}

fn measurable() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("skipping: mesh arithmetic is ~50x slower without optimisation");
        return false;
    }
    true
}

fn large_mesh() -> kreate_mesh::MeshData {
    let mut mesh = kreate_mesh::primitives::grid(10.0, DIVISIONS, DIVISIONS);
    mesh.ensure_selection_layers();
    kreate_mesh::selection::select_all(&mut mesh);
    mesh
}

/// The median of several runs. A single sample measures the machine.
fn median_time(runs: usize, mut f: impl FnMut()) -> Duration {
    let mut samples: Vec<Duration> = (0..runs.max(1))
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[test]
fn a_click_does_not_pay_for_a_render_mesh() {
    // Two claims, one measurement, because they are about the same two numbers
    // and a second test measuring them again only measured the machine after
    // the first had heated it: 540 ms in one test and 958 in the next, for the
    // same call on the same mesh.
    //
    // **A click.** The plan's interaction budget puts its effect in the frame it
    // happened in, and §4.6 names coarse invalidation as the leading cause of
    // hitches in an application of this kind. Reading a selection off a mesh
    // needs three passes over three attribute layers; it was costing a
    // triangulation, a corner-normal solve and a vertex split as well — **559 ms
    // on a million faces, against 27 for the flags alone.**
    //
    // Asserted as a ratio, because that is the claim: the cheap path is cheap
    // *because* it does less, and a ratio survives a slower machine.
    //
    // **A drag.** A frame that moves vertices rebuilds the positions and the
    // normals, and that goes through the whole extraction. **922 ms when first
    // measured, now 559**: it did a hash-map probe and a linear search of the
    // source face for every triangle corner — six million of each — to recover
    // which corner a triangle vertex came from, which `triangulate_corners` now
    // simply reports.
    //
    // That is nowhere near a frame and this does not pretend otherwise. What
    // remains is dominated by `compute_corner_normals`, which floods the smooth
    // group per *corner* rather than per vertex: **401 ms of the 559.** §4.5's
    // answer is that a drag should not come through here at all — the GPU
    // transforms an immutable rest-position buffer and the CPU uploads one small
    // uniform — so this budget guards the shape, not the speed.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let mesh = large_mesh();
    // Warm: the first call of either faults in pages both then reuse.
    let _ = kreate_render::ElementFlags::extract(&mesh);
    let _ = kreate_render::RenderMeshCpu::extract(&mesh);

    let flags = median_time(3, || {
        std::hint::black_box(kreate_render::ElementFlags::extract(&mesh));
    });
    let whole = median_time(3, || {
        std::hint::black_box(kreate_render::RenderMeshCpu::extract(&mesh));
    });
    eprintln!("a million faces: flags alone {flags:?}, whole render mesh {whole:?}");

    assert!(
        flags * 5 < whole,
        "reading the flags took {flags:?} against {whole:?} for the whole render \n         mesh; a click is paying for geometry it does not change"
    );
    // Below the 922 ms this was written to catch and above anything the machine
    // has produced since. The load-dependent spread on this processor is about
    // 1.5x, so a tighter threshold would be testing what else ran first.
    assert!(
        whole < Duration::from_millis(800),
        "extracting a render mesh took {whole:?}; the per-corner search is back"
    );
}
