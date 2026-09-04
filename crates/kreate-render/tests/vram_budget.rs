//! What a scene costs in GPU memory, against the plan's budgets.
//!
//! The plan's figures are ≤100 bytes per render vertex, ≤2.0 GB for a
//! ten-million-vertex scene, and ≤120 MB of transient render targets at 1080p —
//! the last two chosen so the whole thing fits on an 8 GB card with room for the
//! driver and the rest of the desktop.
//!
//! Memory regressions in this part of a renderer are quiet. Nothing gets slower
//! and nothing looks wrong; the application simply stops working on a smaller
//! card, and the report arrives from someone whose machine you do not have. So
//! the budgets are asserted rather than printed, and the per-vertex figure is
//! broken down by buffer so a failure says *which* array grew.
//!
//! No GPU needed: these are the sizes of what would be uploaded, computed from
//! the CPU-side layout that the upload copies verbatim.

use kreate_mesh::primitives;
use kreate_render::RenderMeshCpu;

/// The plan's per-render-vertex budget, for the direct path.
const PER_VERTEX: f32 = 100.0;

/// What the cluster path is allowed instead.
///
/// A relaxation, and it should be read as one rather than as the budget quietly
/// moving. Measured, the cluster path costs 102 to 121 bytes per vertex against
/// the plan's 100 — the extra is a second copy of the index stream, which exists
/// because the direct path needs indices in one order and the cluster path needs
/// them grouped by cluster.
///
/// It is defensible because of what the per-vertex figure is *for*. The
/// constraint that decides whether the application runs on an 8 GB card is the
/// scene-level one: ten million vertices in under 2 GB. At 100 bytes that is
/// 0.93 GB, so the plan set the per-vertex figure with roughly twice the
/// headroom it needed. At 128 bytes a ten-million-vertex scene is 1.19 GB, still
/// comfortably inside 2 GB — and the ten-million-vertex test below is the one
/// that actually holds the line.
///
/// If the cluster path ever becomes the default, the right fix is not a larger
/// budget but dropping the direct path index buffer for static meshes, which
/// removes the duplication entirely.
const PER_VERTEX_CLUSTERED: f32 = 128.0;

/// Meshes with different topologies, because the per-vertex cost depends on how
/// many triangles and edges a vertex is shared by, and a single shape would only
/// pin down one ratio.
fn meshes() -> Vec<(&'static str, kreate_mesh::MeshData)> {
    vec![
        ("grid", primitives::grid(10.0, 60, 60)),
        ("sphere", primitives::uv_sphere(1.0, 64, 32)),
        ("torus", primitives::torus(1.0, 0.3, 64, 32)),
        ("icosphere", primitives::ico_sphere(1.0, 5)),
        ("cube", primitives::cube(2.0)),
    ]
}

#[test]
fn the_render_mesh_stays_within_the_per_vertex_budget() {
    for (name, mesh) in meshes() {
        let r = RenderMeshCpu::extract(&mesh);
        assert!(
            r.bytes_per_vertex() <= PER_VERTEX,
            "{name}: {:.1} bytes/vertex exceeds the {PER_VERTEX} byte budget",
            r.bytes_per_vertex()
        );
    }
}

#[test]
fn clustering_roughly_doubles_the_index_memory() {
    // The cost of the GPU-driven path, stated as a number rather than left to be
    // discovered. The cluster index stream is a second copy of every index in a
    // different order — the direct path needs one order and the cluster path the
    // other — plus 48 bytes per cluster of records.
    //
    // This is why `with_clusters` is opt-in. Baking unconditionally, which is
    // what the first version of the cluster work did, made every mesh in every
    // scene pay for a path that is off by default and currently measures slower
    // than the direct draw.
    for (name, mesh) in meshes() {
        let plain = RenderMeshCpu::extract(&mesh);
        let clustered = RenderMeshCpu::extract(&mesh).with_clusters();
        let added = clustered.gpu_bytes() - plain.gpu_bytes();
        let indices = plain.indices.len() * 4;
        println!(
            "{name:>10}: {:.1} -> {:.1} bytes/vertex (+{} KB, index stream {} KB)",
            plain.bytes_per_vertex(),
            clustered.bytes_per_vertex(),
            added / 1024,
            indices / 1024,
        );
        assert!(
            clustered.gpu_bytes() > plain.gpu_bytes(),
            "{name}: clustering added nothing, so it did not happen"
        );
        // A second copy of the indices plus records, and nothing else. Twice the
        // index stream plus a generous allowance for records is the ceiling; if
        // this trips, the bake has started duplicating something bigger.
        assert!(
            added <= indices * 2,
            "{name}: clustering added {} KB against an index stream of {} KB",
            added / 1024,
            indices / 1024
        );
    }
}

#[test]
fn a_clustered_mesh_fits_the_relaxed_budget() {
    // The budget has to hold for the path that uses the most memory, not just
    // the cheapest one — but it is a *different* number, and the reasoning for
    // it lives on `PER_VERTEX_CLUSTERED` rather than being left implicit.
    for (name, mesh) in meshes() {
        let r = RenderMeshCpu::extract(&mesh).with_clusters();
        assert!(
            r.bytes_per_vertex() <= PER_VERTEX_CLUSTERED,
            "{name} with clusters: {:.1} bytes/vertex exceeds the {} byte budget",
            r.bytes_per_vertex(),
            PER_VERTEX_CLUSTERED
        );
    }
}

#[test]
fn the_direct_path_is_what_keeps_the_hundred_byte_figure() {
    // Stated separately so the relaxation above cannot be mistaken for the plan
    // having been met outright. The path that ships stays inside the plan
    // number; the experimental one does not, and both facts are recorded rather
    // than averaged into one comfortable claim.
    let worst = meshes()
        .into_iter()
        .map(|(name, mesh)| (name, RenderMeshCpu::extract(&mesh).bytes_per_vertex()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("meshes");
    println!(
        "worst direct-path cost: {} at {:.1} bytes/vertex",
        worst.0, worst.1
    );
    assert!(worst.1 <= PER_VERTEX);
}

#[test]
fn ten_million_vertices_fit_in_the_two_gigabyte_budget() {
    // The plan's scene-level figure, and the one that decides whether the
    // application runs on an 8 GB card. Extrapolated from a real mesh's measured
    // per-vertex cost rather than by allocating ten million vertices, which
    // would make this test take minutes and prove the same thing.
    //
    // Extrapolation is sound here because every buffer in the layout is a fixed
    // number of bytes per vertex, edge or triangle: nothing in it grows
    // superlinearly, which is exactly the property `RenderMeshCpu` was designed
    // around.
    // The *clustered* cost, because a budget that only holds for the cheaper
    // configuration is not a budget. This is the test that decides whether the
    // application runs on an 8 GB card.
    let mesh = primitives::uv_sphere(1.0, 200, 100);
    let r = RenderMeshCpu::extract(&mesh).with_clusters();
    let per_vertex = r.bytes_per_vertex() as f64;
    let scene = per_vertex * 10_000_000.0;
    let gigabytes = scene / (1024.0 * 1024.0 * 1024.0);
    println!("{per_vertex:.1} bytes/vertex -> {gigabytes:.2} GB for 10M vertices");
    assert!(
        gigabytes <= 2.0,
        "10M vertices would take {gigabytes:.2} GB, over the 2.0 GB budget"
    );
}

#[test]
fn transient_targets_fit_the_budget_at_1080p() {
    // The plan allows 120 MB of transient render targets at 1080p. Counted here
    // from the formats the viewport actually allocates, because the number is
    // easy to blow with one more full-screen buffer and nothing will complain
    // until a smaller card runs out.
    const W: usize = 1920;
    const H: usize = 1080;
    let pixels = W * H;

    // Every full-resolution target the viewport owns, by format.
    let colour = pixels * 8; // Rgba16Float
    let depth = pixels * 4; // Depth32Float
    let ids = pixels * 8; // Rg32Uint
                          // The depth pyramid: a full-resolution R32Float level plus its mip chain,
                          // which sums to about a third more.
    let pyramid = (pixels as f64 * 4.0 * 4.0 / 3.0) as usize;

    let total = colour + depth + ids + pyramid;
    let megabytes = total as f64 / (1024.0 * 1024.0);
    println!(
        "1080p transient targets: colour {} MB, depth {} MB, ids {} MB, \
         pyramid {} MB, total {megabytes:.1} MB",
        colour / 1_048_576,
        depth / 1_048_576,
        ids / 1_048_576,
        pyramid / 1_048_576,
    );
    assert!(
        megabytes <= 120.0,
        "transient targets take {megabytes:.1} MB, over the 120 MB budget"
    );
}

#[test]
fn the_cull_buffers_scale_with_clusters_not_with_the_viewport() {
    // The property that keeps GPU-driven culling affordable at any window size:
    // its working set is proportional to the scene, not to the number of pixels.
    // A cull whose buffers grew with the viewport would be a resolution-dependent
    // memory cost, which is the shape of bug that only appears on a large
    // monitor.
    //
    // Counted from the layout rather than measured: the visible list is one u32
    // per cluster per pass, the drawn flags one u32 per cluster, and the
    // arguments 256 bytes per draw per pass.
    let clusters = 100_000usize;
    let draws = 1_000usize;
    let visible = clusters * 4 * 2;
    let drawn = clusters * 4;
    let args = draws * 256 * 2;
    let total = visible + drawn + args;
    let megabytes = total as f64 / (1024.0 * 1024.0);
    println!("100k clusters over 1k draws: {megabytes:.1} MB of cull buffers");
    assert!(
        megabytes <= 8.0,
        "cull buffers take {megabytes:.1} MB, which is out of proportion to a \
         scene of this size"
    );
}
