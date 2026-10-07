//! How long a render-mesh extract takes on a sculpt-sized mesh — what every
//! frame of a sculpt stroke pays to show the result.
//! `cargo run --release -p kreate-render --example extract_cost`

use std::time::Instant;

use kreate_mesh::primitives;
use kreate_render::RenderMeshCpu;

fn main() {
    let m = primitives::uv_sphere(1.0, 512, 256);
    println!("verts {}", m.num_verts());
    for _ in 0..3 {
        let t = Instant::now();
        let cpu = RenderMeshCpu::extract(&m);
        println!(
            "extract: {:.1} ms ({} render verts)",
            t.elapsed().as_secs_f64() * 1e3,
            cpu.positions.len()
        );
    }
}
