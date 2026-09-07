//! Where the time goes when a million vertices enter edit mode.
//!
//! A probe, not a budget. It exists so the conversion is optimised against a
//! measurement instead of a guess, and it is kept because the answer is the
//! kind that changes: a future edit to `add_edge` or to the disk splice moves
//! these numbers, and the next person deserves to see which phase moved.
//!
//! Run in release. In a debug build the arithmetic is roughly fifty times
//! slower and the proportions are meaningless.

use std::time::Instant;

use kreate_mesh::bmesh::BMesh;
use kreate_mesh::{primitives, MeshData};

const DIVISIONS: u32 = 1000;

fn measurable() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("skipping: unoptimised timings say nothing about the design");
        return false;
    }
    true
}

#[test]
fn report_where_the_conversion_spends_its_time() {
    if !measurable() {
        return;
    }
    let mesh: MeshData = primitives::grid(10.0, DIVISIONS, DIVISIONS);
    eprintln!(
        "grid: {} verts, {} edges, {} faces, {} corners",
        mesh.num_verts(),
        mesh.num_edges(),
        mesh.num_faces(),
        mesh.num_corners()
    );

    // The whole thing, which is what the budget test measures.
    let start = Instant::now();
    let bm = BMesh::from_mesh_data(&mesh);
    let whole = start.elapsed();
    eprintln!("from_mesh_data total: {whole:?}");
    assert_eq!(bm.num_verts(), mesh.num_verts() as usize);
    assert_eq!(bm.num_faces(), mesh.num_faces() as usize);
    assert_eq!(bm.num_edges(), mesh.num_edges() as usize);

    // And the way back, for scale: leaving edit mode is the same crossing in
    // the other direction and is not on the same budget.
    let start = Instant::now();
    let back = bm.to_mesh_data();
    eprintln!("to_mesh_data total: {:?}", start.elapsed());
    assert_eq!(back.num_faces(), mesh.num_faces());
}
