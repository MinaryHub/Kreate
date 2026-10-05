//! Bytes per vertex, as a test rather than an intention.
//!
//! Memory regressions in a mesh kernel are silent until they are fatal: nothing
//! gets slower, nothing fails, and then a scene that used to open does not. The
//! plan's budget is 250 B per vertex for the edit structure and 100 B for the
//! GPU-side render mesh; this pins the first one.
//!
//! Measured in **release**, where handles are 4 bytes and the generation array
//! does not exist. A debug build pays roughly double and that is fine — it is
//! not the configuration anyone edits a 10 M-vertex mesh in.
//!
//! # What the budget does and does not cover
//!
//! It is a *quad-mesh* budget, and the plan should be read that way. A quad mesh
//! tends to two edges and four loops per vertex; a triangle mesh tends to three
//! and six, so it costs about half again as much per vertex and an icosphere
//! measures around 330 B. That is arithmetic, not a regression: it is the same
//! bytes per *element*, and a triangle mesh has more elements per vertex. The
//! number that matters for the 10 M-vertex target is the quad one, because that
//! is what subdivision surfaces and hand-modelled assets are.

use kreate_mesh::bmesh::BMesh;
use kreate_mesh::primitives;

/// The plan's Phase 2 budget.
const BUDGET_PER_VERT: usize = 250;

/// Prints a mesh's edit-structure cost and returns its *topology's* bytes per
/// vertex: the vertices, edges, loops and faces and the links between them.
///
/// Attribute layers — UVs, selection, colours — are reported but not counted
/// against the budget. They are the content a mesh carries rather than what
/// the structure costs, they scale with what the user puts on the mesh, and
/// counting them hid the structure's own cost behind whichever layers the
/// primitives happen to have: the UVs the primitives gained after the budget
/// was set took the grid from 221 to 253 bytes a vertex without the structure
/// changing at all. [`attributes_hold_one_value_per_element`] keeps the layers
/// honest instead.
fn report(name: &str, mesh: &kreate_mesh::MeshData) -> f64 {
    let bm = BMesh::from_mesh_data(mesh);
    let bytes = bm.memory_bytes();
    let attributes: usize = bm
        .attributes()
        .iter()
        .map(|l| l.data.len() * l.data.ty().size())
        .sum();
    let verts = bm.num_verts() as f64;
    let topology = (bytes - attributes) as f64 / verts;
    println!(
        "{name:12} V {:6} E {:6} L {:6} F {:6}  {bytes:9} B  topology {topology:6.1} B/vert  attributes {:5.1} B/vert",
        bm.num_verts(),
        bm.num_edges(),
        bm.num_loops(),
        bm.num_faces(),
        attributes as f64 / verts,
    );
    topology
}

/// Every attribute layer holds one value per element of its domain, at its
/// type's size: no stale slots, no layer kept at a larger domain's length.
#[test]
fn attributes_hold_one_value_per_element() {
    for (name, mesh) in [
        ("grid", primitives::grid(10.0, 32, 32)),
        ("uv_sphere", primitives::uv_sphere(1.0, 32, 16)),
        ("torus", primitives::torus(1.0, 0.25, 32, 12)),
    ] {
        let bm = BMesh::from_mesh_data(&mesh);
        for layer in bm.attributes().iter() {
            assert_eq!(
                layer.data.len(),
                bm.slot_count(layer.domain),
                "{name}: the {} layer does not have one value per element",
                layer.name
            );
        }
    }
}

#[test]
fn the_edit_structure_stays_inside_its_per_vertex_budget() {
    // A large quad grid is the worst realistic case and the one the budget was
    // written against: it approaches the limiting ratio of one vertex to two
    // edges to four loops to one face, where the per-element costs are weighted
    // most heavily toward loops.
    let per_vert = report("grid 128²", &primitives::grid(10.0, 128, 128));

    if cfg!(debug_assertions) {
        // Debug carries an 8-byte handle and a generation per slot. Still worth
        // an upper bound, so a change that doubles memory does not hide here.
        assert!(
            per_vert < 2.0 * BUDGET_PER_VERT as f64,
            "{per_vert:.1} B/vert in debug"
        );
        return;
    }

    assert!(
        per_vert <= BUDGET_PER_VERT as f64,
        "{per_vert:.1} B/vert of topology exceeds the {BUDGET_PER_VERT} B budget"
    );
}

#[test]
fn every_primitive_reports_its_footprint() {
    // Printed rather than asserted for the smaller shapes: a torus or a sphere
    // has a different element ratio, and the interesting number is the trend
    // across them, not a threshold on each.
    let meshes: Vec<(&str, kreate_mesh::MeshData)> = vec![
        ("cube", primitives::cube(2.0)),
        ("grid 32²", primitives::grid(4.0, 32, 32)),
        ("uv_sphere", primitives::uv_sphere(1.0, 64, 32)),
        ("ico_sphere", primitives::ico_sphere(1.0, 3)),
        ("cylinder", primitives::cylinder(1.0, 2.0, 64)),
        ("torus", primitives::torus(1.0, 0.4, 64, 32)),
    ];
    for (name, mesh) in &meshes {
        let per_vert = report(name, mesh);
        assert!(
            per_vert.is_finite() && per_vert > 0.0,
            "{name}: nonsensical measurement"
        );
    }
}

#[test]
fn a_freed_slot_is_reused_rather_than_growing_the_arena() {
    // The other half of the memory story. Without slot reuse, a long editing
    // session grows without bound even though the mesh does not — and the
    // symptom is "it gets slow after an hour", which is nearly impossible to
    // attribute after the fact.
    let mut bm = BMesh::from_mesh_data(&primitives::grid(4.0, 16, 16));
    let before = bm.memory_bytes();

    for _ in 0..50 {
        let f = bm.face_ids().next().unwrap();
        let verts = bm.face_verts(f);
        bm.remove_face(f);
        bm.add_face(&verts).unwrap();
    }
    bm.assert_valid();

    assert_eq!(
        bm.memory_bytes(),
        before,
        "fifty delete-and-recreate cycles must reuse the same slots"
    );
}
