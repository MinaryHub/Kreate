//! Adversarial sequences against the edit structure.
//!
//! The plan's operator-sequence fuzzer, in the form that costs nothing: a fixed
//! seed, a tiny xorshift, and the validator after every single step. Random
//! *sequences* are what find cycle bugs — each individual operation is easy to
//! get right and the interactions are not — and a fixed seed means a failure is
//! reproducible from the printed step number rather than from a lucky rerun.
//!
//! No `proptest` dependency: shrinking is what `proptest` is for, and here the
//! failing step is already the minimal reproduction because the mesh was valid
//! one step earlier.

use kreate_math::Vec3;
use kreate_mesh::bmesh::BMesh;
use kreate_mesh::{primitives, MeshData};

/// xorshift64*, so the sequence is identical on every platform and every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

fn seeds() -> impl Iterator<Item = u64> {
    (1..=16u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn starting_meshes() -> Vec<(&'static str, MeshData)> {
    vec![
        ("cube", primitives::cube(2.0)),
        ("grid", primitives::grid(2.0, 5, 5)),
        ("uv_sphere", primitives::uv_sphere(1.0, 10, 6)),
        ("torus", primitives::torus(1.0, 0.4, 10, 7)),
        ("cone", primitives::cone(1.0, 2.0, 9)),
    ]
}

#[test]
fn random_removal_sequences_never_leave_an_invalid_mesh() {
    for (name, mesh) in starting_meshes() {
        for seed in seeds() {
            let mut rng = Rng(seed);
            let mut bm = BMesh::from_mesh_data(&mesh);
            bm.assert_valid();

            for step in 0..40 {
                let faces: Vec<_> = bm.face_ids().collect();
                let edges: Vec<_> = bm.edge_ids().collect();
                let verts: Vec<_> = bm.vert_ids().collect();

                match rng.next() % 5 {
                    0 if !faces.is_empty() => {
                        bm.remove_face(faces[rng.below(faces.len())]);
                    }
                    1 if !edges.is_empty() => {
                        bm.remove_edge(edges[rng.below(edges.len())]);
                    }
                    2 if !verts.is_empty() => {
                        bm.remove_vert(verts[rng.below(verts.len())]);
                    }
                    3 if verts.len() >= 2 => {
                        let a = verts[rng.below(verts.len())];
                        let b = verts[rng.below(verts.len())];
                        if a != b {
                            bm.add_edge(a, b).unwrap();
                        }
                    }
                    _ if verts.len() >= 3 => {
                        // A face over three arbitrary existing vertices — the
                        // operation most likely to collide with the topology
                        // already there.
                        let a = verts[rng.below(verts.len())];
                        let b = verts[rng.below(verts.len())];
                        let c = verts[rng.below(verts.len())];
                        if a != b && b != c && a != c {
                            bm.add_face(&[a, b, c]).unwrap();
                        }
                    }
                    _ => {}
                }

                let problems = bm.validate();
                assert!(
                    problems.is_empty(),
                    "{name}, seed {seed:#x}, step {step}: {problems:#?}"
                );
            }
        }
    }
}

#[test]
fn a_round_trip_survives_a_random_edit_sequence() {
    // The property that matters most in practice: whatever an editing session
    // leaves behind must still bake to storage form, and the baked form must
    // itself be valid. A `BMesh` that validates but converts to a broken
    // `MeshData` would corrupt the document on leaving edit mode.
    for (name, mesh) in starting_meshes() {
        for seed in seeds().take(8) {
            let mut rng = Rng(seed);
            let mut bm = BMesh::from_mesh_data(&mesh);

            for _ in 0..25 {
                let faces: Vec<_> = bm.face_ids().collect();
                let verts: Vec<_> = bm.vert_ids().collect();
                match rng.next() % 3 {
                    0 if !faces.is_empty() => {
                        bm.remove_face(faces[rng.below(faces.len())]);
                    }
                    1 if !verts.is_empty() => {
                        let v = verts[rng.below(verts.len())];
                        let p = bm.position(v).unwrap();
                        bm.set_position(v, p + Vec3::splat(0.01));
                    }
                    _ => {
                        bm.remove_loose();
                    }
                }
            }

            bm.assert_valid();
            let baked = bm.to_mesh_data();
            let problems = baked.validate();
            assert!(
                problems.is_empty(),
                "{name}, seed {seed:#x}: baked mesh is invalid: {problems:#?}"
            );

            // And back in again, which is what leaving and re-entering edit mode
            // does. A conversion that loses an invariant shows up here.
            let again = BMesh::from_mesh_data(&baked);
            again.assert_valid();
            assert_eq!(again.num_faces(), baked.num_faces() as usize);
            assert_eq!(again.num_verts(), baked.num_verts() as usize);
            assert_eq!(again.num_edges(), baked.num_edges() as usize);
        }
    }
}

#[test]
fn taking_a_mesh_apart_completely_ends_at_empty_and_valid() {
    // Removal in the least convenient order: always the first live element, so
    // the arena's free list is maximally fragmented and every splice runs
    // against a hole.
    for (name, mesh) in starting_meshes() {
        let mut bm = BMesh::from_mesh_data(&mesh);
        loop {
            let Some(v) = bm.vert_ids().next() else { break };
            bm.remove_vert(v);
            bm.assert_valid();
        }
        assert!(bm.is_empty(), "{name}");
        assert_eq!(bm.num_edges(), 0, "{name}");
        assert_eq!(bm.num_faces(), 0, "{name}");
        assert_eq!(bm.num_loops(), 0, "{name}: loops leaked");
        assert_eq!(bm.to_mesh_data().validate(), Vec::new(), "{name}");
    }
}

#[test]
fn a_dense_non_manifold_fan_stays_sound() {
    // The case a half-edge structure cannot express at all: many faces on one
    // edge. Built deliberately rather than stumbled into, because it is what
    // knife and dissolve produce internally.
    let mut bm = BMesh::new();
    let a = bm.add_vert(Vec3::ZERO);
    let b = bm.add_vert(Vec3::X);
    let spokes: Vec<_> = (0..32)
        .map(|i| {
            let t = i as f32 * 0.2;
            bm.add_vert(Vec3::new(0.5, t.cos(), t.sin()))
        })
        .collect();
    for s in &spokes {
        bm.add_face(&[a, b, *s]).unwrap();
    }

    let shared = bm.edge_between(a, b).unwrap();
    assert_eq!(bm.edge_face_count(shared), 32);
    assert!(!bm.is_manifold_edge(shared));
    bm.assert_valid();

    // And taking it apart one face at a time keeps the radial cycle consistent,
    // which is where an off-by-one in the unlink would show.
    for (i, s) in spokes.iter().enumerate() {
        let f = bm.vert_faces(*s)[0];
        bm.remove_face(f);
        bm.assert_valid();
        assert_eq!(bm.edge_face_count(shared), 31 - i);
    }
    assert!(bm.is_wire_edge(shared));
}
