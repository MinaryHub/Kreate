//! Semantic mesh comparison for round-trip tests.
//!
//! The question a round-trip test asks is "is this the same model", and the
//! answer has to survive everything a file format is allowed to do to it:
//!
//! - **renumber vertices** — every welder and every exporter that splits
//!   vertices does this, so indices carry no information across a round trip;
//! - **reorder faces** — glTF groups faces by material, OBJ groups by object;
//! - **rotate a face's cycle** — a quad written starting from any of its four
//!   corners is the same quad.
//!
//! And it has to refuse the one thing that looks like the above but is not:
//!
//! - **reversing a face's cycle** turns the model inside out. It is the classic
//!   silent bug of any exporter that changes handedness, and a comparison that
//!   sorts a face's vertices instead of rotating its cycle cannot see it.
//!
//! So a face is canonicalised by rotating it to start at its lowest-ranked
//! vertex, direction preserved, and the faces are then sorted. Two models match
//! when their canonical face lists match.
//!
//! # Positions are matched, not compared pairwise
//!
//! Vertices come back renumbered, so vertex *i* of one mesh is not vertex *i* of
//! the other. Positions are ranked in a fixed order and each mesh's faces are
//! rewritten in terms of that rank, which is what makes the comparison
//! independent of numbering without needing to solve an assignment problem.
//!
//! The rank is by quantised coordinate. Quantisation is what lets a tolerance
//! exist at all here, and it has the failure mode every quantisation has: two
//! positions a hair apart can land either side of a bucket edge. That is
//! acceptable for a corpus of authored meshes with coordinates nowhere near the
//! tolerance, and it is why [`Tolerance::position`] should stay far below the
//! smallest feature in the corpus rather than being raised until a test passes.

use glam::Vec3;
use kreate_mesh::MeshData;

#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// How far a position may move and still be the same position.
    ///
    /// Also the quantisation bucket used to rank positions, so it is a statement
    /// about the corpus as much as about the format: it must be far smaller than
    /// the smallest distance between two distinct vertices in the mesh.
    pub position: f32,
}

impl Default for Tolerance {
    fn default() -> Self {
        // A micrometre at Kreate's metre scale. Comfortably above f32 round-trip
        // noise through a text format, comfortably below anything anyone models.
        Self { position: 1e-5 }
    }
}

/// What was different, in terms a person can act on.
#[derive(Clone, Debug)]
pub enum Mismatch {
    VertexCount {
        expected: usize,
        actual: usize,
    },
    FaceCount {
        expected: usize,
        actual: usize,
    },
    /// A position in one mesh has no counterpart in the other.
    MissingVertex {
        position: Vec3,
    },
    /// The face lists differ. Both are given canonically, so the printed forms
    /// are directly comparable.
    Faces {
        expected: Vec<Vec<usize>>,
        actual: Vec<Vec<usize>>,
    },
    /// The same faces, but at least one is wound the other way. Called out
    /// separately because it is a specific, common, and otherwise
    /// hard-to-diagnose bug rather than generic disagreement.
    Winding {
        face: Vec<usize>,
    },
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mismatch::VertexCount { expected, actual } => {
                write!(f, "expected {expected} vertices, got {actual}")
            }
            Mismatch::FaceCount { expected, actual } => {
                write!(f, "expected {expected} faces, got {actual}")
            }
            Mismatch::MissingVertex { position } => {
                write!(f, "no vertex near {position}")
            }
            Mismatch::Faces { expected, actual } => {
                write!(
                    f,
                    "faces differ:\n  expected {expected:?}\n  actual   {actual:?}"
                )
            }
            Mismatch::Winding { face } => write!(
                f,
                "face {face:?} came back wound the other way; the model is inside out"
            ),
        }
    }
}

/// Compares two meshes as models rather than as data structures.
///
/// Ignores vertex numbering, face order, and where each face's cycle begins.
/// Does not ignore winding, face size, or position.
pub fn compare_geometry(
    expected: &MeshData,
    actual: &MeshData,
    tolerance: Tolerance,
) -> Result<(), Mismatch> {
    if expected.num_verts() != actual.num_verts() {
        return Err(Mismatch::VertexCount {
            expected: expected.num_verts() as usize,
            actual: actual.num_verts() as usize,
        });
    }
    if expected.num_faces() != actual.num_faces() {
        return Err(Mismatch::FaceCount {
            expected: expected.num_faces() as usize,
            actual: actual.num_faces() as usize,
        });
    }

    // A shared ranking of positions, so both meshes' faces can be written in the
    // same terms. Built from `expected` and looked up from `actual`, which is
    // also how a moved vertex is caught: it fails to find a rank.
    let ranks = PositionRanks::new(expected.positions(), tolerance.position);
    let mut expected_faces = canonical_faces(expected, &ranks)?;
    let mut actual_faces = canonical_faces(actual, &ranks)?;
    expected_faces.sort();
    actual_faces.sort();

    if expected_faces != actual_faces {
        // Before reporting generic disagreement, check for the specific case
        // worth naming. A reversed face is a different sequence, so it would
        // otherwise be reported as "these two lists differ" and leave the reader
        // to notice that one is the mirror of the other.
        for face in &expected_faces {
            if actual_faces.contains(&reverse_cycle(face)) && !actual_faces.contains(face) {
                return Err(Mismatch::Winding { face: face.clone() });
            }
        }
        return Err(Mismatch::Faces {
            expected: expected_faces,
            actual: actual_faces,
        });
    }
    Ok(())
}

/// Every face as a cycle of position ranks, rotated to start at its lowest.
fn canonical_faces(mesh: &MeshData, ranks: &PositionRanks) -> Result<Vec<Vec<usize>>, Mismatch> {
    let positions = mesh.positions();
    let mut out = Vec::with_capacity(mesh.num_faces() as usize);
    for face in mesh.faces() {
        let mut cycle = Vec::with_capacity(face.len());
        for &v in face {
            let position = positions[v as usize];
            cycle.push(
                ranks
                    .rank_of(position)
                    .ok_or(Mismatch::MissingVertex { position })?,
            );
        }
        out.push(rotate_to_lowest(cycle));
    }
    Ok(out)
}

/// Rotates a cycle so its smallest element leads, keeping the direction.
///
/// Rotation rather than sorting: sorting would make a face equal to its own
/// reverse, which is exactly the bug this comparison exists to catch.
fn rotate_to_lowest(mut cycle: Vec<usize>) -> Vec<usize> {
    if cycle.is_empty() {
        return cycle;
    }
    let start = cycle
        .iter()
        .enumerate()
        .min_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);
    cycle.rotate_left(start);
    cycle
}

fn reverse_cycle(cycle: &[usize]) -> Vec<usize> {
    let mut out = cycle.to_vec();
    out[1..].reverse();
    rotate_to_lowest(out)
}

/// A fixed ordering of positions, so both meshes can name a vertex the same way.
struct PositionRanks {
    /// Quantised key to rank, sorted for binary search.
    table: Vec<([i64; 3], usize)>,
    scale: f64,
}

impl PositionRanks {
    fn new(positions: &[Vec3], tolerance: f32) -> Self {
        let scale = 1.0 / tolerance.max(f32::MIN_POSITIVE) as f64;
        let mut table: Vec<([i64; 3], usize)> = positions
            .iter()
            .map(|p| (quantise(*p, scale), 0))
            .collect::<Vec<_>>();
        // Rank by the quantised key so the numbering is a property of the
        // geometry, not of the order the file happened to list it in.
        table.sort_by_key(|(key, _)| *key);
        for (rank, entry) in table.iter_mut().enumerate() {
            entry.1 = rank;
        }
        Self { table, scale }
    }

    fn rank_of(&self, position: Vec3) -> Option<usize> {
        let key = quantise(position, self.scale);
        // The exact bucket first, then its neighbours: a coordinate that moved
        // by a fraction of the tolerance can still land in the next bucket, and
        // failing on that would make the tolerance a lie.
        if let Ok(i) = self.table.binary_search_by_key(&key, |(k, _)| *k) {
            return Some(self.table[i].1);
        }
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                for dz in -1..=1i64 {
                    let probe = [key[0] + dx, key[1] + dy, key[2] + dz];
                    if let Ok(i) = self.table.binary_search_by_key(&probe, |(k, _)| *k) {
                        return Some(self.table[i].1);
                    }
                }
            }
        }
        None
    }
}

fn quantise(p: Vec3, scale: f64) -> [i64; 3] {
    [
        (p.x as f64 * scale).round() as i64,
        (p.y as f64 * scale).round() as i64,
        (p.z as f64 * scale).round() as i64,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> MeshData {
        MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        )
    }

    #[test]
    fn a_mesh_equals_itself() {
        assert!(compare_geometry(&quad(), &quad(), Tolerance::default()).is_ok());
    }

    #[test]
    fn renumbering_the_vertices_changes_nothing() {
        // What every welder and every splitting exporter does. If this were
        // reported as a difference the comparison would be useless.
        let renumbered = MeshData::from_faces(
            vec![
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            ],
            &[vec![2, 3, 0, 1]],
        );
        assert!(compare_geometry(&quad(), &renumbered, Tolerance::default()).is_ok());
    }

    #[test]
    fn starting_the_cycle_elsewhere_changes_nothing() {
        let rotated = MeshData::from_faces(quad().positions().to_vec(), &[vec![2, 3, 0, 1]]);
        assert!(compare_geometry(&quad(), &rotated, Tolerance::default()).is_ok());
    }

    #[test]
    fn reordering_the_faces_changes_nothing() {
        let a = MeshData::from_faces(
            vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
            &[vec![0, 1, 2], vec![1, 4, 3]],
        );
        let b = MeshData::from_faces(a.positions().to_vec(), &[vec![1, 4, 3], vec![0, 1, 2]]);
        assert!(compare_geometry(&a, &b, Tolerance::default()).is_ok());
    }

    #[test]
    fn reversing_the_winding_is_caught_and_named() {
        // The whole reason this module rotates cycles rather than sorting them.
        let flipped = MeshData::from_faces(quad().positions().to_vec(), &[vec![3, 2, 1, 0]]);
        match compare_geometry(&quad(), &flipped, Tolerance::default()) {
            Err(Mismatch::Winding { .. }) => {}
            other => panic!("expected a winding mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_moved_vertex_is_caught() {
        let moved = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.5),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        );
        assert!(compare_geometry(&quad(), &moved, Tolerance::default()).is_err());
    }

    #[test]
    fn noise_below_the_tolerance_is_not_a_difference() {
        // Text formats do not write f32 exactly. If this failed, every OBJ round
        // trip would fail.
        let jittered = MeshData::from_faces(
            quad()
                .positions()
                .iter()
                .map(|p| *p + Vec3::splat(1e-7))
                .collect(),
            &[vec![0, 1, 2, 3]],
        );
        assert!(compare_geometry(&quad(), &jittered, Tolerance::default()).is_ok());
    }

    #[test]
    fn a_triangulated_quad_is_not_the_same_model() {
        // The n-gon claim depends on this being a difference. If a fan of two
        // triangles compared equal to the quad it came from, every KREATE_ngons
        // test would be vacuous.
        let fan =
            MeshData::from_faces(quad().positions().to_vec(), &[vec![0, 1, 2], vec![0, 2, 3]]);
        assert!(compare_geometry(&quad(), &fan, Tolerance::default()).is_err());
    }
}
