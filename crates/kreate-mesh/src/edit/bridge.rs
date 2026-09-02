//! Bridge edge loops.
//!
//! Joins two open borders with a band of quads: the operator that closes the gap
//! between two extruded shells, caps a tube, or stitches two halves of a model
//! together.
//!
//! # The hard part is not the quads, it is the pairing
//!
//! Given two rings of the same length there are `2n` ways to pair them up — `n`
//! rotations, times two directions — and all but one produce a twisted band.
//! Worse, a twisted bridge is *valid* topology: it validates, it renders, and it
//! looks like a mistake the user made. So the pairing is chosen by minimising
//! total squared distance over all `2n` candidates, which is the answer a user
//! means every time and is cheap at the sizes bridges are actually used on.
//!
//! # What v1 requires
//!
//! Two rings of **equal length**. Bridging rings of different lengths needs
//! either resampling (which moves vertices the user did not ask to move) or a
//! triangle-fan match (which needs a rule for where the fans go). Both are real
//! decisions rather than details, and neither is needed to close the gap between
//! two extrusions — which is what bridge is for.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};
use crate::Domain;

/// Why a bridge could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    /// Fewer or more than two borders were found in the selection.
    NotTwoLoops { found: usize },
    /// The two borders have different vertex counts.
    LengthMismatch { a: usize, b: usize },
    /// A border was too short to bridge.
    TooShort { len: usize },
    /// The mesh refused a face the bridge tried to build.
    Mesh(BmError),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::NotTwoLoops { found } => {
                write!(f, "bridge needs exactly two borders, found {found}")
            }
            BridgeError::LengthMismatch { a, b } => write!(
                f,
                "the borders have {a} and {b} vertices; bridging unequal loops is not supported yet"
            ),
            BridgeError::TooShort { len } => write!(f, "a border of {len} cannot be bridged"),
            BridgeError::Mesh(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<BmError> for BridgeError {
    fn from(e: BmError) -> Self {
        BridgeError::Mesh(e)
    }
}

/// What a bridge produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bridge {
    pub faces: Vec<FaceId>,
}

/// Chains a set of boundary edges into ordered vertex cycles.
///
/// Unlike a face region's outline, these edges carry no winding to follow, so
/// the chain is built from adjacency alone. A vertex on three or more of the
/// given edges is ambiguous and ends the attempt.
pub(crate) fn chain_borders(mesh: &BMesh, edges: &[EdgeId]) -> Option<Vec<Vec<VertId>>> {
    let n = mesh.slot_count(Domain::Point);
    // Up to two neighbours per vertex; a third means the border forks.
    let mut links: Vec<[Option<VertId>; 2]> = vec![[None, None]; n];
    let push = |v: VertId, other: VertId, links: &mut Vec<[Option<VertId>; 2]>| -> bool {
        let slot = v.index() as usize;
        if slot >= links.len() {
            return false;
        }
        for place in links[slot].iter_mut() {
            if place.is_none() {
                *place = Some(other);
                return true;
            }
        }
        false
    };

    let mut ends: Vec<VertId> = Vec::new();
    for &e in edges {
        let [a, b] = mesh.edge_verts(e)?;
        if !push(a, b, &mut links) || !push(b, a, &mut links) {
            return None;
        }
        ends.push(a);
        ends.push(b);
    }
    ends.sort_unstable_by_key(|v| v.index());
    ends.dedup();

    let mut visited = vec![false; n];
    let mut cycles = Vec::new();
    for &start in &ends {
        if visited[start.index() as usize] {
            continue;
        }
        let mut cycle = vec![start];
        visited[start.index() as usize] = true;
        let mut current = start;
        let mut previous: Option<VertId> = None;
        for _ in 0..ends.len() {
            let [x, y] = links[current.index() as usize];
            let next = match (x, y) {
                (Some(x), Some(y)) => {
                    if Some(x) == previous {
                        y
                    } else {
                        x
                    }
                }
                // A border that does not close is an arc, not a loop.
                _ => return None,
            };
            if next == start {
                break;
            }
            if visited[next.index() as usize] {
                return None;
            }
            visited[next.index() as usize] = true;
            cycle.push(next);
            previous = Some(current);
            current = next;
        }
        cycles.push(cycle);
    }
    Some(cycles)
}

/// The alignment of `b` against `a` that pairs them most closely.
///
/// Returns the rotation and whether `b` must be reversed. Brute force over all
/// `2n` candidates: `n` is a loop length, so this is trivial at any size a
/// bridge is used on, and the alternative — matching the nearest pair and
/// walking from there — picks wrong whenever the two loops are offset or twisted
/// relative to each other, which is most of the time.
fn best_alignment(a: &[Vec3], b: &[Vec3]) -> (usize, bool) {
    let n = a.len();
    let mut best = (0usize, false);
    let mut best_cost = f32::INFINITY;
    for &reversed in &[false, true] {
        for rotation in 0..n {
            let mut cost = 0.0;
            for (i, p) in a.iter().enumerate() {
                let j = if reversed {
                    (rotation + n - i) % n
                } else {
                    (rotation + i) % n
                };
                cost += (*p - b[j]).length_squared();
            }
            if cost < best_cost {
                best_cost = cost;
                best = (rotation, reversed);
            }
        }
    }
    best
}

/// Joins two vertex rings with a band of quads.
pub fn bridge_loops(
    mesh: &mut BMesh,
    first: &[VertId],
    second: &[VertId],
) -> Result<Bridge, BridgeError> {
    if first.len() < 3 {
        return Err(BridgeError::TooShort { len: first.len() });
    }
    if first.len() != second.len() {
        return Err(BridgeError::LengthMismatch {
            a: first.len(),
            b: second.len(),
        });
    }
    let n = first.len();
    let pa: Vec<Vec3> = first.iter().filter_map(|v| mesh.position(*v)).collect();
    let pb: Vec<Vec3> = second.iter().filter_map(|v| mesh.position(*v)).collect();
    if pa.len() != n || pb.len() != n {
        return Err(BridgeError::TooShort {
            len: pa.len().min(pb.len()),
        });
    }
    let (rotation, reversed) = best_alignment(&pa, &pb);

    let paired = |i: usize| -> VertId {
        let j = if reversed {
            (rotation + n - i) % n
        } else {
            (rotation + i) % n
        };
        second[j]
    };

    let mut out = Bridge::default();
    for i in 0..n {
        let k = (i + 1) % n;
        let (a, b) = (first[i], first[k]);
        let (c, d) = (paired(i), paired(k));
        // Skip rather than fail on a rung whose two ends coincide: two rings
        // that touch at a point are unusual but not an error, and refusing the
        // whole bridge over one degenerate quad would be worse than building the
        // rest of it.
        if a == b || c == d || a == d || b == c {
            continue;
        }
        out.faces.push(mesh.add_face(&[a, b, d, c])?);
    }

    mesh.select_none();
    for f in &out.faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);
    Ok(out)
}

/// Bridges the two borders found among the selected edges.
pub fn bridge_selected(mesh: &mut BMesh) -> Result<Bridge, BridgeError> {
    let edges = mesh.selected_edges();
    let Some(cycles) = chain_borders(mesh, &edges) else {
        return Err(BridgeError::NotTwoLoops { found: 0 });
    };
    let [a, b] = &cycles[..] else {
        return Err(BridgeError::NotTwoLoops {
            found: cycles.len(),
        });
    };
    let (a, b) = (a.clone(), b.clone());
    bridge_loops(mesh, &a, &b)
}

/// Every open border of the mesh, as ordered vertex cycles.
///
/// What a bridge tool offers when the user has selected nothing: two open ends
/// is the common case, and asking them to select both borders first is friction.
///
/// Wire edges count as border, not just edges with one face. A ring of loose
/// edges — what an extrude with no cap or an imported curve leaves — is exactly
/// the thing a bridge is for, and excluding it makes the tool refuse its own
/// commonest input.
pub fn open_borders(mesh: &BMesh) -> Option<Vec<Vec<VertId>>> {
    let edges: Vec<EdgeId> = mesh
        .edge_ids()
        .filter(|&e| mesh.edge_face_count(e) <= 1)
        .collect();
    if edges.is_empty() {
        return Some(Vec::new());
    }
    chain_borders(mesh, &edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// Two squares, one above the other, each a face with an open border.
    fn two_open_rings(twist: f32) -> (BMesh, Vec<VertId>, Vec<VertId>) {
        let mut m = BMesh::new();
        let lower: Vec<VertId> = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ]
        .iter()
        .map(|p| m.add_vert(*p))
        .collect();
        // The upper ring is authored starting from a different corner and in the
        // opposite direction, which is what a real pair of extruded borders looks
        // like and what a naive index-order pairing gets wrong.
        let (s, c) = (twist.sin(), twist.cos());
        let rot = |x: f32, y: f32| Vec3::new(x * c - y * s, x * s + y * c, 2.0);
        let upper: Vec<VertId> = [
            rot(1.0, 1.0),
            rot(-1.0, 1.0),
            rot(-1.0, -1.0),
            rot(1.0, -1.0),
        ]
        .iter()
        .map(|p| m.add_vert(*p))
        .collect();
        for w in [&lower, &upper] {
            m.add_edge(w[0], w[1]).unwrap();
            m.add_edge(w[1], w[2]).unwrap();
            m.add_edge(w[2], w[3]).unwrap();
            m.add_edge(w[3], w[0]).unwrap();
        }
        (m, lower, upper)
    }

    #[test]
    fn bridging_two_squares_makes_a_tube() {
        let (mut m, a, b) = two_open_rings(0.0);
        let out = bridge_loops(&mut m, &a, &b).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 4);
        assert_eq!(m.num_faces(), 4);
        assert_eq!(m.num_verts(), 8);
        for f in &out.faces {
            assert_eq!(m.face(*f).unwrap().len(), 4);
        }
    }

    #[test]
    fn the_band_is_not_twisted() {
        // The failure this exists to prevent: a twisted bridge is *valid*
        // topology. It validates, it renders, and it looks like a mistake the
        // user made. Every rung of an untwisted band has roughly the length of
        // the gap; a twisted one has rungs running diagonally across it.
        for twist in [0.0f32, 0.4, -0.7, 1.2] {
            let (mut m, a, b) = two_open_rings(twist);
            let out = bridge_loops(&mut m, &a, &b).unwrap();
            m.assert_valid();

            for f in &out.faces {
                let ring: Vec<Vec3> = m
                    .face_verts(*f)
                    .iter()
                    .filter_map(|v| m.position(*v))
                    .collect();
                // The two rungs of the quad, i.e. the sides joining the rings.
                let rungs = [(ring[1] - ring[2]).length(), (ring[0] - ring[3]).length()];
                for r in rungs {
                    assert!(
                        r < 3.2,
                        "twist {twist}: a rung of {r} means the band crossed itself"
                    );
                }
            }
        }
    }

    #[test]
    fn a_reversed_ring_is_matched_rather_than_crossed() {
        // Two borders wound opposite ways — exactly what the two ends of an
        // extruded tube are. Pairing them in index order gives a band that
        // passes through itself.
        let mut m = BMesh::new();
        let ring = |m: &mut BMesh, z: f32, reverse: bool| -> Vec<VertId> {
            let mut pts = [
                Vec3::new(-1.0, -1.0, z),
                Vec3::new(1.0, -1.0, z),
                Vec3::new(1.0, 1.0, z),
                Vec3::new(-1.0, 1.0, z),
            ];
            if reverse {
                pts.reverse();
            }
            pts.iter().map(|p| m.add_vert(*p)).collect()
        };
        let a = ring(&mut m, 0.0, false);
        let b = ring(&mut m, 1.0, true);

        let out = bridge_loops(&mut m, &a, &b).unwrap();
        m.assert_valid();
        for f in &out.faces {
            let pts: Vec<Vec3> = m
                .face_verts(*f)
                .iter()
                .filter_map(|v| m.position(*v))
                .collect();
            // Each quad spans one unit of height and one edge of the square, so
            // its perimeter is about six. A crossed one is much longer.
            let perimeter: f32 = (0..4).map(|i| (pts[i] - pts[(i + 1) % 4]).length()).sum();
            assert!(
                perimeter < 7.0,
                "quad perimeter {perimeter} suggests a cross"
            );
        }
    }

    #[test]
    fn bridging_two_loose_rings_makes_an_open_tube() {
        // The operator's real use: two borders with nothing between them. Note
        // what bridge does *not* do — it does not cap the ends. Building a wall
        // over a tube that already has one would duplicate it, which is why the
        // input here is two bare rings rather than a cylinder with its caps
        // removed.
        let mut m = BMesh::new();
        let ring = |m: &mut BMesh, z: f32| -> Vec<VertId> {
            let n = 8;
            let verts: Vec<VertId> = (0..n)
                .map(|i| {
                    let t = i as f32 / n as f32 * std::f32::consts::TAU;
                    m.add_vert(Vec3::new(t.cos(), t.sin(), z))
                })
                .collect();
            for i in 0..n {
                m.add_edge(verts[i], verts[(i + 1) % n]).unwrap();
            }
            verts
        };
        let lower = ring(&mut m, 0.0);
        let upper = ring(&mut m, 2.0);

        let borders = open_borders(&m).expect("two clean borders");
        assert_eq!(borders.len(), 2);

        let out = bridge_loops(&mut m, &lower, &upper).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 8);
        let rungs = m.edge_ids().filter(|&e| m.edge_face_count(e) == 2).count();
        let rims = m.edge_ids().filter(|&e| m.edge_face_count(e) == 1).count();
        assert_eq!(rungs, 8, "the vertical edges each carry two quads");
        assert_eq!(rims, 16, "and the two rims stay open");
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }

    #[test]
    fn unequal_loops_are_refused_with_the_reason() {
        // Refused rather than resampled: resampling moves vertices the user did
        // not ask to move, and a fan match needs a rule for where the fans go.
        // Both are decisions, not details.
        let mut m = BMesh::new();
        let a: Vec<VertId> = (0..4)
            .map(|i| m.add_vert(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let b: Vec<VertId> = (0..6)
            .map(|i| m.add_vert(Vec3::new(i as f32, 0.0, 1.0)))
            .collect();
        assert_eq!(
            bridge_loops(&mut m, &a, &b),
            Err(BridgeError::LengthMismatch { a: 4, b: 6 })
        );
        assert_eq!(m.num_faces(), 0, "nothing half-built was left behind");
    }

    #[test]
    fn a_border_that_does_not_close_is_refused() {
        // An arc, not a loop: two edges in a line. Chaining it would run off the
        // end, and bridging it would need a rule for the open ends.
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let b = m.add_vert(Vec3::X);
        let c = m.add_vert(Vec3::X * 2.0);
        let e1 = m.add_edge(a, b).unwrap();
        let e2 = m.add_edge(b, c).unwrap();
        assert_eq!(chain_borders(&m, &[e1, e2]), None);
    }

    #[test]
    fn a_forked_border_is_refused() {
        // Three edges meeting at a vertex: there is no single way round.
        let mut m = BMesh::new();
        let hub = m.add_vert(Vec3::ZERO);
        let spokes: Vec<VertId> = (0..3)
            .map(|i| m.add_vert(Vec3::new(i as f32, 1.0, 0.0)))
            .collect();
        let edges: Vec<EdgeId> = spokes
            .iter()
            .map(|s| m.add_edge(hub, *s).unwrap())
            .collect();
        assert_eq!(chain_borders(&m, &edges), None);
    }

    #[test]
    fn a_closed_surface_has_no_borders() {
        let m = BMesh::from_mesh_data(&primitives::cube(2.0));
        assert_eq!(open_borders(&m), Some(Vec::new()));
    }

    #[test]
    fn bridging_the_selection_finds_both_borders() {
        // A cylinder with its caps removed: the two rims are the borders, and
        // bridging them builds a second wall. Unusual as modelling, but it is
        // the shortest way to exercise the path from a selection to two chained
        // borders.
        let mut m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 6));
        let caps: Vec<FaceId> = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 6)
            .collect();
        for f in caps {
            m.remove_face(f);
        }
        m.select_none();
        for e in m.edge_ids().collect::<Vec<_>>() {
            if m.is_boundary_edge(e) {
                m.set_edge_selected(e, true);
            }
        }
        m.flush_selection(SelectMode::Edge);

        let out = bridge_selected(&mut m).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 6);
        let mut selected = m.selected_faces();
        let mut built = out.faces.clone();
        selected.sort_unstable();
        built.sort_unstable();
        assert_eq!(selected, built, "the band is what ends up selected");
    }

    #[test]
    fn one_border_is_not_two() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        m.select_all();
        assert_eq!(
            bridge_selected(&mut m),
            Err(BridgeError::NotTwoLoops { found: 1 })
        );
    }

    #[test]
    fn a_bridged_mesh_still_bakes_to_storage_form() {
        let (mut m, a, b) = two_open_rings(0.3);
        bridge_loops(&mut m, &a, &b).unwrap();
        m.assert_valid();
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }
}
