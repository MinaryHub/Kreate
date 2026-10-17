//! Edge slide: Blender's `G G`. The selected edges move across the faces
//! beside them, each vertex along the edge it has on that side — a loop
//! shifted towards one of its neighbours without leaving the surface.
//!
//! Which side is which is settled along the selection, face by face, so that
//! a whole loop goes one way: a vertex's rail on the plus side is the edge of
//! the plus face that is not the loop's.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use kreate_math::Vec3;

use crate::bmesh::BMesh;
use crate::{ArenaIndex, EdgeId, FaceId, VertId};

/// One vertex that slides: where it starts and where each side takes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rail {
    pub vert: VertId,
    pub from: Vec3,
    /// Where a factor of one takes it; `None` where that side has no face.
    pub plus: Option<Vec3>,
    /// Where a factor of minus one takes it.
    pub minus: Option<Vec3>,
}

/// The edge of face `f` at `v` other than `e`.
fn other_edge_at(mesh: &BMesh, f: FaceId, v: VertId, e: EdgeId) -> Option<EdgeId> {
    for l in mesh.face_loops(f) {
        if mesh.loop_vert(l) != Some(v) {
            continue;
        }
        let out = mesh.loop_edge(l)?;
        let back = mesh.loop_edge(mesh.loop_prev(l)?)?;
        return Some(if out == e { back } else { out });
    }
    None
}

fn face_has_edge(mesh: &BMesh, f: FaceId, e: EdgeId) -> bool {
    mesh.face_loops(f)
        .into_iter()
        .any(|l| mesh.loop_edge(l) == Some(e))
}

fn centroid(mesh: &BMesh, f: FaceId) -> Vec3 {
    let verts = mesh.face_verts(f);
    let sum: Vec3 = verts.iter().filter_map(|v| mesh.position(*v)).sum();
    sum / verts.len().max(1) as f32
}

/// The rails of every vertex of the selected edges.
///
/// Empty when no edge is selected. A vertex whose faces on both sides are
/// the selection's own — the corner of a selected loop — has no rail and is
/// left out.
pub fn rails(mesh: &BMesh) -> Vec<Rail> {
    let mut selected = mesh.selected_edges();
    selected.sort_by_key(|e| e.index());
    let chosen: HashSet<EdgeId> = selected.iter().copied().collect();
    // Each selected edge's faces, with the side each is on.
    let mut sides: HashMap<EdgeId, Vec<(FaceId, bool)>> = HashMap::new();
    for seed in &selected {
        if sides.contains_key(seed) {
            continue;
        }
        let faces = mesh.edge_faces(*seed);
        sides.insert(
            *seed,
            faces
                .iter()
                .enumerate()
                .map(|(i, f)| (*f, i == 0))
                .collect(),
        );
        let mut queue = VecDeque::from([*seed]);
        while let Some(e) = queue.pop_front() {
            let Some(ends) = mesh.edge_verts(e) else {
                continue;
            };
            let known = sides.get(&e).cloned().unwrap_or_default();
            for v in ends {
                for next in mesh.vert_edges(v) {
                    if next == e || !chosen.contains(&next) || sides.contains_key(&next) {
                        continue;
                    }
                    let faces = mesh.edge_faces(next);
                    // Through the rail: the plus face here holds the edge the
                    // plus face of `e` leaves `v` by.
                    let mut plus: Option<FaceId> = None;
                    for (f, side) in &known {
                        let Some(rail) = other_edge_at(mesh, *f, v, e) else {
                            continue;
                        };
                        if chosen.contains(&rail) {
                            continue;
                        }
                        if let Some(g) = faces.iter().find(|g| face_has_edge(mesh, **g, rail)) {
                            plus = Some(if *side {
                                *g
                            } else {
                                // The minus face matched: plus is the other.
                                match faces.iter().find(|h| *h != g) {
                                    Some(h) => *h,
                                    None => {
                                        sides.insert(next, vec![(*g, false)]);
                                        plus = None;
                                        break;
                                    }
                                }
                            });
                            break;
                        }
                    }
                    if sides.contains_key(&next) {
                        queue.push_back(next);
                        continue;
                    }
                    // Not a grid here: by where the faces lie instead.
                    let plus = plus.or_else(|| {
                        let reference = known.iter().find(|(_, s)| *s)?.0;
                        let p = mesh.position(v)?;
                        let towards = (centroid(mesh, reference) - p).normalize_or_zero();
                        faces.iter().copied().max_by(|a, b| {
                            let da = (centroid(mesh, *a) - p).normalize_or_zero().dot(towards);
                            let db = (centroid(mesh, *b) - p).normalize_or_zero().dot(towards);
                            da.total_cmp(&db)
                        })
                    });
                    sides.insert(next, faces.iter().map(|f| (*f, Some(*f) == plus)).collect());
                    queue.push_back(next);
                }
            }
        }
    }

    // Each vertex's far ends on each side, in a stable order.
    let mut targets: BTreeMap<(u32, bool), (VertId, Vec<VertId>)> = BTreeMap::new();
    for e in &selected {
        let Some(ends) = mesh.edge_verts(*e) else {
            continue;
        };
        for (f, side) in sides.get(e).into_iter().flatten() {
            for v in ends {
                let Some(rail) = other_edge_at(mesh, *f, v, *e) else {
                    continue;
                };
                if chosen.contains(&rail) {
                    continue;
                }
                let Some(far) = mesh.edge_other_vert(rail, v) else {
                    continue;
                };
                let entry = targets
                    .entry((v.index(), *side))
                    .or_insert_with(|| (v, Vec::new()));
                if !entry.1.contains(&far) {
                    entry.1.push(far);
                }
            }
        }
    }
    let mean = |vs: &[VertId]| -> Option<Vec3> {
        let points: Vec<Vec3> = vs.iter().filter_map(|v| mesh.position(*v)).collect();
        (!points.is_empty()).then(|| points.iter().sum::<Vec3>() / points.len() as f32)
    };
    let mut out: BTreeMap<u32, Rail> = BTreeMap::new();
    for ((index, side), (v, far)) in &targets {
        let Some(from) = mesh.position(*v) else {
            continue;
        };
        let rail = out.entry(*index).or_insert(Rail {
            vert: *v,
            from,
            plus: None,
            minus: None,
        });
        if *side {
            rail.plus = mean(far);
        } else {
            rail.minus = mean(far);
        }
    }
    out.into_values().collect()
}

/// Slides the selected edges by `factor`: one is all the way to the plus
/// side's far vertices, minus one to the other side's, zero where they are.
/// Returns how many vertices moved.
pub fn edge_slide(mesh: &mut BMesh, factor: f32) -> usize {
    let factor = factor.clamp(-1.0, 1.0);
    if factor == 0.0 {
        return 0;
    }
    let mut moved = 0;
    for rail in rails(mesh) {
        let to = if factor > 0.0 { rail.plus } else { rail.minus };
        if let Some(to) = to {
            mesh.set_position(rail.vert, rail.from.lerp(to, factor.abs()));
            moved += 1;
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// A grid four by four faces, its middle row of edges along X selected.
    fn grid_with_middle_row() -> (BMesh, Vec<VertId>) {
        let mut bm = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let row: Vec<VertId> = bm
            .vert_ids()
            .filter(|v| bm.position(*v).is_some_and(|p| p.y.abs() < 1e-4))
            .collect();
        assert_eq!(row.len(), 5);
        let edges: Vec<EdgeId> = bm
            .edge_ids()
            .filter(|e| {
                bm.edge_verts(*e)
                    .is_some_and(|[a, b]| row.contains(&a) && row.contains(&b))
            })
            .collect();
        assert_eq!(edges.len(), 4);
        for e in edges {
            bm.set_edge_selected(e, true);
        }
        (bm, row)
    }

    #[test]
    fn a_row_slides_as_one_towards_either_neighbour() {
        let (bm, row) = grid_with_middle_row();
        let ys =
            |bm: &BMesh| -> Vec<f32> { row.iter().map(|v| bm.position(*v).unwrap().y).collect() };
        let mut up = bm.clone();
        assert_eq!(edge_slide(&mut up, 0.5), 5);
        let mut down = bm.clone();
        assert_eq!(edge_slide(&mut down, -0.5), 5);
        let (a, b) = (ys(&up), ys(&down));
        // Half a row, all the same way, and the other way for minus.
        for y in &a {
            assert!((y.abs() - 0.5).abs() < 1e-4, "{a:?}");
            assert!((y - a[0]).abs() < 1e-5, "one way: {a:?}");
        }
        for (y, z) in a.iter().zip(&b) {
            assert!((y + z).abs() < 1e-5, "{a:?} vs {b:?}");
        }
        // Along the surface: X untouched.
        for v in &row {
            assert!((up.position(*v).unwrap().x - bm.position(*v).unwrap().x).abs() < 1e-6);
        }
    }

    #[test]
    fn nothing_selected_slides_nothing() {
        let mut bm = BMesh::from_mesh_data(&primitives::grid(2.0, 2, 2));
        assert!(rails(&bm).is_empty());
        assert_eq!(edge_slide(&mut bm, 0.5), 0);
    }
}
