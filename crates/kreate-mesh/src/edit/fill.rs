//! Fill: closing a hole.
//!
//! Two operations on the same input — a closed border — that differ in what they
//! leave behind. [`fill`] makes one n-gon, which is right when the hole is small
//! or flat. [`grid_fill`] makes a quad patch, which is what you want when the
//! hole is part of a surface you intend to keep subdividing, because an n-gon in
//! the middle of a subdivision cage is a pinch point.
//!
//! # Winding is the part that goes wrong
//!
//! A new face must be wound *opposite* to how its neighbours traverse the shared
//! edges, or it comes out inside-out. The border chain arrives in an arbitrary
//! direction — it is built from adjacency, which has no sense of front and
//! back — so the direction has to be taken from a neighbouring face rather than
//! assumed. A filled hole with a reversed normal renders as a black patch and
//! reads as a shading bug.

use kreate_math::Vec3;

use crate::arena::{FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};

/// Why a fill could not be done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FillError {
    /// Fewer or more than one border was found.
    NotOneBorder {
        found: usize,
    },
    /// The border does not close, or forks.
    BorderNotClosed,
    /// A border of fewer than three vertices.
    TooShort {
        len: usize,
    },
    /// A grid fill needs an even number of edges, so the patch has two sides of
    /// each length.
    OddBorder {
        len: usize,
    },
    /// The requested span does not divide the border into a patch.
    BadSpan {
        len: usize,
        span: usize,
    },
    Mesh(BmError),
}

impl std::fmt::Display for FillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FillError::NotOneBorder { found } => {
                write!(f, "fill needs exactly one border, found {found}")
            }
            FillError::BorderNotClosed => write!(f, "the border does not close"),
            FillError::TooShort { len } => write!(f, "a border of {len} cannot be filled"),
            FillError::OddBorder { len } => write!(
                f,
                "a grid fill needs an even number of edges; this border has {len}"
            ),
            FillError::BadSpan { len, span } => write!(
                f,
                "a span of {span} does not divide a border of {len} into a patch"
            ),
            FillError::Mesh(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FillError {}

impl From<BmError> for FillError {
    fn from(e: BmError) -> Self {
        FillError::Mesh(e)
    }
}

/// What a fill produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fill {
    pub faces: Vec<FaceId>,
    /// Interior vertices a grid fill created. Empty for an n-gon fill.
    pub verts: Vec<VertId>,
}

/// Orients a border ring so a face built from it faces the same way as its
/// neighbours.
///
/// Takes the direction from a neighbouring face rather than assuming one: the
/// chain is built from adjacency and has no sense of front and back. A ring with
/// no neighbours at all — a loose loop of wire — is left as it is, since there is
/// nothing to agree with.
fn orient(mesh: &BMesh, ring: &[VertId]) -> Vec<VertId> {
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let Some(edge) = mesh.edge_between(a, b) else {
            continue;
        };
        let faces = mesh.edge_faces(edge);
        let Some(&neighbour) = faces.first() else {
            continue;
        };
        // Does the neighbour walk this edge the same way the ring does?
        let same = mesh
            .face_loops(neighbour)
            .into_iter()
            .any(|l| mesh.loop_edge(l) == Some(edge) && mesh.loop_vert(l) == Some(a));
        if same {
            // Then a face built in ring order would agree with it, which means
            // the two would face opposite ways across the shared edge.
            let mut reversed = ring.to_vec();
            reversed.reverse();
            return reversed;
        }
        return ring.to_vec();
    }
    ring.to_vec()
}

/// Closes a border with a single n-gon.
pub fn fill(mesh: &mut BMesh, ring: &[VertId]) -> Result<Fill, FillError> {
    if ring.len() < 3 {
        return Err(FillError::TooShort { len: ring.len() });
    }
    let ring = orient(mesh, ring);
    let face = mesh.add_face(&ring)?;
    let out = Fill {
        faces: vec![face],
        verts: Vec::new(),
    };
    select(mesh, &out);
    Ok(out)
}

/// The default span for a border: as square a patch as the length allows.
pub fn default_span(len: usize) -> usize {
    (len / 4).max(1)
}

/// Closes a border with a grid of quads.
///
/// The border is read as the outline of an `span × (len/2 - span)` patch, and
/// the interior is a Coons patch — bilinear interpolation from the four sides
/// with the corner contribution removed. Plain bilinear interpolation between
/// two opposite sides ignores the other two, so a curved boundary produces a
/// flat interior that pulls away from it; the Coons correction is what makes the
/// patch actually meet its own edges.
pub fn grid_fill(mesh: &mut BMesh, ring: &[VertId], span: usize) -> Result<Fill, FillError> {
    let n = ring.len();
    if n < 4 {
        return Err(FillError::TooShort { len: n });
    }
    if !n.is_multiple_of(2) {
        return Err(FillError::OddBorder { len: n });
    }
    let half = n / 2;
    let a = span;
    let b = half.saturating_sub(a);
    if a == 0 || b == 0 {
        return Err(FillError::BadSpan { len: n, span });
    }
    let ring = orient(mesh, ring);

    // The four sides, each running from a corner toward the next. `top` and
    // `left` are read backwards so that they run alongside `bottom` and `right`
    // rather than against them.
    let at = |i: usize| ring[i % n];
    let bottom: Vec<VertId> = (0..=a).map(at).collect();
    let right: Vec<VertId> = (a..=a + b).map(at).collect();
    let mut top: Vec<VertId> = (a + b..=2 * a + b).map(at).collect();
    top.reverse();
    let mut left: Vec<VertId> = (2 * a + b..=2 * a + 2 * b).map(at).collect();
    left.reverse();

    // The boundary positions are read out in full before any vertex is added:
    // the interpolation needs them all, and adding a vertex takes the mesh
    // mutably.
    let read = |side: &[VertId]| -> Vec<Vec3> {
        side.iter()
            .map(|v| mesh.position(*v).unwrap_or(Vec3::ZERO))
            .collect()
    };
    let (bottom_p, right_p, top_p, left_p) = (read(&bottom), read(&right), read(&top), read(&left));
    let corners = [bottom_p[0], bottom_p[a], top_p[0], top_p[a]];

    // The grid, `(a + 1) × (b + 1)`, with the sides taken as given and the
    // interior interpolated.
    let mut grid: Vec<Vec<VertId>> = vec![Vec::with_capacity(b + 1); a + 1];
    let mut out = Fill::default();
    for i in 0..=a {
        for j in 0..=b {
            let existing = match (i, j) {
                (_, 0) => Some(bottom[i]),
                (_, _) if j == b => Some(top[i]),
                (0, _) => Some(left[j]),
                (_, _) if i == a => Some(right[j]),
                _ => None,
            };
            let v = match existing {
                Some(v) => v,
                None => {
                    let (u, w) = (i as f32 / a as f32, j as f32 / b as f32);
                    let coons = bottom_p[i] * (1.0 - w)
                        + top_p[i] * w
                        + left_p[j] * (1.0 - u)
                        + right_p[j] * u
                        - (corners[0] * (1.0 - u) * (1.0 - w)
                            + corners[1] * u * (1.0 - w)
                            + corners[2] * (1.0 - u) * w
                            + corners[3] * u * w);
                    let v = mesh.add_vert(coons);
                    out.verts.push(v);
                    v
                }
            };
            grid[i].push(v);
        }
    }

    for i in 0..a {
        for j in 0..b {
            out.faces.push(mesh.add_face(&[
                grid[i][j],
                grid[i + 1][j],
                grid[i + 1][j + 1],
                grid[i][j + 1],
            ])?);
        }
    }
    select(mesh, &out);
    Ok(out)
}

fn select(mesh: &mut BMesh, out: &Fill) {
    mesh.select_none();
    for f in &out.faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);
}

/// The one open border of the mesh, or a reason there is not one.
pub fn the_only_border(mesh: &BMesh) -> Result<Vec<VertId>, FillError> {
    let Some(borders) = super::bridge::open_borders(mesh) else {
        return Err(FillError::BorderNotClosed);
    };
    match borders.len() {
        1 => Ok(borders.into_iter().next().unwrap()),
        found => Err(FillError::NotOneBorder { found }),
    }
}

/// Fills the border the selected edges form, or the mesh's only open border.
pub fn fill_selected(mesh: &mut BMesh, grid: bool, span: Option<usize>) -> Result<Fill, FillError> {
    let selected = mesh.selected_edges();
    let ring = if selected.is_empty() {
        the_only_border(mesh)?
    } else {
        let Some(mut chains) = super::bridge::chain_borders(mesh, &selected) else {
            return Err(FillError::BorderNotClosed);
        };
        match chains.len() {
            1 => chains.pop().unwrap(),
            found => return Err(FillError::NotOneBorder { found }),
        }
    };
    if grid {
        let span = span.unwrap_or_else(|| default_span(ring.len()));
        grid_fill(mesh, &ring, span)
    } else {
        fill(mesh, &ring)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// A cube with one face removed, leaving a square hole.
    fn holed_cube() -> (BMesh, Vec3) {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let top = m
            .face_ids()
            .max_by(|a, b| {
                let za = m.face_normal(*a).normalize_or_zero().z;
                let zb = m.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            })
            .unwrap();
        let normal = m.face_normal(top).normalize_or_zero();
        m.remove_face(top);
        (m, normal)
    }

    #[test]
    fn filling_a_square_hole_closes_the_solid() {
        let (mut m, _) = holed_cube();
        let ring = the_only_border(&m).unwrap();
        assert_eq!(ring.len(), 4);

        let out = fill(&mut m, &ring).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 1);
        assert!(out.verts.is_empty(), "an n-gon fill adds no vertices");
        assert_eq!(m.num_faces(), 6);
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "watertight again");
        }
    }

    #[test]
    fn a_filled_face_points_the_same_way_as_its_neighbours() {
        // The failure this file is mostly about. A reversed fill renders as a
        // black patch and reads as a shading bug rather than a winding one.
        let (mut m, was) = holed_cube();
        let ring = the_only_border(&m).unwrap();
        let out = fill(&mut m, &ring).unwrap();

        let now = m.face_normal(out.faces[0]).normalize_or_zero();
        assert!(
            now.dot(was) > 0.99,
            "the fill faces {now:?} where the removed face faced {was:?}"
        );
    }

    #[test]
    fn the_ring_direction_does_not_decide_the_winding() {
        // The chain is built from adjacency and arrives in an arbitrary
        // direction. Both directions must give the same face.
        let (mut forward, was) = holed_cube();
        let ring = the_only_border(&forward).unwrap();
        let mut backward_mesh = forward.clone();
        let mut backward = ring.clone();
        backward.reverse();

        let a = fill(&mut forward, &ring).unwrap();
        let b = fill(&mut backward_mesh, &backward).unwrap();
        let na = forward.face_normal(a.faces[0]).normalize_or_zero();
        let nb = backward_mesh.face_normal(b.faces[0]).normalize_or_zero();
        assert!(na.dot(was) > 0.99 && nb.dot(was) > 0.99, "{na:?} {nb:?}");
    }

    #[test]
    fn a_loose_wire_ring_is_filled_either_way_round() {
        // Nothing to agree with, so any winding is as good as the other. What
        // must not happen is a refusal.
        let mut m = BMesh::new();
        let ring: Vec<VertId> = (0..6)
            .map(|i| {
                let t = i as f32 / 6.0 * std::f32::consts::TAU;
                m.add_vert(Vec3::new(t.cos(), t.sin(), 0.0))
            })
            .collect();
        for i in 0..6 {
            m.add_edge(ring[i], ring[(i + 1) % 6]).unwrap();
        }
        let out = fill(&mut m, &ring).unwrap();
        m.assert_valid();
        assert_eq!(m.face(out.faces[0]).unwrap().len(), 6);
    }

    #[test]
    fn a_grid_fill_makes_quads_rather_than_an_ngon() {
        // The reason both exist. An n-gon in the middle of a subdivision cage is
        // a pinch point; a quad patch is not.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        // Punch a 2x2 hole out of the middle.
        let middle: Vec<FaceId> = m
            .face_ids()
            .filter(|f| {
                m.face_verts(*f)
                    .iter()
                    .all(|v| m.position(*v).unwrap().truncate().length() < 1.5)
            })
            .collect();
        assert_eq!(middle.len(), 4);
        for f in middle {
            m.remove_face(f);
        }
        m.remove_loose();
        m.assert_valid();

        // A hole in a sheet leaves *two* borders — the hole and the sheet's own
        // outer edge — so the ring has to be named rather than inferred.
        let borders = super::super::bridge::open_borders(&m).unwrap();
        assert_eq!(borders.len(), 2, "the hole and the outer edge");
        let ring = borders
            .into_iter()
            .find(|r| r.len() == 8)
            .expect("the hole's outline");

        let out = grid_fill(&mut m, &ring, default_span(ring.len())).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 4, "a 2x2 patch");
        assert_eq!(out.verts.len(), 1, "one interior vertex");
        for f in &out.faces {
            assert_eq!(m.face(*f).unwrap().len(), 4);
        }
        for e in m.edge_ids() {
            assert!(m.edge_face_count(e) <= 2, "no non-manifold edge");
        }
    }

    #[test]
    fn a_grid_fill_follows_a_boundary_that_only_one_side_bows() {
        // The Coons correction, measured where it is the only thing that can
        // produce the answer. One side is bowed and the other three are flat, so
        // interpolating between the *flat* pair gives a flat interior no matter
        // how pronounced the bow is. Only the terms from the bowed side can lift
        // the interior to meet it.
        //
        // A weaker check — "the interior stays within the boundary's range" —
        // passes with those terms removed, which is why this measures the lift
        // directly.
        let mut m = BMesh::new();
        let a = 3usize;
        let b = 3usize;
        let bow = 1.0f32;

        // A 3x3 patch outline, laid out as a square in XY.
        let mut points: Vec<Vec3> = Vec::new();
        for i in 0..=a {
            // The bottom side, bowed upward in the middle. The corners stay
            // flat, so a flat interior cannot be explained by them.
            let lift = if i == 0 || i == a { 0.0 } else { bow };
            points.push(Vec3::new(i as f32, 0.0, lift));
        }
        for j in 1..=b {
            points.push(Vec3::new(a as f32, j as f32, 0.0));
        }
        for i in (0..a).rev() {
            points.push(Vec3::new(i as f32, b as f32, 0.0));
        }
        for j in (1..b).rev() {
            points.push(Vec3::new(0.0, j as f32, 0.0));
        }
        assert_eq!(points.len(), 2 * (a + b), "the outline of a {a}x{b} patch");

        let ring: Vec<VertId> = points.iter().map(|p| m.add_vert(*p)).collect();
        let n = ring.len();
        for i in 0..n {
            m.add_edge(ring[i], ring[(i + 1) % n]).unwrap();
        }

        let out = grid_fill(&mut m, &ring, a).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), a * b);
        assert_eq!(out.verts.len(), (a - 1) * (b - 1));

        // The interior row nearest the bowed side must be lifted by most of the
        // bow; the far row by much less.
        let heights: Vec<(f32, f32)> = out
            .verts
            .iter()
            .map(|v| {
                let p = m.position(*v).unwrap();
                (p.y, p.z)
            })
            .collect();
        let near = heights
            .iter()
            .filter(|(y, _)| *y < 1.5)
            .map(|(_, z)| *z)
            .fold(f32::MIN, f32::max);
        let far = heights
            .iter()
            .filter(|(y, _)| *y > 1.5)
            .map(|(_, z)| *z)
            .fold(f32::MIN, f32::max);
        assert!(
            near > bow * 0.5,
            "the row beside the bow was lifted only {near}, so the interior is              not meeting its own boundary"
        );
        assert!(far < near, "and the far row should be less lifted: {far}");
    }

    #[test]
    fn an_odd_border_is_refused_with_the_reason() {
        // A patch has two sides of each length, so an odd ring cannot be one.
        // Refused rather than rounded, because the rounding would put a triangle
        // somewhere the user did not choose.
        let mut m = BMesh::new();
        let ring: Vec<VertId> = (0..5)
            .map(|i| {
                let t = i as f32 / 5.0 * std::f32::consts::TAU;
                m.add_vert(Vec3::new(t.cos(), t.sin(), 0.0))
            })
            .collect();
        for i in 0..5 {
            m.add_edge(ring[i], ring[(i + 1) % 5]).unwrap();
        }
        assert_eq!(
            grid_fill(&mut m, &ring, 2),
            Err(FillError::OddBorder { len: 5 })
        );
        assert_eq!(m.num_faces(), 0, "nothing half-built was left behind");

        // And an n-gon fill handles it fine, which is the answer to give.
        assert!(fill(&mut m, &ring).is_ok());
    }

    #[test]
    fn a_bad_span_is_refused() {
        let (mut m, _) = holed_cube();
        let ring = the_only_border(&m).unwrap();
        assert_eq!(
            grid_fill(&mut m, &ring, 2),
            Err(FillError::BadSpan { len: 4, span: 2 })
        );
        assert_eq!(m.num_faces(), 5);
    }

    #[test]
    fn a_closed_solid_has_nothing_to_fill() {
        let m = BMesh::from_mesh_data(&primitives::cube(2.0));
        assert_eq!(
            the_only_border(&m),
            Err(FillError::NotOneBorder { found: 0 })
        );
    }

    #[test]
    fn two_holes_are_refused_rather_than_one_being_picked() {
        let mut m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 6));
        let caps: Vec<FaceId> = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 6)
            .collect();
        for f in caps {
            m.remove_face(f);
        }
        assert_eq!(
            the_only_border(&m),
            Err(FillError::NotOneBorder { found: 2 })
        );
    }

    #[test]
    fn filling_the_selection_goes_through_the_same_path() {
        let (mut m, _) = holed_cube();
        m.select_none();
        for e in m.edge_ids().collect::<Vec<_>>() {
            if m.is_boundary_edge(e) {
                m.set_edge_selected(e, true);
            }
        }
        m.flush_selection(SelectMode::Edge);

        let out = fill_selected(&mut m, false, None).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 1);
        assert_eq!(m.selected_faces(), out.faces);
    }

    #[test]
    fn a_filled_mesh_still_bakes_to_storage_form() {
        let (mut m, _) = holed_cube();
        let ring = the_only_border(&m).unwrap();
        fill(&mut m, &ring).unwrap();
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }
}
