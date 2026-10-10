//! Voxel remesh: the mesh rebuilt as even squares over a grid of voxels —
//! Blender's sculpt-mode Ctrl+R.
//!
//! A sculpt stretches its faces where the brush pulled and crowds them where
//! it pushed; a remesh evens them out so the next stroke has something to
//! work with. It also closes holes and fuses parts that overlap, which is
//! what makes it the way to join shapes into one surface.
//!
//! # How
//!
//! 1. **Distance**: at every grid point near the surface, how far it is from
//!    the nearest triangle.
//! 2. **Inside or out**: along each row of the grid a ray counts the surface
//!    it has gone in through and come out of — the winding number. Anything
//!    wound into is inside, so where two parts overlap is inside both rather
//!    than, as by counting crossings, outside again. That is the sign of the
//!    distance.
//! 3. **Surface nets**: one vertex in each voxel the surface passes through,
//!    at the middle of where the surface crosses its edges, and one square
//!    across every grid edge the surface crosses. Squares, as Blender's
//!    remesh makes, and a closed surface every time.

use kreate_math::Vec3;

use crate::MeshData;

/// The most grid points a remesh will use — a little over 300 MB of working
/// memory at the far end. Finer than this is refused rather than tried.
pub const MAX_POINTS: usize = 40_000_000;

/// Why a remesh could not be done.
#[derive(Clone, Debug, PartialEq)]
pub enum RemeshError {
    /// There is no surface to rebuild.
    Empty,
    /// The voxel size is too small for the mesh: the grid would have this
    /// many points.
    TooFine(usize),
}

impl std::fmt::Display for RemeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemeshError::Empty => write!(f, "the mesh has no faces to remesh"),
            RemeshError::TooFine(n) => write!(
                f,
                "a voxel that small needs {n} grid points, more than {MAX_POINTS}: make the voxel size bigger"
            ),
        }
    }
}

/// The point of triangle `abc` nearest `p`. Ericson's, §5.1.5.
fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Values on the points of a grid: `dims` points along each axis, `h`
/// apart, the first at `origin`, X fastest. Negative is inside.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub origin: Vec3,
    pub h: f32,
    pub dims: [usize; 3],
    pub values: Vec<f32>,
}

impl Field {
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    pub fn point(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin + Vec3::new(i as f32, j as f32, k as f32) * self.h
    }

    pub fn at(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[self.index(i, j, k)]
    }
}

/// Which points of a grid — `dims` points `h` apart from `origin`, X
/// fastest — are inside the closed surface `tris`: wound into, counted along
/// rows in X, so overlapping parts are inside both and inward-facing faces
/// still make a solid.
pub fn winding(tris: &[[Vec3; 3]], origin: Vec3, h: f32, dims: [usize; 3]) -> Vec<bool> {
    let cell_of = |v: f32, axis: usize| (v - origin[axis]) / h;
    // The row is nudged off the grid's exact lines, so it does not run along
    // an edge it would count twice.
    let nudge = Vec3::new(0.0, 0.000_37 * h, 0.000_71 * h);
    // Each crossing and whether the ray goes in (+1) or out (−1) there, by
    // which way the triangle faces.
    let mut rows: Vec<Vec<(f32, i32)>> = vec![Vec::new(); dims[1] * dims[2]];
    for t in tris {
        let (tlo, thi) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
        if cell_of(thi.y, 1) < 0.0 || cell_of(thi.z, 2) < 0.0 {
            continue;
        }
        let j0 = cell_of(tlo.y, 1).floor().max(0.0) as usize;
        let j1 = (cell_of(thi.y, 1).ceil().max(0.0) as usize).min(dims[1] - 1);
        let k0 = cell_of(tlo.z, 2).floor().max(0.0) as usize;
        let k1 = (cell_of(thi.z, 2).ceil().max(0.0) as usize).min(dims[2] - 1);
        for k in k0..=k1 {
            for j in j0..=j1 {
                let q = origin + Vec3::new(0.0, j as f32, k as f32) * h + nudge;
                // Where the line along X at (q.y, q.z) meets the triangle, if
                // it does: barycentric coordinates in the YZ plane.
                let (a, b, c) = (t[0], t[1], t[2]);
                let d = (b.y - a.y) * (c.z - a.z) - (c.y - a.y) * (b.z - a.z);
                if d.abs() < 1e-20 {
                    continue;
                }
                let u = ((q.y - a.y) * (c.z - a.z) - (c.y - a.y) * (q.z - a.z)) / d;
                let v = ((b.y - a.y) * (q.z - a.z) - (q.y - a.y) * (b.z - a.z)) / d;
                if u < 0.0 || v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let x = a.x + (b.x - a.x) * u + (c.x - a.x) * v;
                let facing = (b - a).cross(c - a).x;
                rows[k * dims[1] + j].push((x, if facing < 0.0 { 1 } else { -1 }));
            }
        }
    }
    let mut inside = vec![false; dims.iter().product()];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            let row = &mut rows[k * dims[1] + j];
            row.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut crossed, mut winding) = (0, 0);
            for i in 0..dims[0] {
                let x = origin.x + i as f32 * h;
                while crossed < row.len() && row[crossed].0 < x {
                    winding += row[crossed].1;
                    crossed += 1;
                }
                inside[(k * dims[1] + j) * dims[0] + i] = winding != 0;
            }
        }
    }
    inside
}

/// The mesh rebuilt as squares at `voxel` spacing.
pub fn voxel_remesh(mesh: &MeshData, voxel: f32) -> Result<MeshData, RemeshError> {
    let tri = mesh.triangulate();
    let (tris, _) = tri.indices.as_chunks::<3>();
    let p = mesh.positions();
    let tris: Vec<[Vec3; 3]> = tris
        .iter()
        .map(|t| t.map(|i| p[i as usize]))
        .filter(|t| t.iter().all(|v| v.is_finite()))
        .collect();
    if tris.is_empty() || voxel.is_nan() || voxel <= 0.0 {
        return Err(RemeshError::Empty);
    }
    let h = voxel;
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for t in &tris {
        for v in t {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    // Two voxels of room all round, so the outside is outside everywhere at
    // the grid's edge.
    let origin = lo - Vec3::splat(2.0 * h);
    let extent = (hi - lo) + Vec3::splat(4.0 * h);
    let dims = [extent.x, extent.y, extent.z].map(|e| (e / h).ceil() as usize + 1);
    let points = dims.iter().product::<usize>();
    if points > MAX_POINTS {
        return Err(RemeshError::TooFine(points));
    }
    let band = 2.0 * h;
    let mut field = Field {
        origin,
        h,
        dims,
        values: vec![f32::MAX; points],
    };
    // 1. Distance, near the surface.
    let cell_of = |v: f32, axis: usize| (v - origin[axis]) / h;
    for t in &tris {
        let (tlo, thi) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
        let range = |axis: usize| {
            let a = (cell_of(tlo[axis] - band, axis).floor().max(0.0)) as usize;
            let b = (cell_of(thi[axis] + band, axis).ceil() as usize).min(dims[axis] - 1);
            a..=b
        };
        for k in range(2) {
            for j in range(1) {
                for i in range(0) {
                    let q = field.point(i, j, k);
                    let d = closest_on_triangle(q, t[0], t[1], t[2]).distance(q);
                    let at = field.index(i, j, k);
                    if d < field.values[at] {
                        field.values[at] = d;
                    }
                }
            }
        }
    }
    // 2. Inside or out, by winding. That is the sign of the distance.
    let inside = winding(&tris, origin, h, dims);
    for (value, inside) in field.values.iter_mut().zip(inside) {
        // Never exactly zero: a point on the surface takes its side from the
        // count like any other, rather than being a `-0` that reads as
        // outside.
        let d = value.clamp(1e-5 * h, 3.0 * h);
        *value = if inside { -d } else { d };
    }
    surface_nets(&field).ok_or(RemeshError::Empty)
}

/// The surface where `field` crosses zero, by surface nets: one vertex in
/// each voxel the surface passes through, at the middle of where it crosses
/// the voxel's edges, and one square across every grid edge it crosses.
/// Facing out, from negative to positive. `None` when it crosses nowhere.
pub fn surface_nets(field: &Field) -> Option<MeshData> {
    let dims = field.dims;
    if dims.iter().any(|d| *d < 2) {
        return None;
    }
    // 3. Surface nets.
    let cells = [dims[0] - 1, dims[1] - 1, dims[2] - 1];
    let cell_index = |i: usize, j: usize, k: usize| (k * cells[1] + j) * cells[0] + i;
    let mut vertex_of = vec![u32::MAX; cells.iter().product()];
    let mut positions: Vec<Vec3> = Vec::new();
    const CORNERS: [[usize; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for k in 0..cells[2] {
        for j in 0..cells[1] {
            for i in 0..cells[0] {
                let v: [f32; 8] = CORNERS.map(|c| field.at(i + c[0], j + c[1], k + c[2]));
                let inside = v.iter().filter(|d| **d < 0.0).count();
                if inside == 0 || inside == 8 {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                let mut n = 0.0;
                for (a, b) in EDGES {
                    if (v[a] < 0.0) != (v[b] < 0.0) {
                        let t = v[a] / (v[a] - v[b]);
                        let pa =
                            field.point(i + CORNERS[a][0], j + CORNERS[a][1], k + CORNERS[a][2]);
                        let pb =
                            field.point(i + CORNERS[b][0], j + CORNERS[b][1], k + CORNERS[b][2]);
                        sum += pa.lerp(pb, t);
                        n += 1.0;
                    }
                }
                vertex_of[cell_index(i, j, k)] = positions.len() as u32;
                positions.push(sum / n);
            }
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut quad = |cs: [[usize; 3]; 4], flip: bool| {
        let ids: Vec<u32> = cs
            .iter()
            .map(|c| vertex_of[cell_index(c[0], c[1], c[2])])
            .collect();
        if ids.contains(&u32::MAX) {
            return;
        }
        faces.push(if flip {
            ids.into_iter().rev().collect()
        } else {
            ids
        });
    };
    for k in 1..cells[2] {
        for j in 1..cells[1] {
            for i in 0..cells[0] {
                // The edge from (i, j, k) along X.
                let (a, b) = (field.at(i, j, k), field.at(i + 1, j, k));
                if (a < 0.0) != (b < 0.0) {
                    quad(
                        [[i, j - 1, k - 1], [i, j, k - 1], [i, j, k], [i, j - 1, k]],
                        b < 0.0,
                    );
                }
            }
        }
    }
    for k in 1..cells[2] {
        for j in 0..cells[1] {
            for i in 1..cells[0] {
                let (a, b) = (field.at(i, j, k), field.at(i, j + 1, k));
                if (a < 0.0) != (b < 0.0) {
                    quad(
                        [[i - 1, j, k - 1], [i - 1, j, k], [i, j, k], [i, j, k - 1]],
                        b < 0.0,
                    );
                }
            }
        }
    }
    for k in 0..cells[2] {
        for j in 1..cells[1] {
            for i in 1..cells[0] {
                let (a, b) = (field.at(i, j, k), field.at(i, j, k + 1));
                if (a < 0.0) != (b < 0.0) {
                    quad(
                        [[i - 1, j - 1, k], [i, j - 1, k], [i, j, k], [i - 1, j, k]],
                        b < 0.0,
                    );
                }
            }
        }
    }
    if faces.is_empty() {
        return None;
    }
    Some(MeshData::from_faces(positions, &faces))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn volume(m: &MeshData) -> f32 {
        let p = m.positions();
        let mut v = 0.0;
        for f in 0..m.num_faces() {
            let ring = m.face_verts(f);
            for k in 1..ring.len() - 1 {
                let (a, b, c) = (
                    p[ring[0] as usize],
                    p[ring[k] as usize],
                    p[ring[k + 1] as usize],
                );
                v += a.dot(b.cross(c)) / 6.0;
            }
        }
        v
    }

    /// Every edge between exactly two faces: closed, no seams.
    fn closed(m: &MeshData) -> bool {
        let mut count = std::collections::HashMap::new();
        for f in 0..m.num_faces() {
            let ring = m.face_verts(f);
            for k in 0..ring.len() {
                let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        count.values().all(|c| *c == 2)
    }

    #[test]
    fn a_cube_comes_back_closed_in_squares_and_the_same_size() {
        let cube = primitives::cube(2.0);
        let m = voxel_remesh(&cube, 0.1).unwrap();
        assert!(
            (0..m.num_faces()).all(|f| m.face_verts(f).len() == 4),
            "squares"
        );
        assert!(closed(&m));
        let v = volume(&m);
        assert!((v - 8.0).abs() < 0.4, "volume {v} near 8, facing out");
    }

    #[test]
    fn a_sphere_stays_round() {
        let ball = primitives::uv_sphere(1.0, 32, 16);
        let m = voxel_remesh(&ball, 0.08).unwrap();
        assert!(closed(&m));
        for p in m.positions() {
            assert!((p.length() - 1.0).abs() < 0.08, "{}", p.length());
        }
    }

    #[test]
    fn overlapping_parts_fuse_into_one_surface() {
        let a = primitives::cube(1.0);
        let mut b = primitives::cube(1.0);
        for p in b.positions_mut() {
            p.x += 0.5;
        }
        let (both, _) = crate::edit::join(&[
            crate::edit::Part {
                mesh: &a,
                transform: kreate_math::DAffine3::IDENTITY,
            },
            crate::edit::Part {
                mesh: &b,
                transform: kreate_math::DAffine3::IDENTITY,
            },
        ]);
        let m = voxel_remesh(&both, 0.05).unwrap();
        assert!(closed(&m));
        let v = volume(&m);
        assert!((v - 1.5).abs() < 0.1, "the union, 1.5: {v}");
    }

    #[test]
    fn too_fine_and_nothing_are_said() {
        let cube = primitives::cube(2.0);
        assert!(matches!(
            voxel_remesh(&cube, 0.0001),
            Err(RemeshError::TooFine(_))
        ));
        assert!(matches!(
            voxel_remesh(&MeshData::empty(), 0.1),
            Err(RemeshError::Empty)
        ));
    }
}
