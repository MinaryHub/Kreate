//! Unwrap: each piece of the surface, cut along its seams, laid flat with as
//! little stretch as it allows — Blender's `U` ▸ Unwrap.
//!
//! # Least squares conformal maps
//!
//! Lévy et al.'s LSCM. A map is conformal when it keeps angles, and a
//! triangle maps conformally exactly when one complex equation holds over its
//! corners. A surface can rarely satisfy that everywhere, so the unwrap is the
//! map that comes closest in the least-squares sense, weighted by area. Two
//! corners are pinned — otherwise the whole map could slide and shrink to a
//! point — and the rest is one linear least-squares problem, solved here by
//! conjugate gradients on its normal equations: no dense matrix, no external
//! solver, linear memory.
//!
//! # Islands
//!
//! Faces are joined across every edge that is not a seam and whose other face
//! is also being unwrapped; each connected piece is laid flat on its own, at
//! the size it has on the model, and the pieces are packed together after.
//!
//! # Pins
//!
//! A corner pinned in the UV layer (`uv_pin`) keeps the UV it has. Two pins or
//! more in an island replace the two the solver would pick, so the island is
//! laid around where they are and left there — not flipped, scaled or packed.
//! One pin keeps the island's own shape and moves it so that corner stays put.

use std::collections::HashMap;

use kreate_math::{Vec2, Vec3};

use crate::arena::{ArenaIndex, FaceId, LoopId, VertId};
use crate::attribute::{names, AttrType, Domain};
use crate::bmesh::BMesh;
use crate::edit::uv_pack;

/// Unwraps `faces` into `layer`, packing the pieces with `margin` between
/// them. Returns how many corners were written.
pub fn unwrap(mesh: &mut BMesh, faces: &[FaceId], layer: &str, margin: f32) -> usize {
    let islands = islands(mesh, faces);
    let mut written: Vec<(LoopId, Vec2)> = Vec::new();
    let mut packed: Vec<Vec<LoopId>> = Vec::new();
    let seams = mesh
        .attributes()
        .get_bool(names::SEAM)
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let pins = mesh
        .attributes()
        .get_bool(names::UV_PIN)
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let current = mesh
        .attributes()
        .get_f32x2(layer)
        .map(|s| s.to_vec())
        .unwrap_or_default();
    for island in &islands {
        let Some((uvs, pinned)) = flatten(mesh, island, &seams, &pins, &current) else {
            continue;
        };
        let mut corners = Vec::new();
        for (l, uv) in uvs {
            written.push((l, uv));
            corners.push(l);
        }
        // A pinned island stays where its pins put it.
        if !pinned {
            packed.push(corners);
        }
    }
    if written.is_empty() {
        return 0;
    }
    let len = mesh.slot_count(Domain::Corner);
    mesh.attributes_mut()
        .ensure(layer, Domain::Corner, AttrType::F32x2, len);
    if let Some(values) = mesh.attributes_mut().make_mut_f32x2(layer) {
        for (l, uv) in &written {
            if let Some(slot) = values.get_mut(l.index() as usize) {
                *slot = *uv;
            }
        }
    }
    uv_pack::pack(mesh, layer, &packed, margin);
    written.len()
}

/// The pieces: faces joined across edges that are not seams.
fn islands(mesh: &BMesh, faces: &[FaceId]) -> Vec<Vec<FaceId>> {
    let wanted: std::collections::HashSet<FaceId> = faces.iter().copied().collect();
    let seams = mesh
        .attributes()
        .get_bool(names::SEAM)
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let mut seen: std::collections::HashSet<FaceId> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &start in faces {
        if !seen.insert(start) {
            continue;
        }
        let mut island = vec![start];
        let mut work = vec![start];
        while let Some(f) = work.pop() {
            for l in mesh.face_loops(f) {
                let Some(e) = mesh.loop_edge(l) else { continue };
                if seams.get(e.index() as usize).copied().unwrap_or(false) {
                    continue;
                }
                for g in mesh.edge_faces(e) {
                    if wanted.contains(&g) && seen.insert(g) {
                        island.push(g);
                        work.push(g);
                    }
                }
            }
        }
        out.push(island);
    }
    out
}

/// One island laid flat: a UV per corner, at the island's size on the model.
///
/// The unknowns are not the vertices but the corners joined around each
/// vertex across edges that are not seams: a seam running into the island
/// splits its vertices in two, one for each side of the cut, which is what
/// lets a tube cut once down its side open flat.
fn flatten(
    mesh: &BMesh,
    island: &[FaceId],
    seams: &[bool],
    pins: &[bool],
    current: &[Vec2],
) -> Option<(Vec<(LoopId, Vec2)>, bool)> {
    let in_island: std::collections::HashSet<FaceId> = island.iter().copied().collect();
    let mut parent: HashMap<LoopId, LoopId> = HashMap::new();
    fn root(parent: &mut HashMap<LoopId, LoopId>, mut l: LoopId) -> LoopId {
        while let Some(&p) = parent.get(&l) {
            if p == l {
                break;
            }
            let gp = parent.get(&p).copied().unwrap_or(p);
            parent.insert(l, gp);
            l = gp;
        }
        l
    }
    let loop_at = |f: FaceId, v: VertId| {
        mesh.face_loops(f)
            .into_iter()
            .find(|l| mesh.loop_vert(*l) == Some(v))
    };
    for f in island {
        for l in mesh.face_loops(*f) {
            parent.entry(l).or_insert(l);
        }
    }
    for f in island {
        for l in mesh.face_loops(*f) {
            let Some(e) = mesh.loop_edge(l) else { continue };
            if seams.get(e.index() as usize).copied().unwrap_or(false) {
                continue;
            }
            let Some(ends) = mesh.edge_verts(e) else {
                continue;
            };
            for g in mesh.edge_faces(e) {
                if g == *f || !in_island.contains(&g) {
                    continue;
                }
                for v in ends {
                    if let (Some(a), Some(b)) = (loop_at(*f, v), loop_at(g, v)) {
                        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                        if ra != rb {
                            parent.insert(ra, rb);
                        }
                    }
                }
            }
        }
    }
    // The unknowns, numbered, each with its vertex's position.
    let mut index: HashMap<LoopId, usize> = HashMap::new();
    let mut verts: Vec<VertId> = Vec::new();
    let mut corner_of: Vec<(LoopId, usize)> = Vec::new();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    for f in island {
        let ring: Vec<usize> = mesh
            .face_loops(*f)
            .into_iter()
            .map(|l| {
                let r = root(&mut parent, l);
                let k = *index.entry(r).or_insert_with(|| {
                    verts.push(mesh.loop_vert(l).expect("a loop has a vertex"));
                    verts.len() - 1
                });
                corner_of.push((l, k));
                k
            })
            .collect();
        for i in 1..ring.len().saturating_sub(1) {
            triangles.push([ring[0], ring[i], ring[i + 1]]);
        }
    }
    let n = verts.len();
    if n < 3 || triangles.is_empty() {
        return None;
    }
    let p: Vec<Vec3> = verts
        .iter()
        .map(|v| mesh.position(*v).unwrap_or(Vec3::ZERO))
        .collect();

    // The user's pins: an unknown is pinned when any of its corners is, at
    // that corner's UV.
    let mut pinned_at: Vec<Option<Vec2>> = vec![None; n];
    for (l, k) in &corner_of {
        let i = l.index() as usize;
        if pins.get(i).copied().unwrap_or(false) {
            if let Some(uv) = current.get(i) {
                pinned_at[*k] = Some(*uv);
            }
        }
    }
    let user_pins: Vec<usize> = (0..n).filter(|k| pinned_at[*k].is_some()).collect();

    // Two pins far apart: from any vertex, the farthest; from that, the farthest.
    let far = |from: usize| {
        (0..n)
            .max_by(|a, b| p[*a].distance(p[from]).total_cmp(&p[*b].distance(p[from])))
            .unwrap_or(0)
    };
    let pin_a = far(0);
    let pin_b = far(pin_a);
    if pin_a == pin_b && user_pins.len() < 2 {
        return None;
    }
    let span = p[pin_a].distance(p[pin_b]) as f64;

    // Each triangle's rows of the conformal equation, in real form: two rows
    // per triangle over (u, v) of each corner. Built in the triangle's own
    // plane, scaled by the square root of its area.
    struct Row {
        terms: Vec<(usize, f64)>, // (unknown, coefficient); unknown 2k is u_k, 2k+1 is v_k
    }
    let mut rows: Vec<Row> = Vec::with_capacity(triangles.len() * 2);
    let mut area3d = 0.0f64;
    for t in &triangles {
        let (a, b, c) = (p[t[0]].as_dvec3(), p[t[1]].as_dvec3(), p[t[2]].as_dvec3());
        let e1 = b - a;
        let e2 = c - a;
        let len1 = e1.length();
        let cross = e1.cross(e2).length();
        if len1 < 1e-12 || cross < 1e-12 {
            continue;
        }
        area3d += cross * 0.5;
        let x = e1 / len1;
        let local = [(0.0, 0.0), (len1, 0.0), (e2.dot(x), cross / len1)];
        let scale = 1.0 / (cross * 0.5).sqrt();
        // W_j = (x_l - x_k) + i (y_l - y_k) for (j, k, l) in cyclic order.
        let mut re: Vec<(usize, f64)> = Vec::new();
        let mut im: Vec<(usize, f64)> = Vec::new();
        for (j, corner) in t.iter().enumerate() {
            let (k, l) = ((j + 1) % 3, (j + 2) % 3);
            let wr = (local[l].0 - local[k].0) * scale;
            let wi = (local[l].1 - local[k].1) * scale;
            let (u, v) = (2 * corner, 2 * corner + 1);
            // (wr + i wi)(u + i v) = (wr u - wi v) + i (wi u + wr v)
            re.push((u, wr));
            re.push((v, -wi));
            im.push((u, wi));
            im.push((v, wr));
        }
        rows.push(Row { terms: re });
        rows.push(Row { terms: im });
    }
    if rows.is_empty() {
        return None;
    }

    // Pinned values: the two pins along the U axis, as far apart as on the
    // model, so the solution comes out at roughly the model's scale.
    let unknowns = 2 * n;
    let mut fixed = vec![None::<f64>; unknowns];
    if user_pins.len() >= 2 {
        // The user's pins are the constraints: the island is laid around them.
        for k in &user_pins {
            let uv = pinned_at[*k].expect("pinned");
            fixed[2 * k] = Some(uv.x as f64);
            fixed[2 * k + 1] = Some(uv.y as f64);
        }
    } else {
        fixed[2 * pin_a] = Some(0.0);
        fixed[2 * pin_a + 1] = Some(0.0);
        fixed[2 * pin_b] = Some(span);
        fixed[2 * pin_b + 1] = Some(0.0);
    }

    // Normal equations AᵀA x = -Aᵀ B p over the free unknowns, by CG,
    // matrix-free: A applied row by row each time.
    let free: Vec<usize> = (0..unknowns).filter(|i| fixed[*i].is_none()).collect();
    let mut slot = vec![usize::MAX; unknowns];
    for (k, i) in free.iter().enumerate() {
        slot[*i] = k;
    }
    let apply = |x: &[f64]| -> Vec<f64> {
        // Aᵀ (A x) over the free unknowns only.
        let mut out = vec![0.0; free.len()];
        for row in &rows {
            let mut s = 0.0;
            for (i, c) in &row.terms {
                if slot[*i] != usize::MAX {
                    s += c * x[slot[*i]];
                }
            }
            for (i, c) in &row.terms {
                if slot[*i] != usize::MAX {
                    out[slot[*i]] += c * s;
                }
            }
        }
        out
    };
    let mut rhs = vec![0.0; free.len()];
    for row in &rows {
        let pinned: f64 = row
            .terms
            .iter()
            .filter_map(|(i, c)| fixed[*i].map(|f| c * f))
            .sum();
        for (i, c) in &row.terms {
            if slot[*i] != usize::MAX {
                rhs[slot[*i]] -= c * pinned;
            }
        }
    }
    let mut x = vec![0.0; free.len()];
    let mut r = rhs.clone();
    let mut d = r.clone();
    let mut rr: f64 = r.iter().map(|v| v * v).sum();
    let tolerance = rr.max(1e-30) * 1e-20;
    for _ in 0..(4 * free.len()).max(100) {
        if rr <= tolerance {
            break;
        }
        let ad = apply(&d);
        let dad: f64 = d.iter().zip(&ad).map(|(a, b)| a * b).sum();
        if dad.abs() < 1e-300 {
            break;
        }
        let alpha = rr / dad;
        for k in 0..x.len() {
            x[k] += alpha * d[k];
            r[k] -= alpha * ad[k];
        }
        let next: f64 = r.iter().map(|v| v * v).sum();
        let beta = next / rr;
        rr = next;
        for k in 0..d.len() {
            d[k] = r[k] + beta * d[k];
        }
    }
    let value = |i: usize| fixed[i].unwrap_or_else(|| x[slot[i]]);
    let mut uv: Vec<Vec2> = (0..n)
        .map(|k| Vec2::new(value(2 * k) as f32, value(2 * k + 1) as f32))
        .collect();

    if user_pins.len() >= 2 {
        // Where the pins said: no flip, no scale, no packing.
        if !uv.iter().all(|p| p.is_finite()) {
            return None;
        }
        return Some((corner_of.into_iter().map(|(l, k)| (l, uv[k])).collect(), true));
    }

    // A map can come out mirrored; a mirrored island samples a texture
    // backwards. Flip it if most triangles wound the other way.
    let flipped = triangles
        .iter()
        .filter(|t| {
            let (a, b, c) = (uv[t[0]], uv[t[1]], uv[t[2]]);
            (b - a).perp_dot(c - a) < 0.0
        })
        .count();
    if flipped * 2 > triangles.len() {
        for p in &mut uv {
            p.y = -p.y;
        }
    }
    // The island at its size on the model: its UV area matched to its area.
    let area2d: f64 = triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (uv[t[0]], uv[t[1]], uv[t[2]]);
            ((b - a).perp_dot(c - a) as f64 * 0.5).abs()
        })
        .sum();
    if area2d > 1e-18 {
        let s = (area3d / area2d).sqrt() as f32;
        for p in &mut uv {
            *p *= s;
        }
    }
    // One pin: the island keeps its shape and moves so that corner stays.
    let one_pin = user_pins.first().copied();
    if let Some(k) = one_pin {
        let shift = pinned_at[k].expect("pinned") - uv[k];
        for p in &mut uv {
            *p += shift;
        }
    }
    if !uv.iter().all(|p| p.is_finite()) {
        return None;
    }
    Some((
        corner_of.into_iter().map(|(l, k)| (l, uv[k])).collect(),
        one_pin.is_some(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bmesh::SelectMode;
    use crate::primitives;

    const LAYER: &str = "uv";

    /// UV triangle areas over 3D areas, per triangle: one everywhere is no
    /// stretch at all.
    fn stretch(mesh: &BMesh) -> Vec<f32> {
        let uvs = mesh.attributes().get_f32x2(LAYER).unwrap().to_vec();
        let mut out = Vec::new();
        for f in mesh.face_ids() {
            let loops = mesh.face_loops(f);
            let p: Vec<Vec3> = loops
                .iter()
                .map(|l| mesh.position(mesh.loop_vert(*l).unwrap()).unwrap())
                .collect();
            let t: Vec<Vec2> = loops.iter().map(|l| uvs[l.index() as usize]).collect();
            for i in 1..loops.len() - 1 {
                let a3 = (p[i] - p[0]).cross(p[i + 1] - p[0]).length() * 0.5;
                let a2 = ((t[i] - t[0]).perp_dot(t[i + 1] - t[0]) * 0.5).abs();
                out.push(a2 / a3);
            }
        }
        out
    }

    #[test]
    fn a_flat_grid_unwraps_without_stretch() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(2.0, 4, 4));
        let faces: Vec<FaceId> = m.face_ids().collect();
        assert_eq!(unwrap(&mut m, &faces, LAYER, 0.0), 64);
        let s = stretch(&m);
        let first = s[0];
        assert!(s.iter().all(|x| (x / first - 1.0).abs() < 1e-3), "{s:?}");
    }

    #[test]
    fn a_cube_cut_along_seams_lays_out_flat_and_every_face_keeps_its_shape() {
        // Seams on the edges that cut a cube into a cross: everything but the
        // four edges around the bottom face and one vertical edge... simpler:
        // seam every edge, so each face is its own island.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let n = m.slot_count(Domain::Edge);
        m.attributes_mut()
            .ensure(names::SEAM, Domain::Edge, AttrType::Bool, n);
        for s in m
            .attributes_mut()
            .make_mut_bool(names::SEAM)
            .unwrap()
            .iter_mut()
        {
            *s = true;
        }
        let faces: Vec<FaceId> = m.face_ids().collect();
        assert_eq!(unwrap(&mut m, &faces, LAYER, 0.02), 24);
        // Each square face comes out square: equal stretch on its triangles.
        let s = stretch(&m);
        assert!(s.iter().all(|x| (x / s[0] - 1.0).abs() < 1e-3), "{s:?}");
        // And everything inside the unit square.
        let uvs = m.attributes().get_f32x2(LAYER).unwrap();
        assert!(uvs
            .iter()
            .all(|p| p.x >= -1e-4 && p.x <= 1.0001 && p.y >= -1e-4 && p.y <= 1.0001));
        let _ = SelectMode::Face;
    }

    #[test]
    fn a_cylinder_cut_once_down_its_side_opens_flat() {
        // A tube: the sides of a cylinder without its caps, one seam down it.
        let mut m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 16));
        let sides: Vec<FaceId> = m
            .face_ids()
            .filter(|f| m.face_verts(*f).len() == 4)
            .collect();
        let vertical = m
            .edge_ids()
            .find(|e| {
                let [a, b] = m.edge_verts(*e).unwrap();
                let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
                (pa.x - pb.x).abs() < 1e-5 && (pa.y - pb.y).abs() < 1e-5
            })
            .expect("a side edge");
        let n = m.slot_count(Domain::Edge);
        m.attributes_mut()
            .ensure(names::SEAM, Domain::Edge, AttrType::Bool, n);
        m.attributes_mut().make_mut_bool(names::SEAM).unwrap()[vertical.index() as usize] = true;
        unwrap(&mut m, &sides, LAYER, 0.0);
        // A developable surface: no stretch, so every side triangle alike.
        let uvs = m.attributes().get_f32x2(LAYER).unwrap().to_vec();
        let mut ratios = Vec::new();
        for f in &sides {
            let loops = m.face_loops(*f);
            let p: Vec<Vec3> = loops
                .iter()
                .map(|l| m.position(m.loop_vert(*l).unwrap()).unwrap())
                .collect();
            let t: Vec<Vec2> = loops.iter().map(|l| uvs[l.index() as usize]).collect();
            let a3 = (p[1] - p[0]).cross(p[2] - p[0]).length();
            let a2 = (t[1] - t[0]).perp_dot(t[2] - t[0]).abs();
            ratios.push(a2 / a3);
        }
        assert!(
            ratios.iter().all(|r| (r / ratios[0] - 1.0).abs() < 0.02),
            "{ratios:?}"
        );
    }

    /// Pins the corners at vertices `which`, at the UVs they have now.
    fn pin(m: &mut BMesh, which: &[usize]) {
        let n = m.slot_count(Domain::Corner);
        m.attributes_mut()
            .ensure(names::UV_PIN, Domain::Corner, AttrType::Bool, n);
        let corners: Vec<usize> = m
            .face_ids()
            .flat_map(|f| m.face_loops(f))
            .filter(|l| which.contains(&(m.loop_vert(*l).unwrap().index() as usize)))
            .map(|l| l.index() as usize)
            .collect();
        let flags = m.attributes_mut().make_mut_bool(names::UV_PIN).unwrap();
        for c in corners {
            flags[c] = true;
        }
    }

    fn uv_at_vertex(m: &BMesh, v: usize) -> Vec2 {
        let uvs = m.attributes().get_f32x2(LAYER).unwrap();
        let l = m
            .face_ids()
            .flat_map(|f| m.face_loops(f))
            .find(|l| m.loop_vert(*l).unwrap().index() as usize == v)
            .unwrap();
        uvs[l.index() as usize]
    }

    #[test]
    fn two_pins_stay_where_they_are_and_the_rest_is_laid_around_them() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(2.0, 4, 4));
        let faces: Vec<FaceId> = m.face_ids().collect();
        unwrap(&mut m, &faces, LAYER, 0.0);
        // Two far corners of the grid, moved somewhere new and pinned there.
        let (a, b) = (0usize, m.slot_count(Domain::Point) - 1);
        let (ua, ub) = (Vec2::new(3.0, 1.0), Vec2::new(5.5, 3.5));
        {
            let positions: Vec<(usize, usize)> = m
                .face_ids()
                .flat_map(|f| m.face_loops(f))
                .map(|l| (l.index() as usize, m.loop_vert(l).unwrap().index() as usize))
                .collect();
            let uvs = m.attributes_mut().make_mut_f32x2(LAYER).unwrap();
            for (c, v) in positions {
                if v == a {
                    uvs[c] = ua;
                }
                if v == b {
                    uvs[c] = ub;
                }
            }
        }
        pin(&mut m, &[a, b]);
        unwrap(&mut m, &faces, LAYER, 0.0);
        assert!(uv_at_vertex(&m, a).distance(ua) < 1e-5);
        assert!(uv_at_vertex(&m, b).distance(ub) < 1e-5);
        // Not packed back into the unit square: laid around the pins, and
        // still without stretch.
        let s = stretch(&m);
        assert!(s.iter().all(|x| (x / s[0] - 1.0).abs() < 1e-3), "{s:?}");
    }

    #[test]
    fn one_pin_keeps_the_island_s_shape_and_holds_that_corner() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(2.0, 4, 4));
        let faces: Vec<FaceId> = m.face_ids().collect();
        unwrap(&mut m, &faces, LAYER, 0.0);
        let before = stretch(&m);
        let held = uv_at_vertex(&m, 5) + Vec2::new(2.0, -1.0);
        {
            let positions: Vec<(usize, usize)> = m
                .face_ids()
                .flat_map(|f| m.face_loops(f))
                .map(|l| (l.index() as usize, m.loop_vert(l).unwrap().index() as usize))
                .collect();
            let uvs = m.attributes_mut().make_mut_f32x2(LAYER).unwrap();
            for (c, v) in positions {
                if v == 5 {
                    uvs[c] = held;
                }
            }
        }
        pin(&mut m, &[5]);
        unwrap(&mut m, &faces, LAYER, 0.0);
        assert!(uv_at_vertex(&m, 5).distance(held) < 1e-5);
        let after = stretch(&m);
        assert!((after[0] / before[0] - 1.0).abs() < 1e-3 || after.iter().all(|x| (x / after[0] - 1.0).abs() < 1e-3));
    }
}
