//! Extrude.
//!
//! The most-used operator in a polygon modeller, and the one that decides
//! whether the rest of the toolset feels trustworthy.
//!
//! # Region, not per-face
//!
//! Extruding a region of faces produces *one* raised shell with walls only
//! around its outline. Extruding each face separately produces a shell per face
//! with walls between them — a different operation, and the one users complain
//! about when a tool gets this wrong. The whole difference is which edges count
//! as boundary: here, an edge is boundary when exactly one of the faces on it is
//! in the region.
//!
//! # Order of operations
//!
//! Everything is *read* before anything is written. The boundary directions come
//! from the original faces' loop cycles, and those cycles stop existing the
//! moment the faces are removed — so removal happens last, after the new shell
//! is already built.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};

/// What an extrusion produced.
#[derive(Clone, Debug, Default)]
pub struct Extrusion {
    /// The raised copies of the original faces. The new selection.
    pub new_faces: Vec<FaceId>,
    /// The walls joining the shell to what it was lifted from.
    pub side_faces: Vec<FaceId>,
    /// The vertices that moved, which is what a transform then drags.
    pub new_verts: Vec<VertId>,
    /// The region's area-weighted normal, before the offset was applied.
    pub normal: Vec3,
}

/// Extrudes a region of faces, offsetting the new shell by `offset`.
///
/// Passing `Vec3::ZERO` gives the topology change alone, which is exactly what
/// an interactive extrude wants: build once on mouse-down, then let the modal
/// transform move `new_verts` every frame.
pub fn extrude_faces(
    mesh: &mut BMesh,
    faces: &[FaceId],
    offset: Vec3,
) -> Result<Extrusion, BmError> {
    let faces: Vec<FaceId> = faces
        .iter()
        .copied()
        .filter(|f| mesh.face(*f).is_some())
        .collect();
    if faces.is_empty() {
        return Ok(Extrusion::default());
    }

    // --- read ---------------------------------------------------------------
    let normal = mesh.region_normal(&faces);

    // Membership by slot: a linear `contains` over the region would make the
    // whole operator quadratic, which shows up the first time someone extrudes
    // a whole character rather than one face.
    let mut in_region = vec![false; mesh.slot_count(crate::Domain::Face)];
    for f in &faces {
        in_region[f.index() as usize] = true;
    }

    let mut rings: Vec<Vec<VertId>> = Vec::with_capacity(faces.len());
    let mut boundary: Vec<(VertId, VertId)> = Vec::new();
    for &f in &faces {
        let loops = mesh.face_loops(f);
        let mut ring = Vec::with_capacity(loops.len());
        for &l in &loops {
            let Some(v) = mesh.loop_vert(l) else { continue };
            ring.push(v);

            let Some(e) = mesh.loop_edge(l) else { continue };
            let inside = mesh
                .edge_faces(e)
                .into_iter()
                .filter(|g| in_region[g.index() as usize])
                .count();
            if inside == 1 {
                // Direction taken from this face's own winding, so the wall it
                // grows is wound consistently with it. Reading it from the edge
                // instead gives an arbitrary orientation and half the walls come
                // out inside-out.
                let Some(next) = mesh.loop_next(l) else {
                    continue;
                };
                let Some(b) = mesh.loop_vert(next) else {
                    continue;
                };
                boundary.push((v, b));
            }
        }
        rings.push(ring);
    }

    // --- write --------------------------------------------------------------
    let mut originals: Vec<VertId> = rings.iter().flatten().copied().collect();
    originals.sort_unstable_by_key(|v| v.index());
    originals.dedup();

    let mut copy_of = vec![None; mesh.slot_count(crate::Domain::Point)];
    let mut new_verts = Vec::with_capacity(originals.len());
    for &v in &originals {
        let Some(p) = mesh.position(v) else { continue };
        let n = mesh.add_vert(p + offset);
        // `add_vert` can grow the point domain, so the table is sized after.
        if copy_of.len() <= v.index() as usize {
            copy_of.resize(v.index() as usize + 1, None);
        }
        copy_of[v.index() as usize] = Some(n);
        new_verts.push(n);
    }
    let copied =
        |v: VertId| -> Option<VertId> { copy_of.get(v.index() as usize).copied().flatten() };

    let mut new_faces = Vec::with_capacity(faces.len());
    for ring in &rings {
        let lifted: Vec<VertId> = ring.iter().filter_map(|v| copied(*v)).collect();
        if lifted.len() == ring.len() {
            new_faces.push(mesh.add_face(&lifted)?);
        }
    }

    let mut side_faces = Vec::with_capacity(boundary.len());
    for (a, b) in boundary {
        let (Some(a2), Some(b2)) = (copied(a), copied(b)) else {
            continue;
        };
        // `(a, b, b', a')`: with `a → b` the original winding, this quad's normal
        // points away from the solid. Reversing it produces a shell that is lit
        // from the inside and reads as a shading bug rather than a winding one.
        side_faces.push(mesh.add_face(&[a, b, b2, a2])?);
    }

    for f in &faces {
        mesh.remove_face(*f);
    }

    // A vertex that the region alone used is now interior to nothing: the shell
    // it belonged to has moved up and the walls do not reach it. Leaving it
    // behind litters the mesh with points that no operator will ever touch again
    // and that the user cannot see to delete.
    for v in originals {
        if mesh.vert_faces(v).is_empty() {
            mesh.remove_vert(v);
        }
    }

    // The new shell is what the user goes on to move, so it is what must be
    // selected — anything else and the follow-up transform grabs the wrong thing.
    mesh.select_none();
    for f in &new_faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);

    Ok(Extrusion {
        new_faces,
        side_faces,
        new_verts,
        normal,
    })
}

/// Extrudes the selected faces along the region normal.
///
/// The form a keystroke reaches: `E` extrudes along the normal, and the modal
/// transform that follows constrains to that axis by default. Into the solid
/// — a negative distance — it is a push: see [`push_faces`].
pub fn extrude_selected_faces(mesh: &mut BMesh, distance: f32) -> Result<Extrusion, BmError> {
    let faces = mesh.selected_faces();
    let normal = mesh.region_normal(&faces);
    if distance < 0.0 {
        return push_faces(mesh, &faces, normal * distance);
    }
    extrude_faces(mesh, &faces, normal * distance)
}

/// Faces pushed into the solid, as Fusion's Press Pull cuts: what they sweep
/// through is taken away, not wrapped in walls.
///
/// A plain extrude inward stands a wall on every edge of the region. Where
/// the face beyond an edge is level with the region — a face inset in a
/// larger one — that wall is right: it is the side of the pocket. Where the
/// face beyond already runs the way the push goes — the side of a box under
/// its top — the wall lies on that face, two surfaces in one place, and the
/// box is not cut at all. So every edge vertex whose faces outside the region
/// all run along the push is welded to its moved copy: those faces follow
/// the region down, and the walls on them fold to nothing. The top of a box
/// pushed in makes a shorter box; an inset face pushed in makes a pocket.
///
/// A region that reaches a side face at one end only — a strip of a box's
/// front, at its right-hand edge — slides at that end and stands a wall at
/// the other, and the wall over the top is then a triangle in the top's own
/// plane. Any wall left lying in the plane of the face beyond it is merged
/// into that face, which is then the top with a notch in it.
pub fn push_faces(mesh: &mut BMesh, faces: &[FaceId], offset: Vec3) -> Result<Extrusion, BmError> {
    let dir = offset.normalize_or_zero();
    if dir == Vec3::ZERO {
        return extrude_faces(mesh, faces, offset);
    }
    let mut in_region = vec![false; mesh.slot_count(crate::Domain::Face)];
    for f in faces {
        if let Some(slot) = in_region.get_mut(f.index() as usize) {
            *slot = true;
        }
    }
    let inside = |f: FaceId| in_region.get(f.index() as usize).copied().unwrap_or(false);
    // The region's vertices that slide: on its edge, and every face beyond
    // runs along the push.
    let mut sliding: Vec<VertId> = Vec::new();
    for f in faces {
        for v in mesh.face_verts(*f) {
            if sliding.contains(&v) {
                continue;
            }
            let outside: Vec<FaceId> = mesh
                .vert_faces(v)
                .into_iter()
                .filter(|g| !inside(*g))
                .collect();
            if !outside.is_empty()
                && outside.iter().all(|g| {
                    let n = mesh.face_normal(*g).normalize_or_zero();
                    n != Vec3::ZERO && n.dot(dir).abs() < 0.02
                })
            {
                sliding.push(v);
            }
        }
    }
    // A face beyond follows the push only whole: every corner of it that
    // is on the region must slide, or the one that does drags the face
    // askew — a top quarter pulled in at one corner and not the other. Such
    // a corner stands instead, and its wall cuts a notch in the face.
    let mut region_points: Vec<VertId> = faces.iter().flat_map(|f| mesh.face_verts(*f)).collect();
    region_points.sort_unstable_by_key(|v| v.index());
    region_points.dedup();
    loop {
        let before = sliding.len();
        let snapshot = sliding.clone();
        sliding.retain(|v| {
            mesh.vert_faces(*v)
                .into_iter()
                .filter(|g| !inside(*g))
                .all(|g| {
                    mesh.face_verts(g)
                        .iter()
                        .filter(|u| {
                            region_points
                                .binary_search_by_key(&u.index(), |r| r.index())
                                .is_ok()
                        })
                        .all(|u| snapshot.contains(u))
                })
        });
        if sliding.len() == before {
            break;
        }
    }
    let positions: Vec<(VertId, Vec3)> = sliding
        .iter()
        .filter_map(|v| Some((*v, mesh.position(*v)?)))
        .collect();
    // Every region vertex and where it was, to find its copy afterwards.
    let mut region_verts: Vec<(VertId, Vec3)> = Vec::new();
    for f in faces {
        for v in mesh.face_verts(*f) {
            if !region_verts.iter().any(|(u, _)| *u == v) {
                if let Some(p) = mesh.position(v) {
                    region_verts.push((v, p));
                }
            }
        }
    }
    let out = extrude_faces(mesh, faces, offset)?;
    // Each sliding vertex's copy: the new vertex where it was, moved.
    let mut groups: Vec<Vec<VertId>> = Vec::new();
    for (v, p) in &positions {
        let target = *p + offset;
        if let Some(copy) = out
            .new_verts
            .iter()
            .copied()
            .find(|n| mesh.position(*n).is_some_and(|q| q.distance(target) < 1e-7))
        {
            mesh.set_position(*v, target);
            groups.push(vec![*v, copy]);
        }
    }
    // The vertices the pushed faces stand on now, to find them again: the
    // merge builds every face it touches anew.
    let mut lifted: Vec<VertId> = out.new_verts.clone();
    lifted.extend(groups.iter().map(|g| g[0]));
    // The vertices that stay and their copies: what the walls run between.
    let standing: Vec<(VertId, VertId)> = region_verts
        .iter()
        .filter(|(v, _)| !sliding.contains(v))
        .filter_map(|(v, p)| {
            let target = *p + offset;
            let copy = out
                .new_verts
                .iter()
                .copied()
                .find(|n| mesh.position(*n).is_some_and(|q| q.distance(target) < 1e-7))?;
            Some((*v, copy))
        })
        .collect();
    crate::edit::merge::merge_groups(mesh, &groups, crate::edit::merge::MergeTarget::First)?;
    let lifted: Vec<VertId> = lifted
        .into_iter()
        .filter(|v| mesh.vert(*v).is_some())
        .collect();
    let mut on = vec![false; mesh.slot_count(crate::Domain::Point)];
    for v in &lifted {
        on[v.index() as usize] = true;
    }
    let new_faces: Vec<FaceId> = mesh
        .face_ids()
        .filter(|f| {
            let ring = mesh.face_verts(*f);
            !ring.is_empty()
                && ring
                    .iter()
                    .all(|v| on.get(v.index() as usize).copied().unwrap_or(false))
        })
        .collect();
    let mut new_faces = new_faces;
    let is_new = |f: FaceId, new_faces: &[FaceId]| new_faces.contains(&f);
    // The walls left: faces with an edge from a standing vertex to its copy.
    let walls: Vec<FaceId> = mesh
        .face_ids()
        .filter(|f| !is_new(*f, &new_faces))
        .filter(|f| {
            let ring = mesh.face_verts(*f);
            (0..ring.len()).any(|i| {
                let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
                standing
                    .iter()
                    .any(|(v, c)| (a == *v && b == *c) || (a == *c && b == *v))
            })
        })
        .collect();
    // Each wall lying in the plane of a face beyond it, with that face. The
    // wall faces the other way and covers part of that face: merged across
    // their shared edge — whose two sides run opposite ways, as any edge's
    // do — the face's outline goes round the wall's, and the part the wall
    // covered is cut out of it. The notch.
    let mut joined: Vec<Vec<FaceId>> = Vec::new();
    for w in &walls {
        let nw = mesh.face_normal(*w).normalize_or_zero();
        if nw == Vec3::ZERO {
            continue;
        }
        for l in mesh.face_loops(*w) {
            let Some(e) = mesh.loop_edge(l) else { continue };
            for g in mesh.edge_faces(e) {
                if g == *w || walls.contains(&g) || is_new(g, &new_faces) {
                    continue;
                }
                if mesh.face_normal(g).normalize_or_zero().dot(nw).abs() > 0.9999 {
                    match joined.iter_mut().find(|j| j.contains(&g) || j.contains(w)) {
                        Some(j) => {
                            for f in [*w, g] {
                                if !j.contains(&f) {
                                    j.push(f);
                                }
                            }
                        }
                        None => joined.push(vec![*w, g]),
                    }
                }
            }
        }
    }
    let mut merged = Vec::new();
    for group in &joined {
        merged.extend(crate::edit::dissolve_faces(mesh, group, &[])?.faces);
    }
    unspike(mesh, merged)?;
    // The pushed faces themselves are untouched by that, but look again.
    new_faces.retain(|f| mesh.face(*f).is_some());
    let side_faces: Vec<FaceId> = out
        .side_faces
        .iter()
        .copied()
        .filter(|f| mesh.face(*f).is_some())
        .collect();
    mesh.select_none();
    for f in &new_faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);
    Ok(Extrusion {
        new_faces,
        side_faces,
        new_verts: lifted,
        normal: out.normal,
    })
}

/// Takes the spikes out of `faces`: an outline that runs out to a vertex
/// and straight back, which a notch merged into a face leaves where the
/// notch was the face's whole width. The vertex goes from the outline, and
/// the face beyond the edge the spike ran along takes the point the outline
/// now turns at, so the two still meet edge to edge. A vertex no face uses
/// any more goes too.
fn unspike(mesh: &mut BMesh, mut faces: Vec<FaceId>) -> Result<(), BmError> {
    let position = |mesh: &BMesh, v: VertId| mesh.position(v).unwrap_or(Vec3::ZERO);
    let mut loose: Vec<VertId> = Vec::new();
    // Each pass mends one spike and rebuilds two faces; the bound is only a
    // guard against an outline that will not settle.
    for _ in 0..faces.len() * 8 + 8 {
        let mut mended = false;
        'faces: for fi in 0..faces.len() {
            let f = faces[fi];
            if mesh.face(f).is_none() {
                continue;
            }
            let ring = mesh.face_verts(f);
            let n = ring.len();
            if n < 4 {
                continue;
            }
            for i in 0..n {
                let (pi, ni) = ((i + n - 1) % n, (i + 1) % n);
                let v = ring[i];
                let a = position(mesh, ring[pi]) - position(mesh, v);
                let b = position(mesh, ring[ni]) - position(mesh, v);
                if a.length() < 1e-7 || b.length() < 1e-7 || (a.length() - b.length()).abs() < 1e-7
                {
                    continue;
                }
                if a.normalize().dot(b.normalize()) < 0.9999 {
                    continue;
                }
                // Out along one edge and back along the other: the nearer end
                // is where the outline turns now; the farther, the edge's other
                // end, which the face beyond shares.
                let (near, far) = if a.length() < b.length() {
                    (ring[pi], ring[ni])
                } else {
                    (ring[ni], ring[pi])
                };
                let beyond = mesh.vert_faces(v).into_iter().find(|g| {
                    if *g == f {
                        return false;
                    }
                    let r = mesh.face_verts(*g);
                    (0..r.len()).any(|k| {
                        let (x, y) = (r[k], r[(k + 1) % r.len()]);
                        (x == v && y == far) || (x == far && y == v)
                    })
                });
                let Some(beyond) = beyond else {
                    continue;
                };
                let mut other = mesh.face_verts(beyond);
                let m = other.len();
                if other.contains(&near) {
                    // Both faces fold at `v` through the same two points —
                    // a corner the notch took whole: it leaves both.
                    let k = other.iter().position(|x| *x == v).expect("shares v");
                    let (x, y) = (other[(k + m - 1) % m], other[(k + 1) % m]);
                    if !((x == near && y == far) || (x == far && y == near)) || m < 4 {
                        continue;
                    }
                    other.remove(k);
                } else {
                    let at = (0..m).find(|k| {
                        let (x, y) = (other[*k], other[(k + 1) % m]);
                        (x == v && y == far) || (x == far && y == v)
                    });
                    let Some(at) = at else { continue };
                    other.insert(at + 1, near);
                }
                let mut mine = ring.clone();
                mine.remove(i);
                mesh.remove_face(f);
                mesh.remove_face(beyond);
                let f2 = mesh.add_face(&mine)?;
                let beyond2 = mesh.add_face(&other)?;
                faces[fi] = f2;
                // The face beyond is rebuilt too, under a new name; it may
                // have a spike of its own still.
                match faces.iter().position(|g| *g == beyond) {
                    Some(k) => faces[k] = beyond2,
                    None => faces.push(beyond2),
                }
                loose.push(v);
                mended = true;
                break 'faces;
            }
        }
        if !mended {
            break;
        }
    }
    if loose.is_empty() {
        return Ok(());
    }
    let wires: Vec<_> = mesh.edge_ids().filter(|e| mesh.is_wire_edge(*e)).collect();
    for e in wires {
        mesh.remove_edge(e);
    }
    for v in loose {
        if mesh.vert(v).is_some() && mesh.vert_faces(v).is_empty() {
            mesh.remove_vert(v);
        }
    }
    Ok(())
}

/// What an edge-and-vertex extrusion produced: the new vertices, which a
/// move then drags.
#[derive(Clone, Debug, Default)]
pub struct WireExtrusion {
    pub new_verts: Vec<VertId>,
    pub new_faces: Vec<FaceId>,
}

/// Extrudes the selected faces, then moves each new vertex out along the
/// normals of the faces around it — Blender's Extrude Faces Along Normals.
///
/// Every face moves `distance` along its *own* normal, not the region's: the
/// offset at a vertex is scaled so each face's plane lands exactly that far
/// out (even thickness), so a cube grows into a bigger cube rather than
/// rounding off at its corners. A closed solid, which has no region normal to
/// travel along, inflates.
pub fn extrude_selected_faces_along_normals(
    mesh: &mut BMesh,
    distance: f32,
) -> Result<Extrusion, BmError> {
    let faces = mesh.selected_faces();
    let out = extrude_faces(mesh, &faces, Vec3::ZERO)?;
    offset_along_normals(mesh, &out.new_faces, distance);
    Ok(out)
}

/// Moves the vertices of `faces` out along their faces' normals, each face's
/// plane by `distance`.
pub fn offset_along_normals(mesh: &mut BMesh, faces: &[FaceId], distance: f32) {
    // Read every normal before moving anything: a moved vertex changes the
    // normals of the faces around it.
    let normals: Vec<(FaceId, Vec3)> = faces
        .iter()
        .map(|f| (*f, mesh.face_normal(*f).normalize_or_zero()))
        .filter(|(_, n)| *n != Vec3::ZERO)
        .collect();
    let mut around: Vec<Vec<Vec3>> = vec![Vec::new(); mesh.slot_count(crate::Domain::Point)];
    let mut verts: Vec<VertId> = Vec::new();
    for (f, n) in &normals {
        for v in mesh.face_verts(*f) {
            let slot = &mut around[v.index() as usize];
            if slot.is_empty() {
                verts.push(v);
            }
            slot.push(*n);
        }
    }
    let moves: Vec<(VertId, Vec3)> = verts
        .into_iter()
        .filter_map(|v| {
            let ns = &around[v.index() as usize];
            let dir = ns.iter().copied().sum::<Vec3>().normalize_or_zero();
            if dir == Vec3::ZERO {
                return None;
            }
            // Even thickness: the faces meet `dir` at an angle, and moving a
            // unit along it lifts each face's plane by only the cosine. Clamped
            // so a near-fold does not throw the vertex off to infinity.
            let cosine = ns.iter().map(|n| n.dot(dir)).sum::<f32>() / ns.len() as f32;
            let p = mesh.position(v)?;
            Some((v, p + dir * (distance / cosine.max(0.25))))
        })
        .collect();
    for (v, p) in moves {
        mesh.set_position(v, p);
    }
}

/// Blender's `E` on edges and vertices: each selected edge pulls out a quad
/// behind it, each selected vertex on no selected edge an edge, and the new
/// copies become the selection — to be moved where the pointer takes them.
/// Nothing moves here; the copies sit on the originals.
pub fn extrude_edges_and_verts(mesh: &mut BMesh) -> Result<WireExtrusion, BmError> {
    let mut out = WireExtrusion::default();
    let edges = mesh.selected_edges();
    let mut verts: Vec<VertId> = mesh.selected_verts();
    for e in &edges {
        if let Some([a, b]) = mesh.edge_verts(*e) {
            for v in [a, b] {
                if !verts.contains(&v) {
                    verts.push(v);
                }
            }
        }
    }
    if verts.is_empty() {
        return Ok(out);
    }
    // Read before writing: each edge's winding in the face it borders, so the
    // new quad runs against it and faces the same way.
    let walls: Vec<(VertId, VertId)> = edges
        .iter()
        .filter_map(|e| {
            let [a, b] = mesh.edge_verts(*e)?;
            let along = mesh
                .edge_loops(*e)
                .first()
                .and_then(|l| mesh.loop_vert(*l))
                .map(|first| first == a)
                .unwrap_or(true);
            // A face walks the edge a→b; the wall walks it b→a.
            Some(if along { (b, a) } else { (a, b) })
        })
        .collect();
    let mut copy: Vec<(VertId, VertId)> = Vec::with_capacity(verts.len());
    for v in &verts {
        let Some(p) = mesh.position(*v) else { continue };
        let n = mesh.add_vert(p);
        copy.push((*v, n));
        out.new_verts.push(n);
    }
    let copy_of = |v: VertId| copy.iter().find(|(o, _)| *o == v).map(|(_, n)| *n);
    for (a, b) in walls {
        let (Some(na), Some(nb)) = (copy_of(a), copy_of(b)) else {
            continue;
        };
        out.new_faces.push(mesh.add_face(&[a, b, nb, na])?);
    }
    // A vertex on no selected edge: an edge out of it.
    for (v, n) in &copy {
        if mesh.edge_between(*v, *n).is_none()
            && !edges
                .iter()
                .any(|e| mesh.edge_verts(*e).is_some_and(|ends| ends.contains(v)))
        {
            mesh.add_edge(*v, *n)?;
        }
    }
    mesh.select_none();
    for n in &out.new_verts {
        mesh.set_vert_selected(*n, true);
    }
    mesh.flush_selection(SelectMode::Vert);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn cube() -> BMesh {
        BMesh::from_mesh_data(&primitives::cube(2.0))
    }

    #[test]
    fn along_normals_each_face_moves_its_own_way_by_the_same_distance() {
        // The top and the +X side of a cube, both selected: the top rises by
        // 0.5 and the side moves out by 0.5, so the shared edge goes up *and*
        // out — not along the region's averaged diagonal.
        let mut m = cube();
        for dir in [Vec3::Z, Vec3::X] {
            let f = face_facing(&m, dir);
            m.set_face_selected(f, true);
        }
        m.flush_selection(SelectMode::Face);
        let out = extrude_selected_faces_along_normals(&mut m, 0.5).unwrap();
        assert_eq!(out.new_faces.len(), 2);
        let max = out
            .new_verts
            .iter()
            .filter_map(|v| m.position(*v))
            .fold(Vec3::splat(f32::MIN), Vec3::max);
        assert!((max.z - 1.5).abs() < 1e-5, "{max:?}");
        assert!((max.x - 1.5).abs() < 1e-5, "{max:?}");
        for f in &out.new_faces {
            let n = m.face_normal(*f).normalize_or_zero();
            assert!(n.dot(Vec3::Z) > 0.999 || n.dot(Vec3::X) > 0.999, "{n:?}");
        }
    }

    #[test]
    fn along_normals_a_closed_cube_grows_into_a_bigger_cube() {
        // No region normal, but each face has its own: every face out by 1.
        let mut m = cube();
        let all: Vec<FaceId> = m.face_ids().collect();
        for f in all {
            m.set_face_selected(f, true);
        }
        m.flush_selection(SelectMode::Face);
        let out = extrude_selected_faces_along_normals(&mut m, 1.0).unwrap();
        for v in &out.new_verts {
            let p = m.position(*v).unwrap();
            for c in p.to_array() {
                assert!((c.abs() - 2.0).abs() < 1e-5, "{p:?}");
            }
        }
    }

    /// The face whose normal points most nearly along `dir`.
    fn face_facing(mesh: &BMesh, dir: Vec3) -> FaceId {
        mesh.face_ids()
            .max_by(|a, b| {
                let da = mesh.face_normal(*a).normalize_or_zero().dot(dir);
                let db = mesh.face_normal(*b).normalize_or_zero().dot(dir);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    }

    #[test]
    fn extruding_one_face_of_a_cube_makes_a_box_on_a_box() {
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        let out = extrude_faces(&mut m, &[top], Vec3::Z * 3.0).unwrap();
        m.assert_valid();

        assert_eq!(out.new_faces.len(), 1);
        assert_eq!(out.side_faces.len(), 4, "one wall per boundary edge");
        assert_eq!(out.new_verts.len(), 4);

        assert_eq!(
            m.num_faces(),
            5 + 1 + 4,
            "five original, the lid, four walls"
        );
        assert_eq!(m.num_verts(), 12);
        assert_eq!(m.num_edges(), 20);

        // Still a closed solid: every edge has exactly two faces.
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "the result must stay watertight");
        }
        assert_eq!(
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64,
            2,
            "Euler characteristic of a genus-0 closed surface"
        );

        let lifted = m.face_verts(out.new_faces[0]);
        for v in lifted {
            assert!((m.position(v).unwrap().z - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_walls_face_outward() {
        // The classic silent failure. A reversed side quad still draws, still
        // validates, and still exports — it just lights from the inside, and the
        // report that comes back is "the shading looks wrong".
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        let out = extrude_faces(&mut m, &[top], Vec3::Z * 2.0).unwrap();

        let centre = Vec3::new(0.0, 0.0, 2.0);
        for f in &out.side_faces {
            let n = m.face_normal(*f).normalize_or_zero();
            let verts = m.face_verts(*f);
            let mid: Vec3 =
                verts.iter().filter_map(|v| m.position(*v)).sum::<Vec3>() / verts.len() as f32;
            assert!(
                n.dot(mid - centre) > 0.0,
                "wall {f:?} points inward: normal {n:?} at {mid:?}"
            );
        }
        let lid = m.face_normal(out.new_faces[0]).normalize_or_zero();
        assert!(lid.dot(Vec3::Z) > 0.99, "the lid kept the original winding");
    }

    #[test]
    fn a_region_grows_walls_only_around_its_outline() {
        // The region-versus-per-face distinction, stated as a count. Two adjacent
        // faces share an edge; a per-face extrude would put a wall on it and
        // leave a membrane inside the solid.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        assert_eq!(m.num_faces(), 4);
        let all: Vec<FaceId> = m.face_ids().collect();

        let out = extrude_faces(&mut m, &all, Vec3::Z).unwrap();
        m.assert_valid();
        assert_eq!(out.new_faces.len(), 4);
        assert_eq!(
            out.side_faces.len(),
            8,
            "the grid's outline is eight edges; the four interior edges get no wall"
        );
        assert_eq!(m.num_faces(), 12, "no leftover faces from the old region");
    }

    #[test]
    fn the_interior_of_an_extruded_region_leaves_nothing_behind() {
        // A 3x3 grid has one vertex used only by the region. After the region
        // lifts away, that vertex belongs to nothing.
        let mut m = BMesh::from_mesh_data(&primitives::grid(6.0, 3, 3));
        let before = m.num_verts();
        let all: Vec<FaceId> = m.face_ids().collect();
        extrude_faces(&mut m, &all, Vec3::Z).unwrap();
        m.assert_valid();

        // 16 vertices, of which the 4 interior ones are used only by the region.
        assert_eq!(m.num_verts(), before + 16 - 4);
        for v in m.vert_ids() {
            assert!(
                !m.vert_faces(v).is_empty(),
                "{v:?} was left with no faces at all"
            );
        }
    }

    #[test]
    fn a_zero_offset_extrude_changes_topology_and_nothing_else() {
        // What an interactive extrude does on mouse-down: build the shell, then
        // let the modal transform move it. If this moved anything, the first
        // frame of every extrude would jump.
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        let before: Vec<Vec3> = m.vert_ids().filter_map(|v| m.position(v)).collect();

        let out = extrude_faces(&mut m, &[top], Vec3::ZERO).unwrap();
        m.assert_valid();

        for v in &out.new_verts {
            let p = m.position(*v).unwrap();
            assert!(
                before.iter().any(|q| (*q - p).length() < 1e-6),
                "the copy sits exactly on its original"
            );
        }
        assert_eq!(out.side_faces.len(), 4);
    }

    #[test]
    fn the_new_shell_is_what_ends_up_selected() {
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        m.set_face_selected(top, true);
        m.flush_selection(SelectMode::Face);

        let out = extrude_selected_faces(&mut m, 2.0).unwrap();
        assert_eq!(m.selected_faces(), out.new_faces);
        assert_eq!(m.selected_verts().len(), 4);
        assert!(
            !m.selected_faces().contains(&top),
            "the original is gone, not merely deselected"
        );
    }

    #[test]
    fn extruding_a_boundary_face_of_an_open_surface_works() {
        // A plane has no second face on any edge, so every edge is boundary and
        // the extrusion is an open box with no lid on the bottom.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let out = extrude_faces(&mut m, &[f], Vec3::Z * 2.0).unwrap();
        m.assert_valid();

        assert_eq!(out.side_faces.len(), 4);
        assert_eq!(m.num_faces(), 5);
        assert_eq!(m.num_verts(), 8, "the original four are held by the walls");
    }

    #[test]
    fn extruding_nothing_is_a_no_op_rather_than_an_error() {
        let mut m = cube();
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        let out = extrude_faces(&mut m, &[], Vec3::Z).unwrap();
        assert!(out.new_faces.is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn repeated_extrusions_stack() {
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        m.set_face_selected(top, true);
        m.flush_selection(SelectMode::Face);

        for _ in 0..8 {
            extrude_selected_faces(&mut m, 1.0).unwrap();
            m.assert_valid();
        }
        assert_eq!(m.num_faces(), 5 + 1 + 8 * 4);
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "still watertight after eight");
        }
        let top_z = m
            .selected_verts()
            .iter()
            .filter_map(|v| m.position(*v))
            .map(|p| p.z)
            .fold(f32::MIN, f32::max);
        assert!(
            (top_z - 9.0).abs() < 1e-4,
            "eight metres above the cube's lid"
        );
    }

    #[test]
    fn extruding_every_face_of_a_solid_grows_a_shell() {
        // The whole surface is the region, so there is no boundary at all and no
        // walls: the result is the original solid with a second, detached shell
        // outside it. Unusual, but it must not corrupt anything.
        let mut m = cube();
        let all: Vec<FaceId> = m.face_ids().collect();
        let out = extrude_faces(&mut m, &all, Vec3::ZERO).unwrap();
        m.assert_valid();

        assert!(out.side_faces.is_empty(), "a closed region has no outline");
        assert_eq!(out.new_faces.len(), 6);
        assert_eq!(m.num_faces(), 6, "the original faces were replaced");
        assert_eq!(m.num_verts(), 8, "and their vertices reclaimed");
    }

    #[test]
    fn e_on_an_open_edge_pulls_out_a_quad_that_faces_with_its_neighbour() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let e = m.edge_ids().next().unwrap();
        m.select_none();
        m.set_edge_selected(e, true);
        m.flush_selection(SelectMode::Edge);
        let out = extrude_edges_and_verts(&mut m).unwrap();
        m.assert_valid();
        assert_eq!(out.new_verts.len(), 2);
        assert_eq!(m.num_faces(), 2);
        // The two faces walk the shared edge in opposite directions: the wall
        // is wound as its neighbour is, not flipped.
        let loops = m.edge_loops(e);
        assert_eq!(loops.len(), 2);
        assert_ne!(m.loop_vert(loops[0]), m.loop_vert(loops[1]));
        assert_eq!(m.selected_verts().len(), 2);
    }

    #[test]
    fn e_on_a_lone_vertex_makes_an_edge_out_of_it() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let v = m.vert_ids().next().unwrap();
        m.select_none();
        m.set_vert_selected(v, true);
        let edges_before = m.num_edges();
        let out = extrude_edges_and_verts(&mut m).unwrap();
        m.assert_valid();
        assert_eq!(out.new_verts.len(), 1);
        assert_eq!(m.num_edges(), edges_before + 1);
        assert_eq!(m.num_faces(), 1);
    }
}

#[cfg(test)]
mod push_tests {
    use super::*;
    use crate::measure::{is_closed, mass_properties};
    use crate::primitives;

    fn top(m: &BMesh) -> FaceId {
        m.face_ids()
            .find(|f| m.face_normal(*f).normalize_or_zero().dot(Vec3::Z) > 0.999)
            .unwrap()
    }

    /// Every edge of every face along an axis — nothing pulled askew, as a
    /// corner that slid alone pulls its top quarter — and no outline that
    /// runs out and straight back.
    fn square(m: &BMesh) {
        for g in m.face_ids() {
            let ring: Vec<Vec3> = m
                .face_verts(g)
                .iter()
                .filter_map(|v| m.position(*v))
                .collect();
            for i in 0..ring.len() {
                let d = ring[(i + 1) % ring.len()] - ring[i];
                let along = [d.x, d.y, d.z].iter().filter(|c| c.abs() > 1e-5).count();
                assert!(along <= 1, "a slanted edge in {ring:?}");
                let next = ring[(i + 2) % ring.len()] - ring[(i + 1) % ring.len()];
                assert!(
                    d.normalize_or_zero().dot(next.normalize_or_zero()) > -0.999,
                    "a spike in {ring:?}"
                );
            }
        }
    }

    fn volume(m: &BMesh) -> f64 {
        let data = m.to_mesh_data();
        assert!(is_closed(&data), "the solid is not closed");
        mass_properties(&data, |p| p).volume
    }

    #[test]
    fn a_boxs_top_pushed_in_makes_a_shorter_box() {
        // Fusion's Press Pull on the top of a box: the box loses the slab the
        // face swept through. A plain extrude stood four walls on the box's
        // own sides and took nothing away.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = top(&m);
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        let out = extrude_selected_faces(&mut m, -0.5).unwrap();
        assert_eq!(m.num_faces(), 6, "still a box");
        assert_eq!(m.num_verts(), 8);
        assert!(
            (volume(&m) - 2.0 * 2.0 * 1.5).abs() < 1e-4,
            "{}",
            volume(&m)
        );
        square(&m);
        assert_eq!(out.new_faces.len(), 1, "the top stays selected");
        let n = m.face_normal(out.new_faces[0]).normalize_or_zero();
        assert!(n.dot(Vec3::Z) > 0.999);
    }

    #[test]
    fn an_inset_face_pushed_in_makes_a_pocket() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = top(&m);
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        crate::edit::inset_selected(&mut m, 0.25, 0.0).unwrap();
        extrude_selected_faces(&mut m, -0.5).unwrap();
        // 8, less a 1.5 × 1.5 × 0.5 pocket.
        assert!(
            (volume(&m) - (8.0 - 1.5 * 1.5 * 0.5)).abs() < 1e-4,
            "{}",
            volume(&m)
        );
    }

    /// A box cut into strips by two loop cuts on its front; `strip` picks
    /// which strip to push: 0 at the left edge, 1 in the middle.
    fn strips() -> BMesh {
        // A 3 x 1 x 2 block whose front (-Y) is three faces side by side,
        // and whose top is three faces to match.
        let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let verts = vec![
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(2.0, 0.0, 0.0),
            p(3.0, 0.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(2.0, 1.0, 0.0),
            p(3.0, 1.0, 0.0),
            p(0.0, 0.0, 2.0),
            p(1.0, 0.0, 2.0),
            p(2.0, 0.0, 2.0),
            p(3.0, 0.0, 2.0),
            p(0.0, 1.0, 2.0),
            p(1.0, 1.0, 2.0),
            p(2.0, 1.0, 2.0),
            p(3.0, 1.0, 2.0),
        ];
        let faces: Vec<Vec<u32>> = vec![
            // front, three strips, facing -Y
            vec![0, 1, 9, 8],
            vec![1, 2, 10, 9],
            vec![2, 3, 11, 10],
            // top, three, facing +Z
            vec![8, 9, 13, 12],
            vec![9, 10, 14, 13],
            vec![10, 11, 15, 14],
            // bottom, three, facing -Z
            vec![0, 4, 5, 1],
            vec![1, 5, 6, 2],
            vec![2, 6, 7, 3],
            // back, three, facing +Y
            vec![4, 12, 13, 5],
            vec![5, 13, 14, 6],
            vec![6, 14, 15, 7],
            // left and right ends
            vec![0, 8, 12, 4],
            vec![3, 7, 15, 11],
        ];
        BMesh::from_mesh_data(&crate::MeshData::from_faces(verts, &faces))
    }

    fn front_strip(m: &BMesh, x: f32) -> FaceId {
        m.face_ids()
            .find(|f| {
                let n = m.face_normal(*f).normalize_or_zero();
                let c: Vec3 = m
                    .face_verts(*f)
                    .iter()
                    .filter_map(|v| m.position(*v))
                    .sum::<Vec3>()
                    / 4.0;
                n.dot(-Vec3::Y) > 0.999 && (c.x - x).abs() < 1e-4
            })
            .unwrap()
    }

    #[test]
    fn a_strip_at_the_end_of_a_box_pushed_in_notches_it_with_no_triangles_left() {
        // The picture a user sent: the end strip of a front pushed in left a
        // triangle on the top, in the top's plane, beside the top.
        let mut m = strips();
        let f = front_strip(&m, 2.5);
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        extrude_selected_faces(&mut m, -0.4).unwrap();
        assert!(
            (volume(&m) - (6.0 - 0.4 * 2.0)).abs() < 1e-4,
            "{}",
            volume(&m)
        );
        square(&m);
        for g in m.face_ids() {
            assert!(
                m.face_verts(g).len() > 3,
                "a triangle is left: {:?}",
                m.face_verts(g)
            );
        }
    }

    #[test]
    fn a_strip_in_the_middle_pushed_in_notches_top_and_bottom_cleanly() {
        let mut m = strips();
        let f = front_strip(&m, 1.5);
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        extrude_selected_faces(&mut m, -0.4).unwrap();
        assert!(
            (volume(&m) - (6.0 - 0.4 * 2.0)).abs() < 1e-4,
            "{}",
            volume(&m)
        );
        square(&m);
        // No face lies on another in the top's plane: the top pieces and
        // the walls over the notch are one face per piece.
        let tops = m
            .face_ids()
            .filter(|g| m.face_normal(*g).normalize_or_zero().dot(Vec3::Z) > 0.999)
            .count();
        assert_eq!(tops, 3, "the top's three pieces, the middle one notched");
    }

    #[test]
    fn a_quarter_of_a_face_at_a_corner_pushed_in_leaves_its_neighbours_square() {
        // The second picture: a cube cut in two each way, the front's upper
        // quarter by a side pushed in. Its corner on the cube's edge slid on
        // its own and pulled the top and the side askew.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let all: Vec<FaceId> = m.face_ids().collect();
        crate::edit::subdivide::subdivide_faces(&mut m, &all).unwrap();
        m.select_none();
        // The front (-Y) quarter at the top and the -X side.
        let f = m
            .face_ids()
            .find(|f| {
                let n = m.face_normal(*f).normalize_or_zero();
                let c: Vec3 = m
                    .face_verts(*f)
                    .iter()
                    .filter_map(|v| m.position(*v))
                    .sum::<Vec3>()
                    / 4.0;
                n.dot(-Vec3::Y) > 0.999 && c.x < 0.0 && c.z > 0.0
            })
            .unwrap();
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        extrude_selected_faces(&mut m, -0.5).unwrap();
        assert!((volume(&m) - (8.0 - 0.5)).abs() < 1e-4, "{}", volume(&m));
        square(&m);
    }

    #[test]
    fn outward_is_the_plain_extrude_still() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = top(&m);
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        extrude_selected_faces(&mut m, 0.5).unwrap();
        assert_eq!(m.num_faces(), 10, "walls on every side");
        assert!((volume(&m) - 2.0 * 2.0 * 2.5).abs() < 1e-4);
    }
}
