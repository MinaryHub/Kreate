//! Joining several meshes into one, and separating a selection out of one.
//!
//! Both are document-level operations rather than topology edits: nothing is
//! cut or welded, geometry is only moved between meshes. What makes them
//! interesting is the *attributes*, and specifically that two meshes rarely
//! carry the same ones.
//!
//! # The rules for merging layers
//!
//! - A layer present in any part is present in the result. Parts that lack it
//!   contribute the type's default rather than being dropped, because dropping
//!   a UV set because one of five objects had none is a silent loss of work.
//! - A layer whose *type* or *domain* disagrees between parts cannot be merged.
//!   It is reported rather than quietly discarded — §5.6's rule that an
//!   importer never silently drops data applies just as much here, where the
//!   data is the user's own.
//! - `material_index` is offset per part, so the face-to-material *partition*
//!   survives. Two objects both using index zero mean different things by it,
//!   and there is no object-level slot list yet to say what; keeping the
//!   partition preserves the information that is actually present, and merging
//!   the groups would invent an equality nobody asked for.
//! - A custom split normal is a direction, not a position: it is transformed by
//!   the inverse-transpose, or a non-uniformly scaled join comes out shaded
//!   wrong in a way that looks like a lighting bug.

use glam::{DAffine3, Mat3, Vec3};

use crate::arena::ArenaIndex;
use crate::attribute::{names, AttrData, AttrType, Domain, Layer};
use crate::bmesh::BMesh;
use crate::MeshData;

/// One mesh being joined, with the transform that brings it into the result's
/// space.
pub struct Part<'a> {
    pub mesh: &'a MeshData,
    /// The transform from this mesh's space into the result's.
    ///
    /// For a join this is `target.world.inverse() * source.world`: the point
    /// of a join is that the geometry does not move on screen, and it only
    /// stays put if each part is brought through the target's own transform.
    pub transform: DAffine3,
}

/// What could not be merged.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Layers whose type or domain disagreed between parts.
    ///
    /// Named rather than counted, because the user's next question is *which*
    /// one — and because a count is the kind of message that gets ignored.
    pub conflicted: Vec<String>,
}

/// Joins several meshes into one.
///
/// The parts are concatenated in order, so the first part's elements keep
/// their indices — which is what lets a caller treat the first part as the
/// target and the rest as arrivals.
pub fn join(parts: &[Part<'_>]) -> (MeshData, Report) {
    let mut report = Report::default();
    if parts.is_empty() {
        return (MeshData::empty(), report);
    }

    // --- topology -----------------------------------------------------------
    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    // Where each part's vertices and faces begin in the result, so the
    // attribute pass can place values without recomputing it.
    let mut vert_base: Vec<usize> = Vec::with_capacity(parts.len());
    let mut face_base: Vec<usize> = Vec::with_capacity(parts.len());
    let mut corner_base: Vec<usize> = Vec::with_capacity(parts.len());
    let mut corners = 0usize;

    for part in parts {
        vert_base.push(positions.len());
        face_base.push(faces.len());
        corner_base.push(corners);

        let offset = positions.len() as u32;
        for p in part.mesh.positions() {
            let world = part
                .transform
                .transform_point3(glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64));
            positions.push(Vec3::new(world.x as f32, world.y as f32, world.z as f32));
        }
        for f in 0..part.mesh.num_faces() {
            let ring: Vec<u32> = part.mesh.face_verts(f).iter().map(|v| v + offset).collect();
            corners += ring.len();
            faces.push(ring);
        }
    }

    let mut out = MeshData::from_faces(positions, &faces);

    // --- attributes ---------------------------------------------------------
    let mut names_seen: Vec<&str> = Vec::new();
    for part in parts {
        for layer in part.mesh.attributes().iter() {
            if !names_seen.contains(&&*layer.name) {
                names_seen.push(&layer.name);
            }
        }
    }
    names_seen.sort_unstable();

    for name in names_seen {
        if name == names::POSITION {
            // Already placed, and transformed while it was.
            continue;
        }
        let Some(spec) = agreed_spec(parts, name, &mut report) else {
            continue;
        };
        let (domain, ty) = spec;
        let len = domain_len(&out, domain);
        let mut data = AttrData::zeroed(ty, len);
        let mut material_offset = 0i32;

        for (index, part) in parts.iter().enumerate() {
            let base = match domain {
                Domain::Point => vert_base[index],
                Domain::Face => face_base[index],
                Domain::Corner => corner_base[index],
                // Edges are derived by `from_faces`, and their order is not the
                // concatenation of the parts' edge orders — so an edge layer
                // cannot be placed by an offset. Reported rather than written
                // wrong: a seam or a crease landing on the wrong edge is worse
                // than one that is missing and said so.
                Domain::Edge => {
                    if !report.conflicted.iter().any(|n| n == name) {
                        report.conflicted.push(name.to_string());
                    }
                    break;
                }
            };
            let source = part.mesh.attributes().get(name);
            let count = part_len(part.mesh, domain);
            match source {
                Some(layer) => {
                    place(&mut data, base, &layer.data, count, material_offset, part);
                    if name == names::MATERIAL_INDEX {
                        material_offset += highest_material(&layer.data) + 1;
                    }
                }
                None => {
                    // Left at the type's default. Deliberate: a mesh with no
                    // UVs joined to one with UVs keeps the UVs and reads as
                    // unwrapped-at-the-origin, which is visible and fixable.
                    // Dropping the layer would lose the other mesh's work.
                }
            }
        }

        if report.conflicted.iter().any(|n| n == name) {
            continue;
        }
        out.attributes_mut().insert(name, domain, data);
    }

    (out, report)
}

/// The domain and type every part agrees on for `name`, if they do.
fn agreed_spec(parts: &[Part<'_>], name: &str, report: &mut Report) -> Option<(Domain, AttrType)> {
    let mut spec: Option<(Domain, AttrType)> = None;
    for part in parts {
        let Some(layer) = part.mesh.attributes().get(name) else {
            continue;
        };
        let mine = (layer.domain, layer.data.ty());
        match spec {
            None => spec = Some(mine),
            Some(theirs) if theirs == mine => {}
            Some(_) => {
                if !report.conflicted.iter().any(|n| n == name) {
                    report.conflicted.push(name.to_string());
                }
                return None;
            }
        }
    }
    spec
}

fn domain_len(mesh: &MeshData, domain: Domain) -> usize {
    match domain {
        Domain::Point => mesh.num_verts() as usize,
        Domain::Edge => mesh.num_edges() as usize,
        Domain::Corner => mesh.num_corners() as usize,
        Domain::Face => mesh.num_faces() as usize,
    }
}

fn part_len(mesh: &MeshData, domain: Domain) -> usize {
    domain_len(mesh, domain)
}

/// The largest material index in a layer, or `-1` when there is none.
fn highest_material(data: &AttrData) -> i32 {
    match data {
        AttrData::I32(v) => v.iter().copied().max().unwrap_or(-1),
        _ => -1,
    }
}

/// Copies one part's values into the joined layer at `base`.
fn place(
    dst: &mut AttrData,
    base: usize,
    src: &AttrData,
    count: usize,
    material_offset: i32,
    part: &Part<'_>,
) {
    macro_rules! copy {
        ($variant:ident) => {
            if let (AttrData::$variant(d), AttrData::$variant(s)) = (&mut *dst, src) {
                let target = std::sync::Arc::make_mut(d);
                for i in 0..count.min(s.len()) {
                    if let Some(slot) = target.get_mut(base + i) {
                        *slot = s[i];
                    }
                }
            }
        };
    }

    match src {
        AttrData::F32(_) => copy!(F32),
        AttrData::F32x2(_) => copy!(F32x2),
        AttrData::F32x4(_) => copy!(F32x4),
        AttrData::Bool(_) => copy!(Bool),
        AttrData::U8x4(_) => copy!(U8x4),
        AttrData::I32(s) => {
            if let AttrData::I32(d) = dst {
                let target = std::sync::Arc::make_mut(d);
                for i in 0..count.min(s.len()) {
                    if let Some(slot) = target.get_mut(base + i) {
                        *slot = s[i] + material_offset;
                    }
                }
            }
        }
        AttrData::F32x3(s) => {
            if let AttrData::F32x3(d) = dst {
                // A custom split normal is a direction: the inverse-transpose,
                // not the transform. Under a non-uniform scale the two differ,
                // and using the wrong one comes out as shading that looks like
                // a lighting bug rather than like bad data.
                let normal_matrix = normal_matrix(&part.transform);
                let target = std::sync::Arc::make_mut(d);
                for i in 0..count.min(s.len()) {
                    if let Some(slot) = target.get_mut(base + i) {
                        *slot = (normal_matrix * s[i]).normalize_or_zero();
                    }
                }
            }
        }
    }
}

/// The inverse-transpose of a transform's linear part.
fn normal_matrix(transform: &DAffine3) -> Mat3 {
    let m = transform.matrix3;
    let m = Mat3::from_cols(
        Vec3::new(m.x_axis.x as f32, m.x_axis.y as f32, m.x_axis.z as f32),
        Vec3::new(m.y_axis.x as f32, m.y_axis.y as f32, m.y_axis.z as f32),
        Vec3::new(m.z_axis.x as f32, m.z_axis.y as f32, m.z_axis.z as f32),
    );
    if m.determinant().abs() < 1e-12 {
        // A degenerate transform has no inverse; the identity leaves normals
        // where they were, which is wrong but bounded, and NaN is not.
        Mat3::IDENTITY
    } else {
        m.inverse().transpose()
    }
}

/// Extracts the selected faces into a new mesh and removes them from `mesh`.
///
/// `None` when nothing is selected, or when *everything* is: separating a whole
/// mesh leaves an empty object behind, which is a worse outcome than declining
/// — the user meant to split something.
pub fn separate_selected(mesh: &mut BMesh) -> Option<MeshData> {
    let selected = mesh.selected_faces();
    if selected.is_empty() || selected.len() == mesh.num_faces() {
        return None;
    }

    // Through `MeshData` rather than by building a `BMesh` by hand: the
    // conversion already knows how to carry every attribute domain across, and
    // duplicating that here would be a second place for it to go wrong.
    let whole = mesh.to_mesh_data();
    let keep: Vec<u32> = selected.iter().map(|f| f.index()).collect();
    let taken = subset(&whole, &keep, mesh);

    for f in &selected {
        mesh.remove_face(*f);
    }
    Some(taken)
}

/// A new mesh holding only the named faces of `whole`.
///
/// `faces` are *arena slots*, which is what a `BMesh` face id is; `whole` is
/// the dense conversion of that same mesh, so the two are related by the
/// conversion's own face order rather than by identity. The bmesh is passed in
/// to recover that order.
fn subset(whole: &MeshData, slots: &[u32], mesh: &BMesh) -> MeshData {
    // The conversion emits faces in `face_ids` order, so a slot's dense index
    // is its position in that iteration. Built once rather than searched per
    // face: a linear scan per face is quadratic, and separate is reached on
    // meshes with a hundred thousand faces.
    let mut dense_of = vec![usize::MAX; mesh.slot_count(Domain::Face)];
    for (dense, f) in mesh.face_ids().enumerate() {
        if let Some(slot) = dense_of.get_mut(f.index() as usize) {
            *slot = dense;
        }
    }

    let wanted: Vec<usize> = slots
        .iter()
        .filter_map(|s| dense_of.get(*s as usize).copied())
        .filter(|d| *d != usize::MAX)
        .collect();

    // Vertices are taken in the order the kept faces reference them, and
    // renumbered. A mesh that carried every original vertex would drag the
    // whole model's point data into the new object.
    let mut remap = vec![u32::MAX; whole.num_verts() as usize];
    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut corner_of: Vec<usize> = Vec::new();

    for dense in &wanted {
        let ring = whole.face_verts(*dense as u32);
        let start = whole.face_offsets()[*dense] as usize;
        let mut out_ring = Vec::with_capacity(ring.len());
        for (i, v) in ring.iter().enumerate() {
            let slot = &mut remap[*v as usize];
            if *slot == u32::MAX {
                *slot = positions.len() as u32;
                positions.push(whole.positions()[*v as usize]);
            }
            out_ring.push(*slot);
            corner_of.push(start + i);
        }
        faces.push(out_ring);
    }

    let mut out = MeshData::from_faces(positions, &faces);

    // Attributes, gathered by the index lists just built. Edge layers are
    // dropped for the same reason a join cannot merge them: the new mesh's
    // edges are derived by `from_faces` and are not a subset of the old ones
    // in any order an index list can express.
    let vert_of: Vec<u32> = {
        let mut v = vec![0u32; positions_len(&out)];
        for (old, new) in remap.iter().enumerate() {
            if *new != u32::MAX {
                v[*new as usize] = old as u32;
            }
        }
        v
    };

    let layers: Vec<Layer> = whole.attributes().iter().cloned().collect();
    for layer in layers {
        if &*layer.name == names::POSITION {
            continue;
        }
        let indices: Vec<u32> = match layer.domain {
            Domain::Point => vert_of.clone(),
            Domain::Face => wanted.iter().map(|d| *d as u32).collect(),
            Domain::Corner => corner_of.iter().map(|c| *c as u32).collect(),
            Domain::Edge => continue,
        };
        let gathered = layer.data.gather(&indices);
        out.attributes_mut()
            .insert(&layer.name, layer.domain, gathered);
    }

    out
}

fn positions_len(mesh: &MeshData) -> usize {
    mesh.num_verts() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::FaceId;
    use crate::primitives;
    use glam::Vec2;

    fn part<'a>(mesh: &'a MeshData, transform: DAffine3) -> Part<'a> {
        Part { mesh, transform }
    }

    #[test]
    fn joining_two_meshes_concatenates_their_geometry() {
        let a = primitives::cube(2.0);
        let b = primitives::cube(2.0);
        let shift = DAffine3::from_translation(glam::DVec3::new(5.0, 0.0, 0.0));
        let (out, report) = join(&[part(&a, DAffine3::IDENTITY), part(&b, shift)]);

        assert_eq!(report, Report::default());
        assert_eq!(out.num_verts(), a.num_verts() + b.num_verts());
        assert_eq!(out.num_faces(), a.num_faces() + b.num_faces());
    }

    #[test]
    fn each_part_arrives_through_its_own_transform() {
        // The point of a join: the geometry does not move on screen. It only
        // stays put if each part is brought through the target's transform,
        // which is what the caller passes.
        let a = primitives::cube(2.0);
        let b = primitives::cube(2.0);
        let shift = DAffine3::from_translation(glam::DVec3::new(5.0, 0.0, 0.0));
        let (out, _) = join(&[part(&a, DAffine3::IDENTITY), part(&b, shift)]);

        let far = out.positions().iter().filter(|p| p.x > 3.0).count();
        assert_eq!(far, b.num_verts() as usize, "the second part did not move");
    }

    #[test]
    fn the_first_parts_indices_are_left_alone() {
        // What lets a caller treat the first part as the target: joining into
        // an object must not renumber the geometry that was already there, or
        // a selection or a modifier index captured before the join means
        // something else after it.
        let a = primitives::cube(2.0);
        let b = primitives::uv_sphere(1.0, 8, 4);
        let (out, _) = join(&[part(&a, DAffine3::IDENTITY), part(&b, DAffine3::IDENTITY)]);
        for (i, p) in a.positions().iter().enumerate() {
            assert_eq!(out.positions()[i], *p, "vertex {i} moved");
        }
        for f in 0..a.num_faces() {
            assert_eq!(out.face_verts(f), a.face_verts(f), "face {f} changed");
        }
    }

    #[test]
    fn a_layer_only_one_part_has_survives_the_join() {
        // Dropping a UV set because one of five objects had none is a silent
        // loss of the other four objects' work.
        let mut a = primitives::cube(2.0);
        {
            let corners = a.num_corners() as usize;
            a.attributes_mut()
                .ensure(names::UV, Domain::Corner, AttrType::F32x2, corners);
            if let Some(v) = a.attributes_mut().make_mut_f32x2(names::UV) {
                for (i, uv) in v.iter_mut().enumerate() {
                    *uv = Vec2::new(i as f32, 0.0);
                }
            }
        }
        let b = primitives::cube(2.0);
        let (out, report) = join(&[part(&a, DAffine3::IDENTITY), part(&b, DAffine3::IDENTITY)]);

        assert_eq!(report, Report::default());
        let uvs = out.attributes().get_f32x2(names::UV).expect("kept the set");
        assert_eq!(uvs.len(), out.num_corners() as usize);
        // The first part's values are where they were; the second's are the
        // type's default, which reads as unwrapped at the origin — visible and
        // fixable, unlike a layer that vanished.
        for (i, uv) in uvs.iter().take(a.num_corners() as usize).enumerate() {
            assert_eq!(*uv, Vec2::new(i as f32, 0.0));
        }
        assert_eq!(uvs[a.num_corners() as usize], Vec2::ZERO);
    }

    #[test]
    fn a_layer_whose_type_disagrees_is_reported_and_not_guessed_at() {
        // §5.6's rule that data is never silently dropped applies here too,
        // where the data is the user's own.
        let mut a = primitives::cube(2.0);
        let mut b = primitives::cube(2.0);
        let corners_a = a.num_corners() as usize;
        let corners_b = b.num_corners() as usize;
        a.attributes_mut()
            .ensure("tint", Domain::Corner, AttrType::F32x2, corners_a);
        b.attributes_mut()
            .ensure("tint", Domain::Corner, AttrType::F32x4, corners_b);

        let (out, report) = join(&[part(&a, DAffine3::IDENTITY), part(&b, DAffine3::IDENTITY)]);
        assert_eq!(report.conflicted, vec!["tint".to_string()]);
        assert!(
            out.attributes().get("tint").is_none(),
            "a conflicted layer was written anyway"
        );
    }

    #[test]
    fn material_indices_are_offset_so_the_partition_survives() {
        // Two objects both using index zero mean different things by it, and
        // there is no slot list yet to say what. Keeping the partition
        // preserves the information that is present; merging the groups would
        // invent an equality nobody asked for.
        let mut a = primitives::cube(2.0);
        let mut b = primitives::cube(2.0);
        for (mesh, values) in [(&mut a, [0, 0, 1, 1, 1, 1]), (&mut b, [0, 1, 2, 2, 2, 2])] {
            let faces = mesh.num_faces() as usize;
            mesh.attributes_mut()
                .ensure(names::MATERIAL_INDEX, Domain::Face, AttrType::I32, faces);
            if let Some(v) = mesh.attributes_mut().make_mut_i32(names::MATERIAL_INDEX) {
                for (slot, want) in v.iter_mut().zip(values) {
                    *slot = want;
                }
            }
        }

        let (out, _) = join(&[part(&a, DAffine3::IDENTITY), part(&b, DAffine3::IDENTITY)]);
        let got = out.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
        assert_eq!(&got[..6], &[0, 0, 1, 1, 1, 1]);
        // The second part's highest was 2, the first's was 1, so it starts at 2.
        assert_eq!(&got[6..], &[2, 3, 4, 4, 4, 4]);
    }

    #[test]
    fn a_custom_normal_is_transformed_as_a_direction() {
        // The inverse-transpose, not the transform. Under a non-uniform scale
        // the two differ, and the wrong one comes out as shading that looks
        // like a lighting bug rather than like bad data.
        //
        // The normal has to be *off* the axes for this to say anything. With a
        // normal along X and a diagonal scale, both matrices give a multiple of
        // X and renormalising erases the difference — the first version of this
        // test used exactly that and passed against the wrong matrix.
        let diagonal = Vec3::new(1.0, 1.0, 0.0).normalize();
        let mut b = primitives::cube(2.0);
        let corners = b.num_corners() as usize;
        b.attributes_mut().ensure(
            names::CUSTOM_NORMAL,
            Domain::Corner,
            AttrType::F32x3,
            corners,
        );
        if let Some(v) = b.attributes_mut().make_mut_f32x3(names::CUSTOM_NORMAL) {
            for n in v.iter_mut() {
                *n = diagonal;
            }
        }

        // Squashed hard along X. The two candidate matrices send the diagonal
        // in nearly opposite directions, so the assertion below can fail.
        let squash = DAffine3::from_scale(glam::DVec3::new(0.1, 1.0, 1.0));
        let a = primitives::cube(2.0);
        let (out, _) = join(&[part(&a, DAffine3::IDENTITY), part(&b, squash)]);

        // What each would give, derived here rather than copied from a run.
        let wrong = (Mat3::from_diagonal(Vec3::new(0.1, 1.0, 1.0)) * diagonal).normalize();
        let right = (Mat3::from_diagonal(Vec3::new(10.0, 1.0, 1.0)) * diagonal).normalize();
        assert!(
            (wrong - right).length() > 0.5,
            "the two matrices agree here, so this proves nothing:              {wrong:?} vs {right:?}"
        );

        let normals = out.attributes().get_f32x3(names::CUSTOM_NORMAL).unwrap();
        let n = normals[a.num_corners() as usize];
        assert!(
            (n - right).length() < 1e-4,
            "the normal was transformed by the matrix rather than its              inverse-transpose: {n:?}, expected {right:?}"
        );
        assert!(
            (n.length() - 1.0).abs() < 1e-4,
            "the normal was not renormalised: {}",
            n.length()
        );
    }

    #[test]
    fn an_edge_layer_is_reported_rather_than_placed_wrong() {
        // The joined mesh's edges are derived from its faces, and their order
        // is not the concatenation of the parts' edge orders — so an edge
        // layer cannot be placed by an offset. A crease landing on the wrong
        // edge is worse than one that is missing and said so.
        let mut a = primitives::cube(2.0);
        let edges = a.num_edges() as usize;
        a.attributes_mut()
            .ensure(names::CREASE, Domain::Edge, AttrType::F32, edges);
        let b = primitives::cube(2.0);

        let (out, report) = join(&[part(&a, DAffine3::IDENTITY), part(&b, DAffine3::IDENTITY)]);
        assert_eq!(report.conflicted, vec![names::CREASE.to_string()]);
        assert!(out.attributes().get(names::CREASE).is_none());
    }

    #[test]
    fn joining_nothing_gives_an_empty_mesh() {
        let (out, report) = join(&[]);
        assert_eq!(out.num_verts(), 0);
        assert_eq!(report, Report::default());
    }

    #[test]
    fn separating_a_selection_moves_it_out() {
        // A grid rather than a cube. Two faces of a cube can be *opposite*,
        // using all eight of its vertices between them — so the vertex count
        // could not tell "only the vertices these faces use" from "all of
        // them", which is the claim. A 4x4 grid has twenty-five vertices and
        // one quad uses four.
        let mesh = primitives::grid(4.0, 4, 4);
        let mut bm = BMesh::from_mesh_data(&mesh);
        assert_eq!(bm.num_faces(), 16);

        let chosen: Vec<FaceId> = bm.face_ids().take(1).collect();
        bm.set_face_selected(chosen[0], true);

        let taken = separate_selected(&mut bm).expect("one of sixteen separates");
        assert_eq!(taken.num_faces(), 1);
        assert_eq!(
            bm.num_faces(),
            15,
            "the face was not removed from the source"
        );
        assert_eq!(
            taken.num_verts(),
            4,
            "the new mesh carries vertices its faces do not use"
        );
    }

    #[test]
    fn separating_carries_the_attributes_across() {
        let mut mesh = primitives::cube(2.0);
        {
            let corners = mesh.num_corners() as usize;
            mesh.attributes_mut()
                .ensure(names::UV, Domain::Corner, AttrType::F32x2, corners);
            if let Some(v) = mesh.attributes_mut().make_mut_f32x2(names::UV) {
                for (i, uv) in v.iter_mut().enumerate() {
                    *uv = Vec2::new(i as f32, 0.0);
                }
            }
        }
        let mut bm = BMesh::from_mesh_data(&mesh);
        let chosen: Vec<FaceId> = bm.face_ids().take(1).collect();
        bm.set_face_selected(chosen[0], true);

        let taken = separate_selected(&mut bm).expect("one of six separates");
        let uvs = taken
            .attributes()
            .get_f32x2(names::UV)
            .expect("kept the set");
        assert_eq!(uvs.len(), taken.num_corners() as usize);
        // The values are the ones those corners had, not the type's default.
        assert!(
            uvs.iter().any(|uv| *uv != Vec2::ZERO),
            "the UVs came across as defaults: {uvs:?}"
        );
    }

    #[test]
    fn separating_nothing_or_everything_declines() {
        // Separating a whole mesh leaves an empty object behind, which is a
        // worse outcome than declining — the user meant to split something.
        let mesh = primitives::cube(2.0);
        let mut bm = BMesh::from_mesh_data(&mesh);
        assert!(separate_selected(&mut bm).is_none(), "nothing selected");

        let all: Vec<FaceId> = bm.face_ids().collect();
        for f in &all {
            bm.set_face_selected(*f, true);
        }
        assert!(separate_selected(&mut bm).is_none(), "everything selected");
        assert_eq!(bm.num_faces(), 6, "the source was emptied anyway");
    }

    #[test]
    fn a_join_then_a_separate_gives_the_parts_back() {
        // The round trip that says the two agree about what a part is. Face
        // counts and positions, not indices: neither operation promises to
        // preserve numbering, and asserting it would pin down something that
        // is free to change.
        let a = primitives::cube(2.0);
        let b = primitives::uv_sphere(1.0, 8, 4);
        let shift = DAffine3::from_translation(glam::DVec3::new(6.0, 0.0, 0.0));
        let (joined, _) = join(&[part(&a, DAffine3::IDENTITY), part(&b, shift)]);

        let mut bm = BMesh::from_mesh_data(&joined);
        // The sphere's faces are the ones after the cube's.
        let ids: Vec<FaceId> = bm.face_ids().collect();
        for f in ids.iter().skip(a.num_faces() as usize) {
            bm.set_face_selected(*f, true);
        }
        let taken = separate_selected(&mut bm).expect("the sphere separates");

        assert_eq!(taken.num_faces(), b.num_faces());
        assert_eq!(bm.num_faces(), a.num_faces() as usize);
        // And the separated part is where the join put it, not back at the
        // origin: separate moves geometry between meshes and does not
        // transform it.
        assert!(
            taken.positions().iter().all(|p| p.x > 3.0),
            "the separated part was moved"
        );
    }
}
