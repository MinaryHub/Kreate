//! `MeshData` ↔ `BMesh`.
//!
//! The two directions of entering and leaving edit mode, and the pair the plan
//! calls for a round-trip test on: `MeshData → BMesh → MeshData` must be the
//! identity, modulo element renumbering.
//!
//! Every generic attribute layer survives both crossings. That is the whole
//! argument for the domain model: UVs, creases, seams, vertex colours and
//! selection all ride along without either direction knowing what they are, so
//! adding a layer type later touches no conversion code.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, VertId};
use crate::attribute::{names, AttributeStore, Domain};
use crate::mesh_data::MeshData;

use super::BMesh;

impl BMesh {
    /// Builds an edit structure from storage form.
    ///
    /// Element order is preserved exactly: vertex *i* becomes slot *i*, edge *i*
    /// slot *i*, and face *f*'s corners become consecutive loop slots in winding
    /// order. That is what lets the attribute store be adopted wholesale instead
    /// of gathered, and it makes the round-trip an identity rather than a
    /// permutation on an unedited mesh.
    pub fn from_mesh_data(mesh: &MeshData) -> Self {
        let mut bm = BMesh::new();

        let positions = mesh.positions();
        let verts: Vec<VertId> = positions.iter().map(|p| bm.add_vert(*p)).collect();

        // Edges before faces, so they take slots in the source's order rather
        // than in the order the faces happen to reach them. Without this the
        // edge attribute layers would need a gather, and creases and seams would
        // land on the wrong edges if anyone forgot it.
        for [a, b] in mesh.edges() {
            let (Some(&a), Some(&b)) = (verts.get(*a as usize), verts.get(*b as usize)) else {
                continue;
            };
            let _ = bm.add_edge(a, b);
        }

        for face in mesh.faces() {
            let ids: Vec<VertId> = face
                .iter()
                .filter_map(|v| verts.get(*v as usize).copied())
                .collect();
            if ids.len() != face.len() {
                // A corner referenced a vertex that does not exist. `validate`
                // on the source reports it; dropping the face here keeps the
                // edit structure sound rather than propagating the corruption.
                continue;
            }
            let _ = bm.add_face(&ids);
        }

        // Position is inline in `Vert`, so it must not also exist as a layer —
        // two copies means whichever one an operator happens to read decides the
        // answer.
        let mut attributes = mesh.attributes().clone();
        attributes.remove(names::POSITION);
        for domain in Domain::ALL {
            attributes.resize_domain(domain, bm.slot_count(domain));
        }
        bm.attributes = attributes;
        bm
    }

    /// Bakes the edit structure back to storage form.
    ///
    /// Elements are renumbered densely in slot order, which drops the holes an
    /// editing session leaves behind. Every attribute layer is gathered through
    /// the same renumbering, so nothing can be reindexed out of step.
    ///
    /// **Wire edges do not survive.** [`MeshData`]'s edge array is derived from
    /// its faces, so an edge with no face has nowhere to go. Loose edges are an
    /// edit-mode intermediate here, not document data.
    pub fn to_mesh_data(&self) -> MeshData {
        // --- vertices ---------------------------------------------------------
        let vert_slots: Vec<u32> = self.vert_ids().map(|v| v.index()).collect();
        let mut vert_index = vec![u32::MAX; self.slot_count(Domain::Point)];
        for (new, slot) in vert_slots.iter().enumerate() {
            vert_index[*slot as usize] = new as u32;
        }
        let positions: Vec<Vec3> = self.verts.values().map(|v| v.co).collect();

        // --- faces and corners ------------------------------------------------
        let face_slots: Vec<u32> = self.face_ids().map(|f| f.index()).collect();
        let mut faces: Vec<Vec<u32>> = Vec::with_capacity(face_slots.len());
        let mut loop_slots: Vec<u32> = Vec::new();
        for f in self.face_ids() {
            let loops = self.face_loops(f);
            let mut corners = Vec::with_capacity(loops.len());
            for l in &loops {
                let Some(v) = self.loop_vert(*l) else {
                    continue;
                };
                corners.push(vert_index[v.index() as usize]);
                loop_slots.push(l.index());
            }
            faces.push(corners);
        }

        let mut mesh = MeshData::from_faces(positions, &faces);

        // --- attribute layers -------------------------------------------------
        // Edges get their own pass: `from_faces` derives the edge array in its
        // own canonical order, which is not the edit structure's slot order, so
        // the layers have to be gathered through the pairing rather than copied.
        let edge_slots = self.edge_gather(&vert_index, &mesh);

        let mut attributes = AttributeStore::new();
        for layer in self.attributes.iter() {
            let order: &[u32] = match layer.domain {
                Domain::Point => &vert_slots,
                Domain::Edge => &edge_slots,
                Domain::Corner => &loop_slots,
                Domain::Face => &face_slots,
            };
            attributes.insert(&layer.name, layer.domain, layer.data.gather(order));
        }
        // Position is regenerated from `Vert::co` above, not gathered.
        for layer in attributes.iter() {
            mesh.attributes_mut()
                .insert(&layer.name, layer.domain, layer.data.clone());
        }
        mesh
    }

    /// For each edge of the baked mesh, the edit-structure slot it came from.
    ///
    /// `u32::MAX` where there is no match, which `gather` turns into the layer's
    /// default: an edge the edit structure did not have gets a zero crease
    /// rather than someone else's.
    fn edge_gather(&self, vert_index: &[u32], baked: &MeshData) -> Vec<u32> {
        // Sorted pairs plus binary search rather than a hash map: iteration
        // order here decides the output, and a hasher with a per-run seed would
        // make the result depend on which run it was.
        let mut pairs: Vec<([u32; 2], u32)> = self
            .edge_ids()
            .filter_map(|e| {
                let [a, b] = self.edge_verts(e)?;
                let mut key = [
                    *vert_index.get(a.index() as usize)?,
                    *vert_index.get(b.index() as usize)?,
                ];
                key.sort_unstable();
                Some((key, e.index()))
            })
            .collect();
        pairs.sort_unstable();

        baked
            .edges()
            .iter()
            .map(|[a, b]| {
                let mut key = [*a, *b];
                key.sort_unstable();
                match pairs.binary_search_by_key(&key, |(k, _)| *k) {
                    Ok(i) => pairs[i].1,
                    Err(_) => u32::MAX,
                }
            })
            .collect()
    }
}

impl MeshData {
    /// Enters edit mode.
    pub fn to_bmesh(&self) -> BMesh {
        BMesh::from_mesh_data(self)
    }
}

impl From<&MeshData> for BMesh {
    fn from(mesh: &MeshData) -> Self {
        BMesh::from_mesh_data(mesh)
    }
}

impl From<&BMesh> for MeshData {
    fn from(bm: &BMesh) -> Self {
        bm.to_mesh_data()
    }
}

/// Convenience for tests and operators: the positions an edit structure holds,
/// in slot order.
impl BMesh {
    pub fn positions(&self) -> Vec<Vec3> {
        self.verts.values().map(|v| v.co).collect()
    }

    /// Replaces the whole attribute store, resizing every domain to match.
    ///
    /// Used when adopting layers built elsewhere; keeps the arrays parallel so
    /// the caller cannot leave the mesh in a state the validator would reject.
    pub fn set_attributes(&mut self, mut attributes: AttributeStore) {
        attributes.remove(names::POSITION);
        for domain in Domain::ALL {
            attributes.resize_domain(domain, self.slot_count(domain));
        }
        self.attributes = attributes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{AttrData, AttrType};
    use kreate_math::Vec2;
    use std::sync::Arc;

    fn assert_same_shape(a: &MeshData, b: &MeshData) {
        assert_eq!(a.num_verts(), b.num_verts(), "vertex count");
        assert_eq!(a.num_faces(), b.num_faces(), "face count");
        assert_eq!(a.num_edges(), b.num_edges(), "edge count");
        assert_eq!(a.num_corners(), b.num_corners(), "corner count");
        assert_eq!(a.positions(), b.positions(), "positions");
        assert_eq!(a.face_offsets(), b.face_offsets(), "face offsets");
        assert_eq!(a.corner_verts(), b.corner_verts(), "corner verts");
        assert_eq!(a.edges(), b.edges(), "edges");
    }

    #[test]
    fn a_round_trip_is_the_identity_on_every_primitive() {
        // The property the plan calls for first, because everything downstream —
        // undo, autosave, file IO — assumes entering and leaving edit mode
        // changes nothing.
        let meshes: Vec<(&str, MeshData)> = vec![
            ("plane", crate::primitives::plane(2.0)),
            ("grid", crate::primitives::grid(2.0, 3, 4)),
            ("cube", crate::primitives::cube(2.0)),
            ("uv_sphere", crate::primitives::uv_sphere(1.0, 12, 6)),
            ("ico_sphere", crate::primitives::ico_sphere(1.0, 1)),
            ("cylinder", crate::primitives::cylinder(1.0, 2.0, 8)),
            ("cone", crate::primitives::cone(1.0, 2.0, 8)),
            ("torus", crate::primitives::torus(1.0, 0.4, 12, 8)),
        ];
        for (name, mesh) in meshes {
            let bm = BMesh::from_mesh_data(&mesh);
            bm.assert_valid();
            let back = bm.to_mesh_data();
            back.assert_valid();
            assert_same_shape(&mesh, &back);
            assert_eq!(
                mesh.num_verts() as usize,
                bm.num_verts(),
                "{name}: vertex count through the edit structure"
            );
        }
    }

    #[test]
    fn generic_layers_survive_both_crossings() {
        // The point of the domain model. None of the conversion code knows what
        // a UV or a crease is; if this passes, adding a layer type later needs
        // no conversion work at all.
        let mut mesh = crate::primitives::cube(2.0);
        let corners = mesh.num_corners() as usize;
        let edges = mesh.num_edges() as usize;
        let verts = mesh.num_verts() as usize;
        let faces = mesh.num_faces() as usize;

        mesh.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(
                (0..corners)
                    .map(|i| Vec2::new(i as f32, -(i as f32)))
                    .collect(),
            )),
        );
        mesh.attributes_mut().insert(
            names::CREASE,
            Domain::Edge,
            AttrData::F32(Arc::new((0..edges).map(|i| i as f32 * 0.1).collect())),
        );
        mesh.attributes_mut().insert(
            names::SELECT_VERT,
            Domain::Point,
            AttrData::Bool(Arc::new((0..verts).map(|i| i % 3 == 0).collect())),
        );
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(Arc::new((0..faces).map(|i| i as i32 * 7).collect())),
        );

        let back = BMesh::from_mesh_data(&mesh).to_mesh_data();

        assert_eq!(
            back.attributes().get_f32x2("uv"),
            mesh.attributes().get_f32x2("uv"),
            "corner layer"
        );
        assert_eq!(
            back.attributes().get_f32(names::CREASE),
            mesh.attributes().get_f32(names::CREASE),
            "edge layer — the one that needs the pairing, not a copy"
        );
        assert_eq!(
            back.attributes().get_bool(names::SELECT_VERT),
            mesh.attributes().get_bool(names::SELECT_VERT),
            "point layer"
        );
        assert_eq!(
            back.attributes().get_i32(names::MATERIAL_INDEX),
            mesh.attributes().get_i32(names::MATERIAL_INDEX),
            "face layer"
        );
    }

    #[test]
    fn an_edge_layer_follows_its_edge_and_not_its_index() {
        // The failure this guards: `to_mesh_data` re-derives the edge array in
        // its own canonical order, so copying edge layers across by index puts
        // every crease and seam on the wrong edge. It looks right on a mesh
        // whose orders happen to agree, which is most of them.
        let mut mesh = crate::primitives::cube(2.0);
        let n = mesh.num_edges() as usize;
        mesh.attributes_mut().insert(
            names::SHARP_EDGE,
            Domain::Edge,
            AttrData::Bool(Arc::new((0..n).map(|i| i == 5).collect())),
        );
        let marked = mesh.edges()[5];

        let back = BMesh::from_mesh_data(&mesh).to_mesh_data();
        let sharp = back.attributes().get_bool(names::SHARP_EDGE).unwrap();
        let found: Vec<[u32; 2]> = back
            .edges()
            .iter()
            .zip(sharp)
            .filter(|(_, s)| **s)
            .map(|(e, _)| *e)
            .collect();
        assert_eq!(
            found,
            vec![marked],
            "the same pair of vertices, not index 5"
        );
    }

    #[test]
    fn editing_between_the_crossings_reaches_the_result() {
        let mesh = crate::primitives::cube(2.0);
        let mut bm = BMesh::from_mesh_data(&mesh);

        let f = bm.face_ids().next().unwrap();
        assert!(bm.remove_face(f));
        bm.assert_valid();

        let back = bm.to_mesh_data();
        back.assert_valid();
        assert_eq!(back.num_faces(), 5);
        assert_eq!(back.num_verts(), 8, "the hole keeps its border");
        assert_eq!(back.num_edges(), 12);
    }

    #[test]
    fn deleted_elements_are_renumbered_away() {
        // Slots left free by an edit must not become vertices at the origin.
        let mesh = crate::primitives::grid(2.0, 4, 4);
        let mut bm = BMesh::from_mesh_data(&mesh);
        let v = bm.vert_ids().next().unwrap();
        bm.remove_vert(v);
        bm.remove_loose();
        bm.assert_valid();

        let back = bm.to_mesh_data();
        back.assert_valid();
        assert_eq!(back.num_verts() as usize, bm.num_verts());
        assert!(
            back.positions().iter().all(|p| p.is_finite()),
            "no default-valued leftovers"
        );
        assert_eq!(back.validate(), Vec::new());
    }

    #[test]
    fn a_wire_edge_is_dropped_and_that_is_documented_behaviour() {
        // Not a defect to fix silently: `MeshData`'s edge array is derived from
        // its faces, so there is nowhere to put an edge with no face. Asserted
        // rather than left to be discovered.
        let mut bm = BMesh::from_mesh_data(&crate::primitives::plane(2.0));
        let a = bm.add_vert(Vec3::new(5.0, 0.0, 0.0));
        let b = bm.add_vert(Vec3::new(6.0, 0.0, 0.0));
        bm.add_edge(a, b).unwrap();
        assert_eq!(bm.num_edges(), 5);

        let back = bm.to_mesh_data();
        assert_eq!(back.num_edges(), 4, "the wire has nowhere to go");
        assert_eq!(back.num_verts(), 6, "but its endpoints are still vertices");
        assert_eq!(back.validate(), Vec::new());
    }

    #[test]
    fn positions_do_not_exist_twice() {
        let bm = BMesh::from_mesh_data(&crate::primitives::cube(2.0));
        assert!(
            !bm.attributes().contains(names::POSITION),
            "position lives in Vert::co"
        );
        assert_eq!(bm.validate(), Vec::new());
        assert_eq!(bm.positions().len(), 8);
    }

    #[test]
    fn adopting_an_attribute_store_keeps_the_arrays_parallel() {
        let mut bm = BMesh::from_mesh_data(&crate::primitives::cube(2.0));
        let mut store = AttributeStore::new();
        store.ensure("uv", Domain::Corner, AttrType::F32x2, 3);
        store.insert(
            names::POSITION,
            Domain::Point,
            AttrData::Bool(Arc::new(Vec::new())),
        );
        bm.set_attributes(store);
        assert_eq!(bm.validate(), Vec::new(), "resized to the slot counts");
        assert_eq!(bm.attributes().get_f32x2("uv").unwrap().len(), 24);
    }
}
