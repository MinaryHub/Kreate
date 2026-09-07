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

use crate::arena::{ArenaIndex, EdgeId, LoopId, VertId};
use crate::attribute::{names, AttributeStore, Domain};
use crate::mesh_data::MeshData;

use super::{BMesh, DiskLink, Edge, Face, Loop, Vert};

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
        bm.build_elements(mesh);

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

    /// Fills the arenas and links every cycle, in bulk.
    ///
    /// Split out from [`Self::from_mesh_data`] because it is the whole of the
    /// cost, and because what it must *not* do is the interesting part.
    ///
    /// # Why not `add_edge` and `add_face`
    ///
    /// Those are the right API for an editing operation, where one element
    /// arrives at a time and the structure has to be sound after each. They are
    /// the wrong one for a conversion, where the entire answer is known up
    /// front, and the difference is not a constant factor:
    ///
    /// - `add_edge` asks `edge_between`, which walks a vertex's disk cycle
    ///   *and allocates a `Vec` to do it*. Two million edges, plus four million
    ///   more calls from `add_face`, is six million allocations and six million
    ///   walks to answer a question the source already answered.
    /// - Each `disk_insert` and `radial_insert` splices one element into a
    ///   cycle, which is three or four scattered writes into arenas far larger
    ///   than any cache.
    ///
    /// Here the counts are known, so each cycle is built once from a contiguous
    /// list: two counting passes and a linear link, with no search and no
    /// per-element allocation. Measured on a million-vertex grid — 2 M edges,
    /// 4 M corners — this took the conversion from 919 ms to the figure in
    /// `tests/build_phases.rs`.
    ///
    /// The element *order* is the same as the incremental path gave: vertex
    /// *i* in slot *i*, edge *i* in slot *i*, and a face's corners in
    /// consecutive slots in winding order. That is load-bearing — the
    /// attribute store is adopted wholesale rather than gathered — so it is
    /// asserted by the round-trip test rather than merely intended.
    fn build_elements(&mut self, mesh: &MeshData) {
        let num_verts = mesh.num_verts() as usize;

        // --- vertices ---------------------------------------------------------
        for p in mesh.positions() {
            self.verts.insert(Vert { co: *p, edge: None });
        }

        // --- edges, unlinked --------------------------------------------------
        //
        // Disk links are placeholders until the pass below; nothing may read
        // them in between.
        let placeholder_edge = EdgeId::from_parts(0, 0);
        let mut edge_ids: Vec<EdgeId> = Vec::with_capacity(mesh.num_edges() as usize);
        for [a, b] in mesh.edges() {
            let (a, b) = (*a as usize, *b as usize);
            if a == b || a >= num_verts || b >= num_verts {
                // A degenerate or dangling edge, skipped as the incremental
                // path skipped it. **Unreachable today**: every `MeshData`
                // constructor ends in `rebuild_edges`, which derives the array
                // from the faces and can produce neither — as
                // `the_edge_array_is_always_normalised_in_range_and_non_degenerate`
                // asserts, and as a falsification sweep confirmed by deleting
                // this branch without failing anything.
                //
                // It stays because the alternative to skipping is indexing out
                // of bounds, and a panic in the mesh kernel on a malformed file
                // is a worse answer than a dropped edge. Kept as a guard, not
                // claimed as tested.
                continue;
            }
            let id = self.edges.insert(Edge {
                verts: [
                    VertId::from_parts(a as u32, 0),
                    VertId::from_parts(b as u32, 0),
                ],
                disk: [DiskLink {
                    prev: placeholder_edge,
                    next: placeholder_edge,
                }; 2],
                first_loop: None,
            });
            edge_ids.push(id);
        }

        // --- the vertex-to-edge map, and the disk cycles from it --------------
        //
        // One CSR rather than a hash map: the keys are dense vertex indices, so
        // counting sort is both faster and allocation-free per element. It is
        // then reused to find each corner's edge below, which is the lookup
        // `add_edge` was paying for.
        let mut degree = vec![0u32; num_verts + 1];
        for e in &edge_ids {
            let [a, b] = self.edges[*e].verts;
            degree[a.index() as usize + 1] += 1;
            degree[b.index() as usize + 1] += 1;
        }
        for i in 1..degree.len() {
            degree[i] += degree[i - 1];
        }
        let offsets = degree;
        let mut cursor = offsets.clone();
        let mut incident: Vec<EdgeId> = vec![placeholder_edge; offsets[num_verts] as usize];
        for e in &edge_ids {
            let [a, b] = self.edges[*e].verts;
            for v in [a, b] {
                let slot = &mut cursor[v.index() as usize];
                incident[*slot as usize] = *e;
                *slot += 1;
            }
        }

        for v in 0..num_verts {
            let from = offsets[v] as usize;
            let to = offsets[v + 1] as usize;
            if from == to {
                continue;
            }
            let ring = &incident[from..to];
            self.verts[VertId::from_parts(v as u32, 0)].edge = Some(ring[0]);
            let n = ring.len();
            for i in 0..n {
                let e = ring[i];
                let which = usize::from(self.edges[e].verts[1].index() as usize == v);
                // Cyclic, and correct for a single edge: `prev` and `next` both
                // become the edge itself, which is the invariant `disk_insert`
                // establishes for the first edge of a disk.
                self.edges[e].disk[which] = DiskLink {
                    prev: ring[(i + n - 1) % n],
                    next: ring[(i + 1) % n],
                };
            }
        }

        // --- faces and loops --------------------------------------------------
        let placeholder_loop = LoopId::from_parts(0, 0);
        let mut loop_edge: Vec<EdgeId> = Vec::with_capacity(mesh.num_corners() as usize);
        let mut loop_ids: Vec<LoopId> = Vec::with_capacity(mesh.num_corners() as usize);

        for face in mesh.faces() {
            let n = face.len();
            if n < 3 || face.iter().any(|v| *v as usize >= num_verts) {
                // A corner referencing a vertex that does not exist, or a face
                // too short to be one. Dropped rather than propagated, as
                // before.
                continue;
            }
            let mut repeated = false;
            for (i, v) in face.iter().enumerate() {
                if face[..i].contains(v) {
                    repeated = true;
                    break;
                }
            }
            if repeated {
                continue;
            }

            let f = self.faces.insert(Face {
                first_loop: placeholder_loop,
                len: n as u32,
            });

            let first = self.loops.len();
            for i in 0..n {
                let v = VertId::from_parts(face[i], 0);
                let next_v = VertId::from_parts(face[(i + 1) % n], 0);
                let e = match self.find_edge(&offsets, &incident, v, next_v) {
                    Some(e) => e,
                    // The source's edge array did not contain this face's edge.
                    // `MeshData` derives its edges from its faces, so this only
                    // happens on a hand-built mesh — and creating it here, with
                    // the incremental splice, keeps the structure sound at the
                    // cost of being slow on exactly the meshes nobody profiles.
                    None => match self.add_edge(v, next_v) {
                        Ok(e) => e,
                        Err(_) => continue,
                    },
                };
                let id = self.loops.insert(Loop {
                    vert: v,
                    edge: e,
                    face: f,
                    next: placeholder_loop,
                    prev: placeholder_loop,
                    radial_next: placeholder_loop,
                    radial_prev: placeholder_loop,
                });
                loop_ids.push(id);
                loop_edge.push(e);
            }

            // The loop cycle, over the ids just pushed.
            let ids = &loop_ids[first..];
            let n = ids.len();
            if n == 0 {
                continue;
            }
            for i in 0..n {
                let l = ids[i];
                self.loops[l].next = ids[(i + 1) % n];
                self.loops[l].prev = ids[(i + n - 1) % n];
            }
            self.faces[f].first_loop = ids[0];
        }

        // --- radial cycles ----------------------------------------------------
        //
        // The same counting sort, keyed by edge slot. Two passes over the loop
        // list against four million individual splices.
        let edge_slots = self.edges.slot_count();
        let mut per_edge = vec![0u32; edge_slots + 1];
        for e in &loop_edge {
            per_edge[e.index() as usize + 1] += 1;
        }
        for i in 1..per_edge.len() {
            per_edge[i] += per_edge[i - 1];
        }
        let radial_offsets = per_edge;
        let mut cursor = radial_offsets.clone();
        let mut radial: Vec<LoopId> = vec![placeholder_loop; radial_offsets[edge_slots] as usize];
        for (l, e) in loop_ids.iter().zip(&loop_edge) {
            let slot = &mut cursor[e.index() as usize];
            radial[*slot as usize] = *l;
            *slot += 1;
        }

        for slot in 0..edge_slots {
            let from = radial_offsets[slot] as usize;
            let to = radial_offsets[slot + 1] as usize;
            if from == to {
                continue;
            }
            let ring = &radial[from..to];
            let e = EdgeId::from_parts(slot as u32, 0);
            self.edges[e].first_loop = Some(ring[0]);
            let n = ring.len();
            for i in 0..n {
                let l = ring[i];
                self.loops[l].radial_prev = ring[(i + n - 1) % n];
                self.loops[l].radial_next = ring[(i + 1) % n];
            }
        }
    }

    /// The edge between two vertices, through the CSR built above.
    ///
    /// A scan over one vertex's incident edges, which on any real mesh is four
    /// to six. This is the question `edge_between` answers by walking a disk
    /// cycle and allocating a `Vec`; here the answer is a slice.
    fn find_edge(
        &self,
        offsets: &[u32],
        incident: &[EdgeId],
        a: VertId,
        b: VertId,
    ) -> Option<EdgeId> {
        let from = *offsets.get(a.index() as usize)? as usize;
        let to = *offsets.get(a.index() as usize + 1)? as usize;
        incident.get(from..to)?.iter().copied().find(|e| {
            let verts = self.edges[*e].verts;
            verts == [a, b] || verts == [b, a]
        })
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
