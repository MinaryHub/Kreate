//! Element selection.
//!
//! Selection is an attribute layer, not a side table. That one decision means it
//! survives the `MeshData` ↔ `BMesh` crossing for free, snapshots with the same
//! copy-on-write machinery as everything else, and reaches the GPU through the
//! same path as a crease — no separate serialisation, no separate undo, no
//! separate invalidation.
//!
//! # Flushing
//!
//! The three domains are not independent: an edge whose endpoints are both
//! selected reads as selected to a user, whether or not anything wrote to the
//! edge layer. Blender calls reconciling them *flushing*, and which direction it
//! runs depends on the active mode. Getting this wrong is not subtle — the user
//! selects two vertices, the edge between them does not highlight, and the tool
//! feels broken.

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::attribute::{names, AttrType, Domain};

use super::BMesh;

/// Which element type the user is selecting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SelectMode {
    #[default]
    Vert,
    Edge,
    Face,
}

impl SelectMode {
    pub fn domain(self) -> Domain {
        match self {
            SelectMode::Vert => Domain::Point,
            SelectMode::Edge => Domain::Edge,
            SelectMode::Face => Domain::Face,
        }
    }

    pub fn layer(self) -> &'static str {
        match self {
            SelectMode::Vert => names::SELECT_VERT,
            SelectMode::Edge => names::SELECT_EDGE,
            SelectMode::Face => names::SELECT_FACE,
        }
    }
}

impl BMesh {
    /// Creates the three selection layers if they are missing.
    ///
    /// Called by every selection accessor, so a mesh that has never been
    /// selected in behaves like one where nothing is selected rather than
    /// panicking or silently doing nothing.
    pub fn ensure_selection_layers(&mut self) {
        for mode in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face] {
            let len = self.slot_count(mode.domain());
            self.attributes
                .ensure(mode.layer(), mode.domain(), AttrType::Bool, len);
        }
    }

    /// Copies the selection layers from a `MeshData` that shares this topology.
    ///
    /// Selection lives authoritatively in the document mesh — a click writes it
    /// there directly, without decoding anything — so an edit structure held
    /// across operations has to be told before an operator reads from it. Three
    /// arrays of bools; at a million vertices it is a megabyte of memcpy against
    /// the two seconds a re-decode would cost.
    ///
    /// Only valid while the topology matches, which is exactly when the session
    /// keeps its structure rather than rebuilding: element `i` of the mesh is
    /// slot `i` of the arena for as long as nothing has been added or removed.
    pub fn adopt_selection(&mut self, mesh: &crate::MeshData) {
        self.ensure_selection_layers();
        // The document numbers elements densely and this structure numbers
        // them by arena slot, and the two agree only until something is
        // removed. `to_mesh_data` walks the live elements in slot order and
        // gives each the next index, so the document's element *n* is the
        // n-th live slot — which is how `adopt_positions` pairs them, and how
        // this must too.
        //
        // It did not. It copied slot for slot, so after the first extrude —
        // which deletes the original face and appends new ones, leaving a hole
        // in the arena — the document's selected face landed on whichever
        // slot happened to share its number. The next E extruded that one.
        // "Each press extrudes a different face" was the report, and it was
        // exactly right.
        let live: [Vec<u32>; 3] = [
            self.vert_ids().map(|v| v.index()).collect(),
            self.edge_ids().map(|e| e.index()).collect(),
            self.face_ids().map(|f| f.index()).collect(),
        ];
        for (mode, slots) in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face]
            .into_iter()
            .zip(live)
        {
            let Some(source) = mesh.attributes().get_bool(mode.layer()) else {
                continue;
            };
            let source = source.to_vec();
            let Some(target) = self.attributes.make_mut_bool(mode.layer()) else {
                continue;
            };
            // Every slot cleared first, so a dead one cannot keep a stale
            // selection that a later element reuses.
            target.fill(false);
            for (rank, slot) in slots.into_iter().enumerate() {
                let Some(on) = source.get(rank) else { break };
                if let Some(t) = target.get_mut(slot as usize) {
                    *t = *on;
                }
            }
        }
    }

    /// Copies a mesh’s vertex positions in, in the order it stores them.
    ///
    /// The counterpart of [`BMesh::adopt_selection`] for the other thing that
    /// changes without changing topology. A transform rewrites the document’s
    /// positions array and nothing else — see `ops::mesh`’s transform section
    /// for why it must not go through the edit structure — so a session whose
    /// connectivity is still current can pick the new positions up instead of
    /// decoding the whole mesh again. **Measured: an extrude after a drag costs
    /// 38 ms with this and 81 ms without, at two hundred thousand vertices.**
    ///
    /// The pairing is by *rank among live vertices*, not by arena slot, because
    /// that is the numbering [`BMesh::to_mesh_data`] emits: it walks the live
    /// vertices in slot order and gives each the next index. A mesh that this
    /// structure did not produce may have any numbering at all, which is why
    /// the caller has to have established that the connectivity matches first.
    ///
    /// Bounded by both lengths, like `adopt_selection`, so a mismatch truncates
    /// rather than panicking.
    pub fn adopt_positions(&mut self, mesh: &crate::MeshData) {
        let positions = mesh.positions();
        for (index, (_, vert)) in self.verts.iter_mut().enumerate() {
            let Some(p) = positions.get(index) else { break };
            vert.co = *p;
        }
    }

    fn selected_at(&self, layer: &str, slot: u32) -> bool {
        self.attributes
            .get_bool(layer)
            .and_then(|s| s.get(slot as usize))
            .copied()
            .unwrap_or(false)
    }

    fn set_selected_at(&mut self, mode: SelectMode, slot: u32, on: bool) {
        self.ensure_selection_layers();
        if let Some(v) = self.attributes.make_mut_bool(mode.layer()) {
            if let Some(slot) = v.get_mut(slot as usize) {
                *slot = on;
            }
        }
    }

    #[inline]
    pub fn is_vert_selected(&self, v: VertId) -> bool {
        self.selected_at(names::SELECT_VERT, v.index())
    }

    #[inline]
    pub fn is_edge_selected(&self, e: EdgeId) -> bool {
        self.selected_at(names::SELECT_EDGE, e.index())
    }

    #[inline]
    pub fn is_face_selected(&self, f: FaceId) -> bool {
        self.selected_at(names::SELECT_FACE, f.index())
    }

    pub fn set_vert_selected(&mut self, v: VertId, on: bool) {
        self.set_selected_at(SelectMode::Vert, v.index(), on);
    }

    pub fn set_edge_selected(&mut self, e: EdgeId, on: bool) {
        self.set_selected_at(SelectMode::Edge, e.index(), on);
    }

    pub fn set_face_selected(&mut self, f: FaceId, on: bool) {
        self.set_selected_at(SelectMode::Face, f.index(), on);
    }

    pub fn selected_verts(&self) -> Vec<VertId> {
        self.vert_ids()
            .filter(|v| self.is_vert_selected(*v))
            .collect()
    }

    pub fn selected_edges(&self) -> Vec<EdgeId> {
        self.edge_ids()
            .filter(|e| self.is_edge_selected(*e))
            .collect()
    }

    pub fn selected_faces(&self) -> Vec<FaceId> {
        self.face_ids()
            .filter(|f| self.is_face_selected(*f))
            .collect()
    }

    /// Deselects every element in every domain.
    pub fn select_none(&mut self) {
        self.ensure_selection_layers();
        for mode in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face] {
            if let Some(v) = self.attributes.make_mut_bool(mode.layer()) {
                v.fill(false);
            }
        }
    }

    /// Selects every element in every domain.
    ///
    /// Writes only live slots: a freed slot has no element, and marking it would
    /// make whatever is created there next arrive pre-selected.
    pub fn select_all(&mut self) {
        self.ensure_selection_layers();
        let verts: Vec<u32> = self.vert_ids().map(|v| v.index()).collect();
        let edges: Vec<u32> = self.edge_ids().map(|e| e.index()).collect();
        let faces: Vec<u32> = self.face_ids().map(|f| f.index()).collect();
        for (mode, slots) in [
            (SelectMode::Vert, verts),
            (SelectMode::Edge, edges),
            (SelectMode::Face, faces),
        ] {
            if let Some(v) = self.attributes.make_mut_bool(mode.layer()) {
                v.fill(false);
                for s in slots {
                    v[s as usize] = true;
                }
            }
        }
    }

    /// Inverts the selection in the active mode, then flushes.
    pub fn select_invert(&mut self, mode: SelectMode) {
        self.ensure_selection_layers();
        match mode {
            SelectMode::Vert => {
                for v in self.vert_ids().collect::<Vec<_>>() {
                    let on = !self.is_vert_selected(v);
                    self.set_vert_selected(v, on);
                }
            }
            SelectMode::Edge => {
                for e in self.edge_ids().collect::<Vec<_>>() {
                    let on = !self.is_edge_selected(e);
                    self.set_edge_selected(e, on);
                }
            }
            SelectMode::Face => {
                for f in self.face_ids().collect::<Vec<_>>() {
                    let on = !self.is_face_selected(f);
                    self.set_face_selected(f, on);
                }
            }
        }
        self.flush_selection(mode);
    }

    /// Reconciles the three domains, given which one the user was working in.
    ///
    /// Vertex and edge modes flush *upward* — a face is selected when all of its
    /// corners are, which is what makes dragging a box over a quad select the
    /// quad. Face mode flushes *downward*: selecting a face selects everything it
    /// is made of, so a subsequent switch to vertex mode keeps the same region.
    pub fn flush_selection(&mut self, mode: SelectMode) {
        self.ensure_selection_layers();
        match mode {
            SelectMode::Vert => {
                self.flush_up_from_verts();
            }
            SelectMode::Edge => {
                // Endpoints follow the edges, and faces follow the *edges* —
                // never back through the vertices.
                //
                // Re-deriving edges from vertices here is the bug it looks like a
                // simplification: selecting a ring of edges around a tube selects
                // every vertex on it, and every perpendicular edge then has both
                // endpoints selected too. The user asked for a ring and gets a
                // band, and the tool that runs next sees a shape nobody chose.
                for e in self.edge_ids().collect::<Vec<_>>() {
                    if self.is_edge_selected(e) {
                        if let Some([a, b]) = self.edge_verts(e) {
                            self.set_vert_selected(a, true);
                            self.set_vert_selected(b, true);
                        }
                    }
                }
                for f in self.face_ids().collect::<Vec<_>>() {
                    let loops = self.face_loops(f);
                    let on = !loops.is_empty()
                        && loops
                            .iter()
                            .all(|l| self.loop_edge(*l).is_some_and(|e| self.is_edge_selected(e)));
                    self.set_face_selected(f, on);
                }
            }
            SelectMode::Face => {
                for f in self.face_ids().collect::<Vec<_>>() {
                    let on = self.is_face_selected(f);
                    if !on {
                        continue;
                    }
                    for l in self.face_loops(f) {
                        if let Some(v) = self.loop_vert(l) {
                            self.set_vert_selected(v, true);
                        }
                        if let Some(e) = self.loop_edge(l) {
                            self.set_edge_selected(e, true);
                        }
                    }
                }
            }
        }
    }

    /// An edge is selected when both endpoints are; a face when all its corners
    /// are.
    fn flush_up_from_verts(&mut self) {
        for e in self.edge_ids().collect::<Vec<_>>() {
            let on = self
                .edge_verts(e)
                .is_some_and(|[a, b]| self.is_vert_selected(a) && self.is_vert_selected(b));
            self.set_edge_selected(e, on);
        }
        for f in self.face_ids().collect::<Vec<_>>() {
            let loops = self.face_loops(f);
            let on = !loops.is_empty()
                && loops
                    .iter()
                    .all(|l| self.loop_vert(*l).is_some_and(|v| self.is_vert_selected(v)));
            self.set_face_selected(f, on);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_mesh_that_was_never_selected_in_reads_as_empty() {
        let m = BMesh::from_mesh_data(&primitives::cube(2.0));
        assert!(m.selected_verts().is_empty());
        assert!(m.selected_faces().is_empty());
        assert!(!m.is_vert_selected(m.vert_ids().next().unwrap()));
    }

    #[test]
    fn selecting_a_face_selects_what_it_is_made_of() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);

        assert_eq!(m.selected_verts().len(), 4);
        assert_eq!(m.selected_edges().len(), 4);
        assert_eq!(m.selected_faces(), vec![f]);
        m.assert_valid();
    }

    #[test]
    fn selecting_every_corner_of_a_face_selects_the_face() {
        // The rule that makes a box select over a quad select the quad. Without
        // it the user drags over a face, sees four dots light up, and the face
        // tools stay greyed out.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        let verts = m.face_verts(f);
        for v in &verts {
            m.set_vert_selected(*v, true);
        }
        m.flush_selection(SelectMode::Vert);

        assert!(m.is_face_selected(f));
        assert_eq!(m.selected_edges().len(), 4);
    }

    #[test]
    fn three_of_four_corners_does_not_select_the_face() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        for v in m.face_verts(f).into_iter().take(3) {
            m.set_vert_selected(v, true);
        }
        m.flush_selection(SelectMode::Vert);
        assert!(!m.is_face_selected(f));
        assert_eq!(
            m.selected_edges().len(),
            2,
            "only the edges with both ends selected"
        );
    }

    #[test]
    fn an_edge_selection_reaches_its_endpoints_and_stops() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let e = m.edge_ids().next().unwrap();
        m.set_edge_selected(e, true);
        m.flush_selection(SelectMode::Edge);

        assert_eq!(m.selected_verts().len(), 2);
        assert_eq!(m.selected_edges(), vec![e]);
        assert!(m.selected_faces().is_empty(), "one edge is not a face");
    }

    #[test]
    fn an_edge_ring_does_not_spread_sideways() {
        // The bug that hides behind a plausible simplification. Selecting a ring
        // of edges around a tube selects every vertex on it; if faces and edges
        // were then re-derived from those vertices, every perpendicular edge
        // would join the selection too. The user asked for a ring and would get
        // a band — and the next tool would act on a shape nobody chose.
        let mut m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 8));
        let caps: Vec<_> = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 8)
            .collect();
        for f in caps {
            m.remove_face(f);
        }
        let border: Vec<_> = m.edge_ids().filter(|&e| m.is_boundary_edge(e)).collect();
        assert_eq!(border.len(), 16, "two rings of eight");
        for e in &border {
            m.set_edge_selected(*e, true);
        }
        m.flush_selection(SelectMode::Edge);

        assert_eq!(
            m.selected_edges().len(),
            16,
            "the eight vertical edges must stay out of it"
        );
        assert_eq!(m.selected_verts().len(), 16);
        assert!(m.selected_faces().is_empty());
    }

    #[test]
    fn a_face_is_selected_when_all_of_its_edges_are() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        for l in m.face_loops(f) {
            if let Some(e) = m.loop_edge(l) {
                m.set_edge_selected(e, true);
            }
        }
        m.flush_selection(SelectMode::Edge);
        assert!(m.is_face_selected(f));
        assert_eq!(m.selected_faces(), vec![f]);
    }

    #[test]
    fn select_all_and_none_are_inverses() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        m.select_all();
        assert_eq!(m.selected_verts().len(), 8);
        assert_eq!(m.selected_edges().len(), 12);
        assert_eq!(m.selected_faces().len(), 6);
        m.select_none();
        assert!(m.selected_verts().is_empty());
        assert!(m.selected_edges().is_empty());
        assert!(m.selected_faces().is_empty());
    }

    #[test]
    fn select_all_does_not_arm_freed_slots() {
        // Arena slots are recycled and the selection layer is indexed by slot.
        // Marking a free slot means the next element created there arrives
        // already selected, and the next operator silently acts on it.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        m.remove_face(f);
        m.select_all();

        let verts = m.face_verts(m.face_ids().next().unwrap());
        let recreated = m.add_face(&verts).unwrap();
        assert!(
            !m.is_face_selected(recreated),
            "a face created in a recycled slot must start unselected"
        );
        m.assert_valid();
    }

    #[test]
    fn inverting_in_vertex_mode_flushes_the_other_domains() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        for v in m.face_verts(f) {
            m.set_vert_selected(v, true);
        }
        m.select_invert(SelectMode::Vert);
        assert_eq!(m.selected_verts().len(), 4, "the other four corners");
        assert!(!m.is_face_selected(f));
        assert!(
            m.selected_faces().len() == 1,
            "the opposite face, which the inverted corners complete"
        );
    }

    #[test]
    fn selection_survives_the_round_trip_to_storage_form() {
        // The reason selection is an attribute layer at all.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        m.set_face_selected(f, true);
        m.flush_selection(SelectMode::Face);
        let chosen = m.selected_verts().len();

        let baked = m.to_mesh_data();
        let again = BMesh::from_mesh_data(&baked);
        assert_eq!(again.selected_verts().len(), chosen);
        assert_eq!(again.selected_faces().len(), 1);
        assert_eq!(
            baked
                .attributes()
                .get_bool(names::SELECT_FACE)
                .map(|s| s.iter().filter(|x| **x).count()),
            Some(1)
        );
    }

    #[test]
    fn a_selection_adopted_after_a_deletion_lands_on_the_right_face() {
        // The bug as the user met it. Extruding a face deletes it and appends
        // new ones, which leaves a hole in the arena; the document numbers
        // faces densely, so from then on slot and index disagree. The
        // selection has to be paired the way `to_mesh_data` numbers, by rank
        // among the live faces, or the face the document says is selected is
        // not the one this structure thinks is.
        use crate::primitives;
        let mut bm = BMesh::from_mesh_data(&primitives::cube(2.0));
        let top = bm
            .face_ids()
            .max_by(|a, b| {
                let za = bm.face_normal(*a).z;
                let zb = bm.face_normal(*b).z;
                za.partial_cmp(&zb).unwrap()
            })
            .unwrap();
        let out = crate::edit::extrude_faces(&mut bm, &[top], glam::Vec3::Z).unwrap();
        let lid = out.new_faces[0];
        // The extrude deleted the top and appended a lid and four walls, so
        // there is a hole where the top was and every face after it has a
        // slot one higher than its dense index. Without that the bug cannot
        // show, so the fixture says so.
        let slots: Vec<u32> = bm.face_ids().map(|f| f.index()).collect();
        assert!(
            slots.iter().any(|s| *s as usize >= slots.len()),
            "the fixture needs a hole in the arena for the bug to show: {slots:?}"
        );

        // What the document holds: the lid selected, by its dense index.
        let mut data = bm.to_mesh_data();
        let dense_lid = bm.face_ids().position(|f| f == lid).unwrap();
        crate::selection::set_selected(&mut data, Domain::Face, dense_lid as u32, true);

        bm.select_none();
        bm.adopt_selection(&data);
        assert_eq!(
            bm.selected_faces(),
            vec![lid],
            "the document's selected face was adopted onto a different one"
        );
    }
}
