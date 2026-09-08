//! The edit-mode mesh, kept between operations.
//!
//! `BMesh` is the edit structure: loop, radial and disk cycles, everything an
//! operator needs to walk a neighbourhood. `MeshData` is the document form: flat,
//! compact, copy-on-write. Converting between them costs a pass over every
//! vertex, edge, corner and face.
//!
//! Without a session, every operator pays that twice — decode on the way in,
//! encode on the way out. Extrude a face on a million-vertex mesh and the
//! conversion costs more than the extrusion by a wide margin. Held across
//! operations, the decode happens once when edit mode is entered.
//!
//! # The encode still happens per operation, and that is deliberate
//!
//! What comes out is what the renderer draws and what undo records, so it has to
//! exist after every step. But `MeshData` is copy-on-write per array, so an
//! encode that produces the same topology hands back arrays the caches can
//! recognise — see [`crate::Document::mesh_arc`]. Snapshotting the `BMesh`
//! instead, which is the obvious alternative, would cost around 200 bytes a
//! vertex per undo step against the document form's fifty.
//!
//! # Selection is not here
//!
//! Selection changes go straight to the document mesh — see
//! `kreate_mesh::selection`. They are the most frequent thing a user does in
//! edit mode and they need no adjacency, so making them wait on a session that
//! may not exist would be the wrong shape. The session is invalidated when one
//! happens, which costs a decode on the next *topology* edit and keeps the two
//! paths from having to agree about anything.

use std::sync::Arc;

use kreate_mesh::BMesh;

use crate::document::MeshId;
use crate::Document;

/// One mesh's edit structure, held while edit mode is open.
pub struct EditSession {
    mesh: MeshId,
    bmesh: BMesh,
    /// The mesh the `BMesh` was decoded from, held for comparison.
    ///
    /// A session is current while the document's *topology* arrays are still
    /// the ones it decoded — pointer identity, the same free and exact test the
    /// renderer and the modifier cache use. Anything that changes the
    /// connectivity by another route, an undo above all, makes it stale, and a
    /// session that trusted itself would silently undo the user's last action.
    ///
    /// **Topology, not the whole mesh.** Two things change without changing
    /// connectivity, and treating either as staleness costs a full decode on
    /// the *next* operation — two seconds at a million vertices, which is worse
    /// than having no session at all:
    ///
    /// - A selection click rewrites one layer and leaves everything else
    ///   shared.
    /// - A transform rewrites the positions array and leaves everything else
    ///   shared, which is the whole point of it declaring its write set.
    ///
    /// Both are copied in when an operator asks for the structure, rather than
    /// decoded again. What is *not* excused is a change to the connectivity:
    /// that is a different mesh, and the session has to be built from it.
    source: Arc<kreate_mesh::MeshData>,
    /// Which edge each corner walks along, for the mesh as it currently is.
    ///
    /// Topology, so it survives any number of selection changes — and that is
    /// why it is worth holding. Rebuilding it is a sorted pass over every edge:
    /// at a million faces that is four hundred milliseconds, on every click,
    /// for a flush that otherwise takes two.
    ///
    /// An `Arc` so a caller can take it without borrowing the session, which it
    /// cannot do while also holding the mesh it is about to write.
    corner_edges: Arc<Vec<u32>>,
}

impl EditSession {
    /// Builds a session for a mesh, decoding it once.
    pub fn open(doc: &Document, mesh: MeshId) -> Option<Self> {
        let data = doc.mesh_arc(mesh)?.clone();
        Some(Self {
            mesh,
            bmesh: data.to_bmesh(),
            corner_edges: Arc::new(data.corner_edges()),
            source: data,
        })
    }

    /// The corner-to-edge map for the mesh as the document currently has it.
    ///
    /// Cheap to take — one refcount — so a caller can hold it while it borrows
    /// the mesh mutably.
    pub fn corner_edges(&self) -> Arc<Vec<u32>> {
        self.corner_edges.clone()
    }

    pub fn mesh(&self) -> MeshId {
        self.mesh
    }

    /// Whether this session is still the right one to use.
    ///
    /// It has to be about the same mesh *and* be looking at the same document it
    /// last agreed with. The version check is what makes a stale session
    /// harmless rather than destructive.
    pub fn matches(&self, doc: &Document, mesh: MeshId) -> bool {
        self.mesh == mesh
            && doc
                .mesh(mesh)
                .is_some_and(|current| self.source.same_topology(current))
    }

    /// The edit structure with the document's selection and positions copied
    /// in.
    ///
    /// Both are authoritative in the document, because both change by routes
    /// that never touch the edit structure: a click writes a selection layer
    /// and a transform writes positions, neither needing adjacency and neither
    /// willing to pay for a decode. So both have to be brought across before an
    /// operator reads them. Four arrays against a two-second decode.
    ///
    /// This is the other half of `matches` accepting a moved mesh as current.
    /// Without the copy, an extrude after a drag would build its new faces from
    /// the positions the vertices had *before* the drag — silently undoing the
    /// user's last action, which is the exact failure the staleness check
    /// exists to prevent.
    pub fn bmesh_for(&mut self, doc: &Document) -> &mut BMesh {
        if let Some(current) = doc.mesh(self.mesh) {
            self.bmesh.adopt_selection(current);
            // Skipped when the positions array is still the one that was
            // decoded, which is the common case: the copy is a pass over every
            // vertex, and paying it on every selection click would give back
            // what holding the session was for.
            if !self.source.same_geometry(current) {
                self.bmesh.adopt_positions(current);
            }
        }
        &mut self.bmesh
    }

    /// The edit structure, to read.
    ///
    /// Read-only, and deliberately without the selection copy `bmesh_for`
    /// does: this is for the interface, which draws what is there rather than
    /// preparing it for an edit. The UV editor is the caller — a UV is a
    /// corner, and corner *slots* are what the UV operators take, so the panel
    /// has to read the arena's numbering rather than the document's dense one.
    pub fn bmesh(&self) -> &BMesh {
        &self.bmesh
    }

    /// The edit structure, for an operator to work on.
    pub fn bmesh_mut(&mut self) -> &mut BMesh {
        &mut self.bmesh
    }

    /// Writes the edit structure back to the document.
    ///
    /// Called after every operator that changed anything: the result is what the
    /// renderer draws and what undo records.
    pub fn bake(&mut self, doc: &mut Document) -> bool {
        let baked = self.bmesh.to_mesh_data();
        let corner_edges = baked.corner_edges();
        let Some(slot) = doc.mesh_mut(self.mesh) else {
            return false;
        };
        *slot = baked;
        // The topology may have changed, so the map has to be rebuilt with it.
        // Keeping a stale one would put creases and selections on the wrong
        // edges, which shows up long after the operator that caused it.
        self.corner_edges = Arc::new(corner_edges);
        self.source = doc
            .mesh_arc(self.mesh)
            .cloned()
            .expect("just written above");
        true
    }

    /// Rebuilds from the document, discarding whatever the session held.
    ///
    /// For when something else changed the mesh — a selection click, an undo, a
    /// modifier applied. Cheaper than being wrong.
    pub fn reopen(&mut self, doc: &Document) -> bool {
        let Some(data) = doc.mesh_arc(self.mesh).cloned() else {
            return false;
        };
        self.bmesh = data.to_bmesh();
        self.corner_edges = Arc::new(data.corner_edges());
        self.source = data;
        true
    }
}

/// The sessions edit mode has open, one per mesh.
///
/// Edit mode is a set of meshes — see `kreate_ops::ops::mesh::edit_targets` — and
/// each of them needs its own edit structure for the same reason the first one
/// does: without it every operator pays a decode and an encode of the whole
/// mesh, which at a million faces is most of a second. One session and a set of
/// meshes would have put that cost back for every object after the active one.
///
/// A `Vec` and not a map. The set is the user's object selection, so it is tens
/// of entries at the very most, and a linear scan of ten `MeshId`s is faster
/// than hashing one. It also keeps the order the sessions were opened in, which
/// is `edit_targets`' order, which is the order operators walk them in — so a
/// multi-mesh edit is deterministic without anyone arranging for it.
#[derive(Default)]
pub struct EditSessions {
    open: Vec<EditSession>,
}

impl EditSessions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.open.is_empty()
    }

    pub fn len(&self) -> usize {
        self.open.len()
    }

    /// Which meshes have a session, in the order they were opened.
    pub fn meshes(&self) -> impl Iterator<Item = MeshId> + '_ {
        self.open.iter().map(|s| s.mesh())
    }

    /// Opens sessions for exactly `meshes`, keeping any already open.
    ///
    /// Kept rather than rebuilt because a session is expensive to make and
    /// cheap to check: entering edit mode on three objects and then changing
    /// which is active must not decode all three again. A mesh that has left
    /// the set is dropped, which is what stops the collection growing with
    /// every mode change.
    ///
    /// A mesh whose session cannot be opened — it is not in the document — is
    /// left out rather than reported. The caller asked for a set, and an
    /// operator reaching for a missing one converts for itself, which is the
    /// same thing that happens outside edit mode.
    pub fn open_for(&mut self, doc: &Document, meshes: &[MeshId]) {
        self.open.retain(|s| meshes.contains(&s.mesh()));
        for mesh in meshes {
            if self.open.iter().any(|s| s.mesh() == *mesh) {
                continue;
            }
            if let Some(session) = EditSession::open(doc, *mesh) {
                self.open.push(session);
            }
        }
        // Back into the caller's order, so operators walk the meshes in the
        // order `edit_targets` named them however the set was arrived at.
        self.open.sort_by_key(|s| {
            meshes
                .iter()
                .position(|m| *m == s.mesh())
                .unwrap_or(usize::MAX)
        });
    }

    pub fn get(&self, mesh: MeshId) -> Option<&EditSession> {
        self.open.iter().find(|s| s.mesh() == mesh)
    }

    pub fn get_mut(&mut self, mesh: MeshId) -> Option<&mut EditSession> {
        self.open.iter_mut().find(|s| s.mesh() == mesh)
    }

    pub fn clear(&mut self) {
        self.open.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};

    fn doc_with_cube() -> (Document, MeshId) {
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc.selection.add(object);
        (doc, mesh)
    }

    #[test]
    fn a_session_opens_on_the_mesh_it_was_asked_for() {
        let (doc, mesh) = doc_with_cube();
        let session = EditSession::open(&doc, mesh).expect("the mesh is there");
        assert_eq!(session.mesh(), mesh);
        assert!(session.matches(&doc, mesh));
    }

    #[test]
    fn a_session_survives_a_selection_change() {
        // Selection is not geometry. Treating a click as staleness costs a full
        // decode on the *next* operation — two seconds at a million vertices,
        // which is worse than having no session at all.
        let (mut doc, mesh) = doc_with_cube();
        let session = EditSession::open(&doc, mesh).expect("the mesh is there");

        kreate_mesh::selection::select_all(doc.mesh_mut(mesh).unwrap());
        assert!(
            session.matches(&doc, mesh),
            "a click must not invalidate the edit structure"
        );
    }

    #[test]
    fn a_session_survives_a_move() {
        // A move is not a change of connectivity, so the structure the session
        // holds is still the right one — it has the wrong numbers in it, and
        // `bmesh_for` copies the right ones across. Treating a move as
        // staleness would make an extrude after a drag pay for decoding the
        // whole mesh again, which is the cost holding a session exists to
        // avoid.
        let (mut doc, mesh) = doc_with_cube();
        let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

        doc.mesh_mut(mesh).unwrap().positions_mut()[0].y += 1.0;
        assert!(session.matches(&doc, mesh), "a move is not staleness");

        let moved = doc.mesh(mesh).unwrap().positions()[0];
        let bm = session.bmesh_for(&doc);
        let first = bm.vert_ids().next().expect("the first vertex");
        let adopted = bm.position(first).expect("it has a position");
        assert_eq!(adopted, moved, "the move was not copied in");
    }

    #[test]
    fn a_session_is_stale_once_the_connectivity_changes() {
        // The check that makes a stale session harmless: a mesh replaced by
        // another route — an undo above all — must not be overwritten by what
        // the session still holds. No amount of copying arrays across makes the
        // held structure describe a different mesh.
        let (mut doc, mesh) = doc_with_cube();
        let session = EditSession::open(&doc, mesh).expect("the mesh is there");

        *doc.mesh_mut(mesh).unwrap() = kreate_mesh::primitives::cylinder(1.0, 2.0, 12);
        assert!(
            !session.matches(&doc, mesh),
            "a session must not claim to be current after a change of topology"
        );
    }

    #[test]
    fn the_structure_picks_up_a_selection_made_elsewhere() {
        // The other half of not invalidating on a click: an operator reading the
        // selection has to see the click, or an extrude acts on nothing.
        let (mut doc, mesh) = doc_with_cube();
        let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");
        assert!(session.bmesh_for(&doc).selected_faces().is_empty());

        kreate_mesh::selection::select_all(doc.mesh_mut(mesh).unwrap());
        assert_eq!(
            session.bmesh_for(&doc).selected_faces().len(),
            6,
            "the structure did not see the selection"
        );
    }

    #[test]
    fn a_session_is_not_used_for_another_mesh() {
        let (mut doc, mesh) = doc_with_cube();
        let other = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let session = EditSession::open(&doc, mesh).expect("the mesh is there");
        assert!(!session.matches(&doc, other));
    }

    #[test]
    fn baking_writes_the_edit_structure_back() {
        let (mut doc, mesh) = doc_with_cube();
        let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

        let faces: Vec<_> = session.bmesh_mut().face_ids().collect();
        kreate_mesh::edit::subdivide_faces(session.bmesh_mut(), &faces).expect("a cube subdivides");
        assert!(session.bake(&mut doc));

        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 24);
        assert!(
            session.matches(&doc, mesh),
            "a session is current again once it has written"
        );
    }

    #[test]
    fn a_session_survives_several_edits_without_reopening() {
        // The point of holding it: the decode happens once, not once per
        // operator.
        let (mut doc, mesh) = doc_with_cube();
        let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

        for _ in 0..3 {
            assert!(session.matches(&doc, mesh), "still current between edits");
            let faces: Vec<_> = session.bmesh_mut().face_ids().collect();
            kreate_mesh::edit::subdivide_faces(session.bmesh_mut(), &faces).expect("subdivides");
            session.bake(&mut doc);
        }
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6 * 4 * 4 * 4);
    }

    #[test]
    fn reopening_picks_up_what_someone_else_did() {
        let (mut doc, mesh) = doc_with_cube();
        let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

        // Someone else replaces the document mesh outright.
        let replaced = kreate_mesh::primitives::uv_sphere(1.0, 8, 4);
        *doc.mesh_mut(mesh).unwrap() = replaced;
        assert!(!session.matches(&doc, mesh));

        assert!(session.reopen(&doc));
        assert!(session.matches(&doc, mesh));
        assert_eq!(
            session.bmesh_mut().face_ids().count(),
            doc.mesh(mesh).unwrap().num_faces() as usize
        );
    }

    #[test]
    fn a_session_outlives_the_object_that_opened_it() {
        // Deleting the object does not delete its mesh — that is what datablock
        // indirection means, and it is why a session keys on the mesh rather
        // than on the object.
        let (mut doc, mesh) = doc_with_cube();
        let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");
        let object = doc.objects().next().unwrap().0;
        doc.remove_object(object);

        assert!(
            session.reopen(&doc),
            "the datablock outlives the object that referenced it"
        );
    }

    #[test]
    fn opening_a_mesh_another_document_holds_gives_nothing() {
        // A handle from elsewhere names nothing here. Returning `None` rather
        // than panicking matters because the caller is the edit-mode toggle, and
        // a document swapped underneath it — a file opened — is ordinary.
        let (_, mesh) = doc_with_cube();
        let empty = Document::new();
        assert!(EditSession::open(&empty, mesh).is_none());
    }
}

#[cfg(test)]
mod session_set_tests {
    use super::*;
    use crate::document::{Object, ObjectData};

    fn doc_with(n: usize) -> (Document, Vec<MeshId>) {
        let mut doc = Document::new();
        let mut meshes = Vec::new();
        for i in 0..n {
            let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0 + i as f32));
            doc.add_object(Object::new(format!("Cube{i}"), ObjectData::Mesh(mesh)));
            meshes.push(mesh);
        }
        (doc, meshes)
    }

    #[test]
    fn a_session_is_opened_for_every_mesh_asked_for() {
        let (doc, meshes) = doc_with(3);
        let mut sessions = EditSessions::new();
        sessions.open_for(&doc, &meshes);
        assert_eq!(sessions.len(), 3);
        for mesh in &meshes {
            assert!(sessions.get(*mesh).is_some(), "no session for {mesh:?}");
        }
    }

    #[test]
    fn opening_again_keeps_the_sessions_that_are_still_wanted() {
        // The reason this is `open_for` and not `open`: a session costs a
        // decode of the whole mesh, and changing which object is active must
        // not pay for all of them again.
        //
        // Observed through *staleness* rather than through the address of the
        // decoded structure, which was the first attempt and does not work: a
        // collection that clears and rebuilds frees each `BMesh` immediately
        // before allocating the next, so the replacement lands at the same
        // address and the comparison passes. A falsification sweep caught that
        // — the break survived.
        //
        // A session knows whether it still agrees with the document. Change the
        // mesh out from under it and a *kept* session stays stale, while one
        // that was decoded again would come back current. Nothing about that
        // depends on the allocator.
        let (mut doc, meshes) = doc_with(3);
        let mut sessions = EditSessions::new();
        sessions.open_for(&doc, &meshes);
        assert!(
            sessions
                .get(meshes[1])
                .is_some_and(|s| s.matches(&doc, meshes[1])),
            "the fixture starts stale, so it cannot tell kept from rebuilt"
        );

        *doc.mesh_mut(meshes[1]).expect("a mesh") = kreate_mesh::primitives::cube(3.0);
        sessions.open_for(&doc, &meshes);
        assert!(
            !sessions
                .get(meshes[1])
                .expect("a session")
                .matches(&doc, meshes[1]),
            "the session was decoded again"
        );
        // And the ones that were not disturbed are still current, so the
        // assertion above is about keeping and not about breaking.
        for mesh in [meshes[0], meshes[2]] {
            assert!(
                sessions.get(mesh).is_some_and(|s| s.matches(&doc, mesh)),
                "session for {mesh:?} went stale on its own"
            );
        }
    }

    #[test]
    fn a_mesh_that_leaves_the_set_loses_its_session() {
        // Otherwise the collection grows with every mode change, holding a
        // `BMesh` — two hundred bytes a vertex — for every mesh ever edited.
        let (doc, meshes) = doc_with(3);
        let mut sessions = EditSessions::new();
        sessions.open_for(&doc, &meshes);

        sessions.open_for(&doc, &meshes[..1]);
        assert_eq!(sessions.len(), 1);
        assert!(sessions.get(meshes[0]).is_some());
        assert!(sessions.get(meshes[2]).is_none());
    }

    #[test]
    fn the_sessions_are_in_the_order_they_were_asked_for() {
        // Operators walk the set, so the order decides which mesh an edit
        // touches first — and a deterministic answer is worth having for free.
        let (doc, meshes) = doc_with(3);
        let mut sessions = EditSessions::new();
        sessions.open_for(&doc, &meshes);
        assert_eq!(sessions.meshes().collect::<Vec<_>>(), meshes);

        // Reversed, and after the set has already been built once — which is
        // the case a `retain`-then-push would get wrong: the kept ones stay
        // where they were and the new ones land at the end.
        let flipped: Vec<MeshId> = meshes.iter().rev().copied().collect();
        sessions.open_for(&doc, &flipped);
        assert_eq!(sessions.meshes().collect::<Vec<_>>(), flipped);
    }

    #[test]
    fn a_mesh_the_document_does_not_have_is_left_out() {
        let (doc, meshes) = doc_with(1);
        let mut other = Document::new();
        let stranger = other.add_mesh(kreate_mesh::primitives::cube(1.0));

        let mut sessions = EditSessions::new();
        sessions.open_for(&doc, &[meshes[0], stranger]);
        assert_eq!(sessions.len(), 1);
        assert!(sessions.get(meshes[0]).is_some());
    }

    #[test]
    fn an_empty_set_closes_everything() {
        let (doc, meshes) = doc_with(2);
        let mut sessions = EditSessions::new();
        sessions.open_for(&doc, &meshes);
        sessions.open_for(&doc, &[]);
        assert!(sessions.is_empty());
    }
}
