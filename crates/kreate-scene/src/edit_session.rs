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
    /// A session is current while the document's geometry arrays are still the
    /// ones it decoded — pointer identity, the same free and exact test the
    /// renderer and the modifier cache use. Anything that changes the geometry
    /// by another route, an undo above all, makes it stale, and a session that
    /// trusted itself would silently undo the user's last action.
    ///
    /// **Geometry, not the whole mesh.** A selection click rewrites a layer and
    /// leaves everything else shared, and treating that as staleness costs a
    /// full decode on the *next* operation — two seconds at a million vertices,
    /// which is worse than having no session at all. Selection is instead copied
    /// in when an operator asks for the structure.
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
                .is_some_and(|current| self.source.same_geometry(current))
    }

    /// The edit structure with the document's selection copied in.
    ///
    /// Selection is authoritative in the document — clicks write it there
    /// without decoding anything — so it has to be brought across before an
    /// operator reads it. Three arrays of bools against a two-second decode.
    pub fn bmesh_for(&mut self, doc: &Document) -> &mut BMesh {
        if let Some(current) = doc.mesh(self.mesh) {
            self.bmesh.adopt_selection(current);
        }
        &mut self.bmesh
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
    fn a_session_is_stale_once_the_geometry_changes() {
        // The check that makes a stale session harmless: geometry written by
        // another route — an undo above all — must not be overwritten by what
        // the session still holds.
        let (mut doc, mesh) = doc_with_cube();
        let session = EditSession::open(&doc, mesh).expect("the mesh is there");

        doc.mesh_mut(mesh).unwrap().positions_mut()[0].y += 1.0;
        assert!(
            !session.matches(&doc, mesh),
            "a session must not claim to be current after a geometry change"
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
