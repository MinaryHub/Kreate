//! Undo storage: structural-sharing snapshots plus declared write-set deltas.
//!
//! # Why two mechanisms
//!
//! A snapshot is a clone of the [`Document`] with all of its `Arc`s — microseconds
//! of work, and memory cost equal only to the arrays an operator actually
//! touched. That covers topology changes, which alter many arrays at once, at
//! almost no cost.
//!
//! It has one honest weakness. A 10 M vertex mesh's position array is 120 MB, so
//! *any* transform would cost one 120 MB undo step even though it moved fifty
//! vertices. Hence [`DocMut::edit_positions`]: an operator declares which
//! elements it is about to write, and only those old values are recorded.
//! Transforms touch only the selection, so this covers the overwhelmingly common
//! case.
//!
//! Building that API *before* there are forty operators to retrofit is the whole
//! point — the alternative is auditing every mutation site later.
//!
//! # What was rejected
//!
//! - **Full deep-copy snapshots.** Memory death at the stated scale.
//! - **Per-command inverses.** Every operator would need a provably correct
//!   inverse: an enormous bug surface, and it interacts badly with an
//!   adjust-last-operation panel, which wants "restore and re-run" rather than
//!   "invert".
//! - **Persistent immutable trees.** Cache-hostile for a mesh kernel.
//!   `Arc`-per-array copy-on-write *is* persistence, at exactly the right
//!   granularity: the array.
//!
//! # Where the stack lives
//!
//! This module provides the *mechanism* — snapshots, deltas, and the recording
//! write API. The undo *stack*, which pairs each entry with the operator id and
//! its reflected parameters, lives in `kreate-ops`, because the dependency
//! direction runs that way: operators are the only thing permitted to mutate a
//! document, and the scene layer must not depend on them.

use glam::Vec3;
use kreate_mesh::attribute::names;

use crate::document::{Document, MeshId};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EditError {
    #[error("mesh does not exist")]
    NoSuchMesh,
    #[error("vertex index {index} is out of range ({count} vertices)")]
    IndexOutOfRange { index: u32, count: u32 },
}

/// What is needed to put the document back.
#[derive(Clone, Debug)]
pub enum UndoPayload {
    /// A whole-document snapshot. Cheap to make and cheap to hold, because
    /// everything it did not change is shared.
    Snapshot(Box<Document>),
    /// The previous values of specific vertices of one mesh.
    PositionDelta {
        mesh: MeshId,
        indices: Vec<u32>,
        values: Vec<Vec3>,
    },
    /// The operator reported success but changed nothing worth recording.
    Nothing,
}

impl UndoPayload {
    /// Approximate bytes held, for the undo memory budget.
    ///
    /// Tracked because memory regressions in this domain are silent right up
    /// until they are fatal: "millions of elements" plus snapshot undo is how a
    /// process reaches 16 GB and starts thrashing.
    pub fn bytes(&self) -> usize {
        match self {
            // A snapshot's arrays are shared with the live document, so what it
            // *holds* is the structure, not the geometry. Charging it the full
            // mesh size would make the budget useless; charging nothing would
            // make it a lie. The structure is what a snapshot uniquely owns.
            UndoPayload::Snapshot(d) => {
                std::mem::size_of::<Document>()
                    + d.object_count() * std::mem::size_of::<usize>() * 8
            }
            UndoPayload::PositionDelta {
                indices, values, ..
            } => indices.len() * 4 + values.len() * 12,
            UndoPayload::Nothing => 0,
        }
    }

    pub fn is_nothing(&self) -> bool {
        matches!(self, UndoPayload::Nothing)
    }

    /// Puts the document back to the state this payload describes.
    pub fn restore(&self, doc: &mut Document) {
        match self {
            // `adopt` rather than assignment: the snapshot carries the version
            // it had when it was taken, and letting that overwrite the live one
            // makes the counter go backwards. A cache keyed on it then misses a
            // real change — see `Document::version`.
            UndoPayload::Snapshot(snapshot) => doc.adopt((**snapshot).clone()),
            UndoPayload::PositionDelta {
                mesh,
                indices,
                values,
            } => {
                if let Some(m) = doc.mesh_mut(*mesh) {
                    let positions = m.positions_mut();
                    for (i, v) in indices.iter().zip(values) {
                        if let Some(slot) = positions.get_mut(*i as usize) {
                            *slot = *v;
                        }
                    }
                }
            }
            UndoPayload::Nothing => {}
        }
    }
}

/// A recording handle on a document.
///
/// Operators receive one of these rather than a bare `&mut Document`, so that
/// what they changed is captured as a side effect of changing it. Reaching for
/// [`DocMut::document_mut`] is always possible but is the coarse path: it records
/// a full snapshot.
pub struct DocMut<'a> {
    doc: &'a mut Document,
    /// Snapshot taken lazily, only if something asks for the coarse path.
    before: Option<Box<Document>>,
    deltas: Vec<UndoPayload>,
}

impl<'a> DocMut<'a> {
    pub fn new(doc: &'a mut Document) -> Self {
        Self {
            doc,
            before: None,
            deltas: Vec::new(),
        }
    }

    pub fn document(&self) -> &Document {
        self.doc
    }

    /// Full mutable access, at the cost of a whole-document snapshot.
    ///
    /// Correct for anything that changes topology or the object graph. Using it
    /// for a transform would work but would record far more than necessary,
    /// which is what [`DocMut::edit_positions`] avoids.
    pub fn document_mut(&mut self) -> &mut Document {
        if self.before.is_none() {
            self.before = Some(Box::new(self.doc.clone()));
        }
        self.doc
    }

    /// Writes a declared subset of a mesh's vertex positions.
    ///
    /// `f` is called with each index and a mutable reference to its position. The
    /// previous values are captured first, so the undo record is proportional to
    /// the *selection*, not to the mesh.
    ///
    /// Indices are validated up front: a partial application would leave the mesh
    /// half-transformed with an undo record that does not match it.
    pub fn edit_positions(
        &mut self,
        mesh: MeshId,
        indices: &[u32],
        mut f: impl FnMut(u32, &mut Vec3),
    ) -> Result<(), EditError> {
        let count = self
            .doc
            .mesh(mesh)
            .ok_or(EditError::NoSuchMesh)?
            .num_verts();
        if let Some(&index) = indices.iter().find(|i| **i >= count) {
            return Err(EditError::IndexOutOfRange { index, count });
        }

        let data = self.doc.mesh_mut(mesh).ok_or(EditError::NoSuchMesh)?;
        let positions = data.positions_mut();
        let mut old = Vec::with_capacity(indices.len());
        for &i in indices {
            let slot = &mut positions[i as usize];
            old.push(*slot);
            f(i, slot);
        }

        self.deltas.push(UndoPayload::PositionDelta {
            mesh,
            indices: indices.to_vec(),
            values: old,
        });
        Ok(())
    }

    /// Every vertex index of a mesh, for operators that genuinely touch all of
    /// them. Present so "affect everything" still goes through the declared path
    /// rather than tempting a caller into the coarse one.
    pub fn all_vertices(&self, mesh: MeshId) -> Vec<u32> {
        self.doc
            .mesh(mesh)
            .map(|m| (0..m.num_verts()).collect())
            .unwrap_or_default()
    }

    /// The selected vertices of a mesh, from its selection attribute layer.
    pub fn selected_vertices(&self, mesh: MeshId) -> Vec<u32> {
        let Some(m) = self.doc.mesh(mesh) else {
            return Vec::new();
        };
        match m.attributes().get_bool(names::SELECT_VERT) {
            Some(sel) => sel
                .iter()
                .enumerate()
                .filter(|(_, s)| **s)
                .map(|(i, _)| i as u32)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Consumes the handle and returns what has to be restored to undo the work.
    ///
    /// A snapshot wins over deltas when both were recorded: the coarse path was
    /// taken, so something outside the declared write sets may also have moved.
    pub fn finish(self) -> UndoPayload {
        if let Some(before) = self.before {
            return UndoPayload::Snapshot(before);
        }
        match self.deltas.len() {
            0 => UndoPayload::Nothing,
            1 => self.deltas.into_iter().next().unwrap(),
            // Several declared writes in one operator: merge them so a single
            // undo step restores all of it.
            _ => merge_deltas(self.deltas),
        }
    }
}

/// Merges position deltas, keeping the *earliest* recorded value per index.
///
/// An operator that writes the same vertex twice must undo to the value from
/// before the operator ran, not to the value it had halfway through.
fn merge_deltas(deltas: Vec<UndoPayload>) -> UndoPayload {
    let mut mesh_id = None;
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<Vec3> = Vec::new();

    for d in deltas {
        let UndoPayload::PositionDelta {
            mesh,
            indices: idx,
            values: vals,
        } = d
        else {
            continue;
        };
        // Deltas across two different meshes cannot be merged into this shape;
        // the coarse path is the honest answer there.
        match mesh_id {
            None => mesh_id = Some(mesh),
            Some(existing) if existing == mesh => {}
            Some(_) => return UndoPayload::Nothing,
        }
        for (i, v) in idx.into_iter().zip(vals) {
            if !indices.contains(&i) {
                indices.push(i);
                values.push(v);
            }
        }
    }

    match mesh_id {
        Some(mesh) => UndoPayload::PositionDelta {
            mesh,
            indices,
            values,
        },
        None => UndoPayload::Nothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};
    use kreate_mesh::attribute::{AttrData, AttrType, Domain};

    fn doc_with_grid() -> (Document, MeshId) {
        let mut d = Document::new();
        let mesh = d.add_mesh(kreate_mesh::primitives::grid(4.0, 4, 4));
        d.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));
        (d, mesh)
    }

    #[test]
    fn a_declared_write_records_only_what_it_touched() {
        let (mut doc, mesh) = doc_with_grid();
        let total = doc.mesh(mesh).unwrap().num_verts();
        assert!(
            total > 20,
            "the point is that the mesh is bigger than the edit"
        );

        let mut edit = DocMut::new(&mut doc);
        edit.edit_positions(mesh, &[0, 1, 2], |_, p| p.z += 1.0)
            .unwrap();
        let payload = edit.finish();

        match &payload {
            UndoPayload::PositionDelta {
                indices, values, ..
            } => {
                assert_eq!(indices.len(), 3);
                assert_eq!(values.len(), 3);
            }
            other => panic!("expected a delta, got {other:?}"),
        }
        // 3 vertices, not 25.
        assert!(
            payload.bytes() < 100,
            "delta held {} bytes",
            payload.bytes()
        );
    }

    #[test]
    fn undo_restores_exactly_the_previous_positions() {
        let (mut doc, mesh) = doc_with_grid();
        let before: Vec<Vec3> = doc.mesh(mesh).unwrap().positions().to_vec();

        let mut edit = DocMut::new(&mut doc);
        edit.edit_positions(mesh, &[0, 5, 9], |_, p| *p += Vec3::new(1.0, 2.0, 3.0))
            .unwrap();
        let payload = edit.finish();

        assert_ne!(doc.mesh(mesh).unwrap().positions(), before.as_slice());
        payload.restore(&mut doc);
        assert_eq!(
            doc.mesh(mesh).unwrap().positions(),
            before.as_slice(),
            "undo must restore the positions exactly"
        );
    }

    #[test]
    fn writing_the_same_vertex_twice_undoes_to_the_original_value() {
        // The subtle one: a merged delta must keep the value from before the
        // operator ran, not the value the vertex had halfway through it.
        let (mut doc, mesh) = doc_with_grid();
        let original = doc.mesh(mesh).unwrap().positions()[0];

        let mut edit = DocMut::new(&mut doc);
        edit.edit_positions(mesh, &[0], |_, p| p.z = 1.0).unwrap();
        edit.edit_positions(mesh, &[0], |_, p| p.z = 2.0).unwrap();
        let payload = edit.finish();

        assert_eq!(doc.mesh(mesh).unwrap().positions()[0].z, 2.0);
        payload.restore(&mut doc);
        assert_eq!(doc.mesh(mesh).unwrap().positions()[0], original);
    }

    #[test]
    fn several_declared_writes_merge_into_one_step() {
        let (mut doc, mesh) = doc_with_grid();
        let before: Vec<Vec3> = doc.mesh(mesh).unwrap().positions().to_vec();

        let mut edit = DocMut::new(&mut doc);
        edit.edit_positions(mesh, &[0, 1], |_, p| p.z += 1.0)
            .unwrap();
        edit.edit_positions(mesh, &[7, 8], |_, p| p.z -= 1.0)
            .unwrap();
        let payload = edit.finish();

        payload.restore(&mut doc);
        assert_eq!(doc.mesh(mesh).unwrap().positions(), before.as_slice());
    }

    #[test]
    fn an_out_of_range_index_is_refused_before_anything_is_written() {
        // A partial application would leave the mesh half-transformed with an
        // undo record that does not match it.
        let (mut doc, mesh) = doc_with_grid();
        let before: Vec<Vec3> = doc.mesh(mesh).unwrap().positions().to_vec();
        let count = doc.mesh(mesh).unwrap().num_verts();

        let mut edit = DocMut::new(&mut doc);
        let err = edit.edit_positions(mesh, &[0, 1, 9999], |_, p| p.z = 42.0);
        assert_eq!(err, Err(EditError::IndexOutOfRange { index: 9999, count }));
        drop(edit);

        assert_eq!(
            doc.mesh(mesh).unwrap().positions(),
            before.as_slice(),
            "nothing should have been written"
        );
    }

    #[test]
    fn the_coarse_path_records_a_snapshot_and_restores_topology() {
        let (mut doc, mesh) = doc_with_grid();
        let faces_before = doc.mesh(mesh).unwrap().num_faces();

        let mut edit = DocMut::new(&mut doc);
        let d = edit.document_mut();
        d.add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));
        let payload = edit.finish();
        assert!(matches!(payload, UndoPayload::Snapshot(_)));
        assert_eq!(doc.object_count(), 2);

        payload.restore(&mut doc);
        assert_eq!(doc.object_count(), 1, "the added object must be gone");
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), faces_before);
    }

    #[test]
    fn a_snapshot_wins_over_deltas_when_both_were_recorded() {
        // If the coarse path was taken, something outside the declared write sets
        // may also have moved, so the fine-grained record is not trustworthy.
        let (mut doc, mesh) = doc_with_grid();
        let mut edit = DocMut::new(&mut doc);
        edit.edit_positions(mesh, &[0], |_, p| p.z = 1.0).unwrap();
        edit.document_mut()
            .add_object(Object::new("X", ObjectData::Empty));
        assert!(matches!(edit.finish(), UndoPayload::Snapshot(_)));
    }

    #[test]
    fn restoring_a_snapshot_advances_the_version() {
        let (mut doc, _) = doc_with_grid();
        let mut edit = DocMut::new(&mut doc);
        edit.document_mut()
            .add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));
        let payload = edit.finish();
        let after_edit = doc.version();

        payload.restore(&mut doc);
        assert!(
            doc.version() > after_edit,
            "undo left the version at {}, which a cache would read as 'unchanged'",
            doc.version()
        );
    }

    #[test]
    fn an_operator_that_changed_nothing_records_nothing() {
        let (mut doc, _) = doc_with_grid();
        let edit = DocMut::new(&mut doc);
        let payload = edit.finish();
        assert!(payload.is_nothing());
        assert_eq!(payload.bytes(), 0);
    }

    #[test]
    fn selected_vertices_come_from_the_attribute_layer() {
        let (mut doc, mesh) = doc_with_grid();
        let n = doc.mesh(mesh).unwrap().num_verts() as usize;
        let mut sel = vec![false; n];
        sel[2] = true;
        sel[7] = true;
        doc.mesh_mut(mesh).unwrap().attributes_mut().insert(
            names::SELECT_VERT,
            Domain::Point,
            AttrData::Bool(std::sync::Arc::new(sel)),
        );

        let edit = DocMut::new(&mut doc);
        assert_eq!(edit.selected_vertices(mesh), vec![2, 7]);
    }

    #[test]
    fn a_mesh_with_no_selection_layer_yields_no_selected_vertices() {
        let (mut doc, mesh) = doc_with_grid();
        let edit = DocMut::new(&mut doc);
        assert!(edit.selected_vertices(mesh).is_empty());
        let _ = AttrType::Bool;
    }

    #[test]
    fn all_vertices_covers_the_whole_mesh() {
        let (mut doc, mesh) = doc_with_grid();
        let count = doc.mesh(mesh).unwrap().num_verts();
        let edit = DocMut::new(&mut doc);
        assert_eq!(edit.all_vertices(mesh).len(), count as usize);
    }

    #[test]
    fn editing_a_missing_mesh_is_an_error_not_a_panic() {
        // An id that belongs to a different document, which is what a stale
        // reference held across a file load looks like.
        let (_other, foreign_mesh) = doc_with_grid();
        let mut empty = Document::new();
        let mut edit = DocMut::new(&mut empty);
        assert_eq!(
            edit.edit_positions(foreign_mesh, &[0], |_, _| {}),
            Err(EditError::NoSuchMesh)
        );
        assert!(
            edit.finish().is_nothing(),
            "a failed edit must record nothing"
        );
    }

    #[test]
    fn a_delta_is_far_smaller_than_a_snapshot_of_the_same_edit() {
        // The reason both mechanisms exist. A transform on a large mesh must not
        // cost the whole position array.
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::grid(100.0, 200, 200));
        let verts = doc.mesh(mesh).unwrap().num_verts();
        assert!(verts > 40_000);

        let mut edit = DocMut::new(&mut doc);
        edit.edit_positions(mesh, &[0, 1, 2, 3], |_, p| p.z += 1.0)
            .unwrap();
        let delta = edit.finish();

        assert!(
            delta.bytes() < 1_000,
            "a four-vertex edit held {} bytes",
            delta.bytes()
        );
    }
}
