//! Extracting the UV editor's view from the edit structure.
//!
//! A pure function, in the library rather than in the binary, so it can be
//! tested. It was a private method on `App` until a falsification sweep pointed
//! out that nothing exercised it: the panel tests supply their own data, so
//! publishing the wrong numbering entirely would have gone unnoticed.
//!
//! # Why the arena's numbering and not the document's
//!
//! The UV operators take *corner slots* — arena slots, which have gaps in them
//! once anything has been deleted. The document's corners are dense. Publishing
//! dense indices would therefore select the wrong corners on any mesh that had
//! been edited: silently, only sometimes, and only after a delete. So the view
//! is read from the `BMesh` and indexed the way the operators index.

use kreate_mesh::attribute::{names, Domain};
use kreate_mesh::bmesh::BMesh;
use kreate_mesh::ArenaIndex;
use kreate_ui::UvData;

/// What the UV panel should draw for this edit structure.
///
/// `None` when the mesh has no UV layer at all, which the panel says out loud
/// rather than drawing an empty square — that is a different state from "there
/// is no mesh being edited", and telling them apart is the point of returning
/// an `Option` here and another one at the call site.
pub fn from_bmesh(bm: &BMesh) -> Option<UvData> {
    let slots = bm.slot_count(Domain::Corner);
    let uvs = bm.attributes().get_f32x2(names::UV)?;
    let flags = bm.attributes().get_bool(names::SELECT_UV);

    let mut data = UvData {
        // Indexed by slot, including the free ones: the panel looks corners up
        // by slot, so the array has to be slot-shaped even where that means
        // carrying a value nothing points at. Cheaper than a second indirection
        // on every lookup.
        uvs: (0..slots)
            .map(|i| {
                let uv = uvs.get(i).copied().unwrap_or_default();
                egui::vec2(uv.x, uv.y)
            })
            .collect(),
        selected: (0..slots)
            .map(|i| flags.and_then(|f| f.get(i)).copied().unwrap_or(false))
            .collect(),
        offsets: Vec::with_capacity(bm.num_faces() + 1),
        corners: Vec::new(),
        pinned: {
            let pins = bm.attributes().get_bool(names::UV_PIN);
            (0..slots)
                .map(|i| pins.and_then(|f| f.get(i)).copied().unwrap_or(false))
                .collect()
        },
    };

    data.offsets.push(0);
    for f in bm.face_ids() {
        for l in bm.face_loops(f) {
            data.corners.push(l.index());
        }
        data.offsets.push(data.corners.len() as u32);
    }
    Some(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_mesh::edit::uv_project::{self, Projection};
    use kreate_mesh::edit::{uv_edit, DeleteMode};
    use kreate_mesh::{arena::FaceId, primitives};

    fn cube_with_uvs() -> BMesh {
        let mesh = primitives::cube(2.0);
        let mut bm = BMesh::from_mesh_data(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        uv_project::project(&mut bm, &faces, Projection::Box, uv_project::DEFAULT_LAYER);
        bm
    }

    #[test]
    fn a_mesh_without_uvs_publishes_nothing() {
        // The layer is removed rather than a bare primitive used: primitives
        // arrive with coordinates now, and a fixture that relied on them not
        // doing so was relying on a gap rather than on a rule. What is being
        // checked is that a mesh *without* a layer shows nothing, which is
        // still a state a mesh can be in — an imported one, or one whose layer
        // somebody deleted.
        let mut mesh = primitives::cube(2.0);
        mesh.attributes_mut().remove(uv_project::DEFAULT_LAYER);
        let bm = BMesh::from_mesh_data(&mesh);
        assert!(from_bmesh(&bm).is_none());
    }

    #[test]
    fn every_face_is_published_with_its_own_corners() {
        let bm = cube_with_uvs();
        let data = from_bmesh(&bm).expect("the cube has UVs");
        assert_eq!(data.offsets.len(), 7, "six faces plus the end offset");
        assert_eq!(data.corners.len(), 24);

        let view = data.view();
        assert_eq!(view.face_count(), 6);
    }

    #[test]
    fn the_published_indices_are_arena_slots_and_not_dense_positions() {
        // The defect this module exists to prevent, and the one the panel's own
        // tests cannot see because they supply their own data.
        //
        // Deleting a face frees its loop slots, so the arena's numbering gains
        // a gap while a dense numbering would close it. Publishing dense
        // indices would hand the operators corner numbers that name *different*
        // corners — quietly, and only on a mesh that had been edited.
        let mut bm = cube_with_uvs();
        let doomed = bm.face_ids().next().unwrap();
        bm.set_face_selected(doomed, true);
        kreate_mesh::edit::delete(&mut bm, DeleteMode::Faces);

        let data = from_bmesh(&bm).expect("the rest still has UVs");
        // The premise: after a delete the two numberings genuinely differ, so
        // the assertion below can fail.
        let max = data.corners.iter().copied().max().unwrap_or(0) as usize;
        assert!(
            max >= data.corners.len(),
            "no gap was left in the arena, so this test proves nothing: \
             {} corners, highest slot {max}",
            data.corners.len()
        );

        // Every published index resolves to a live corner, and to the one the
        // face actually owns.
        let mut expected: Vec<u32> = Vec::new();
        for f in bm.face_ids() {
            for l in bm.face_loops(f) {
                expected.push(l.index());
            }
        }
        assert_eq!(data.corners, expected);
        for slot in &data.corners {
            assert!(
                bm.corner_at(*slot).is_some(),
                "published slot {slot} does not resolve to a corner"
            );
        }
    }

    #[test]
    fn the_arrays_are_slot_shaped_so_a_lookup_by_slot_lands() {
        // A gap in the arena means the highest slot exceeds the corner count,
        // and the panel indexes by slot. Arrays sized to the *count* would put
        // every corner past the gap out of range, and the panel would silently
        // draw nothing for them.
        let mut bm = cube_with_uvs();
        let doomed = bm.face_ids().next().unwrap();
        bm.set_face_selected(doomed, true);
        kreate_mesh::edit::delete(&mut bm, DeleteMode::Faces);

        let data = from_bmesh(&bm).unwrap();
        let slots = bm.slot_count(Domain::Corner);
        assert_eq!(data.uvs.len(), slots);
        assert_eq!(data.selected.len(), slots);
        for slot in &data.corners {
            assert!((*slot as usize) < data.uvs.len());
        }
    }

    #[test]
    fn the_selection_is_published_against_the_same_slots() {
        let mut bm = cube_with_uvs();
        let first = bm.face_ids().next().unwrap();
        let chosen = bm.face_loops(first);
        uv_edit::set_selection(&mut bm, &chosen, uv_edit::SelectAction::Replace);

        let data = from_bmesh(&bm).unwrap();
        let selected: Vec<u32> = data
            .corners
            .iter()
            .copied()
            .filter(|slot| data.selected[*slot as usize])
            .collect();
        let mut expected: Vec<u32> = chosen.iter().map(|l| l.index()).collect();
        expected.sort_unstable();
        let mut got = selected;
        got.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn the_uvs_are_published_as_they_are_stored() {
        // No flip here. The V flip belongs to the *screen* mapping, and every
        // format's flip belongs to the IO boundary; a third one in between
        // would be a third place to get it wrong.
        let bm = cube_with_uvs();
        let stored = bm
            .attributes()
            .get_f32x2(uv_project::DEFAULT_LAYER)
            .unwrap();
        let data = from_bmesh(&bm).unwrap();
        for slot in &data.corners {
            let i = *slot as usize;
            assert_eq!(data.uvs[i].x, stored[i].x);
            assert_eq!(data.uvs[i].y, stored[i].y);
        }
    }
}
