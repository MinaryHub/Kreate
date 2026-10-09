//! The sculpted mesh as sculpt mode shows it: its mask in the first UV
//! coordinate and its face sets in the second, where the solid shader's
//! sculpt overlay reads them (`Draw::sculpt_overlay`).
//!
//! The coordinates are kept and handed out again while the mask and the face
//! sets are the same arrays, so a stroke that only moves vertices leaves every
//! layer but the positions as it was — which is what lets the renderer take
//! the cheap path that rewrites positions alone.

use std::sync::Arc;

use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::sculpt::{FACE_SET, MASK};
use kreate_mesh::MeshData;

/// What the shown copy was made from, by address.
type Key = (usize, usize, usize);

#[derive(Default)]
pub struct SculptDisplay {
    last: Option<(Arc<MeshData>, Arc<MeshData>)>,
    coords: Option<(Key, AttrData)>,
}

fn layer_address(mesh: &MeshData, name: &str) -> usize {
    match mesh.attributes().get(name).map(|l| &l.data) {
        Some(AttrData::F32(a)) => Arc::as_ptr(a) as usize,
        Some(AttrData::I32(a)) => Arc::as_ptr(a) as usize,
        _ => 0,
    }
}

impl SculptDisplay {
    /// The copy of `mesh` to draw.
    pub fn show(&mut self, mesh: &Arc<MeshData>) -> Option<Arc<MeshData>> {
        if let Some((input, out)) = &self.last {
            if Arc::ptr_eq(input, mesh) {
                return Some(out.clone());
            }
        }
        let key = (
            layer_address(mesh, MASK),
            layer_address(mesh, FACE_SET),
            mesh.corner_verts().as_ptr() as usize,
        );
        let coords = match &self.coords {
            Some((k, data)) if *k == key => data.clone(),
            _ => {
                let mask = kreate_mesh::sculpt::mask_of(mesh);
                let sets = kreate_mesh::sculpt::face_sets_of(mesh);
                let mut uv = vec![glam::Vec2::ZERO; mesh.corner_verts().len()];
                for f in 0..mesh.num_faces() {
                    let set = sets[f as usize] as f32;
                    for c in mesh.face_corner_range(f) {
                        uv[c] = glam::Vec2::new(mask[mesh.corner_verts()[c] as usize], set);
                    }
                }
                let data = AttrData::F32x2(Arc::new(uv));
                self.coords = Some((key, data.clone()));
                data
            }
        };
        let mut out = (**mesh).clone();
        out.attributes_mut().insert(names::UV, Domain::Corner, coords);
        let out = Arc::new(out);
        self.last = Some((mesh.clone(), out.clone()));
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_and_face_sets_ride_in_the_coordinates_and_are_kept() {
        let mut m = kreate_mesh::primitives::grid(2.0, 2, 2);
        let n = m.num_verts() as usize;
        m.attributes_mut().insert(MASK, Domain::Point, AttrData::F32(Arc::new(vec![1.0; n])));
        let nf = m.num_faces() as usize;
        m.attributes_mut().insert(FACE_SET, Domain::Face, AttrData::I32(Arc::new(vec![3; nf])));
        let m = Arc::new(m);
        let mut d = SculptDisplay::default();
        let shown = d.show(&m).unwrap();
        let uv = shown.attributes().get_f32x2(names::UV).unwrap();
        assert!(uv.iter().all(|c| *c == glam::Vec2::new(1.0, 3.0)));
        // Only the positions moved: the coordinates are the same array.
        let mut moved = (*m).clone();
        moved.positions_mut()[0].z = 1.0;
        let again = d.show(&Arc::new(moved)).unwrap();
        let a = shown.attributes().get_f32x2(names::UV).unwrap().as_ptr();
        let b = again.attributes().get_f32x2(names::UV).unwrap().as_ptr();
        assert_eq!(a, b);
    }
}
