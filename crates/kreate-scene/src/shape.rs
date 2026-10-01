//! Shape keys: the same mesh in other shapes, mixed by weight.
//!
//! Blender's, for faces and corrective shapes. The mesh's own positions are
//! the **basis**; each key is a point layer `shape:<name>` holding how far
//! every vertex moves in that shape, and what is drawn is the basis plus each
//! key's offset times its value.
//!
//! # Offsets, not positions
//!
//! Blender stores a key as absolute positions. Offsets here, for the edit
//! that adds a vertex: a new vertex gets nothing in every layer it did not
//! have, and nothing as an offset is "where the basis has it" — as an
//! absolute position it would be the origin, and the vertex would fly there
//! the moment the key was turned up.
//!
//! # Editing a key
//!
//! With a key active, edit mode edits that shape: on the way in the mesh's
//! positions become the key's shape and the layer keeps the basis; on the way
//! out they are traded back and the layer is the offset again. The object
//! records which key is traded in, so evaluation and a save in the middle
//! both know.
//!
//! Names and values live on the object, so two objects sharing a mesh share
//! its shapes and each has its own expression; the layers live on the mesh.

use std::sync::Arc;

use kreate_math::Vec3;
use kreate_mesh::attribute::{AttrData, Domain};
use kreate_mesh::MeshData;

/// One key on an object: its name and how far it is turned up.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeKey {
    pub name: String,
    pub value: f32,
}

/// The layer a key's offsets are in.
pub fn layer(name: &str) -> String {
    format!("shape:{name}")
}

fn offsets(mesh: &MeshData, name: &str) -> Option<Arc<Vec<Vec3>>> {
    match mesh.attributes().get(&layer(name)).map(|l| &l.data) {
        Some(AttrData::F32x3(v)) if v.len() == mesh.num_verts() as usize => Some(v.clone()),
        _ => None,
    }
}

/// The mesh with the keys mixed in: `None` when no key is turned up, so the
/// caller keeps the mesh it had.
pub fn mix(mesh: &MeshData, keys: &[ShapeKey]) -> Option<MeshData> {
    let live: Vec<(Arc<Vec<Vec3>>, f32)> = keys
        .iter()
        .filter(|k| k.value != 0.0 && k.value.is_finite())
        .filter_map(|k| Some((offsets(mesh, &k.name)?, k.value)))
        .collect();
    if live.is_empty() {
        return None;
    }
    let mut out = mesh.clone();
    for (i, p) in out.positions_mut().iter_mut().enumerate() {
        for (d, v) in &live {
            *p += d[i] * *v;
        }
    }
    Some(out)
}

/// A new key's layer: no offset anywhere — the basis — or, from the mix, the
/// shape the keys make now.
pub fn add_layer(mesh: &mut MeshData, name: &str, from_mix: &[ShapeKey]) {
    let n = mesh.num_verts() as usize;
    let mut d = vec![Vec3::ZERO; n];
    if let Some(mixed) = mix(mesh, from_mix) {
        for (i, (now, base)) in mixed.positions().iter().zip(mesh.positions()).enumerate() {
            d[i] = *now - *base;
        }
    }
    mesh.attributes_mut().insert(&layer(name), Domain::Point, AttrData::F32x3(Arc::new(d)));
}

/// Into edit mode on key `name`: the positions become its shape, and the
/// layer holds the basis meanwhile. `false` when the mesh has no such key.
pub fn trade_in(mesh: &mut MeshData, name: &str) -> bool {
    let Some(d) = offsets(mesh, name) else { return false };
    let basis: Vec<Vec3> = mesh.positions().to_vec();
    for (p, off) in mesh.positions_mut().iter_mut().zip(d.iter()) {
        *p += *off;
    }
    mesh.attributes_mut().insert(&layer(name), Domain::Point, AttrData::F32x3(Arc::new(basis)));
    true
}

/// Out of edit mode on key `name`: what was edited becomes the key's offsets
/// and the basis comes back. A vertex made during the edit has no basis in the
/// layer (it reads as the origin): it joins the basis where it was made, with
/// no offset.
pub fn trade_out(mesh: &mut MeshData, name: &str) -> bool {
    let n = mesh.num_verts() as usize;
    let basis: Vec<Vec3> = match mesh.attributes().get(&layer(name)).map(|l| &l.data) {
        Some(AttrData::F32x3(v)) if v.len() == n => v.to_vec(),
        _ => return false,
    };
    let edited: Vec<Vec3> = mesh.positions().to_vec();
    let mut d = vec![Vec3::ZERO; n];
    let mut base = basis;
    for i in 0..n {
        if base[i] == Vec3::ZERO && edited[i] != Vec3::ZERO {
            base[i] = edited[i];
        }
        d[i] = edited[i] - base[i];
    }
    mesh.positions_mut().copy_from_slice(&base);
    mesh.attributes_mut().insert(&layer(name), Domain::Point, AttrData::F32x3(Arc::new(d)));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str, value: f32) -> ShapeKey {
        ShapeKey { name: name.into(), value }
    }

    #[test]
    fn a_key_moves_the_mesh_by_its_value() {
        let mut m = kreate_mesh::primitives::cube(2.0);
        add_layer(&mut m, "Tall", &[]);
        let fresh = mix(&m, &[key("Tall", 1.0)]).map(|x| x.positions().to_vec());
        assert!(fresh.is_none_or(|p| p == m.positions()), "a new key moved something");
        // Edit the key: the top four up by two.
        assert!(trade_in(&mut m, "Tall"));
        for p in m.positions_mut() {
            if p.z > 0.0 {
                p.z += 2.0;
            }
        }
        assert!(trade_out(&mut m, "Tall"));
        assert_eq!(m.bounds().max.z, 1.0, "the basis did not come back");
        let half = mix(&m, &[key("Tall", 0.5)]).unwrap();
        assert!((half.bounds().max.z - 2.0).abs() < 1e-6);
        let full = mix(&m, &[key("Tall", 1.0)]).unwrap();
        assert!((full.bounds().max.z - 3.0).abs() < 1e-6);
        // A second key from the mix holds what the first made.
        add_layer(&mut m, "Copy", &[key("Tall", 1.0)]);
        let copy = mix(&m, &[key("Copy", 1.0)]).unwrap();
        assert!((copy.bounds().max.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_vertex_added_while_editing_a_key_stays_where_it_was_made() {
        let mut m = kreate_mesh::primitives::plane(2.0);
        add_layer(&mut m, "K", &[]);
        trade_in(&mut m, "K");
        // As an edit that adds a vertex leaves the layers: zero for the new one.
        let mut bm = kreate_mesh::BMesh::from_mesh_data(&m);
        bm.add_vert(Vec3::new(5.0, 5.0, 0.0));
        let mut m = bm.to_mesh_data();
        trade_out(&mut m, "K");
        assert!(m.positions().contains(&Vec3::new(5.0, 5.0, 0.0)), "the new vertex moved");
        let full = mix(&m, &[key("K", 1.0)]);
        assert!(full.is_none() || full.unwrap().positions().contains(&Vec3::new(5.0, 5.0, 0.0)));
    }
}
