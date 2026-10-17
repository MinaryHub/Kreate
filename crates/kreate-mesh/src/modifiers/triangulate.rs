//! The triangulate modifier: every face cut into triangles, the way the
//! renderer cuts it — so what goes to a game engine, a printer or a boolean
//! is what was seen.

use crate::attribute::Domain;
use crate::MeshData;

/// Every face of `input` as triangles, its corner and face layers carried
/// to the triangles they came from.
pub fn triangulate(input: &MeshData) -> MeshData {
    if input.is_empty() {
        return input.clone();
    }
    let tri = input.triangulate_corners();
    let corner_verts = input.corner_verts();
    let (tris, _) = tri.corners.as_chunks::<3>();
    let faces: Vec<Vec<u32>> = tris
        .iter()
        .map(|c| c.iter().map(|k| corner_verts[*k as usize]).collect())
        .collect();
    let mut out = MeshData::from_faces(input.positions().to_vec(), &faces);
    let mut attributes = input.attributes().clone();
    attributes.gather_domain(Domain::Corner, &tri.corners);
    attributes.gather_domain(Domain::Face, &tri.tri_to_face);
    // The edges are new; their layers do not carry.
    let edge_layers: Vec<String> = attributes
        .iter_domain(Domain::Edge)
        .map(|l| l.name.to_string())
        .collect();
    for name in edge_layers {
        attributes.remove(&name);
    }
    let edges = out.num_edges() as usize;
    *out.attributes_mut() = attributes;
    out.attributes_mut().resize_domain(Domain::Edge, edges);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_cube_becomes_twelve_triangles_with_its_uvs() {
        let cube = primitives::cube(2.0);
        let out = triangulate(&cube);
        assert_eq!(out.num_faces(), 12);
        assert!(out.faces().all(|f| f.len() == 3));
        assert_eq!(out.num_verts(), 8);
        assert_eq!(
            out.attributes()
                .get_f32x2(crate::attribute::names::UV)
                .map(|u| u.len()),
            Some(36)
        );
        assert!(crate::measure::is_closed(&out));
    }
}
