//! The weld modifier: vertices closer than a distance merged into one, and
//! the faces that collapse with them dropped — what an import in pieces, a
//! mirror with a seam, or a scan with doubled points wants.

use kreate_math::Vec3;

use crate::attribute::Domain;
use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeldParams {
    /// Vertices nearer than this become one, in the mesh's units.
    pub distance: f32,
}

impl Default for WeldParams {
    fn default() -> Self {
        Self { distance: 0.001 }
    }
}

/// `input` with its near-coincident vertices merged.
///
/// Through a grid of cells the merge distance wide: a vertex looks in its
/// own cell and the ones round it, so two across a cell boundary still
/// meet. The first of a group keeps its place; the rest go to it.
pub fn weld(input: &MeshData, params: &WeldParams) -> MeshData {
    let d = params.distance;
    if input.is_empty() || d <= 0.0 || !d.is_finite() {
        return input.clone();
    }
    let positions = input.positions();
    let n = positions.len();
    let cell = |p: Vec3| -> [i64; 3] {
        [
            (p.x / d).floor() as i64,
            (p.y / d).floor() as i64,
            (p.z / d).floor() as i64,
        ]
    };
    let mut grid: std::collections::HashMap<[i64; 3], Vec<u32>> = std::collections::HashMap::new();
    let mut target: Vec<u32> = (0..n as u32).collect();
    let mut kept: Vec<u32> = Vec::new();
    let mut new_index = vec![u32::MAX; n];
    for (i, p) in positions.iter().enumerate() {
        let c = cell(*p);
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        for j in list {
                            if positions[*j as usize].distance(*p) <= d {
                                found = Some(*j);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        match found {
            Some(j) => target[i] = j,
            None => {
                grid.entry(c).or_default().push(i as u32);
                new_index[i] = kept.len() as u32;
                kept.push(i as u32);
            }
        }
    }
    if kept.len() == n {
        return input.clone();
    }
    let remap = |v: u32| new_index[target[v as usize] as usize];
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut face_source: Vec<u32> = Vec::new();
    let mut corner_source: Vec<u32> = Vec::new();
    for f in 0..input.num_faces() {
        let range = input.face_corner_range(f);
        let verts = input.face_verts(f);
        let mut ring: Vec<u32> = Vec::with_capacity(verts.len());
        let mut corners: Vec<u32> = Vec::with_capacity(verts.len());
        for (k, v) in verts.iter().enumerate() {
            let m = remap(*v);
            // A run of the same vertex is one corner.
            if ring.last() != Some(&m) {
                ring.push(m);
                corners.push(range.start as u32 + k as u32);
            }
        }
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
            corners.pop();
        }
        if ring.len() >= 3 {
            faces.push(ring);
            face_source.push(f);
            corner_source.extend(corners);
        }
    }
    let kept_positions: Vec<Vec3> = kept.iter().map(|i| positions[*i as usize]).collect();
    let mut out = MeshData::from_faces(kept_positions, &faces);
    let mut attributes = input.attributes().clone();
    attributes.gather_domain(Domain::Point, &kept);
    attributes.gather_domain(Domain::Face, &face_source);
    attributes.gather_domain(Domain::Corner, &corner_source);
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
    fn two_halves_meeting_at_a_seam_become_one_closed_mesh() {
        // A cube in two halves, each with its own vertices along the seam.
        let cube = primitives::cube(2.0);
        let positions: Vec<Vec3> = cube.positions().to_vec();
        let mut doubled = positions.clone();
        doubled.extend(positions.iter().copied());
        let faces: Vec<Vec<u32>> = cube
            .faces()
            .enumerate()
            .map(|(i, f)| {
                let shift = if i % 2 == 0 { 0 } else { 8 };
                f.iter().map(|v| v + shift).collect()
            })
            .collect();
        let torn = MeshData::from_faces(doubled, &faces);
        assert!(!crate::measure::is_closed(&torn));
        let out = weld(&torn, &WeldParams { distance: 1e-4 });
        assert_eq!(out.num_verts(), 8);
        assert_eq!(out.num_faces(), 6);
        assert!(crate::measure::is_closed(&out));
        // Nothing near enough: as it was.
        let same = weld(&cube, &WeldParams { distance: 1e-4 });
        assert_eq!(same.num_verts(), 8);
    }

    #[test]
    fn a_face_that_collapses_goes() {
        // A triangle with two corners a hair apart: welded, it is a line.
        let m = MeshData::from_faces(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0001, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2], vec![0, 2, 3]],
        );
        let out = weld(&m, &WeldParams { distance: 0.01 });
        assert_eq!(out.num_verts(), 3);
        assert_eq!(out.num_faces(), 1);
    }
}
