//! OFF (Object File Format): Geomview's, and the one geometry-processing
//! papers and datasets use.
//!
//! A word `OFF`, three counts, the vertices, the faces — each face its own
//! vertex count then its indices, n-gons as written. Variants put letters
//! before `OFF` for what each vertex carries beyond its position (`N` a
//! normal, `C` a colour, `ST` a texture coordinate); those are read past.
//! Comments run from `#` to the end of a line.
//!
//! No units or axes, so the same assumptions as PLY, said in the notes.

use crate::ir::{ImportMesh, ImportScene, LossyNote, SourceMetadata};
use glam::Vec3;

#[derive(Debug, thiserror::Error)]
pub enum OffError {
    #[error("not an OFF file: it does not start with `OFF`")]
    NotOff,
    #[error("the file ends before {what}")]
    Truncated { what: String },
    #[error("`{word}` is not a number")]
    Number { word: String },
}

/// Parses an OFF file into the IR: one mesh, at the origin.
pub fn parse_off(text: &str, name: &str) -> Result<ImportScene, OffError> {
    // Comments out first, then one stream of words, each with its line: the
    // counts and vertices may be laid out across lines any way at all, but a
    // face's colour follows its indices on the face's own line.
    let words: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .flat_map(|(n, l)| {
            l.split('#')
                .next()
                .unwrap_or("")
                .split_ascii_whitespace()
                .map(move |w| (n, w))
        })
        .collect();
    let keyword = words.first().map(|w| w.1).ok_or(OffError::NotOff)?;
    let Some(prefix) = keyword.strip_suffix("OFF") else {
        return Err(OffError::NotOff);
    };
    if prefix.contains('4') || prefix.contains('n') {
        // Homogeneous or n-dimensional coordinates: not a surface in space.
        return Err(OffError::NotOff);
    }
    // What each vertex carries after its position, by the prefix's letters.
    let mut per_vertex_extra = 0;
    if prefix.contains("ST") {
        per_vertex_extra += 2;
    }
    if prefix.contains('C') {
        per_vertex_extra += 4;
    }
    if prefix.contains('N') {
        per_vertex_extra += 3;
    }
    let mut at = 1;
    let next = |at: &mut usize, what: &str| -> Result<(usize, f64), OffError> {
        let (line, word) = *words.get(*at).ok_or_else(|| OffError::Truncated {
            what: what.to_string(),
        })?;
        *at += 1;
        let v = word.parse::<f64>().map_err(|_| OffError::Number {
            word: word.to_string(),
        })?;
        Ok((line, v))
    };
    let vertex_count = next(&mut at, "the vertex count")?.1.max(0.0) as usize;
    let face_count = next(&mut at, "the face count")?.1.max(0.0) as usize;
    let _edges = next(&mut at, "the edge count")?;

    let mut source = SourceMetadata::new("off");
    source.notes.push(LossyNote::AssumedUnitScale {
        assumed: "1 metre per unit (OFF declares none)".into(),
    });
    source.notes.push(LossyNote::AssumedAxisConvention {
        assumed: "Z up (OFF declares none)".into(),
    });

    let mut positions = Vec::with_capacity(vertex_count.min(1 << 24));
    for _ in 0..vertex_count {
        let x = next(&mut at, "a vertex")?.1 as f32;
        let y = next(&mut at, "a vertex")?.1 as f32;
        let z = next(&mut at, "a vertex")?.1 as f32;
        for _ in 0..per_vertex_extra {
            next(&mut at, "a vertex")?;
        }
        positions.push(Vec3::new(x, y, z));
    }
    let mut faces = Vec::with_capacity(face_count.min(1 << 24));
    let mut coloured = false;
    for i in 0..face_count {
        let what = format!("face {i}");
        let (line, n) = next(&mut at, &what)?;
        let n = n.max(0.0) as usize;
        let mut face = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            face.push(next(&mut at, &what)?.1.max(0.0) as u32);
        }
        // Anything left on the face's line is its colour.
        while words.get(at).is_some_and(|w| w.0 == line) {
            at += 1;
            coloured = true;
        }
        if face.iter().any(|v| *v as usize >= positions.len()) {
            source.notes.push(LossyNote::OutOfRangeIndex {
                mesh: name.to_string(),
                face: i,
            });
            continue;
        }
        let mut distinct = face.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < 3 {
            source.notes.push(LossyNote::DroppedDegenerateFace {
                mesh: name.to_string(),
                face: i,
            });
            continue;
        }
        faces.push(face);
    }
    if coloured || prefix.contains('C') {
        source.notes.push(LossyNote::Unsupported {
            what: "colours".into(),
        });
    }
    let mut scene = ImportScene::new(source);
    scene.meshes.push(ImportMesh {
        name: name.to_string(),
        positions,
        faces,
        corner_uvs: None,
        corner_normals: None,
        face_materials: None,
        smoothing_groups: None,
    });
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_of_quads_reads_with_its_colours_stepped_over() {
        let text = "OFF
# a unit cube
8 6 12
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
4 0 3 2 1
4 4 5 6 7 255 0 0
4 0 1 5 4
4 1 2 6 5 0.5 0.5 0.5 1.0
4 2 3 7 6
4 3 0 4 7
";
        let scene = parse_off(text, "cube").expect("parses");
        let mesh = &scene.meshes[0];
        assert_eq!(mesh.positions.len(), 8);
        assert_eq!(mesh.faces.len(), 6);
        assert_eq!(mesh.faces[3], vec![1, 2, 6, 5]);
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { .. })));
    }

    #[test]
    fn per_vertex_extras_are_read_past_and_a_short_file_is_an_error() {
        let text = "NOFF 3 1 0\n0 0 0 0 0 1\n1 0 0 0 0 1\n0 1 0 0 0 1\n3 0 1 2\n";
        let scene = parse_off(text, "t").expect("parses");
        assert_eq!(scene.meshes[0].positions[2], Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(scene.meshes[0].faces, vec![vec![0, 1, 2]]);
        assert!(matches!(
            parse_off("OFF 3 1 0\n0 0 0\n", "t"),
            Err(OffError::Truncated { .. })
        ));
        assert!(matches!(parse_off("ply", "t"), Err(OffError::NotOff)));
    }
}
