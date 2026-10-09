//! PLY (the Stanford polygon format), in all three of its encodings.
//!
//! What scanners and photogrammetry write, and what point-cloud and research
//! tools pass around. A header names the file's elements and the properties
//! each one has, in order and with their types; then the elements follow as
//! text, or as little- or big-endian binary. So a reader reads the header and
//! then does exactly what it said — there is no fixed layout to assume.
//!
//! # What is read
//!
//! Vertices (`x y z`, and `nx ny nz` and `s t` / `u v` / `texture_u
//! texture_v` when present) and faces (`vertex_indices` or `vertex_index`,
//! n-gons as written). Every other element and property is stepped over by its
//! declared type, so a file with extra data still reads; colours per vertex are
//! named in the notes rather than dropped silently.
//!
//! # What has to be assumed
//!
//! PLY says nothing of units or axes. Like STL, it is taken as Z up and in
//! metres, and the notes say so: a scan that arrives lying on its back is
//! one rotation away, a scan that arrives a hundred times too big is
//! something a user should be told was a guess.

use crate::ir::{ImportMesh, ImportScene, LossyNote, SourceMetadata};
use glam::{Vec2, Vec3};

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("not a PLY file: it does not start with `ply`")]
    NotPly,
    #[error("header line {line}: {what}")]
    Header { line: usize, what: String },
    #[error("the {element} data ends early, at item {item}")]
    Truncated { element: String, item: usize },
    #[error("the {element} data has something that is not a number: {what}")]
    Number { element: String, what: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    Ascii,
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
struct Property {
    name: String,
    /// `Some(count type)` for a list.
    list: Option<Scalar>,
    ty: Scalar,
}

#[derive(Clone, Debug)]
struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

/// Reads one value at a time from whichever encoding the body is in.
struct Reader<'a> {
    encoding: Encoding,
    bytes: &'a [u8],
    at: usize,
    /// For ASCII: the body's words, one value each.
    words: std::str::SplitAsciiWhitespace<'a>,
}

impl Reader<'_> {
    fn value(&mut self, ty: Scalar) -> Result<f64, String> {
        if self.encoding == Encoding::Ascii {
            let word = self.words.next().ok_or("the file ends early")?;
            return word
                .parse::<f64>()
                .map_err(|_| format!("`{word}` is not a number"));
        }
        let n = ty.size();
        let Some(raw) = self.bytes.get(self.at..self.at + n) else {
            return Err("the file ends early".into());
        };
        self.at += n;
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(raw);
        if self.encoding == Encoding::Big {
            b[..n].reverse();
        }
        Ok(match ty {
            Scalar::I8 => b[0] as i8 as f64,
            Scalar::U8 => b[0] as f64,
            Scalar::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Scalar::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            Scalar::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Scalar::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Scalar::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Scalar::F64 => f64::from_le_bytes(b),
        })
    }
}

/// Parses a PLY file into the IR: one mesh, at the origin.
pub fn parse_ply(bytes: &[u8], name: &str) -> Result<ImportScene, PlyError> {
    if !bytes.starts_with(b"ply") {
        return Err(PlyError::NotPly);
    }
    // The header is text up to `end_header` and its line break, whatever the
    // body is.
    let marker = b"end_header";
    let end = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .ok_or(PlyError::Header {
            line: 0,
            what: "no `end_header`".into(),
        })?;
    let mut body = end + marker.len();
    // The one line break after it: `\n` or `\r\n`.
    if bytes.get(body) == Some(&b'\r') {
        body += 1;
    }
    if bytes.get(body) == Some(&b'\n') {
        body += 1;
    }
    let header = String::from_utf8_lossy(&bytes[..end]);

    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    for (i, line) in header.lines().enumerate() {
        let line_no = i + 1;
        let bad = |what: &str| PlyError::Header {
            line: line_no,
            what: what.to_string(),
        };
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["format", form, _version] => {
                encoding = Some(match *form {
                    "ascii" => Encoding::Ascii,
                    "binary_little_endian" => Encoding::Little,
                    "binary_big_endian" => Encoding::Big,
                    other => return Err(bad(&format!("unknown format `{other}`"))),
                })
            }
            ["element", name, count] => elements.push(Element {
                name: name.to_string(),
                count: count
                    .parse()
                    .map_err(|_| bad(&format!("`{count}` is not a count")))?,
                properties: Vec::new(),
            }),
            ["property", "list", count, ty, name] => {
                let property = Property {
                    name: name.to_string(),
                    list: Some(
                        Scalar::parse(count)
                            .ok_or_else(|| bad(&format!("unknown type `{count}`")))?,
                    ),
                    ty: Scalar::parse(ty).ok_or_else(|| bad(&format!("unknown type `{ty}`")))?,
                };
                elements
                    .last_mut()
                    .ok_or_else(|| bad("a property before any element"))?
                    .properties
                    .push(property);
            }
            ["property", ty, name] => {
                let property = Property {
                    name: name.to_string(),
                    list: None,
                    ty: Scalar::parse(ty).ok_or_else(|| bad(&format!("unknown type `{ty}`")))?,
                };
                elements
                    .last_mut()
                    .ok_or_else(|| bad("a property before any element"))?
                    .properties
                    .push(property);
            }
            _ => {}
        }
    }
    let encoding = encoding.ok_or(PlyError::Header {
        line: 0,
        what: "no `format` line".into(),
    })?;

    let text = if encoding == Encoding::Ascii {
        std::str::from_utf8(&bytes[body..]).unwrap_or("")
    } else {
        ""
    };
    let mut reader = Reader {
        encoding,
        bytes,
        at: body,
        words: text.split_ascii_whitespace(),
    };

    let mut source = SourceMetadata::new("ply");
    source.notes.push(LossyNote::AssumedUnitScale {
        assumed: "1 metre per unit (PLY declares none)".into(),
    });
    source.notes.push(LossyNote::AssumedAxisConvention {
        assumed: "Z up (PLY declares none)".into(),
    });

    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut coloured = false;
    for element in &elements {
        let is_vertex = element.name == "vertex";
        let is_face = element.name == "face";
        let find = |names: &[&str]| {
            element
                .properties
                .iter()
                .position(|p| names.contains(&p.name.as_str()))
        };
        let (px, py, pz) = (find(&["x"]), find(&["y"]), find(&["z"]));
        let (nx, ny, nz) = (find(&["nx"]), find(&["ny"]), find(&["nz"]));
        let tu = find(&["s", "u", "texture_u", "texture_s"]);
        let tv = find(&["t", "v", "texture_v", "texture_t"]);
        let has_normals = is_vertex && nx.is_some() && ny.is_some() && nz.is_some();
        let has_uvs = is_vertex && tu.is_some() && tv.is_some();
        coloured |= is_vertex && find(&["red", "r", "diffuse_red"]).is_some();
        let indices = find(&["vertex_indices", "vertex_index"]);
        if is_vertex {
            positions.reserve(element.count.min(1 << 24));
        }
        let mut values = vec![0.0f64; element.properties.len()];
        for item in 0..element.count {
            let fail = |what: String| {
                if what.contains("ends early") {
                    PlyError::Truncated {
                        element: element.name.clone(),
                        item,
                    }
                } else {
                    PlyError::Number {
                        element: element.name.clone(),
                        what,
                    }
                }
            };
            let mut face = Vec::new();
            for (k, property) in element.properties.iter().enumerate() {
                match property.list {
                    None => values[k] = reader.value(property.ty).map_err(fail)?,
                    Some(count_ty) => {
                        let n = reader.value(count_ty).map_err(fail)?;
                        let n = if n.is_finite() && n >= 0.0 {
                            n as usize
                        } else {
                            0
                        };
                        let keep = is_face && Some(k) == indices;
                        for _ in 0..n {
                            let v = reader.value(property.ty).map_err(fail)?;
                            if keep {
                                face.push(v.max(0.0) as u32);
                            }
                        }
                    }
                }
            }
            if is_vertex {
                let get = |i: Option<usize>| i.map_or(0.0, |i| values[i] as f32);
                positions.push(Vec3::new(get(px), get(py), get(pz)));
                if has_normals {
                    normals.push(Vec3::new(get(nx), get(ny), get(nz)));
                }
                if has_uvs {
                    uvs.push(Vec2::new(get(tu), get(tv)));
                }
            } else if is_face {
                faces.push(face);
            }
        }
    }
    if coloured {
        source.notes.push(LossyNote::Unsupported {
            what: "vertex colours".into(),
        });
    }

    // Faces naming a vertex that is not there, or fewer than three distinct
    // ones, are dropped and said to be.
    let count = positions.len() as u32;
    let mut kept = Vec::with_capacity(faces.len());
    for (i, face) in faces.into_iter().enumerate() {
        if face.iter().any(|v| *v >= count) {
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
        kept.push(face);
    }
    // Per vertex in the file, per corner in the IR.
    let per_corner = |values: &[Vec3]| -> Vec<Vec3> {
        kept.iter()
            .flat_map(|f| f.iter().map(|v| values[*v as usize]))
            .collect()
    };
    let corner_normals =
        (normals.len() == positions.len() && !normals.is_empty()).then(|| per_corner(&normals));
    let corner_uvs = (uvs.len() == positions.len() && !uvs.is_empty()).then(|| {
        kept.iter()
            .flat_map(|f| f.iter().map(|v| uvs[*v as usize]))
            .collect()
    });
    let mut scene = ImportScene::new(source);
    scene.meshes.push(ImportMesh {
        name: name.to_string(),
        positions,
        faces: kept,
        corner_uvs,
        corner_normals,
        face_materials: None,
        smoothing_groups: None,
    });
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUAD: &str = "ply
format ascii 1.0
comment a unit square and a triangle on it
element vertex 5
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float s
property float t
property uchar red
property uchar green
property uchar blue
element face 2
property list uchar int vertex_indices
end_header
0 0 0 0 0 1 0 0 255 0 0
1 0 0 0 0 1 1 0 255 0 0
1 1 0 0 0 1 1 1 255 0 0
0 1 0 0 0 1 0 1 255 0 0
0.5 0.5 1 0 0 1 0.5 0.5 0 0 0
4 0 1 2 3
3 0 1 4
";

    #[test]
    fn an_ascii_file_keeps_its_quad_normals_and_uvs() {
        let scene = parse_ply(QUAD.as_bytes(), "square").expect("parses");
        let mesh = &scene.meshes[0];
        assert_eq!(mesh.positions.len(), 5);
        assert_eq!(mesh.faces, vec![vec![0, 1, 2, 3], vec![0, 1, 4]]);
        assert_eq!(mesh.corner_normals.as_ref().map(Vec::len), Some(7));
        assert_eq!(
            mesh.corner_uvs.as_ref().map(|u| u[2]),
            Some(Vec2::new(1.0, 1.0))
        );
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what.contains("colour"))));
    }

    /// The same square, written in binary of either order.
    fn binary(big: bool) -> Vec<u8> {
        let order = if big { "big" } else { "little" };
        let mut out = format!(
            "ply\nformat binary_{order}_endian 1.0\nelement vertex 4\nproperty double x\nproperty double y\nproperty double z\nproperty short extra\nelement face 1\nproperty list uchar uint vertex_index\nend_header\n"
        )
        .into_bytes();
        for (x, y) in [(0.0f64, 0.0f64), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)] {
            for v in [x, y, -1.5] {
                out.extend(if big {
                    v.to_be_bytes()
                } else {
                    v.to_le_bytes()
                });
            }
            out.extend(if big {
                7i16.to_be_bytes()
            } else {
                7i16.to_le_bytes()
            });
        }
        out.push(4);
        for i in [0u32, 1, 2, 3] {
            out.extend(if big {
                i.to_be_bytes()
            } else {
                i.to_le_bytes()
            });
        }
        out
    }

    #[test]
    fn binary_files_of_either_byte_order_read_alike() {
        for big in [false, true] {
            let scene = parse_ply(&binary(big), "b").expect("parses");
            let mesh = &scene.meshes[0];
            assert_eq!(mesh.positions[2], Vec3::new(2.0, 2.0, -1.5), "big: {big}");
            assert_eq!(mesh.faces, vec![vec![0, 1, 2, 3]]);
            assert!(mesh.corner_normals.is_none());
        }
    }

    #[test]
    fn a_short_file_and_a_bad_index_are_reported() {
        let mut short = binary(false);
        short.truncate(short.len() - 3);
        assert!(matches!(
            parse_ply(&short, "s"),
            Err(PlyError::Truncated { .. })
        ));
        let bad = QUAD.replace("3 0 1 4", "3 0 1 9");
        let scene = parse_ply(bad.as_bytes(), "b").expect("parses");
        assert_eq!(scene.meshes[0].faces.len(), 1);
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::OutOfRangeIndex { .. })));
        assert!(matches!(parse_ply(b"solid x", "x"), Err(PlyError::NotPly)));
    }
}
