//! STL, in both of its forms.
//!
//! The simplest format that matters, and the one that most needs the importer to
//! do work the file does not: STL is a bag of independent triangles with no
//! shared vertices, no units, no axis convention, no names and no materials.
//!
//! # Welding is not optional here
//!
//! Every triangle carries its own three vertices, so a cube arrives as thirty-six
//! points where eight would do, and no two triangles share an edge. Such a mesh
//! renders perfectly and survives no edit at all — extrude moves one triangle and
//! leaves a hole behind it, subdivide leaves cracks along every seam. So the
//! importer welds, and [`StlOptions::weld_epsilon`] is the one knob that matters.
//!
//! # What the file does not say, and what is assumed instead
//!
//! Units and orientation are both absent from the format. STL is overwhelmingly
//! used for 3D printing, where the convention is millimetres and Z up, so that is
//! what is assumed — and *recorded* as an assumption in
//! [`crate::ir::SourceMetadata::notes`] rather than applied silently. A model
//! that comes in a thousand times too large is the commonest STL complaint there
//! is, and the answer is always the unit assumption.
//!
//! # Facet normals
//!
//! Each triangle stores a normal, which is redundant with its winding and is
//! frequently zero or wrong — plenty of exporters never fill it in. It is not
//! used as geometry. It is used for the one thing it is good for: when a stored
//! normal is valid and points opposite to the winding, the triangle is wound
//! backwards, and the importer flips it. That is a real defect in real files, and
//! fixing it is the difference between a model that shades correctly and one
//! that is inside out in patches.
//!
//! # Reading a hostile file
//!
//! A binary STL states its triangle count in the header, and nothing else in the
//! file corroborates it. A count of four billion asks for a 200 GB allocation,
//! and an allocation failure *aborts* the process — no error, no unwinding. So
//! the count is checked against the file's actual length before anything is
//! sized by it. This is the same defect that was found in `.kmesh` by fuzzing,
//! and it is written down here because the shape recurs: any count that arrives
//! from a file and reaches an allocation needs a bound that also came from the
//! file.

use crate::ir::{ImportMesh, ImportScene, LossyNote, SourceMetadata};
use glam::Vec3;
use kreate_math::AxisConvention;
use kreate_mesh::MeshData;

/// Bytes per triangle in a binary STL: three normals, nine coordinates, and a
/// two-byte attribute count nobody agrees on the meaning of.
const TRIANGLE_BYTES: usize = 50;
/// The fixed header before the triangle count.
const HEADER_BYTES: usize = 80;

#[derive(Clone, Copy, Debug)]
pub struct StlOptions {
    /// Assumed source axis convention.
    ///
    /// Z up, because STL's overwhelming use is 3D printing and every slicer
    /// treats it that way.
    pub axis: AxisConvention,
    /// Assumed metres per source unit. Millimetres, for the same reason.
    pub unit_scale: f32,
    /// How close two vertices must be to be treated as one.
    ///
    /// Not zero, and not a guess either: STL stores coordinates as `f32`, and an
    /// exporter that computed the same corner twice by two different routes can
    /// easily differ in the last bit. A tolerance of exactly zero would leave
    /// those unmerged and the mesh subtly torn.
    ///
    /// Relative to the model, so a value here is in source units before scaling.
    pub weld_epsilon: f32,
}

impl Default for StlOptions {
    fn default() -> Self {
        Self {
            axis: AxisConvention::KREATE,
            // Millimetres.
            unit_scale: 0.001,
            weld_epsilon: 1e-5,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StlError {
    #[error("truncated: expected at least {expected} bytes, found {found}")]
    Truncated { expected: usize, found: usize },
    #[error(
        "the header claims {claimed} triangles, which needs {needed} bytes, but the file is {found}"
    )]
    ImpossibleTriangleCount {
        claimed: u32,
        needed: usize,
        found: usize,
    },
    #[error("line {line}: {what}")]
    Ascii { line: usize, what: String },
}

/// Parses either form, choosing between them by structure rather than by the
/// leading word.
///
/// The usual test — "does it start with `solid`" — is wrong, and wrong in the
/// direction that corrupts data: plenty of binary writers fill the 80-byte
/// header with a description that begins with the word `solid`, and such a file
/// read as ASCII produces nothing or garbage. The file's *length* is decisive,
/// because a binary STL is exactly `84 + 50n` bytes and an ASCII one almost
/// never is.
pub fn parse_stl(bytes: &[u8], options: &StlOptions) -> Result<ImportScene, StlError> {
    match classify(bytes) {
        Form::Binary => parse_binary(bytes, options),
        Form::Ascii => parse_ascii(bytes, options),
    }
}

enum Form {
    Binary,
    Ascii,
}

/// Decides which form a file is in.
///
/// The order of the tests is the whole design, and the last one is there because
/// of a defect this got wrong first time round.
///
/// 1. A length of exactly `84 + 50n` for the count in the header is decisive:
///    only a binary file is that size by construction.
/// 2. Otherwise, text that contains `facet` is an ASCII file. Checked by content
///    rather than by the leading `solid`, because binary writers put descriptions
///    in the header that begin with that word and a reader dispatching on it gets
///    nothing out of such a file.
/// 3. Otherwise, anything long enough to hold a binary header **is treated as
///    binary** — which means a file whose triangle count is corrupt comes back as
///    an error naming the count. The first version fell through to the ASCII
///    reader here, which found no facets and returned an empty mesh: a corrupt
///    file imported as nothing at all, with no error and nothing to explain it.
fn classify(bytes: &[u8]) -> Form {
    if bytes.len() >= HEADER_BYTES + 4 {
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);
        let expected = (count as usize)
            .checked_mul(TRIANGLE_BYTES)
            .and_then(|t| t.checked_add(HEADER_BYTES + 4));
        // Trailing bytes are tolerated: some writers pad.
        if matches!(expected, Some(n) if bytes.len() >= n) {
            return Form::Binary;
        }
    }
    let looks_like_text = bytes
        .windows(5)
        .take(4096)
        .any(|w| w.eq_ignore_ascii_case(b"facet"));
    if looks_like_text {
        return Form::Ascii;
    }
    if bytes.len() >= HEADER_BYTES + 4 {
        Form::Binary
    } else {
        Form::Ascii
    }
}

fn parse_binary(bytes: &[u8], options: &StlOptions) -> Result<ImportScene, StlError> {
    if bytes.len() < HEADER_BYTES + 4 {
        return Err(StlError::Truncated {
            expected: HEADER_BYTES + 4,
            found: bytes.len(),
        });
    }
    let claimed = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);

    // Checked against the file before it is used to size anything. A count that
    // arrives from a file and reaches an allocation needs a bound that also came
    // from the file.
    let needed = (claimed as usize)
        .checked_mul(TRIANGLE_BYTES)
        .and_then(|n| n.checked_add(HEADER_BYTES + 4))
        .ok_or(StlError::ImpossibleTriangleCount {
            claimed,
            needed: usize::MAX,
            found: bytes.len(),
        })?;
    if needed > bytes.len() {
        return Err(StlError::ImpossibleTriangleCount {
            claimed,
            needed,
            found: bytes.len(),
        });
    }

    let mut triangles = Vec::with_capacity(claimed as usize);
    for i in 0..claimed as usize {
        let at = HEADER_BYTES + 4 + i * TRIANGLE_BYTES;
        let f = |k: usize| -> f32 {
            let o = at + k * 4;
            f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        };
        triangles.push(Triangle {
            normal: Vec3::new(f(0), f(1), f(2)),
            corners: [
                Vec3::new(f(3), f(4), f(5)),
                Vec3::new(f(6), f(7), f(8)),
                Vec3::new(f(9), f(10), f(11)),
            ],
        });
    }
    Ok(assemble(triangles, "stl", options))
}

fn parse_ascii(bytes: &[u8], options: &StlOptions) -> Result<ImportScene, StlError> {
    let text = String::from_utf8_lossy(bytes);
    let mut triangles = Vec::new();
    let mut normal = Vec3::ZERO;
    let mut corners: Vec<Vec3> = Vec::new();

    for (number, source_line) in text.lines().enumerate() {
        let line = number + 1;
        let mut words = source_line.split_whitespace();
        let Some(keyword) = words.next() else {
            continue;
        };
        match keyword {
            "facet" => {
                // `facet normal x y z`. A malformed normal is not fatal: the
                // normal is advisory, and refusing the whole file over one is
                // worse than ignoring it.
                normal = if words.next() == Some("normal") {
                    read_vec3(&mut words).unwrap_or(Vec3::ZERO)
                } else {
                    Vec3::ZERO
                };
                corners.clear();
            }
            "vertex" => {
                let v = read_vec3(&mut words).ok_or_else(|| StlError::Ascii {
                    line,
                    what: "a vertex needs three numbers".to_string(),
                })?;
                corners.push(v);
            }
            "endfacet" => {
                if corners.len() == 3 {
                    triangles.push(Triangle {
                        normal,
                        corners: [corners[0], corners[1], corners[2]],
                    });
                } else if !corners.is_empty() {
                    return Err(StlError::Ascii {
                        line,
                        what: format!("a facet with {} vertices", corners.len()),
                    });
                }
                corners.clear();
            }
            _ => {}
        }
    }
    Ok(assemble(triangles, "stl", options))
}

fn read_vec3<'a>(words: &mut impl Iterator<Item = &'a str>) -> Option<Vec3> {
    let x = words.next()?.parse().ok()?;
    let y = words.next()?.parse().ok()?;
    let z = words.next()?.parse().ok()?;
    Some(Vec3::new(x, y, z))
}

struct Triangle {
    normal: Vec3,
    corners: [Vec3; 3],
}

/// Turns triangles into a welded mesh, recording what had to be assumed.
fn assemble(triangles: Vec<Triangle>, name: &str, options: &StlOptions) -> ImportScene {
    let mut source = SourceMetadata::new("stl");
    source.axis = options.axis;
    source.unit_scale = options.unit_scale;
    source.notes.push(LossyNote::AssumedUnitScale {
        assumed: format!("{} metres per unit (STL declares none)", options.unit_scale),
    });
    source.notes.push(LossyNote::AssumedAxisConvention {
        assumed: "Z up (STL declares none)".to_string(),
    });

    let mut positions = Vec::with_capacity(triangles.len() * 3);
    let mut faces = Vec::with_capacity(triangles.len());
    let mut flipped = 0usize;
    for t in &triangles {
        let mut c = t.corners;
        // The stored normal's one legitimate use. When it is present and points
        // against the winding, the triangle is wound backwards — a real and
        // common defect, and one that shows up as patches of a model being lit
        // from the inside.
        let wound = (c[1] - c[0]).cross(c[2] - c[0]);
        if t.normal.length_squared() > 1e-20
            && wound.length_squared() > 1e-20
            && wound.dot(t.normal) < 0.0
        {
            c.swap(1, 2);
            flipped += 1;
        }
        let base = positions.len() as u32;
        positions.extend_from_slice(&c);
        faces.push(vec![base, base + 1, base + 2]);
    }
    if flipped > 0 {
        source.notes.push(LossyNote::Unsupported {
            what: format!("{flipped} triangles were wound against their stored normal and flipped"),
        });
    }

    let mut mesh = ImportMesh {
        name: name.to_string(),
        positions,
        faces,
        corner_uvs: None,
        corner_normals: None,
        corner_colors: None,
        face_materials: None,
        smoothing_groups: None,
    };
    // The step that makes the result editable rather than merely viewable.
    mesh.weld(options.weld_epsilon, &mut source.notes);

    let mut scene = ImportScene::new(source);
    scene.meshes.push(mesh);
    scene
}

/// Writes a binary STL.
///
/// Binary rather than ASCII: it is a third the size, and it stores the
/// coordinates exactly rather than through a decimal round trip.
///
/// Lossy by nature, and not quietly so — the caller is told what went. STL has
/// no n-gons, no names, no materials, no UVs and no vertex sharing, so a mesh
/// leaves as bare triangles. There is no extension mechanism to record what was
/// dropped in, the way glTF has, so the record has to travel back to the caller.
pub fn write_stl(mesh: &MeshData, options: &StlOptions) -> (Vec<u8>, Vec<LossyNote>) {
    let mut notes = Vec::new();
    let positions = mesh.positions();
    let scale = if options.unit_scale != 0.0 {
        1.0 / options.unit_scale
    } else {
        1.0
    };
    let to_source = options.axis.from_kreate();

    // A fan per face. STL is triangles only, so an n-gon has to be split, and a
    // fan is the right choice for the convex faces that dominate: it adds no
    // vertices and needs no arithmetic.
    let mut triangles: Vec<[Vec3; 3]> = Vec::new();
    let mut ngons = 0usize;
    for face in mesh.faces() {
        if face.len() < 3 {
            continue;
        }
        if face.len() > 3 {
            ngons += 1;
        }
        for i in 1..face.len() - 1 {
            triangles.push([
                to_source * positions[face[0] as usize] * scale,
                to_source * positions[face[i] as usize] * scale,
                to_source * positions[face[i + 1] as usize] * scale,
            ]);
        }
    }
    if ngons > 0 {
        notes.push(LossyNote::Unsupported {
            what: format!("{ngons} n-gons were triangulated; STL has no way to record them"),
        });
    }
    if mesh.attributes().iter().any(|l| l.name.starts_with("uv")) {
        notes.push(LossyNote::Unsupported {
            what: "UVs were dropped; STL stores none".to_string(),
        });
    }

    let mut out = Vec::with_capacity(HEADER_BYTES + 4 + triangles.len() * TRIANGLE_BYTES);
    let mut header = [0u8; HEADER_BYTES];
    // Deliberately not beginning with "solid": a binary file whose header starts
    // with that word is what makes the naive ASCII/binary test misread it, and
    // there is no reason to hand that problem to the next program.
    let banner = b"Kreate binary STL";
    header[..banner.len()].copy_from_slice(banner);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(triangles.len() as u32).to_le_bytes());

    for t in &triangles {
        // Written from the winding rather than left at zero. It is redundant, but
        // a reader that trusts it — and some do — should find it agreeing with
        // the geometry rather than contradicting it.
        let normal = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        for v in [normal, t[0], t[1], t[2]] {
            out.extend_from_slice(&v.x.to_le_bytes());
            out.extend_from_slice(&v.y.to_le_bytes());
            out.extend_from_slice(&v.z.to_le_bytes());
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    (out, notes)
}
