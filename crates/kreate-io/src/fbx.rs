//! FBX, binary and ASCII: what most other tools and asset stores hand over.
//!
//! Autodesk publishes no specification, but the container is well understood:
//! a tree of named records, each with a list of typed values and children of
//! its own. Binary files store that tree with sizes and type codes, large
//! arrays optionally zlib-compressed; ASCII files write the same tree as
//! `Name: values { children }`. Both are read into one [`Node`] tree, and the
//! scene is taken from that.
//!
//! # What is read
//!
//! - **Meshes:** `Geometry` records — control points, polygons as written
//!   (n-gons kept), normals and the first UV set in any of FBX's mapping and
//!   reference modes, and per-polygon material indices.
//! - **Placement:** `Model` records and their parents, with translation,
//!   rotation (in its rotation order, with pre- and post-rotation) and scale.
//!   A geometric offset — a transform on a mesh that its children do not
//!   inherit — is applied to that mesh's points.
//! - **Materials:** diffuse colour and factor, emission, opacity, and a
//!   roughness from the shininess; a diffuse texture, embedded or beside the
//!   file.
//! - **Conventions:** `GlobalSettings` says which axis is up and how many
//!   centimetres a unit is, and both are applied.
//!
//! - **Skinning:** a `Skin` deformer's clusters — each a bone, the control
//!   points it moves and how much, and where bone and mesh stood at the bind
//!   — as the IR's skin and per-vertex weights, the four strongest a vertex.
//! - **Animation:** each `AnimationStack` — a take, as Mixamo or Maya write
//!   them — as a clip: every model's translation, rotation and scale curves,
//!   keyed where the curves are keyed and read straight between.
//! - **Cameras and lights**, as the IR has them.
//!
//! Blend shapes are named in the notes and left behind; pivots are ignored,
//! which is right for the many files whose pivots are zero and an
//! approximation for the rest.

use crate::ir::{
    ImportImage, ImportMaterial, ImportMesh, ImportNode, ImportNodeData, ImportScene,
    ImportTransform, LossyNote, SourceMetadata,
};
use glam::{Mat4, Quat, Vec2, Vec3};
use kreate_math::{AxisConvention, Handedness, UpAxis};

#[derive(Debug, thiserror::Error)]
pub enum FbxError {
    #[error("not an FBX file")]
    NotFbx,
    #[error("the file ends inside a record at byte {0}")]
    Truncated(usize),
    #[error("a record at byte {at}: {what}")]
    Corrupt { at: usize, what: String },
    #[error("line {line}: {what}")]
    Ascii { line: usize, what: String },
}

/// One value of a record.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Text(String),
    Bytes(Vec<u8>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl Value {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    /// An array as numbers, whatever it was stored as.
    fn floats(&self) -> Vec<f64> {
        match self {
            Value::Floats(v) => v.clone(),
            Value::Ints(v) => v.iter().map(|i| *i as f64).collect(),
            Value::Int(i) => vec![*i as f64],
            Value::Float(f) => vec![*f],
            _ => Vec::new(),
        }
    }

    fn ints(&self) -> Vec<i64> {
        match self {
            Value::Ints(v) => v.clone(),
            Value::Floats(v) => v.iter().map(|f| *f as i64).collect(),
            Value::Int(i) => vec![*i],
            _ => Vec::new(),
        }
    }
}

/// One record: its name, its values and its children.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    pub name: String,
    pub values: Vec<Value>,
    pub children: Vec<Node>,
}

impl Node {
    fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }

    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }

    fn value(&self, i: usize) -> Option<&Value> {
        self.values.get(i)
    }

    /// A `Properties70` entry's values after its name and three type words —
    /// or, for an older file's `Properties60`, after its name and two.
    fn property(&self, key: &str) -> Option<&[Value]> {
        for (block, skip) in [("Properties70", 4), ("Properties60", 3)] {
            if let Some(found) = self.child(block).and_then(|props| {
                props
                    .children
                    .iter()
                    .find(|p| p.value(0).and_then(Value::as_str) == Some(key))
            }) {
                return Some(found.values.get(skip..).unwrap_or(&[]));
            }
        }
        None
    }

    fn property_vec3(&self, key: &str) -> Option<Vec3> {
        let v = self.property(key)?;
        Some(Vec3::new(
            v.first()?.as_f64()? as f32,
            v.get(1)?.as_f64()? as f32,
            v.get(2)?.as_f64()? as f32,
        ))
    }

    fn property_f64(&self, key: &str) -> Option<f64> {
        self.property(key)?.first()?.as_f64()
    }
}

// --- the container ----------------------------------------------------------

const MAGIC: &[u8] = b"Kaydara FBX Binary  \0";

/// The record tree of a file, binary or ASCII.
pub fn parse_tree(bytes: &[u8]) -> Result<Vec<Node>, FbxError> {
    if bytes.starts_with(MAGIC) {
        parse_binary(bytes)
    } else {
        let text = std::str::from_utf8(bytes).map_err(|_| FbxError::NotFbx)?;
        if !text.trim_start().starts_with(';') && !text.contains("FBXHeaderExtension") {
            return Err(FbxError::NotFbx);
        }
        parse_ascii(text)
    }
}

fn parse_binary(bytes: &[u8]) -> Result<Vec<Node>, FbxError> {
    if bytes.len() < 27 {
        return Err(FbxError::Truncated(bytes.len()));
    }
    let version = u32::from_le_bytes([bytes[23], bytes[24], bytes[25], bytes[26]]);
    let wide = version >= 7500;
    let mut reader = Binary { bytes, wide };
    let mut at = 27;
    let mut nodes = Vec::new();
    while let Some((node, next)) = reader.node(at, 0)? {
        nodes.push(node);
        at = next;
    }
    Ok(nodes)
}

struct Binary<'a> {
    bytes: &'a [u8],
    wide: bool,
}

impl Binary<'_> {
    fn take(&self, at: usize, n: usize) -> Result<&[u8], FbxError> {
        self.bytes
            .get(at..at.checked_add(n).ok_or(FbxError::Truncated(at))?)
            .ok_or(FbxError::Truncated(at))
    }

    fn u32(&self, at: usize) -> Result<u32, FbxError> {
        let b = self.take(at, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&self, at: usize) -> Result<u64, FbxError> {
        let b = self.take(at, 8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// The record at `at`, and where the next one starts; `None` for the
    /// empty record that ends a list, or the end of the file.
    fn node(&mut self, at: usize, depth: usize) -> Result<Option<(Node, usize)>, FbxError> {
        if depth > 64 {
            return Err(FbxError::Corrupt {
                at,
                what: "records nested too deep".into(),
            });
        }
        let word = if self.wide { 8 } else { 4 };
        if at + 3 * word + 1 > self.bytes.len() {
            return Ok(None);
        }
        let read = |s: &Self, k: usize| -> Result<usize, FbxError> {
            Ok(if s.wide {
                s.u64(at + k * 8)? as usize
            } else {
                s.u32(at + k * 4)? as usize
            })
        };
        let end = read(self, 0)?;
        let count = read(self, 1)?;
        let _list_len = read(self, 2)?;
        let name_len = self.take(at + 3 * word, 1)?[0] as usize;
        if end == 0 {
            return Ok(None);
        }
        if end <= at || end > self.bytes.len() {
            return Err(FbxError::Corrupt {
                at,
                what: format!("it claims to end at byte {end}"),
            });
        }
        let mut cursor = at + 3 * word + 1;
        let name = String::from_utf8_lossy(self.take(cursor, name_len)?).into_owned();
        cursor += name_len;
        let mut values = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let (value, next) = self.value(cursor)?;
            values.push(value);
            cursor = next;
        }
        let mut children = Vec::new();
        while cursor < end {
            match self.node(cursor, depth + 1)? {
                Some((child, next)) => {
                    children.push(child);
                    cursor = next;
                }
                None => break,
            }
        }
        Ok(Some((
            Node {
                name,
                values,
                children,
            },
            end,
        )))
    }

    fn value(&self, at: usize) -> Result<(Value, usize), FbxError> {
        let code = self.take(at, 1)?[0];
        let at = at + 1;
        let b = |n| self.take(at, n);
        Ok(match code {
            b'Y' => {
                let v = b(2)?;
                (Value::Int(i16::from_le_bytes([v[0], v[1]]) as i64), at + 2)
            }
            b'C' => (Value::Int(b(1)?[0] as i64), at + 1),
            b'I' => (Value::Int(self.u32(at)? as i32 as i64), at + 4),
            b'F' => (Value::Float(f32::from_bits(self.u32(at)?) as f64), at + 4),
            b'D' => (Value::Float(f64::from_bits(self.u64(at)?)), at + 8),
            b'L' => (Value::Int(self.u64(at)? as i64), at + 8),
            b'S' | b'R' => {
                let n = self.u32(at)? as usize;
                let raw = self.take(at + 4, n)?;
                let value = if code == b'S' {
                    Value::Text(String::from_utf8_lossy(raw).into_owned())
                } else {
                    Value::Bytes(raw.to_vec())
                };
                (value, at + 4 + n)
            }
            b'f' | b'd' | b'l' | b'i' | b'b' => {
                let count = self.u32(at)? as usize;
                let encoding = self.u32(at + 4)?;
                let stored = self.u32(at + 8)? as usize;
                let raw = self.take(at + 12, stored)?;
                let size = match code {
                    b'd' | b'l' => 8,
                    b'f' | b'i' => 4,
                    _ => 1,
                };
                let wanted = count.checked_mul(size).ok_or(FbxError::Corrupt {
                    at,
                    what: "an array too large".into(),
                })?;
                // Bounded by what the bytes can hold: a count from the file
                // does not size an allocation on its own say-so.
                if wanted > raw.len().saturating_mul(1032).max(1 << 20) {
                    return Err(FbxError::Corrupt {
                        at,
                        what: format!("an array of {count} claims more than its bytes hold"),
                    });
                }
                let data = if encoding == 1 {
                    use std::io::Read;
                    let mut out = Vec::with_capacity(wanted);
                    flate2::read::ZlibDecoder::new(raw)
                        .take(wanted as u64)
                        .read_to_end(&mut out)
                        .map_err(|e| FbxError::Corrupt {
                            at,
                            what: format!("a compressed array: {e}"),
                        })?;
                    out
                } else {
                    raw.to_vec()
                };
                if data.len() < wanted {
                    return Err(FbxError::Corrupt {
                        at,
                        what: "an array shorter than its count".into(),
                    });
                }
                let value = match code {
                    b'd' => Value::Floats(
                        data.as_chunks::<8>()
                            .0
                            .iter()
                            .take(count)
                            .map(|c| f64::from_le_bytes(*c))
                            .collect(),
                    ),
                    b'f' => Value::Floats(
                        data.as_chunks::<4>()
                            .0
                            .iter()
                            .take(count)
                            .map(|c| f32::from_le_bytes(*c) as f64)
                            .collect(),
                    ),
                    b'l' => Value::Ints(
                        data.as_chunks::<8>()
                            .0
                            .iter()
                            .take(count)
                            .map(|c| i64::from_le_bytes(*c))
                            .collect(),
                    ),
                    b'i' => Value::Ints(
                        data.as_chunks::<4>()
                            .0
                            .iter()
                            .take(count)
                            .map(|c| i32::from_le_bytes(*c) as i64)
                            .collect(),
                    ),
                    _ => Value::Ints(data.iter().take(count).map(|b| *b as i64).collect()),
                };
                (value, at + 12 + stored)
            }
            other => {
                return Err(FbxError::Corrupt {
                    at,
                    what: format!("an unknown value type `{}`", other as char),
                })
            }
        })
    }
}

/// The ASCII form: `Name: value, value { children }`, `;` comments, and
/// arrays as `*count { a: values }`.
fn parse_ascii(text: &str) -> Result<Vec<Node>, FbxError> {
    #[derive(Debug, PartialEq)]
    enum Token {
        Key(String),
        Text(String),
        Number(String),
        Open,
        Close,
    }
    let mut tokens: Vec<(usize, Token)> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line_no = n + 1;
        let mut chars = line.char_indices().peekable();
        while let Some(&(i, c)) = chars.peek() {
            match c {
                ';' => break,
                ' ' | '\t' | ',' | '\r' => {
                    chars.next();
                }
                '{' => {
                    tokens.push((line_no, Token::Open));
                    chars.next();
                }
                '}' => {
                    tokens.push((line_no, Token::Close));
                    chars.next();
                }
                '"' => {
                    chars.next();
                    let start = i + 1;
                    let mut end = line.len();
                    for (j, d) in chars.by_ref() {
                        if d == '"' {
                            end = j;
                            break;
                        }
                    }
                    tokens.push((line_no, Token::Text(line[start..end].to_string())));
                }
                _ => {
                    let start = i;
                    let mut end = line.len();
                    while let Some(&(j, d)) = chars.peek() {
                        if matches!(d, ' ' | '\t' | ',' | '{' | '}' | '"' | '\r') {
                            end = j;
                            break;
                        }
                        chars.next();
                    }
                    let word = &line[start..end];
                    if let Some(key) = word.strip_suffix(':') {
                        tokens.push((line_no, Token::Key(key.to_string())));
                    } else if let Some(count) = word.strip_prefix('*') {
                        // An array's count: the values follow in `{ a: ... }`.
                        let _ = count;
                    } else if word.parse::<f64>().is_ok() {
                        tokens.push((line_no, Token::Number(word.to_string())));
                    } else {
                        tokens.push((line_no, Token::Text(word.to_string())));
                    }
                }
            }
        }
    }

    fn number(word: &str) -> Value {
        if !word.contains(['.', 'e', 'E']) {
            if let Ok(i) = word.parse::<i64>() {
                return Value::Int(i);
            }
        }
        Value::Float(word.parse().unwrap_or(0.0))
    }

    // A record: its key, then values up to the next key, `{` or `}`.
    fn record(tokens: &[(usize, Token)], at: &mut usize, depth: usize) -> Result<Node, FbxError> {
        let (line, Token::Key(name)) = &tokens[*at] else {
            return Err(FbxError::Ascii {
                line: tokens[*at].0,
                what: "expected a name".into(),
            });
        };
        if depth > 64 {
            return Err(FbxError::Ascii {
                line: *line,
                what: "records nested too deep".into(),
            });
        }
        *at += 1;
        let mut node = Node {
            name: name.clone(),
            ..Default::default()
        };
        while let Some((_, token)) = tokens.get(*at) {
            match token {
                Token::Number(w) => node.values.push(number(w)),
                Token::Text(t) => node.values.push(Value::Text(t.clone())),
                _ => break,
            }
            *at += 1;
        }
        if matches!(tokens.get(*at), Some((_, Token::Open))) {
            *at += 1;
            loop {
                match tokens.get(*at) {
                    Some((_, Token::Close)) => {
                        *at += 1;
                        break;
                    }
                    Some((_, Token::Key(_))) => node.children.push(record(tokens, at, depth + 1)?),
                    Some((l, _)) => {
                        return Err(FbxError::Ascii {
                            line: *l,
                            what: "a value where a name belongs".into(),
                        })
                    }
                    None => break,
                }
            }
        }
        // An array written `*n { a: ... }`: its values become the record's.
        if node.values.is_empty() && node.children.len() == 1 && node.children[0].name == "a" {
            let values = std::mem::take(&mut node.children[0].values);
            node.children.clear();
            let any_float = values.iter().any(|v| matches!(v, Value::Float(_)));
            node.values.push(if any_float {
                Value::Floats(values.iter().filter_map(Value::as_f64).collect())
            } else {
                Value::Ints(values.iter().filter_map(Value::as_i64).collect())
            });
        }
        Ok(node)
    }

    let mut nodes = Vec::new();
    let mut at = 0;
    while at < tokens.len() {
        match &tokens[at] {
            (_, Token::Key(_)) => nodes.push(record(&tokens, &mut at, 0)?),
            (line, _) => {
                return Err(FbxError::Ascii {
                    line: *line,
                    what: "a value outside any record".into(),
                })
            }
        }
    }
    Ok(nodes)
}

// --- the scene ----------------------------------------------------------------

/// An object's name without its class: binary files write `Name\0\x01Class`,
/// ASCII files `Class::Name`. A class with no name is called by its class.
fn clean_name(raw: &str) -> String {
    let (name, class) = match raw.split_once("\u{0}\u{1}") {
        Some((name, class)) => (name, class),
        None => raw
            .split_once("::")
            .map_or((raw, ""), |(class, name)| (name, class)),
    };
    if name.is_empty() && !class.is_empty() {
        class.to_string()
    } else {
        name.to_string()
    }
}

/// Euler angles in degrees, in FBX's rotation order (`0` is XYZ: X turns
/// first), as one rotation.
fn euler(degrees: Vec3, order: i64) -> Quat {
    let r = degrees * (std::f32::consts::PI / 180.0);
    let (x, y, z) = (
        Quat::from_rotation_x(r.x),
        Quat::from_rotation_y(r.y),
        Quat::from_rotation_z(r.z),
    );
    match order {
        1 => y * z * x, // XZY
        2 => x * z * y, // YZX
        3 => z * x * y, // YXZ
        4 => y * x * z, // ZXY
        5 => x * y * z, // ZYX
        _ => z * y * x, // XYZ
    }
}

/// A layer of per-corner values — normals, UVs — read once: how its values
/// are laid on the mesh, and the values and their indices.
struct Layer {
    mapping: String,
    direct: bool,
    data: Vec<f64>,
    index: Vec<i64>,
    stride: usize,
}

impl Layer {
    fn read(layer: &Node, data: &str, index: &str, stride: usize) -> Self {
        let word = |key: &str, default: &str| {
            layer
                .child(key)
                .and_then(|n| n.value(0))
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string()
        };
        Self {
            mapping: word("MappingInformationType", "ByPolygonVertex"),
            direct: word("ReferenceInformationType", "Direct") == "Direct",
            data: layer
                .child(data)
                .and_then(|n| n.value(0))
                .map(Value::floats)
                .unwrap_or_default(),
            index: layer
                .child(index)
                .and_then(|n| n.value(0))
                .map(Value::ints)
                .unwrap_or_default(),
            stride,
        }
    }

    /// The value at a corner of a polygon, on a vertex.
    fn at(&self, polygon: usize, corner: usize, vertex: usize) -> Option<&[f64]> {
        let at = match self.mapping.as_str() {
            "ByPolygonVertex" => corner,
            "ByVertex" | "ByVertice" | "ByControlPoint" => vertex,
            "ByPolygon" => polygon,
            _ => 0,
        };
        let at = if self.direct {
            at
        } else {
            usize::try_from(*self.index.get(at)?).ok()?
        };
        self.data
            .get(at * self.stride..at * self.stride + self.stride)
    }
}

/// A camera attribute as the IR has it: its vertical angle from the lens and
/// film when the file gives them, its field of view otherwise.
fn camera_of(attribute: &Node, name: String) -> crate::ir::ImportCamera {
    let focal = attribute.property_f64("FocalLength");
    // Inches, as FBX keeps film backs.
    let film = attribute.property_f64("FilmHeight");
    let fov_y = match (focal, film) {
        (Some(f), Some(h)) if f > 0.0 && h > 0.0 => 2.0 * ((h * 25.4 / 2.0) / f).atan(),
        _ => attribute
            .property_f64("FieldOfViewY")
            .or_else(|| attribute.property_f64("FieldOfView"))
            .unwrap_or(40.0)
            .to_radians(),
    };
    let ortho = attribute.property_f64("CameraProjectionType") == Some(1.0);
    crate::ir::ImportCamera {
        name,
        fov_y: (!ortho).then_some(fov_y as f32),
        ortho_width: ortho.then(|| attribute.property_f64("OrthoZoom").unwrap_or(1.0) as f32),
        clip_near: attribute.property_f64("NearPlane").unwrap_or(10.0) as f32,
        clip_far: attribute.property_f64("FarPlane").map(|f| f as f32),
    }
}

/// A light attribute as the IR has it: point, sun or spot, its colour, and
/// an amount read from FBX's unitless intensity.
fn light_of(attribute: &Node, name: String) -> Option<crate::ir::ImportLight> {
    use crate::ir::ImportLightKind;
    let colour = attribute.property_vec3("Color").unwrap_or(Vec3::ONE);
    let amount = attribute.property_f64("Intensity").unwrap_or(100.0) as f32;
    let kind = match attribute.property_f64("LightType").unwrap_or(0.0) as i64 {
        1 => ImportLightKind::Directional,
        2 => {
            let outer = attribute
                .property_f64("OuterAngle")
                .or_else(|| attribute.property_f64("Cone angle"))
                .unwrap_or(45.0)
                .to_radians() as f32
                * 0.5;
            let inner = attribute
                .property_f64("InnerAngle")
                .map_or(outer * 0.8, |a| a.to_radians() as f32 * 0.5)
                .min(outer);
            ImportLightKind::Spot { inner, outer }
        }
        // A point, and an area light that a point stands in for.
        _ => ImportLightKind::Point,
    };
    let intensity = match kind {
        ImportLightKind::Directional => amount * 10.0,
        _ => amount,
    };
    Some(crate::ir::ImportLight {
        name,
        kind,
        color: [colour.x, colour.y, colour.z],
        intensity,
        range: None,
    })
}

/// Reads an FBX file into the IR.
///
/// `file` finds a texture the file names but does not carry: its path as the
/// file wrote it, to the bytes, or `None`.
pub fn parse_fbx(
    bytes: &[u8],
    file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<ImportScene, FbxError> {
    let tree = parse_tree(bytes)?;
    let top = |name: &str| tree.iter().find(|n| n.name == name);
    let mut source = SourceMetadata::new("fbx");

    let empty = Node::default();
    let objects = top("Objects").unwrap_or(&empty);
    // Which axis is up, and how big a unit is: at the top of the file, or —
    // as FBX 7.0 wrote it — among the objects.
    let settings = top("GlobalSettings").or_else(|| objects.child("GlobalSettings"));
    let setting = |key: &str| settings.and_then(|s| s.property_f64(key));
    let up = setting("UpAxis").unwrap_or(1.0) as i64;
    source.axis = AxisConvention {
        up: if up == 2 { UpAxis::ZUp } else { UpAxis::YUp },
        handedness: Handedness::Right,
    };
    if settings.is_none() {
        source.notes.push(LossyNote::AssumedAxisConvention {
            assumed: "Y up (the file has no GlobalSettings)".into(),
        });
    } else if up == 0 || setting("UpAxisSign").is_some_and(|s| s < 0.0) {
        source.notes.push(LossyNote::AssumedAxisConvention {
            assumed: "Y up (the file's up axis is one Kreate does not convert)".into(),
        });
    }
    source.unit_scale = (setting("UnitScaleFactor").unwrap_or(1.0) * 0.01) as f32;
    source.generator = top("FBXHeaderExtension")
        .and_then(|h| h.child("Creator"))
        .and_then(|c| c.value(0))
        .and_then(Value::as_str)
        .map(str::to_string);

    let id_of = |n: &Node| n.value(0).and_then(Value::as_i64);
    // Child to parent, in the file's order: which mesh a model wears, which
    // model is under which, which materials a model has.
    let mut links: Vec<(i64, i64, Option<String>)> = Vec::new();
    for c in top("Connections")
        .map(|c| c.children_named("C").collect::<Vec<_>>())
        .unwrap_or_default()
    {
        let (Some(child), Some(parent)) = (
            c.value(1).and_then(Value::as_i64),
            c.value(2).and_then(Value::as_i64),
        ) else {
            continue;
        };
        let property = c.value(3).and_then(Value::as_str).map(str::to_string);
        links.push((child, parent, property));
    }
    let parents_of = |id: i64| links.iter().filter(move |l| l.0 == id).map(|l| l.1);
    let children_of = |id: i64| links.iter().filter(move |l| l.1 == id);

    let mut scene = ImportScene::new(source);

    // Textures: each `Texture` record's image, by the texture's id.
    let mut image_of_texture: Vec<(i64, usize)> = Vec::new();
    for texture in objects.children_named("Texture") {
        let Some(id) = id_of(texture) else { continue };
        // Embedded, in a `Video` the texture is connected from; else beside
        // the file, by the names it gives.
        let embedded = links
            .iter()
            .filter(|l| l.1 == id)
            .filter_map(|l| objects.children.iter().find(|o| id_of(o) == Some(l.0)))
            .find(|o| o.name == "Video")
            .and_then(|v| v.child("Content"))
            .and_then(|c| match c.value(0) {
                Some(Value::Bytes(b)) if !b.is_empty() => Some(b.clone()),
                _ => None,
            });
        let named = ["RelativeFilename", "FileName", "Filename"]
            .iter()
            .filter_map(|k| {
                texture
                    .child(k)
                    .and_then(|n| n.value(0))
                    .and_then(Value::as_str)
            })
            .map(str::to_string)
            .collect::<Vec<_>>();
        let bytes = embedded.or_else(|| named.iter().find_map(|n| file(n)));
        let name = clean_name(
            texture
                .value(1)
                .and_then(Value::as_str)
                .unwrap_or("Texture"),
        );
        let Some(bytes) = bytes else {
            scene.source.notes.push(LossyNote::Unsupported {
                what: format!("the texture {name}: its image was not found"),
            });
            continue;
        };
        match image::load_from_memory(&bytes) {
            Ok(decoded) => {
                let rgba = decoded.to_rgba8();
                scene.images.push(ImportImage {
                    name: name.clone(),
                    width: rgba.width(),
                    height: rgba.height(),
                    pixels: rgba.into_raw(),
                });
                image_of_texture.push((id, scene.images.len() - 1));
            }
            Err(_) => scene.source.notes.push(LossyNote::Unsupported {
                what: format!("the texture {name}: an image Kreate cannot read"),
            }),
        }
    }

    // Materials, by id.
    let mut material_of: Vec<(i64, usize)> = Vec::new();
    for m in objects.children_named("Material") {
        let Some(id) = id_of(m) else { continue };
        let diffuse = m
            .property_vec3("DiffuseColor")
            .or_else(|| m.property_vec3("Diffuse"))
            .unwrap_or(Vec3::splat(0.8));
        let factor = m.property_f64("DiffuseFactor").unwrap_or(1.0) as f32;
        let opacity = m
            .property_f64("Opacity")
            .or_else(|| m.property_f64("TransparencyFactor").map(|t| 1.0 - t))
            .unwrap_or(1.0) as f32;
        let emissive = m.property_vec3("EmissiveColor").unwrap_or(Vec3::ZERO)
            * m.property_f64("EmissiveFactor").unwrap_or(1.0) as f32;
        // Blender's reading of a Phong exponent as a roughness.
        let roughness =
            m.property_f64("ShininessExponent")
                .or_else(|| m.property_f64("Shininess"))
                .map_or(0.5, |n| 1.0 - (n.clamp(0.0, 100.0).sqrt() / 10.0)) as f32;
        let base = diffuse * factor;
        let texture = children_of(id)
            .filter(|l| {
                l.2.as_deref()
                    .is_some_and(|p| p == "DiffuseColor" || p == "Diffuse" || p == "Maya|baseColor")
            })
            .find_map(|l| {
                image_of_texture
                    .iter()
                    .find(|(t, _)| *t == l.0)
                    .map(|(_, i)| *i)
            });
        scene.materials.push(ImportMaterial {
            name: clean_name(m.value(1).and_then(Value::as_str).unwrap_or("Material")),
            base_color: [base.x, base.y, base.z, opacity.clamp(0.0, 1.0)],
            roughness: roughness.clamp(0.0, 1.0),
            emissive,
            base_color_texture: texture,
            ..Default::default()
        });
        material_of.push((id, scene.materials.len() - 1));
    }

    // Meshes, by geometry id.
    let mut mesh_of: Vec<(i64, usize)> = Vec::new();
    for g in objects.children_named("Geometry") {
        if g.value(2).and_then(Value::as_str) != Some("Mesh") {
            continue;
        }
        let Some(id) = id_of(g) else { continue };
        let name = clean_name(g.value(1).and_then(Value::as_str).unwrap_or("Mesh"));
        let points = g
            .child("Vertices")
            .and_then(|v| v.value(0))
            .map(Value::floats);
        let indices = g
            .child("PolygonVertexIndex")
            .and_then(|v| v.value(0))
            .map(Value::ints);
        let (Some(points), Some(indices)) = (points, indices) else {
            continue;
        };
        let positions: Vec<Vec3> = points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
            .collect();
        // A negative index ends a polygon: its vertex is the bits inverted.
        let mut polygons: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut current = Vec::new();
        for (corner, i) in indices.iter().enumerate() {
            let (vertex, last) = if *i < 0 { (!*i, true) } else { (*i, false) };
            current.push((vertex.max(0) as usize, corner));
            if last {
                polygons.push(std::mem::take(&mut current));
            }
        }
        let normals_layer = g
            .child("LayerElementNormal")
            .map(|l| Layer::read(l, "Normals", "NormalsIndex", 3));
        let uv_layer = g
            .child("LayerElementUV")
            .map(|l| Layer::read(l, "UV", "UVIndex", 2));
        // The array is the indices themselves, whatever the reference type
        // says: one for all, or one a polygon.
        let material_layer = g
            .child("LayerElementMaterial")
            .map(|l| Layer::read(l, "", "Materials", 1));
        let mut faces = Vec::with_capacity(polygons.len());
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut materials = Vec::new();
        let mut notes = Vec::new();
        for (p, polygon) in polygons.iter().enumerate() {
            if polygon.iter().any(|(v, _)| *v >= positions.len()) {
                notes.push(LossyNote::OutOfRangeIndex {
                    mesh: name.clone(),
                    face: p,
                });
                continue;
            }
            if polygon.len() < 3 {
                notes.push(LossyNote::DroppedDegenerateFace {
                    mesh: name.clone(),
                    face: p,
                });
                continue;
            }
            faces.push(polygon.iter().map(|(v, _)| *v as u32).collect::<Vec<_>>());
            for (v, corner) in polygon {
                if let Some(layer) = &normals_layer {
                    let n = layer.at(p, *corner, *v).unwrap_or(&[0.0, 0.0, 1.0]);
                    normals.push(Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32));
                }
                if let Some(layer) = &uv_layer {
                    let t = layer.at(p, *corner, *v).unwrap_or(&[0.0, 0.0]);
                    uvs.push(Vec2::new(t[0] as f32, t[1] as f32));
                }
            }
            if let Some(layer) = &material_layer {
                // The array is the indices themselves, whatever the
                // reference type says: one for all, or one a polygon.
                let m = if layer.mapping == "ByPolygon" {
                    layer.index.get(p)
                } else {
                    layer.index.first()
                };
                materials.push(m.copied().unwrap_or(0) as i32);
            }
        }
        scene.source.notes.extend(notes);
        scene.meshes.push(ImportMesh {
            name,
            positions,
            faces,
            corner_uvs: (!uvs.is_empty()).then_some(uvs),
            corner_normals: (!normals.is_empty()).then_some(normals),
            corner_colors: None,
            // Indices into the model's own material list for now; made the
            // scene's once the model is known.
            face_materials: (!materials.is_empty()).then_some(materials),
            smoothing_groups: None,
        });
        mesh_of.push((id, scene.meshes.len() - 1));
    }

    // Models, placed, their parents found afterwards: a parent may come
    // after its child.
    let models: Vec<&Node> = objects.children_named("Model").collect();
    let mut lights_assumed = false;
    // The meshes as the file has them, for a second model wearing one: the
    // first may have moved its points by a geometric offset.
    let pristine = scene.meshes.clone();
    let mut node_of: Vec<(i64, usize)> = Vec::new();
    for model in &models {
        let Some(id) = id_of(model) else { continue };
        let order = model.property_f64("RotationOrder").unwrap_or(0.0) as i64;
        let rotation = euler(
            model.property_vec3("PreRotation").unwrap_or(Vec3::ZERO),
            order,
        ) * euler(
            model.property_vec3("Lcl Rotation").unwrap_or(Vec3::ZERO),
            order,
        ) * euler(
            model.property_vec3("PostRotation").unwrap_or(Vec3::ZERO),
            order,
        )
        .inverse();
        let mut node = ImportNode::new(clean_name(
            model.value(1).and_then(Value::as_str).unwrap_or("Model"),
        ));
        node.transform = ImportTransform {
            translation: model.property_vec3("Lcl Translation").unwrap_or(Vec3::ZERO),
            rotation,
            scale: model.property_vec3("Lcl Scaling").unwrap_or(Vec3::ONE),
        };
        // The mesh it wears, with the model's materials in its order.
        let geometry = children_of(id)
            .filter(|l| l.2.is_none())
            .find_map(|l| mesh_of.iter().find(|(g, _)| *g == l.0).map(|(_, m)| *m));
        if let Some(mesh) = geometry {
            let own: Vec<usize> = children_of(id)
                .filter_map(|l| material_of.iter().find(|(m, _)| *m == l.0).map(|(_, i)| *i))
                .collect();
            // A mesh worn by a second model with other materials is a copy.
            let mut mesh = mesh;
            let offset = Mat4::from_scale_rotation_translation(
                model.property_vec3("GeometricScaling").unwrap_or(Vec3::ONE),
                euler(
                    model
                        .property_vec3("GeometricRotation")
                        .unwrap_or(Vec3::ZERO),
                    0,
                ),
                model
                    .property_vec3("GeometricTranslation")
                    .unwrap_or(Vec3::ZERO),
            );
            let shared = node_of
                .iter()
                .any(|(_, n)| matches!(scene.nodes[*n].data, ImportNodeData::Mesh(m) if m == mesh));
            if shared {
                scene.meshes.push(pristine[mesh].clone());
                mesh = scene.meshes.len() - 1;
            }
            let target = &mut scene.meshes[mesh];
            if offset != Mat4::IDENTITY {
                let normal = Mat4::from_mat3(glam::Mat3::from_mat4(offset).inverse().transpose());
                for p in &mut target.positions {
                    *p = offset.transform_point3(*p);
                }
                if let Some(ns) = &mut target.corner_normals {
                    for n in ns {
                        *n = normal.transform_vector3(*n).normalize_or_zero();
                    }
                }
            }
            match (&mut target.face_materials, own.is_empty()) {
                (Some(list), false) => {
                    for m in list.iter_mut() {
                        *m = own.get(*m as usize).map_or(-1, |i| *i as i32);
                    }
                }
                (None, false) => {
                    target.face_materials = Some(vec![own[0] as i32; target.faces.len()]);
                }
                (Some(_), true) => target.face_materials = None,
                (None, true) => {}
            }
            node.data = ImportNodeData::Mesh(mesh);
        } else if let Some(attribute) = children_of(id).filter(|l| l.2.is_none()).find_map(|l| {
            objects
                .children
                .iter()
                .find(|o| id_of(o) == Some(l.0) && o.name == "NodeAttribute")
        }) {
            // A camera or a light: FBX's look along +X and shine along -Y
            // in their own space, where the IR's, as glTF's, face -Z. Turned
            // here, after the model's own rotation, so the rest is the same
            // placement every format arrives with.
            let name = node.name.clone();
            match attribute.value(2).and_then(Value::as_str) {
                Some("Camera") => {
                    scene.cameras.push(camera_of(attribute, name));
                    node.data = ImportNodeData::Camera(scene.cameras.len() - 1);
                    node.transform.rotation *= Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
                }
                Some("Light") => {
                    if let Some(light) = light_of(attribute, name) {
                        scene.lights.push(light);
                        node.data = ImportNodeData::Light(scene.lights.len() - 1);
                        node.transform.rotation *=
                            Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
                        lights_assumed = true;
                    }
                }
                _ => {}
            }
        }
        scene.nodes.push(node);
        node_of.push((id, scene.nodes.len() - 1));
    }
    for (id, node) in &node_of {
        scene.nodes[*node].parent =
            parents_of(*id).find_map(|p| node_of.iter().find(|(m, _)| *m == p).map(|(_, n)| *n));
    }
    read_skins(objects, &links, &node_of, &mesh_of, &mut scene);
    read_takes(objects, &links, &models, &node_of, &mut scene);
    read_blend_shapes(objects, &links, &node_of, &mesh_of, &mut scene);
    if lights_assumed {
        scene.source.notes.push(LossyNote::Unsupported {
            what: "FBX lights have no physical unit: an intensity of 100 was taken as \
                   100 candela (a bright bulb), and a directional one's as 1000 lux"
                .into(),
        });
    }
    Ok(scene)
}

/// FBX's clock: ticks a second.
const TICKS: f64 = 46_186_158_000.0;

/// A matrix of sixteen numbers, as FBX writes them: column by column.
fn matrix(node: &Node, name: &str) -> Option<Mat4> {
    let v = node.child(name)?.value(0)?.floats();
    (v.len() == 16).then(|| {
        let a: [f32; 16] = std::array::from_fn(|i| v[i] as f32);
        Mat4::from_cols_array(&a)
    })
}

/// A node's placement in the file's space: its parents' and its own.
fn file_world(scene: &ImportScene, node: usize) -> Mat4 {
    let mut out = Mat4::IDENTITY;
    let mut at = Some(node);
    for _ in 0..=scene.nodes.len() {
        let Some(n) = at.and_then(|i| scene.nodes.get(i)) else {
            break;
        };
        let t = &n.transform;
        out = Mat4::from_scale_rotation_translation(t.scale, t.rotation, t.translation) * out;
        at = n.parent;
    }
    out
}

/// Every `Skin` deformer as the IR's skin of the models that wear its
/// geometry, and its clusters' weights as their meshes' vertex skins.
fn read_skins(
    objects: &Node,
    links: &[(i64, i64, Option<String>)],
    node_of: &[(i64, usize)],
    mesh_of: &[(i64, usize)],
    scene: &mut ImportScene,
) {
    let id_of = |n: &Node| n.value(0).and_then(Value::as_i64);
    let deformer = |kind: &'static str| {
        objects
            .children_named("Deformer")
            .filter(move |d| d.value(2).and_then(Value::as_str) == Some(kind))
    };
    let clusters: Vec<&Node> = deformer("Cluster").collect();
    for skin in deformer("Skin") {
        let Some(skin_id) = id_of(skin) else { continue };
        // The geometry it deforms, and the models wearing it.
        let Some(geometry) = links
            .iter()
            .find(|l| l.0 == skin_id && mesh_of.iter().any(|(g, _)| *g == l.1))
            .map(|l| l.1)
        else {
            continue;
        };
        let wearers: Vec<usize> = links
            .iter()
            .filter(|l| l.0 == geometry)
            .filter_map(|l| node_of.iter().find(|(m, _)| *m == l.1).map(|(_, n)| *n))
            .collect();
        // Where the mesh is, in the file's space, as its models place it.
        // A cluster's own `Transform` is meant to say so, but what writers
        // put there differs — Mixamo writes the bone's bind undone instead —
        // and the models do not lie.
        let mesh_world = wearers
            .first()
            .map(|n| file_world(scene, *n))
            .unwrap_or(Mat4::IDENTITY);
        // Its clusters, each with the bone that moves it.
        let mut joints = Vec::new();
        let mut binds = Vec::new();
        let mut weights: Vec<(Vec<i64>, Vec<f64>)> = Vec::new();
        for cluster in &clusters {
            let Some(cluster_id) = id_of(cluster) else {
                continue;
            };
            if !links.iter().any(|l| l.0 == cluster_id && l.1 == skin_id) {
                continue;
            }
            let Some(bone) = links
                .iter()
                .filter(|l| l.1 == cluster_id)
                .find_map(|l| node_of.iter().find(|(m, _)| *m == l.0).map(|(_, n)| *n))
            else {
                continue;
            };
            joints.push(bone);
            // From the mesh to the bone, at the bind: the bone's placement
            // then (`TransformLink`, which every writer agrees on) undone
            // after the mesh's.
            binds.push(matrix(cluster, "TransformLink").map(|link| link.inverse() * mesh_world));
            weights.push((
                cluster
                    .child("Indexes")
                    .and_then(|v| v.value(0))
                    .map(Value::ints)
                    .unwrap_or_default(),
                cluster
                    .child("Weights")
                    .and_then(|v| v.value(0))
                    .map(Value::floats)
                    .unwrap_or_default(),
            ));
        }
        if joints.is_empty() || joints.len() > usize::from(u16::MAX) {
            continue;
        }
        // One skeleton, one skin: Mixamo binds a body and its joints as two
        // meshes to one set of bones, and a bone can belong to one armature.
        // A skin that shares a bone with one read before joins it, its
        // clusters renumbered into the first's joints.
        let s = match scene
            .skins
            .iter()
            .position(|k| k.joints.iter().any(|j| joints.contains(j)))
        {
            Some(s) => {
                let first = &mut scene.skins[s];
                let mut renumbered = Vec::with_capacity(joints.len());
                for (j, bind) in joints.iter().zip(&binds) {
                    let at = match first.joints.iter().position(|k| k == j) {
                        Some(at) => at,
                        None => {
                            first.joints.push(*j);
                            match (&mut first.inverse_bind, bind) {
                                (Some(list), Some(m)) => list.push(*m),
                                _ => first.inverse_bind = None,
                            }
                            first.joints.len() - 1
                        }
                    };
                    renumbered.push(at);
                }
                // The weights now count the first skin's joints.
                weights = renumbered.iter().zip(weights).fold(
                    vec![(Vec::new(), Vec::new()); scene.skins[s].joints.len()],
                    |mut out, (at, (i, w))| {
                        out[*at].0.extend(i);
                        out[*at].1.extend(w);
                        out
                    },
                );
                s
            }
            None => {
                // Named as Blender names it, not after the deformer: a skin
                // is called after its mesh, and the mesh keeps that name.
                scene.skins.push(crate::ir::ImportSkin {
                    name: "Armature".into(),
                    joints,
                    inverse_bind: binds.into_iter().collect(),
                });
                weights.resize(
                    scene.skins.last().map_or(0, |k| k.joints.len()),
                    (Vec::new(), Vec::new()),
                );
                scene.skins.len() - 1
            }
        };
        for node in wearers {
            let ImportNodeData::Mesh(m) = scene.nodes[node].data else {
                continue;
            };
            scene.nodes[node].skin = Some(s);
            let count = scene.meshes[m].positions.len();
            let mut each: Vec<Vec<(u16, f32)>> = vec![Vec::new(); count];
            for (k, (indexes, ws)) in weights.iter().enumerate() {
                for (i, w) in indexes.iter().zip(ws) {
                    if let Some(slot) = usize::try_from(*i).ok().and_then(|i| each.get_mut(i)) {
                        if *w > 0.0 {
                            slot.push((k as u16, *w as f32));
                        }
                    }
                }
            }
            // The four that move it most, summing to one.
            let mut vertex = crate::ir::VertexSkin::default();
            for mut slot in each {
                slot.sort_by(|a, b| b.1.total_cmp(&a.1));
                slot.truncate(4);
                let total: f32 = slot.iter().map(|(_, w)| w).sum();
                let mut j = [0u16; 4];
                let mut w = [0.0f32; 4];
                for (i, (joint, weight)) in slot.iter().enumerate() {
                    j[i] = *joint;
                    w[i] = if total > 0.0 { weight / total } else { 0.0 };
                }
                vertex.joints.push(j);
                vertex.weights.push(w);
            }
            if scene.vertex_skins.len() <= m {
                scene.vertex_skins.resize(m + 1, None);
            }
            scene.vertex_skins[m] = Some(vertex);
        }
    }
    // The bones no vertex follows — a finger's tip, an eye, the top of the
    // head — are still bones: their parent's bind carried down by their own
    // placement. Left out, each became an empty hung from a joint.
    let limbs: Vec<usize> = node_of
        .iter()
        .filter(|(id, _)| {
            objects.children_named("Model").any(|m| {
                id_of(m) == Some(*id) && m.value(2).and_then(Value::as_str) == Some("LimbNode")
            })
        })
        .map(|(_, n)| *n)
        .collect();
    let local = |n: &ImportNode| {
        Mat4::from_scale_rotation_translation(
            n.transform.scale,
            n.transform.rotation,
            n.transform.translation,
        )
    };
    for skin in &mut scene.skins {
        loop {
            let mut grew = false;
            for &limb in &limbs {
                if skin.joints.contains(&limb) {
                    continue;
                }
                let Some(parent) = scene.nodes[limb].parent else {
                    continue;
                };
                let Some(k) = skin.joints.iter().position(|j| *j == parent) else {
                    continue;
                };
                skin.joints.push(limb);
                if let Some(list) = &mut skin.inverse_bind {
                    // From the mesh to the child: its parent's, then its own
                    // placement undone.
                    let bind = local(&scene.nodes[limb]).inverse() * list[k];
                    list.push(bind);
                }
                grew = true;
            }
            if !grew {
                break;
            }
        }
    }
}

/// One curve: its keys in seconds and its values.
struct Curve {
    times: Vec<f64>,
    values: Vec<f64>,
}

impl Curve {
    /// The value at `t`, straight between keys and held past either end.
    fn at(&self, t: f64) -> Option<f64> {
        let n = self.times.len().min(self.values.len());
        if n == 0 {
            return None;
        }
        let i = self.times[..n].partition_point(|k| *k <= t);
        Some(match i {
            0 => self.values[0],
            i if i >= n => self.values[n - 1],
            i => {
                let (t0, t1) = (self.times[i - 1], self.times[i]);
                let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
                self.values[i - 1] + (self.values[i] - self.values[i - 1]) * f
            }
        })
    }
}

/// One blend-shape channel as read: the weight range each of its targets
/// covers, its percent, its keys, and the curve node keying it.
type ShapeChannel = (Vec<(f32, f32)>, f32, Option<Curve>, Option<i64>);

/// Every `BlendShape` deformer as its mesh's morph targets: each channel a
/// target — or, with in-between shapes (`FullWeights`), a run of them, one
/// from each shape to the next, turned up over its stretch of the channel's
/// percent — its `DeformPercent` the models' weight, and its keys a morph
/// channel in the take they are keyed in.
fn read_blend_shapes(
    objects: &Node,
    links: &[(i64, i64, Option<String>)],
    node_of: &[(i64, usize)],
    mesh_of: &[(i64, usize)],
    scene: &mut ImportScene,
) {
    let id_of = |n: &Node| n.value(0).and_then(Value::as_i64);
    let find = |id: i64| objects.children.iter().find(|o| id_of(o) == Some(id));
    let children = |id: i64, kind: &'static str| -> Vec<&Node> {
        links
            .iter()
            .filter(|l| l.1 == id)
            .filter_map(|l| find(l.0))
            .filter(|n| n.value(2).and_then(Value::as_str) == Some(kind))
            .collect()
    };
    for deformer in objects
        .children_named("Deformer")
        .filter(|d| d.value(2).and_then(Value::as_str) == Some("BlendShape"))
    {
        let Some(deformer_id) = id_of(deformer) else {
            continue;
        };
        let Some((geometry, mesh)) = links
            .iter()
            .filter(|l| l.0 == deformer_id)
            .find_map(|l| mesh_of.iter().find(|(g, _)| *g == l.1).copied())
        else {
            continue;
        };
        let count = scene.meshes[mesh].positions.len();
        let wearers: Vec<usize> = links
            .iter()
            .filter(|l| l.0 == geometry)
            .filter_map(|l| node_of.iter().find(|(m, _)| *m == l.1).map(|(_, n)| *n))
            .collect();
        let mut targets: Vec<(String, Vec<Vec3>)> = Vec::new();
        // Per channel: its weight ranges, its percent and its keys.
        let mut channels: Vec<ShapeChannel> = Vec::new();
        for channel in children(deformer_id, "BlendShapeChannel") {
            let Some(channel_id) = id_of(channel) else {
                continue;
            };
            let name = clean_name(channel.value(1).and_then(Value::as_str).unwrap_or("Shape"));
            let shapes = children(channel_id, "Shape");
            if shapes.is_empty() {
                continue;
            }
            let full_weights: Vec<f64> = channel
                .child("FullWeights")
                .and_then(|v| v.value(0))
                .map(Value::floats)
                .unwrap_or_default();
            // Each shape, its offsets spread to every point, at its weight.
            let mut stops: Vec<(f32, Vec<Vec3>)> = shapes
                .iter()
                .enumerate()
                .map(|(k, shape)| {
                    let indexes = shape
                        .child("Indexes")
                        .and_then(|v| v.value(0))
                        .map(Value::ints)
                        .unwrap_or_default();
                    let offsets = shape
                        .child("Vertices")
                        .and_then(|v| v.value(0))
                        .map(Value::floats)
                        .unwrap_or_default();
                    let mut full = vec![Vec3::ZERO; count];
                    for (i, o) in indexes.iter().zip(offsets.as_chunks::<3>().0) {
                        if let Some(slot) = full.get_mut(*i as usize) {
                            *slot = Vec3::new(o[0] as f32, o[1] as f32, o[2] as f32);
                        }
                    }
                    let weight = full_weights.get(k).map_or(1.0, |w| (*w / 100.0) as f32);
                    (weight, full)
                })
                .collect();
            stops.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut ranges = Vec::new();
            let (mut low, mut before) = (0.0f32, vec![Vec3::ZERO; count]);
            for (k, (w, at)) in stops.iter().enumerate() {
                let step: Vec<Vec3> = at.iter().zip(&before).map(|(a, b)| *a - *b).collect();
                let key = if k + 1 == stops.len() {
                    name.clone()
                } else {
                    format!("{name}_inbetween{}", k + 1)
                };
                targets.push((key, step));
                ranges.push((low, *w));
                low = *w;
                before = at.clone();
            }
            let percent = channel.property_f64("DeformPercent").unwrap_or(0.0) as f32;
            // Its keys: a curve node on its DeformPercent, and the curve
            // under it — and the take that node's layer is in.
            let keyed = links
                .iter()
                .filter(|l| l.1 == channel_id && l.2.as_deref() == Some("DeformPercent"))
                .filter_map(|l| {
                    find(l.0)
                        .filter(|n| n.name == "AnimationCurveNode")
                        .map(|n| (l.0, n))
                })
                .next();
            let curve = keyed.and_then(|(node_id, _)| {
                links
                    .iter()
                    .filter(|l| l.1 == node_id)
                    .filter_map(|l| find(l.0))
                    .find(|n| n.name == "AnimationCurve")
                    .and_then(|c| {
                        let times: Vec<f64> = c
                            .child("KeyTime")
                            .and_then(|v| v.value(0))
                            .map(Value::ints)
                            .unwrap_or_default()
                            .iter()
                            .map(|t| *t as f64 / TICKS)
                            .collect();
                        let values = c
                            .child("KeyValueFloat")
                            .and_then(|v| v.value(0))
                            .map(Value::floats)
                            .unwrap_or_default();
                        (!times.is_empty() && !values.is_empty()).then_some(Curve { times, values })
                    })
            });
            channels.push((ranges, percent, curve, keyed.map(|(id, _)| id)));
        }
        if targets.is_empty() {
            continue;
        }
        // A channel's percent as its targets' weights.
        let spread = |ranges: &[(f32, f32)], percent: f32, out: &mut Vec<f32>| {
            let w = percent / 100.0;
            if ranges.len() == 1 {
                out.push(w);
            } else {
                for (low, high) in ranges {
                    out.push(((w - low) / (high - low).max(1e-6)).clamp(0.0, 1.0));
                }
            }
        };
        let mut rest = Vec::new();
        for (ranges, percent, _, _) in &channels {
            spread(ranges, *percent, &mut rest);
        }
        if scene.vertex_morphs.len() < scene.meshes.len() {
            scene.vertex_morphs.resize(scene.meshes.len(), Vec::new());
        }
        scene.vertex_morphs[mesh] = targets;
        for node in &wearers {
            if scene.node_weights.len() <= *node {
                scene.node_weights.resize(node + 1, Vec::new());
            }
            scene.node_weights[*node] = rest.clone();
        }
        // The keys, from the take's start: what any channel is keyed at.
        if channels.iter().all(|c| c.2.is_none()) {
            continue;
        }
        let mut times: Vec<f64> = channels
            .iter()
            .filter_map(|c| c.2.as_ref())
            .flat_map(|c| c.times.iter().copied())
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
        // The take: the stack above the keyed node's layer, and its start.
        let stack = channels
            .iter()
            .find_map(|c| c.3)
            .and_then(|node| links.iter().find(|l| l.0 == node).map(|l| l.1))
            .and_then(|layer| links.iter().find(|l| l.0 == layer).map(|l| l.1))
            .and_then(find);
        let first = times.first().copied().unwrap_or(0.0);
        let start = stack
            .and_then(|s| s.property_f64("LocalStart"))
            .map(|t| t / TICKS)
            .filter(|t| *t <= first)
            .unwrap_or(first);
        let name = stack
            .and_then(|s| s.value(1).and_then(Value::as_str))
            .map(clean_name)
            .unwrap_or_else(|| "Take".into());
        let mut values = Vec::with_capacity(times.len() * rest.len());
        for t in &times {
            for (ranges, percent, curve, _) in &channels {
                let p = curve
                    .as_ref()
                    .and_then(|c| c.at(*t))
                    .map_or(*percent, |v| v as f32);
                spread(ranges, p, &mut values);
            }
        }
        for node in wearers {
            scene.morph_channels.push(crate::ir::ImportMorphChannel {
                animation: name.clone(),
                node,
                interpolation: crate::ir::ChannelInterpolation::Linear,
                times: times.iter().map(|t| (t - start) as f32).collect(),
                values: values.clone(),
            });
        }
    }
}

/// Every `AnimationStack` — a take — as a clip of the IR: each model's
/// moving parts as channels, keyed where any of its curves is keyed.
fn read_takes(
    objects: &Node,
    links: &[(i64, i64, Option<String>)],
    models: &[&Node],
    node_of: &[(i64, usize)],
    scene: &mut ImportScene,
) {
    use crate::ir::{ChannelInterpolation, ChannelValues, ImportAnimation, ImportChannel};
    let id_of = |n: &Node| n.value(0).and_then(Value::as_i64);
    let find = |id: i64| objects.children.iter().find(|o| id_of(o) == Some(id));
    for stack in objects.children_named("AnimationStack") {
        let Some(stack_id) = id_of(stack) else {
            continue;
        };
        let layers: Vec<i64> = links
            .iter()
            .filter(|l| l.1 == stack_id)
            .filter(|l| find(l.0).is_some_and(|o| o.name == "AnimationLayer"))
            .map(|l| l.0)
            .collect();
        // Model id, then what it animates and its three curves.
        let mut moving: Vec<(i64, &str, [Option<Curve>; 3], Vec3)> = Vec::new();
        for link in links.iter().filter(|l| layers.contains(&l.1)) {
            let Some(node) = find(link.0).filter(|o| o.name == "AnimationCurveNode") else {
                continue;
            };
            let Some((model, part)) = links.iter().find_map(|l| {
                (l.0 == link.0)
                    .then_some(())
                    .and(l.2.as_deref())
                    .filter(|p| matches!(*p, "Lcl Translation" | "Lcl Rotation" | "Lcl Scaling"))
                    .map(|p| (l.1, p))
            }) else {
                continue;
            };
            let defaults = Vec3::new(
                node.property_f64("d|X").unwrap_or(0.0) as f32,
                node.property_f64("d|Y").unwrap_or(0.0) as f32,
                node.property_f64("d|Z").unwrap_or(0.0) as f32,
            );
            let mut curves: [Option<Curve>; 3] = [None, None, None];
            for l in links.iter().filter(|l| l.1 == link.0) {
                let axis = match l.2.as_deref() {
                    Some("d|X") => 0,
                    Some("d|Y") => 1,
                    Some("d|Z") => 2,
                    _ => continue,
                };
                let Some(curve) = find(l.0).filter(|o| o.name == "AnimationCurve") else {
                    continue;
                };
                let times: Vec<f64> = curve
                    .child("KeyTime")
                    .and_then(|v| v.value(0))
                    .map(Value::ints)
                    .unwrap_or_default()
                    .iter()
                    .map(|t| *t as f64 / TICKS)
                    .collect();
                let values = curve
                    .child("KeyValueFloat")
                    .and_then(|v| v.value(0))
                    .map(Value::floats)
                    .unwrap_or_default();
                if !times.is_empty() && !values.is_empty() {
                    curves[axis] = Some(Curve { times, values });
                }
            }
            if curves.iter().any(Option::is_some) {
                moving.push((model, part, curves, defaults));
            }
        }
        if moving.is_empty() {
            continue;
        }
        // From the take's start, or its first key.
        let first = moving
            .iter()
            .flat_map(|(_, _, c, _)| c.iter().flatten())
            .filter_map(|c| c.times.first().copied())
            .fold(f64::INFINITY, f64::min);
        let start = stack
            .property_f64("LocalStart")
            .map(|t| t / TICKS)
            .filter(|t| *t <= first)
            .unwrap_or(first);
        let mut clip = ImportAnimation {
            name: clean_name(stack.value(1).and_then(Value::as_str).unwrap_or("Take")),
            channels: Vec::new(),
        };
        let mut ids: Vec<i64> = moving.iter().map(|m| m.0).collect();
        ids.sort_unstable();
        ids.dedup();
        for model_id in ids {
            let (Some(node), Some(model)) = (
                node_of
                    .iter()
                    .find(|(m, _)| *m == model_id)
                    .map(|(_, n)| *n),
                models.iter().find(|m| id_of(m) == Some(model_id)),
            ) else {
                continue;
            };
            let parts: Vec<&(i64, &str, [Option<Curve>; 3], Vec3)> =
                moving.iter().filter(|m| m.0 == model_id).collect();
            let mut times: Vec<f64> = parts
                .iter()
                .flat_map(|(_, _, c, _)| c.iter().flatten())
                .flat_map(|c| c.times.iter().copied())
                .collect();
            times.sort_by(f64::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
            let order = model.property_f64("RotationOrder").unwrap_or(0.0) as i64;
            let pre = euler(
                model.property_vec3("PreRotation").unwrap_or(Vec3::ZERO),
                order,
            );
            let post = euler(
                model.property_vec3("PostRotation").unwrap_or(Vec3::ZERO),
                order,
            );
            // A camera or light was turned to face the IR's way; its
            // keys are turned the same.
            let turn = match scene.nodes[node].data {
                ImportNodeData::Camera(_) => Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
                ImportNodeData::Light(_) => Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                _ => Quat::IDENTITY,
            };
            let seconds: Vec<f32> = times.iter().map(|t| (t - start) as f32).collect();
            for (_, part, curves, defaults) in parts {
                let value = |t: f64, still: Vec3| {
                    Vec3::new(
                        curves[0].as_ref().and_then(|c| c.at(t)).map_or(
                            if curves[0].is_none() {
                                still.x
                            } else {
                                defaults.x
                            },
                            |v| v as f32,
                        ),
                        curves[1].as_ref().and_then(|c| c.at(t)).map_or(
                            if curves[1].is_none() {
                                still.y
                            } else {
                                defaults.y
                            },
                            |v| v as f32,
                        ),
                        curves[2].as_ref().and_then(|c| c.at(t)).map_or(
                            if curves[2].is_none() {
                                still.z
                            } else {
                                defaults.z
                            },
                            |v| v as f32,
                        ),
                    )
                };
                let values = match *part {
                    "Lcl Translation" => {
                        let still = model.property_vec3("Lcl Translation").unwrap_or(*defaults);
                        ChannelValues::Translation(times.iter().map(|t| value(*t, still)).collect())
                    }
                    "Lcl Rotation" => {
                        let still = model.property_vec3("Lcl Rotation").unwrap_or(*defaults);
                        ChannelValues::Rotation(
                            times
                                .iter()
                                .map(|t| {
                                    pre * euler(value(*t, still), order) * post.inverse() * turn
                                })
                                .collect(),
                        )
                    }
                    _ => {
                        let still = model.property_vec3("Lcl Scaling").unwrap_or(Vec3::ONE);
                        ChannelValues::Scale(times.iter().map(|t| value(*t, still)).collect())
                    }
                };
                clip.channels.push(ImportChannel {
                    node,
                    interpolation: ChannelInterpolation::Linear,
                    times: seconds.clone(),
                    values,
                });
            }
        }
        if !clip.channels.is_empty() {
            scene.animations.push(clip);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quad and a triangle, Y up, in metres (100 cm a unit), one model
    /// under another with a material each face: what a small export looks
    /// like in ASCII.
    const ASCII: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	Creator: "Test"
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UpAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Geometry: 100, "Geometry::Shape", "Mesh" {
		Vertices: *15 {
			a: 0,0,0,1,0,0,1,1,0,0,1,0,0.5,2,0
		}
		PolygonVertexIndex: *7 {
			a: 0,1,2,-4,3,2,-5
		}
		LayerElementNormal: 0 {
			MappingInformationType: "ByVertice"
			ReferenceInformationType: "Direct"
			Normals: *15 {
				a: 0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
			}
		}
		LayerElementUV: 0 {
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *4 {
				a: 0,0,1,1
			}
			UVIndex: *7 {
				a: 0,1,1,0,0,1,1
			}
		}
		LayerElementMaterial: 0 {
			MappingInformationType: "ByPolygon"
			ReferenceInformationType: "IndexToDirect"
			Materials: *2 {
				a: 1,0
			}
		}
	}
	Model: 200, "Model::Root", "Null" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",1,2,3
		}
	}
	Model: 300, "Model::Shape", "Mesh" {
		Properties70:  {
			P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,90
			P: "Lcl Scaling", "Lcl Scaling", "", "A",2,2,2
		}
	}
	Material: 400, "Material::Red", "" {
		Properties70:  {
			P: "DiffuseColor", "Color", "", "A",1,0,0
		}
	}
	Material: 500, "Material::Blue", "" {
		Properties70:  {
			P: "DiffuseColor", "Color", "", "A",0,0,1
			P: "ShininessExponent", "Number", "", "A",100
		}
	}
	AnimationStack: 600, "AnimStack::Take", "" {
	}
}
Connections:  {
	C: "OO",300,200
	C: "OO",200,0
	C: "OO",100,300
	C: "OO",400,300
	C: "OO",500,300
}
"#;

    fn none(_: &str) -> Option<Vec<u8>> {
        None
    }

    #[test]
    fn an_ascii_file_comes_through_with_its_hierarchy_and_materials() {
        let scene = parse_fbx(ASCII.as_bytes(), &none).expect("reads");
        assert_eq!(scene.source.generator.as_deref(), Some("Test"));
        assert_eq!(scene.source.axis.up, UpAxis::YUp);
        assert!(
            (scene.source.unit_scale - 1.0).abs() < 1e-6,
            "100 cm a unit"
        );
        let mesh = &scene.meshes[0];
        assert_eq!(mesh.faces, vec![vec![0, 1, 2, 3], vec![3, 2, 4]]);
        assert_eq!(mesh.corner_normals.as_ref().map(Vec::len), Some(7));
        assert_eq!(
            mesh.corner_uvs.as_ref().map(|u| u[1]),
            Some(Vec2::new(1.0, 1.0))
        );
        // The model's materials in its order: Red then Blue, so the quad's
        // index one is Blue.
        assert_eq!(scene.materials[0].name, "Red");
        assert_eq!(mesh.face_materials, Some(vec![1, 0]));
        assert!(scene.materials[1].roughness < 0.01);
        // Shape under Root, turned and scaled.
        assert_eq!(scene.nodes.len(), 2);
        assert_eq!(scene.nodes[0].name, "Root");
        assert_eq!(
            scene.nodes[0].transform.translation,
            Vec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(scene.nodes[1].parent, Some(0));
        assert_eq!(scene.nodes[1].data, ImportNodeData::Mesh(0));
        let turned = scene.nodes[1].transform.rotation * Vec3::X;
        assert!((turned - Vec3::Y).length() < 1e-5, "{turned}");
        assert_eq!(scene.nodes[1].transform.scale, Vec3::splat(2.0));
        assert!(scene.animations.is_empty(), "an empty take is no clip");
    }

    #[test]
    fn an_import_lands_in_the_document_z_up_and_in_metres() {
        // Y up in the file, so the root's (1, 2, 3) is (1, -3, 2) here; a
        // hundred centimetres a unit, so metres as written.
        let scene = parse_fbx(ASCII.as_bytes(), &none).expect("reads");
        let mut doc = kreate_scene::Document::new();
        let (made, _) = crate::build_into(&scene, &mut doc);
        assert_eq!(made.len(), 2);
        let root = doc
            .objects()
            .find(|(_, o)| o.name == "Root")
            .map(|(id, _)| id)
            .expect("the root");
        let at = doc.world_transform(root).translation;
        assert!(
            (at - kreate_math::DVec3::new(1.0, -3.0, 2.0)).length() < 1e-5,
            "{at}"
        );
        let shape = doc
            .objects()
            .find(|(_, o)| o.name == "Shape")
            .map(|(id, _)| id)
            .expect("the shape");
        assert_eq!(doc.object(shape).and_then(|o| o.parent), Some(root));
    }

    #[test]
    fn a_camera_and_a_light_arrive_facing_where_fbx_faced_them() {
        // FBX's camera looks along its +X and its light shines along -Y; at
        // rest, in a Y-up file, that is along +X and straight down.
        let text = ASCII.replace(
            "\tAnimationStack: 600",
            r#"	NodeAttribute: 700, "NodeAttribute::Cam", "Camera" {
		Properties70:  {
			P: "FocalLength", "Number", "", "A",50
			P: "FilmHeight", "Number", "", "A",0.945
		}
	}
	Model: 710, "Model::Cam", "Camera" {
	}
	NodeAttribute: 800, "NodeAttribute::Sun", "Light" {
		Properties70:  {
			P: "LightType", "enum", "", "",1
			P: "Intensity", "Number", "", "A",50
		}
	}
	Model: 810, "Model::Sun", "Light" {
	}
	AnimationStack: 600"#,
        )
        .replace(
            "\tC: \"OO\",500,300\n",
            "\tC: \"OO\",500,300\n\tC: \"OO\",700,710\n\tC: \"OO\",710,0\n\tC: \"OO\",800,810\n\tC: \"OO\",810,0\n",
        );
        let scene = parse_fbx(text.as_bytes(), &none).expect("reads");
        assert_eq!(scene.cameras.len(), 1);
        let fov = scene.cameras[0].fov_y.expect("perspective");
        // A 50 mm lens on a 24 mm film back: about 27 degrees.
        assert!(
            (fov.to_degrees() - 27.0).abs() < 1.0,
            "{}",
            fov.to_degrees()
        );
        assert_eq!(scene.lights.len(), 1);
        assert!(matches!(
            scene.lights[0].kind,
            crate::ir::ImportLightKind::Directional
        ));
        assert_eq!(scene.lights[0].intensity, 500.0);
        let mut doc = kreate_scene::Document::new();
        crate::build_into(&scene, &mut doc);
        let facing = |name: &str| {
            let (id, _) = doc.objects().find(|(_, o)| o.name == name).expect("there");
            doc.world_transform(id).matrix3 * kreate_math::DVec3::NEG_Z
        };
        let cam = facing("Cam");
        assert!(
            (cam - kreate_math::DVec3::X).length() < 1e-4,
            "the camera looks {cam}"
        );
        let sun = facing("Sun");
        assert!(
            (sun - kreate_math::DVec3::NEG_Z).length() < 1e-4,
            "the sun shines {sun}"
        );
    }

    /// Writes a record tree as binary FBX, as a writer of the given version
    /// would — arrays compressed or not.
    fn write_binary(nodes: &[Node], version: u32, compress: bool) -> Vec<u8> {
        fn value(out: &mut Vec<u8>, v: &Value, compress: bool) {
            match v {
                Value::Int(i) => {
                    out.push(b'L');
                    out.extend(i.to_le_bytes());
                }
                Value::Float(f) => {
                    out.push(b'D');
                    out.extend(f.to_le_bytes());
                }
                Value::Text(s) => {
                    out.push(b'S');
                    out.extend((s.len() as u32).to_le_bytes());
                    out.extend(s.as_bytes());
                }
                Value::Bytes(b) => {
                    out.push(b'R');
                    out.extend((b.len() as u32).to_le_bytes());
                    out.extend(b);
                }
                Value::Ints(v) => {
                    let raw: Vec<u8> = v.iter().flat_map(|i| (*i as i32).to_le_bytes()).collect();
                    array(out, b'i', v.len(), &raw, compress);
                }
                Value::Floats(v) => {
                    let raw: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                    array(out, b'd', v.len(), &raw, compress);
                }
            }
        }
        fn array(out: &mut Vec<u8>, code: u8, count: usize, raw: &[u8], compress: bool) {
            out.push(code);
            out.extend((count as u32).to_le_bytes());
            let data = if compress {
                use std::io::Write;
                let mut e =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                e.write_all(raw).expect("compressed");
                e.finish().expect("compressed")
            } else {
                raw.to_vec()
            };
            out.extend((compress as u32).to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
            out.extend(data);
        }
        fn node(out: &mut Vec<u8>, n: &Node, wide: bool, compress: bool) {
            let word = if wide { 8 } else { 4 };
            let start = out.len();
            out.extend(vec![0u8; 3 * word]);
            out.push(n.name.len() as u8);
            out.extend(n.name.as_bytes());
            let values_start = out.len();
            for v in &n.values {
                value(out, v, compress);
            }
            let values_len = out.len() - values_start;
            if !n.children.is_empty() {
                for c in &n.children {
                    node(out, c, wide, compress);
                }
                out.extend(vec![0u8; 3 * word + 1]);
            }
            let end = out.len();
            let put = |out: &mut Vec<u8>, k: usize, v: usize| {
                if wide {
                    out[start + k * 8..start + k * 8 + 8]
                        .copy_from_slice(&(v as u64).to_le_bytes());
                } else {
                    out[start + k * 4..start + k * 4 + 4]
                        .copy_from_slice(&(v as u32).to_le_bytes());
                }
            };
            put(out, 0, end);
            put(out, 1, n.values.len());
            put(out, 2, values_len);
        }
        let wide = version >= 7500;
        let mut out = MAGIC.to_vec();
        out.extend([0x1a, 0x00]);
        out.extend(version.to_le_bytes());
        for n in nodes {
            node(&mut out, n, wide, compress);
        }
        out.extend(vec![0u8; if wide { 25 } else { 13 }]);
        out
    }

    /// Two bones, a strip skinned to them — the bottom to the root, the top
    /// to the tip — and a take that turns the tip a quarter about Z in a
    /// second. The clusters' `Transform` is Mixamo's: the bone's bind undone,
    /// not the mesh's placement.
    const SKINNED: &str = r#"; FBX 7.4.0 project file
GlobalSettings:  {
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Geometry: 100, "Geometry::Strip", "Mesh" {
		Vertices: *12 {
			a: 0,0,0,1,0,0,0,2,0,1,2,0
		}
		PolygonVertexIndex: *4 {
			a: 0,1,3,-3
		}
	}
	Model: 110, "Model::Body", "Mesh" {
	}
	Deformer: 120, "Deformer::Skin Body", "Skin" {
	}
	Deformer: 130, "SubDeformer::Cluster Root", "Cluster" {
		Indexes: *2 {
			a: 0,1
		}
		Weights: *2 {
			a: 1,1
		}
		Transform: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
	}
	Deformer: 140, "SubDeformer::Cluster Tip", "Cluster" {
		Indexes: *2 {
			a: 2,3
		}
		Weights: *2 {
			a: 1,1
		}
		Transform: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,-1,0,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1
		}
	}
	Model: 150, "Model::Root", "LimbNode" {
	}
	Model: 160, "Model::Tip", "LimbNode" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",0,1,0
		}
	}
	Model: 170, "Model::Tip_End", "LimbNode" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",0,1,0
		}
	}
	AnimationStack: 900, "AnimStack::Wave", "" {
	}
	AnimationLayer: 901, "AnimLayer::Base", "" {
	}
	AnimationCurveNode: 902, "AnimCurveNode::R", "" {
		Properties70:  {
			P: "d|X", "Number", "", "A",0
			P: "d|Y", "Number", "", "A",0
			P: "d|Z", "Number", "", "A",0
		}
	}
	AnimationCurve: 903, "AnimCurve::", "" {
		KeyTime: *2 {
			a: 0,46186158000
		}
		KeyValueFloat: *2 {
			a: 0,90
		}
	}
}
Connections:  {
	C: "OO",100,110
	C: "OO",110,0
	C: "OO",120,100
	C: "OO",130,120
	C: "OO",140,120
	C: "OO",150,130
	C: "OO",160,140
	C: "OO",150,0
	C: "OO",160,150
	C: "OO",170,160
	C: "OO",901,900
	C: "OO",902,901
	C: "OP",902,160,"Lcl Rotation"
	C: "OP",903,902,"d|Z"
}
"#;

    const SHAPED: &str = r#"; FBX 7.4.0 project file
GlobalSettings:  {
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",2
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Geometry: 100, "Geometry::Face", "Mesh" {
		Vertices: *9 {
			a: 0,0,0,1,0,0,0,1,0
		}
		PolygonVertexIndex: *3 {
			a: 0,1,-3
		}
	}
	Model: 110, "Model::Face", "Mesh" {
	}
	Deformer: 120, "Deformer::Morphs", "BlendShape" {
	}
	Deformer: 130, "SubDeformer::Smile", "BlendShapeChannel" {
		DeformPercent: 25
		FullWeights: *2 {
			a: 50,100
		}
		Properties70:  {
			P: "DeformPercent", "Number", "", "A",25
		}
	}
	Geometry: 140, "Geometry::Smile_half", "Shape" {
		Indexes: *1 {
			a: 2
		}
		Vertices: *3 {
			a: 0,0,0.4
		}
	}
	Geometry: 150, "Geometry::Smile", "Shape" {
		Indexes: *1 {
			a: 2
		}
		Vertices: *3 {
			a: 0,0,0.5
		}
	}
	AnimationStack: 900, "AnimStack::Grin", "" {
	}
	AnimationLayer: 901, "AnimLayer::Base", "" {
	}
	AnimationCurveNode: 902, "AnimCurveNode::DeformPercent", "" {
	}
	AnimationCurve: 903, "AnimCurve::", "" {
		KeyTime: *2 {
			a: 0,46186158000
		}
		KeyValueFloat: *2 {
			a: 0,75
		}
	}
}
Connections:  {
	C: "OO",100,110
	C: "OO",110,0
	C: "OO",120,100
	C: "OO",130,120
	C: "OO",140,130
	C: "OO",150,130
	C: "OO",901,900
	C: "OO",902,901
	C: "OP",902,130,"DeformPercent"
	C: "OP",903,902,"d|DeformPercent"
}
"#;

    #[test]
    fn blend_shapes_come_in_with_their_in_betweens_and_keys() {
        let scene = parse_fbx(SHAPED.as_bytes(), &none).expect("reads");
        let targets = &scene.vertex_morphs[0];
        let names: Vec<&str> = targets.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["Smile_inbetween1", "Smile"]);
        assert_eq!(targets[0].1[2], Vec3::new(0.0, 0.0, 0.4));
        assert!((targets[1].1[2].z - 0.1).abs() < 1e-6);
        // 25% is halfway to the in-between at 50%.
        let face = scene.nodes.iter().position(|n| n.name == "Face").unwrap();
        assert_eq!(scene.node_weights[face], vec![0.5, 0.0]);
        // Keyed from 0% to 75%: at the end, the in-between whole and half
        // the rest.
        let keyed = &scene.morph_channels[0];
        assert_eq!(keyed.times.len(), 2);
        assert_eq!(&keyed.values[2..], &[1.0, 0.5]);
        let mut doc = kreate_scene::Document::new();
        crate::build_into(&scene, &mut doc);
        assert!(doc.objects().any(|(_, o)| o.shape_keys.len() == 2));
    }

    #[test]
    fn a_skinned_take_plays_its_bones_and_moves_the_mesh_with_them() {
        use kreate_scene::document::ObjectData;
        use kreate_scene::modifier::EvalCtx;
        let scene = parse_fbx(SKINNED.as_bytes(), &none).expect("reads");
        assert_eq!(scene.skins.len(), 1);
        assert_eq!(scene.animations.len(), 1);
        assert_eq!(scene.animations[0].name, "Wave");
        let mut doc = kreate_scene::Document::new();
        crate::build_into(&scene, &mut doc);
        let (_, rig) = doc
            .objects()
            .find(|(_, o)| matches!(o.data, ObjectData::Armature(_)))
            .expect("a rig");
        let (_, armature) = doc
            .armature_of(
                doc.objects()
                    .find(|(_, o)| matches!(o.data, ObjectData::Armature(_)))
                    .map(|(id, _)| id)
                    .expect("rig"),
            )
            .expect("bones");
        // The tip's end, weighted by nothing, is a bone too, not an empty.
        assert_eq!(armature.bones.len(), 3, "root, tip and its end");
        assert!(rig.action.is_some(), "the take plays");
        assert!(
            !doc.objects()
                .any(|(_, o)| matches!(o.data, ObjectData::Empty)),
            "no joint left over as an empty"
        );
        let body = doc
            .objects()
            .find(|(_, o)| o.name == "Body")
            .map(|(id, _)| id)
            .expect("the body");
        let top = |doc: &mut kreate_scene::Document, frame: f64| -> Vec<kreate_math::DVec3> {
            doc.animate(frame);
            let mesh = kreate_scene::ModifierCache::new()
                .evaluated(doc, body, &EvalCtx::still())
                .expect("a mesh");
            let world = doc.world_transform(body);
            let mut out: Vec<_> = mesh
                .positions()
                .iter()
                .map(|p| world.transform_point3(p.as_dvec3()))
                .filter(|p| p.z > 0.5)
                .collect();
            // By height, then across: the order the expected points are in.
            out.sort_by(|a, b| {
                (a.z * 1e3)
                    .round()
                    .total_cmp(&(b.z * 1e3).round())
                    .then(a.x.total_cmp(&b.x))
            });
            out
        };
        // Z up here: the file's (x, y, z) is (x, -z, y).
        let start = doc.playback.start as f64;
        let near = |a: &[kreate_math::DVec3], b: &[[f64; 3]]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(p, q)| (*p - kreate_math::DVec3::from_array(*q)).length() < 1e-3)
        };
        let rest = top(&mut doc, start);
        assert!(near(&rest, &[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]), "{rest:?}");
        // A second on, the top has swung a quarter about the tip, to its left.
        let end = start + doc.playback.fps;
        let swung = top(&mut doc, end);
        assert!(
            near(&swung, &[[-1.0, 0.0, 1.0], [-1.0, 0.0, 2.0]]),
            "{swung:?}"
        );
    }

    #[test]
    fn binary_files_old_and_new_compressed_or_not_read_as_the_ascii_does() {
        let tree = parse_ascii(ASCII).expect("parses");
        let from_ascii = parse_fbx(ASCII.as_bytes(), &none).expect("reads");
        for (version, compress) in [(7400, false), (7400, true), (7500, true)] {
            let bytes = write_binary(&tree, version, compress);
            assert_eq!(
                parse_tree(&bytes).expect("parses"),
                tree,
                "{version} {compress}"
            );
            let scene = parse_fbx(&bytes, &none).expect("reads");
            assert_eq!(scene.meshes[0].faces, from_ascii.meshes[0].faces);
            assert_eq!(scene.nodes.len(), 2);
        }
    }

    #[test]
    fn damage_is_an_error_not_a_hang_or_a_huge_allocation() {
        let tree = parse_ascii(ASCII).expect("parses");
        let bytes = write_binary(&tree, 7400, true);
        for cut in [30, 100, bytes.len() / 2] {
            let _ = parse_fbx(&bytes[..cut], &none);
        }
        // An array claiming four billion entries in a few bytes.
        let mut bad = MAGIC.to_vec();
        bad.extend([0x1a, 0x00]);
        bad.extend(7400u32.to_le_bytes());
        let mut record = vec![0u8; 12];
        record.push(1);
        record.push(b'X');
        record.push(b'd');
        record.extend(u32::MAX.to_le_bytes());
        record.extend(0u32.to_le_bytes());
        record.extend(8u32.to_le_bytes());
        record.extend([0u8; 8]);
        let end = (bad.len() + record.len()) as u32;
        record[0..4].copy_from_slice(&end.to_le_bytes());
        record[4..8].copy_from_slice(&1u32.to_le_bytes());
        bad.extend(record);
        assert!(parse_fbx(&bad, &none).is_err());
        assert!(matches!(parse_fbx(b"hello", &none), Err(FbxError::NotFbx)));
    }
}
