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
//! Animation, skinning, blend shapes, cameras and lights are named in the
//! notes and left behind; pivots are ignored, which is right for the many
//! files whose pivots are zero and an approximation for the rest.

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
                        data.chunks_exact(8)
                            .take(count)
                            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                            .collect(),
                    ),
                    b'f' => Value::Floats(
                        data.chunks_exact(4)
                            .take(count)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])) as f64)
                            .collect(),
                    ),
                    b'l' => Value::Ints(
                        data.chunks_exact(8)
                            .take(count)
                            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                            .collect(),
                    ),
                    b'i' => Value::Ints(
                        data.chunks_exact(4)
                            .take(count)
                            .map(|c| i32::from_le_bytes(c.try_into().unwrap_or([0; 4])) as i64)
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
    let mut unsupported: Vec<&str> = Vec::new();
    for o in &objects.children {
        let what = match (o.name.as_str(), o.value(2).and_then(Value::as_str)) {
            ("AnimationCurve" | "AnimationCurveNode" | "AnimationStack", _) => "animation",
            ("Deformer", Some("Skin")) => "skinning",
            ("Deformer", Some("BlendShape")) => "blend shapes",
            ("NodeAttribute", Some("Camera")) => "cameras",
            ("NodeAttribute", Some("Light")) => "lights",
            _ => continue,
        };
        if !unsupported.contains(&what) {
            unsupported.push(what);
        }
    }
    for what in unsupported {
        scene
            .source
            .notes
            .push(LossyNote::Unsupported { what: what.into() });
    }

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
            .chunks_exact(3)
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
        }
        scene.nodes.push(node);
        node_of.push((id, scene.nodes.len() - 1));
    }
    for (id, node) in &node_of {
        scene.nodes[*node].parent =
            parents_of(*id).find_map(|p| node_of.iter().find(|(m, _)| *m == p).map(|(_, n)| *n));
    }
    Ok(scene)
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
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what == "animation")));
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
