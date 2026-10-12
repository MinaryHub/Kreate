//! STEP (ISO 10303-21) read: the parts a CAD program sends, as meshes.
//!
//! A CAD file describes solids exactly — a hole is a cylinder, a fillet a
//! torus or a spline — and Kreate holds meshes, so every face is cut into
//! triangles here, finely enough that the facets stay within a small fraction
//! of the model's size of the true surface. What comes through:
//!
//! - **Solids and surfaces**: `MANIFOLD_SOLID_BREP`, `BREP_WITH_VOIDS`,
//!   `FACETED_BREP`, `SHELL_BASED_SURFACE_MODEL` — the forms AP203, AP214
//!   and AP242 write.
//! - **Surfaces**: planes, cylinders, cones, spheres, tori, B-splines
//!   (rational or not), surfaces of revolution and of extrusion, offsets.
//!   Anything else is drawn as a flat patch over its edges, and said.
//! - **Assemblies**: each part once, placed as often as the assembly places
//!   it, as a hierarchy of objects.
//! - **Colours**: a solid's or a face's colour becomes a material.
//! - **Units**: the file's length unit, whatever it is.
//!
//! The mesh is watertight where the solid is: an edge two faces share is cut
//! into the same points for both, so they meet without a crack, and each face
//! keeps its own normals, so its edges stay sharp and its curves smooth.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::{PI, TAU};
use std::rc::Rc;

use glam::{DAffine3, DQuat, DVec2, DVec3, Vec3};

use crate::ir::{
    ImportMaterial, ImportMesh, ImportNode, ImportNodeData, ImportScene, ImportTransform,
    LossyNote, SourceMetadata,
};
use crate::scene_file::{ReadProgress, ReadStage};

#[derive(Debug, thiserror::Error)]
pub enum StepError {
    #[error("not a STEP file: it does not start with `ISO-10303-21`")]
    NotStep,
    #[error("the file has no entities Kreate can read")]
    Empty,
    #[error("the file holds no solid or surface Kreate can read")]
    NoGeometry,
    #[error("cancelled")]
    Cancelled,
}

/// Parses a STEP file into the IR: a node for each part as the assembly
/// places it, a mesh for each part.
pub fn parse_step(text: &str, name: &str) -> Result<ImportScene, StepError> {
    parse_step_with(text, name, &ReadProgress::default())
}

/// [`parse_step`], saying how far it has got and stopping when asked.
pub fn parse_step_with(
    text: &str,
    name: &str,
    progress: &ReadProgress,
) -> Result<ImportScene, StepError> {
    let model = Model::parse(text, progress)?;
    let mut source = SourceMetadata::new("step");
    let unit = model.unit("LENGTH_UNIT");
    match unit {
        Some(u) if u.is_normal() => source.unit_scale = u as f32,
        _ => {
            source.unit_scale = 1e-3;
            source.notes.push(LossyNote::AssumedUnitScale {
                assumed: "millimetres (the file names no length unit)".into(),
            });
        }
    }
    let angle = model.unit("PLANE_ANGLE_UNIT").unwrap_or(1.0);
    if model.skipped > 0 {
        source.notes.push(LossyNote::Unsupported {
            what: format!("{} STEP entities that could not be read", model.skipped),
        });
    }

    let mut reader = Reader::new(&model, angle, progress);
    let mut scene = ImportScene::new(source);
    reader.read_into(&mut scene, name)?;
    if progress.cancelled() {
        return Err(StepError::Cancelled);
    }
    if reader.rough > 0 {
        scene.source.notes.push(LossyNote::Unsupported {
            what: format!(
                "{} STEP faces that could not be followed exactly (drawn flat over their edges)",
                reader.rough
            ),
        });
    }
    if !reader.unknown_surfaces.is_empty() {
        let kinds: Vec<_> = reader.unknown_surfaces.iter().cloned().collect();
        scene.source.notes.push(LossyNote::Unsupported {
            what: format!("STEP surfaces of kind {}", kinds.join(", ")),
        });
    }
    if model.ids.iter().any(|id| {
        model
            .get(*id)
            .is_some_and(|e| e.is("TESSELLATED_SHAPE_REPRESENTATION"))
    }) {
        scene.source.notes.push(LossyNote::Unsupported {
            what: "STEP tessellated (AP242 mesh) geometry".into(),
        });
    }
    Ok(scene)
}

// ---------------------------------------------------------------------------
// The file: entities by number.

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Ref(u32),
    Num(f64),
    Str(String),
    Enum(String),
    List(Vec<Value>),
    Typed(String, Vec<Value>),
    Unset,
}

impl Value {
    fn id(&self) -> Option<u32> {
        match self {
            Value::Ref(r) => Some(*r),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            Value::Typed(_, a) => a.first()?.num(),
            _ => None,
        }
    }
    fn list(&self) -> &[Value] {
        match self {
            Value::List(l) => l,
            _ => &[],
        }
    }
    fn text(&self) -> &str {
        match self {
            Value::Str(s) => s,
            _ => "",
        }
    }
    fn word(&self) -> &str {
        match self {
            Value::Enum(e) => e,
            _ => "",
        }
    }
    fn flag(&self) -> Option<bool> {
        match self.word() {
            "T" => Some(true),
            "F" => Some(false),
            _ => None,
        }
    }
}

fn num(args: &[Value], i: usize) -> f64 {
    args.get(i).and_then(Value::num).unwrap_or(0.0)
}

fn flag(args: &[Value], i: usize) -> bool {
    args.get(i).and_then(Value::flag).unwrap_or(true)
}

const UNSET: Value = Value::Unset;

fn arg(args: &[Value], i: usize) -> &Value {
    args.get(i).unwrap_or(&UNSET)
}

/// One instance: a simple entity is one part, a complex one several.
#[derive(Debug)]
struct Entity {
    parts: Vec<(String, Vec<Value>)>,
}

impl Entity {
    fn kind(&self) -> &str {
        &self.parts[0].0
    }
    fn args(&self) -> &[Value] {
        &self.parts[0].1
    }
    fn complex(&self) -> bool {
        self.parts.len() > 1
    }
    fn part(&self, name: &str) -> Option<&[Value]> {
        self.parts
            .iter()
            .find(|p| p.0 == name)
            .map(|p| p.1.as_slice())
    }
    fn is(&self, name: &str) -> bool {
        self.parts.iter().any(|p| p.0 == name)
    }
}

struct Model {
    ents: HashMap<u32, Entity>,
    /// Every id, in order: what anything that walks the whole file walks, so
    /// a file reads the same way every time.
    ids: Vec<u32>,
    /// Entities that did not parse and were stepped over.
    skipped: usize,
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

type Parse<T> = Result<T, ()>;

impl Parser<'_> {
    fn skip(&mut self) {
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i.min(self.s.len())..].starts_with(b"/*") {
                match find(&self.s[self.i + 2..], b"*/") {
                    Some(k) => self.i += k + 4,
                    None => self.i = self.s.len(),
                }
            } else {
                break;
            }
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.s.get(self.i).copied()
    }
    fn expect(&mut self, c: u8) -> Parse<()> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(())
        }
    }
    /// Past the next `;` outside a string.
    fn skip_statement(&mut self) {
        while let Some(&c) = self.s.get(self.i) {
            self.i += 1;
            match c {
                b';' => return,
                b'\'' => {
                    while let Some(&c) = self.s.get(self.i) {
                        self.i += 1;
                        if c == b'\'' {
                            if self.s.get(self.i) == Some(&b'\'') {
                                self.i += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
                b'/' if self.s.get(self.i) == Some(&b'*') => match find(&self.s[self.i..], b"*/") {
                    Some(k) => self.i += k + 2,
                    None => self.i = self.s.len(),
                },
                _ => {}
            }
        }
    }
    fn keyword(&mut self) -> Parse<String> {
        self.skip();
        let start = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric() || matches!(self.s[self.i], b'_' | b'!'))
        {
            self.i += 1;
        }
        if start == self.i {
            return Err(());
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).to_ascii_uppercase())
    }
    fn integer(&mut self) -> Parse<u32> {
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            self.i += 1;
        }
        std::str::from_utf8(&self.s[start..self.i])
            .ok()
            .and_then(|t| t.parse().ok())
            .ok_or(())
    }
    fn args(&mut self) -> Parse<Vec<Value>> {
        self.expect(b'(')?;
        let mut out = Vec::new();
        if self.peek() == Some(b')') {
            self.i += 1;
            return Ok(out);
        }
        loop {
            out.push(self.value()?);
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b')') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => return Err(()),
            }
        }
    }
    fn value(&mut self) -> Parse<Value> {
        match self.peek() {
            Some(b'#') => {
                self.i += 1;
                Ok(Value::Ref(self.integer()?))
            }
            Some(b'\'') => self.string(),
            Some(b'"') => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i] != b'"' {
                    self.i += 1;
                }
                let t = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
                self.i += 1;
                Ok(Value::Str(t))
            }
            Some(b'.') => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i] != b'.' {
                    self.i += 1;
                }
                let w = String::from_utf8_lossy(&self.s[start..self.i]).to_ascii_uppercase();
                self.i += 1;
                Ok(Value::Enum(w))
            }
            Some(b'$') | Some(b'*') => {
                self.i += 1;
                Ok(Value::Unset)
            }
            Some(b'(') => Ok(Value::List(self.args()?)),
            Some(c) if c == b'-' || c == b'+' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.s.len()
                    && matches!(
                        self.s[self.i],
                        b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
                    )
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i])
                    .ok()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(Value::Num)
                    .ok_or(())
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let k = self.keyword()?;
                let a = self.args()?;
                Ok(Value::Typed(k, a))
            }
            _ => Err(()),
        }
    }
    fn string(&mut self) -> Parse<Value> {
        self.i += 1;
        let mut raw = Vec::new();
        loop {
            match self.s.get(self.i) {
                None => return Err(()),
                Some(b'\'') => {
                    if self.s.get(self.i + 1) == Some(&b'\'') {
                        raw.push(b'\'');
                        self.i += 2;
                    } else {
                        self.i += 1;
                        break;
                    }
                }
                Some(&c) => {
                    raw.push(c);
                    self.i += 1;
                }
            }
        }
        Ok(Value::Str(decode(&raw)))
    }
    /// `#12=NAME(...);` or `#12=(A(...)B(...));`
    fn statement(&mut self) -> Parse<(u32, Entity)> {
        self.expect(b'#')?;
        let id = self.integer()?;
        self.expect(b'=')?;
        let mut parts = Vec::new();
        if self.peek() == Some(b'(') {
            self.i += 1;
            loop {
                if self.peek() == Some(b')') {
                    self.i += 1;
                    break;
                }
                let k = self.keyword()?;
                let a = self.args()?;
                parts.push((k, a));
            }
        } else {
            let k = self.keyword()?;
            let a = self.args()?;
            parts.push((k, a));
        }
        self.expect(b';')?;
        if parts.is_empty() {
            return Err(());
        }
        Ok((id, Entity { parts }))
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// A string's escapes: `\X2\…\X0\` (UTF-16), `\X4\…\X0\`, `\X\hh`, `\S\c`.
fn decode(raw: &[u8]) -> String {
    let mut out = String::new();
    let mut plain: Vec<u8> = Vec::new();
    let flush = |plain: &mut Vec<u8>, out: &mut String| {
        out.push_str(&String::from_utf8_lossy(plain));
        plain.clear();
    };
    let mut i = 0;
    while i < raw.len() {
        let rest = &raw[i..];
        if rest.starts_with(b"\\X2\\") || rest.starts_with(b"\\X4\\") {
            flush(&mut plain, &mut out);
            let width = if rest[2] == b'2' { 4 } else { 8 };
            let mut j = i + 4;
            let mut units = Vec::new();
            while j + width <= raw.len() && raw[j] != b'\\' {
                if let Some(v) = std::str::from_utf8(&raw[j..j + width])
                    .ok()
                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                {
                    units.push(v);
                }
                j += width;
            }
            if width == 4 {
                let u16s: Vec<u16> = units.iter().map(|u| *u as u16).collect();
                out.push_str(&String::from_utf16_lossy(&u16s));
            } else {
                out.extend(units.iter().filter_map(|u| char::from_u32(*u)));
            }
            if raw[j..].starts_with(b"\\X0\\") {
                j += 4;
            }
            i = j;
            continue;
        }
        if rest.starts_with(b"\\X\\") && rest.len() >= 5 {
            if let Some(v) = std::str::from_utf8(&rest[3..5])
                .ok()
                .and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                flush(&mut plain, &mut out);
                out.push(v as char);
                i += 5;
                continue;
            }
        }
        if rest.starts_with(b"\\S\\") && rest.len() >= 4 {
            flush(&mut plain, &mut out);
            out.push((rest[3] as u32 + 128).try_into().unwrap_or('?'));
            i += 4;
            continue;
        }
        if rest.starts_with(b"\\\\") {
            plain.push(b'\\');
            i += 2;
            continue;
        }
        if rest.starts_with(b"\\P") && rest.len() >= 4 && rest[3] == b'\\' {
            i += 4;
            continue;
        }
        plain.push(raw[i]);
        i += 1;
    }
    flush(&mut plain, &mut out);
    out
}

impl Model {
    fn parse(text: &str, progress: &ReadProgress) -> Result<Self, StepError> {
        let head = text.trim_start_matches('\u{feff}').trim_start();
        if !head.starts_with("ISO-10303-21") {
            return Err(StepError::NotStep);
        }
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        let mut ents = HashMap::new();
        let mut skipped = 0;
        progress.start(ReadStage::Parsing, text.len() as u64);
        let mut count = 0u32;
        loop {
            count = count.wrapping_add(1);
            if count.is_multiple_of(4096) {
                progress.advance(p.i as u64);
                if progress.cancelled() {
                    return Err(StepError::Cancelled);
                }
            }
            match p.peek() {
                None => break,
                Some(b'#') => {
                    let start = p.i;
                    match p.statement() {
                        Ok((id, e)) => {
                            ents.insert(id, e);
                        }
                        Err(()) => {
                            // One bad instance costs itself, not the file.
                            p.i = start;
                            p.skip_statement();
                            skipped += 1;
                        }
                    }
                }
                // The header, and the section markers.
                Some(_) => p.skip_statement(),
            }
        }
        if ents.is_empty() {
            return Err(StepError::Empty);
        }
        let mut ids: Vec<u32> = ents.keys().copied().collect();
        ids.sort_unstable();
        Ok(Self { ents, ids, skipped })
    }

    fn get(&self, id: u32) -> Option<&Entity> {
        self.ents.get(&id)
    }
    fn at(&self, v: &Value) -> Option<&Entity> {
        self.get(v.id()?)
    }
    fn part(&self, v: &Value, name: &str) -> Option<&[Value]> {
        self.at(v)?.part(name)
    }
    fn point(&self, v: &Value) -> Option<DVec3> {
        let a = self.part(v, "CARTESIAN_POINT")?;
        let c = arg(a, 1).list();
        if c.is_empty() {
            return None;
        }
        Some(DVec3::new(num(c, 0), num(c, 1), num(c, 2)))
    }
    fn direction(&self, v: &Value) -> Option<DVec3> {
        let a = self.part(v, "DIRECTION")?;
        let c = arg(a, 1).list();
        DVec3::new(num(c, 0), num(c, 1), num(c, 2)).try_normalize()
    }
    fn vector(&self, v: &Value) -> Option<DVec3> {
        let a = self.part(v, "VECTOR")?;
        Some(self.direction(arg(a, 1))? * num(a, 2))
    }
    fn vertex(&self, v: &Value) -> Option<DVec3> {
        let a = self.part(v, "VERTEX_POINT")?;
        self.point(arg(a, 1))
    }
    /// A placement as a frame; the world's when there is none.
    fn frame(&self, v: &Value) -> Frame {
        let identity = Frame::new(DVec3::ZERO, DVec3::Z, None);
        let Some(e) = self.at(v) else {
            return identity;
        };
        if let Some(a) = e.part("AXIS2_PLACEMENT_3D") {
            let o = self.point(arg(a, 1)).unwrap_or(DVec3::ZERO);
            let z = self.direction(arg(a, 2)).unwrap_or(DVec3::Z);
            return Frame::new(o, z, self.direction(arg(a, 3)));
        }
        if let Some(a) = e.part("AXIS2_PLACEMENT_2D") {
            let o = self.point(arg(a, 1)).unwrap_or(DVec3::ZERO);
            return Frame::new(o, DVec3::Z, self.direction(arg(a, 2)));
        }
        if let Some(a) = e.part("AXIS1_PLACEMENT") {
            let o = self.point(arg(a, 1)).unwrap_or(DVec3::ZERO);
            let z = self.direction(arg(a, 2)).unwrap_or(DVec3::Z);
            return Frame::new(o, z, None);
        }
        if let Some(a) = e.part("CARTESIAN_TRANSFORMATION_OPERATOR_3D") {
            // (name, description, axis1, axis2, local_origin, scale, axis3)
            let o = self.point(arg(a, 4)).unwrap_or(DVec3::ZERO);
            let x = self.direction(arg(a, 2));
            let z = self.direction(arg(a, 6)).unwrap_or(DVec3::Z);
            return Frame::new(o, z, x);
        }
        identity
    }

    /// Metres (or radians) per unit of the kind the file assigns globally.
    fn unit(&self, kind: &str) -> Option<f64> {
        for id in &self.ids {
            let Some(a) = self
                .get(*id)
                .and_then(|e| e.part("GLOBAL_UNIT_ASSIGNED_CONTEXT"))
            else {
                continue;
            };
            for u in arg(a, 0).list() {
                if let Some(f) = self.unit_factor(u, kind, 0) {
                    return Some(f);
                }
            }
        }
        None
    }
    fn unit_factor(&self, v: &Value, kind: &str, depth: u32) -> Option<f64> {
        let e = self.at(v)?;
        if depth > 4 || !e.is(kind) {
            return None;
        }
        if let Some(si) = e.part("SI_UNIT") {
            return Some(si_prefix(arg(si, 0).word()));
        }
        if let Some(c) = e.part("CONVERSION_BASED_UNIT") {
            let m = self.at(arg(c, 1))?;
            let a = [
                "MEASURE_WITH_UNIT",
                "LENGTH_MEASURE_WITH_UNIT",
                "PLANE_ANGLE_MEASURE_WITH_UNIT",
            ]
            .iter()
            .filter_map(|k| m.part(k))
            .find(|a| a.len() >= 2)?;
            let value = arg(a, 0).num()?;
            let base = self.unit_factor(arg(a, 1), kind, depth + 1)?;
            return Some(value * base);
        }
        None
    }
}

fn si_prefix(word: &str) -> f64 {
    match word {
        "EXA" => 1e18,
        "PETA" => 1e15,
        "TERA" => 1e12,
        "GIGA" => 1e9,
        "MEGA" => 1e6,
        "KILO" => 1e3,
        "HECTO" => 1e2,
        "DECA" => 1e1,
        "DECI" => 1e-1,
        "CENTI" => 1e-2,
        "MILLI" => 1e-3,
        "MICRO" => 1e-6,
        "NANO" => 1e-9,
        "PICO" => 1e-12,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Geometry.

#[derive(Clone, Copy, Debug)]
struct Frame {
    o: DVec3,
    x: DVec3,
    y: DVec3,
    z: DVec3,
}

impl Frame {
    fn new(o: DVec3, z: DVec3, x: Option<DVec3>) -> Self {
        let z = z.try_normalize().unwrap_or(DVec3::Z);
        let x = x
            .map(|x| x - z * x.dot(z))
            .and_then(DVec3::try_normalize)
            .unwrap_or_else(|| {
                let a = if z.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
                (a - z * a.dot(z)).normalize()
            });
        Self {
            o,
            x,
            y: z.cross(x),
            z,
        }
    }
    fn local(&self, p: DVec3) -> DVec3 {
        let d = p - self.o;
        DVec3::new(d.dot(self.x), d.dot(self.y), d.dot(self.z))
    }
    fn at(&self, l: DVec3) -> DVec3 {
        self.o + self.x * l.x + self.y * l.y + self.z * l.z
    }
    fn matrix(&self) -> DAffine3 {
        DAffine3::from_cols(self.x, self.y, self.z, self.o)
    }
}

/// A non-uniform rational B-spline: a curve when `n_v` is one.
#[derive(Clone, Debug)]
struct Nurbs {
    pu: usize,
    pv: usize,
    ku: Vec<f64>,
    kv: Vec<f64>,
    nu: usize,
    nv: usize,
    /// Row-major, `nu` rows of `nv`.
    ctrl: Vec<DVec3>,
    w: Vec<f64>,
    closed: [bool; 2],
}

fn find_span(n: usize, p: usize, t: f64, k: &[f64]) -> usize {
    if t >= k[n] {
        let mut s = n - 1;
        while s > p && k[s] >= k[n] {
            s -= 1;
        }
        return s;
    }
    if t <= k[p] {
        let mut s = p;
        while s + 1 < n && k[s + 1] <= t {
            s += 1;
        }
        return s;
    }
    let (mut lo, mut hi) = (p, n);
    let mut mid = (lo + hi) / 2;
    while t < k[mid] || t >= k[mid + 1] {
        if t < k[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

/// The `p + 1` basis functions that are not zero at `t` (The NURBS Book, A2.2).
fn basis(span: usize, t: f64, p: usize, k: &[f64], n: &mut [f64; 16]) {
    let mut left = [0.0; 16];
    let mut right = [0.0; 16];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = t - k[span + 1 - j];
        right[j] = k[span + j] - t;
        let mut saved = 0.0;
        for r in 0..j {
            let den = right[r + 1] + left[j - r];
            let temp = if den.abs() > 0.0 { n[r] / den } else { 0.0 };
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
}

impl Nurbs {
    // Every field once, checked against the others: a struct literal would
    // skip the checks.
    #[allow(clippy::too_many_arguments)]
    fn new(
        pu: usize,
        pv: usize,
        ku: Vec<f64>,
        kv: Vec<f64>,
        nu: usize,
        nv: usize,
        ctrl: Vec<DVec3>,
        w: Vec<f64>,
        closed: [bool; 2],
    ) -> Option<Self> {
        if pu > 15
            || pv > 15
            || nu <= pu
            || nv <= pv
            || ctrl.len() != nu * nv
            || w.len() != ctrl.len()
        {
            return None;
        }
        if ku.len() != nu + pu + 1 || kv.len() != nv + pv + 1 {
            return None;
        }
        if ku[nu] <= ku[pu] || kv[nv] <= kv[pv] {
            return None;
        }
        Some(Self {
            pu,
            pv,
            ku,
            kv,
            nu,
            nv,
            ctrl,
            w,
            closed,
        })
    }
    fn domain(&self, axis: usize) -> (f64, f64) {
        if axis == 0 {
            (self.ku[self.pu], self.ku[self.nu])
        } else {
            (self.kv[self.pv], self.kv[self.nv])
        }
    }
    fn fold(&self, axis: usize, t: f64) -> f64 {
        let (lo, hi) = self.domain(axis);
        if self.closed[axis] {
            lo + (t - lo).rem_euclid(hi - lo)
        } else {
            t.clamp(lo, hi)
        }
    }
    fn eval(&self, u: f64, v: f64) -> DVec3 {
        let u = self.fold(0, u);
        let v = self.fold(1, v);
        let su = find_span(self.nu, self.pu, u, &self.ku);
        let sv = find_span(self.nv, self.pv, v, &self.kv);
        let mut bu = [0.0; 16];
        let mut bv = [0.0; 16];
        basis(su, u, self.pu, &self.ku, &mut bu);
        basis(sv, v, self.pv, &self.kv, &mut bv);
        let mut sum = DVec3::ZERO;
        let mut wsum = 0.0;
        for (i, bu) in bu.iter().enumerate().take(self.pu + 1) {
            let row = (su - self.pu + i) * self.nv;
            for (j, bv) in bv.iter().enumerate().take(self.pv + 1) {
                let k = row + sv - self.pv + j;
                let b = bu * bv * self.w[k];
                sum += self.ctrl[k] * b;
                wsum += b;
            }
        }
        if wsum.abs() > 0.0 {
            sum / wsum
        } else {
            sum
        }
    }
}

fn expand_knots(mults: &[Value], knots: &[Value]) -> Vec<f64> {
    let mut out = Vec::new();
    for (m, k) in mults.iter().zip(knots) {
        let m = m.num().unwrap_or(1.0).max(0.0) as usize;
        let k = k.num().unwrap_or(0.0);
        out.extend(std::iter::repeat_n(k, m.min(64)));
    }
    out
}

/// A knot vector for a spline that does not write one.
fn made_knots(kind: &str, n: usize, p: usize) -> Vec<f64> {
    match kind {
        "UNIFORM" => (0..n + p + 1).map(|i| i as f64).collect(),
        "BEZIER" if p > 0 && (n - 1).is_multiple_of(p) => {
            let segments = (n - 1) / p;
            let mut k = vec![0.0; p + 1];
            for s in 1..segments {
                k.extend(std::iter::repeat_n(s as f64, p));
            }
            k.extend(std::iter::repeat_n(segments as f64, p + 1));
            k
        }
        _ => {
            // Clamped and uniform inside.
            let inner = n.saturating_sub(p + 1);
            let mut k = vec![0.0; p + 1];
            for i in 1..=inner {
                k.push(i as f64 / (inner + 1) as f64);
            }
            k.extend(std::iter::repeat_n(1.0, p + 1));
            k
        }
    }
}

fn spline_kind(e: &Entity) -> &'static str {
    if e.is("BEZIER_CURVE") || e.is("BEZIER_SURFACE") {
        "BEZIER"
    } else if e.is("UNIFORM_CURVE") || e.is("UNIFORM_SURFACE") {
        "UNIFORM"
    } else {
        "QUASI_UNIFORM"
    }
}

#[derive(Debug)]
enum Curve {
    Line { p: DVec3, d: DVec3 },
    Conic { f: Frame, a: f64, b: f64 },
    Spline(Nurbs),
    Polyline(Vec<DVec3>),
    Unknown,
}

impl Curve {
    fn eval(&self, t: f64) -> DVec3 {
        match self {
            Curve::Line { p, d } => *p + *d * t,
            Curve::Conic { f, a, b } => f.at(DVec3::new(a * t.cos(), b * t.sin(), 0.0)),
            Curve::Spline(s) => s.eval(t, 0.0),
            Curve::Polyline(pts) => {
                let last = pts.len().saturating_sub(1);
                let t = t.clamp(0.0, last as f64);
                let i = (t.floor() as usize).min(last.saturating_sub(1));
                let f = t - i as f64;
                pts[i].lerp(pts[(i + 1).min(last)], f)
            }
            Curve::Unknown => DVec3::ZERO,
        }
    }
    fn domain(&self) -> Option<(f64, f64)> {
        match self {
            Curve::Spline(s) => Some(s.domain(0)),
            Curve::Conic { .. } => Some((0.0, TAU)),
            Curve::Polyline(p) => Some((0.0, p.len().saturating_sub(1) as f64)),
            _ => None,
        }
    }
    fn periodic(&self) -> Option<f64> {
        match self {
            Curve::Conic { .. } => Some(TAU),
            Curve::Spline(s) if s.closed[0] => {
                let (lo, hi) = s.domain(0);
                Some(hi - lo)
            }
            _ => None,
        }
    }
    /// The parameter of the point nearest `p`.
    fn closest(&self, p: DVec3, hint: Option<f64>) -> f64 {
        match self {
            Curve::Line { p: o, d } => (p - *o).dot(*d) / d.length_squared().max(1e-300),
            Curve::Conic { f, a, b } => {
                let l = f.local(p);
                (l.y / b).atan2(l.x / a)
            }
            Curve::Polyline(pts) => {
                let mut best = (f64::MAX, 0.0);
                for i in 0..pts.len().saturating_sub(1) {
                    let (a, b) = (pts[i], pts[i + 1]);
                    let d = b - a;
                    let t = ((p - a).dot(d) / d.length_squared().max(1e-300)).clamp(0.0, 1.0);
                    let dist = (a + d * t).distance_squared(p);
                    if dist < best.0 {
                        best = (dist, i as f64 + t);
                    }
                }
                best.1
            }
            Curve::Spline(s) => {
                let (lo, hi) = s.domain(0);
                let n = (s.nu * 8).clamp(16, 512);
                let step = (hi - lo) / n as f64;
                let mut best = (f64::MAX, lo);
                let look = |t: f64, best: &mut (f64, f64)| {
                    let d = s.eval(t, 0.0).distance_squared(p);
                    if d < best.0 {
                        *best = (d, t);
                    }
                };
                for i in 0..=n {
                    look(lo + step * i as f64, &mut best);
                }
                if let Some(h) = hint {
                    look(s.fold(0, h), &mut best);
                }
                golden(
                    |t| s.eval(t, 0.0).distance_squared(p),
                    best.1 - step,
                    best.1 + step,
                    lo,
                    hi,
                )
            }
            Curve::Unknown => 0.0,
        }
    }
}

/// The minimum of `f` on `[a, b]` (inside `[lo, hi]`), by golden section.
fn golden(f: impl Fn(f64) -> f64, a: f64, b: f64, lo: f64, hi: f64) -> f64 {
    let (mut a, mut b) = (a.max(lo), b.min(hi));
    let g = 0.618_033_988_749_895;
    let mut c = b - (b - a) * g;
    let mut d = a + (b - a) * g;
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * g;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * g;
            fd = f(d);
        }
    }
    (a + b) * 0.5
}

#[derive(Debug)]
enum Surface {
    Plane(Frame),
    Cylinder(Frame, f64),
    /// Radius at the frame, and the tangent of the half angle.
    Cone(Frame, f64, f64),
    Sphere(Frame, f64),
    Torus(Frame, f64, f64),
    Spline(Box<Nurbs>, Vec<(DVec2, DVec3)>),
    /// The curve turned about the axis, `e` the direction the curve lies in.
    Revolution {
        curve: Rc<Curve>,
        o: DVec3,
        axis: DVec3,
        e: DVec3,
    },
    Extrusion {
        curve: Rc<Curve>,
        dir: DVec3,
    },
    Offset(Rc<Surface>, f64),
    Unknown,
}

impl Surface {
    fn eval(&self, q: DVec2) -> DVec3 {
        let (u, v) = (q.x, q.y);
        match self {
            Surface::Plane(f) => f.at(DVec3::new(u, v, 0.0)),
            Surface::Cylinder(f, r) => f.at(DVec3::new(r * u.cos(), r * u.sin(), v)),
            Surface::Cone(f, r, t) => {
                let rr = r + v * t;
                f.at(DVec3::new(rr * u.cos(), rr * u.sin(), v))
            }
            Surface::Sphere(f, r) => f.at(DVec3::new(
                r * v.cos() * u.cos(),
                r * v.cos() * u.sin(),
                r * v.sin(),
            )),
            Surface::Torus(f, big, small) => {
                let rr = big + small * v.cos();
                f.at(DVec3::new(rr * u.cos(), rr * u.sin(), small * v.sin()))
            }
            Surface::Spline(s, _) => s.eval(u, v),
            Surface::Revolution { curve, o, axis, .. } => {
                DQuat::from_axis_angle(*axis, u) * (curve.eval(v) - *o) + *o
            }
            Surface::Extrusion { curve, dir } => curve.eval(u) + *dir * v,
            Surface::Offset(b, d) => b.eval(q) + b.normal(q) * *d,
            Surface::Unknown => DVec3::ZERO,
        }
    }

    /// The unit normal, `∂u × ∂v` — the direction STEP's own normal points.
    fn normal(&self, q: DVec2) -> DVec3 {
        let (u, v) = (q.x, q.y);
        match self {
            Surface::Plane(f) => f.z,
            Surface::Cylinder(f, _) => f.x * u.cos() + f.y * u.sin(),
            Surface::Cone(f, r, t) => {
                let n = (f.x * u.cos() + f.y * u.sin() - f.z * *t).normalize();
                if r + v * t < 0.0 {
                    -n
                } else {
                    n
                }
            }
            Surface::Sphere(f, _) | Surface::Torus(f, _, _) => {
                f.x * (v.cos() * u.cos()) + f.y * (v.cos() * u.sin()) + f.z * v.sin()
            }
            Surface::Offset(b, _) => b.normal(q),
            _ => self.fd_normal(q),
        }
    }

    fn spans(&self) -> DVec2 {
        match self {
            Surface::Spline(s, _) => {
                let (a, b) = s.domain(0);
                let (c, d) = s.domain(1);
                DVec2::new(b - a, d - c)
            }
            Surface::Revolution { curve, .. } => {
                DVec2::new(TAU, curve.domain().map(|(a, b)| b - a).unwrap_or(1.0))
            }
            Surface::Extrusion { curve, .. } => {
                DVec2::new(curve.domain().map(|(a, b)| b - a).unwrap_or(1.0), 1.0)
            }
            Surface::Offset(b, _) => b.spans(),
            _ => DVec2::ONE,
        }
    }

    fn fd_normal(&self, q: DVec2) -> DVec3 {
        let span = self.spans();
        let h = span * 1e-6;
        let mut q = q;
        for _ in 0..4 {
            let du = self.eval(q + DVec2::new(h.x, 0.0)) - self.eval(q - DVec2::new(h.x, 0.0));
            let dv = self.eval(q + DVec2::new(0.0, h.y)) - self.eval(q - DVec2::new(0.0, h.y));
            let n = du.cross(dv);
            if n.length() > 1e-9 * du.length() * dv.length() && n.length() > 0.0 {
                return n.normalize();
            }
            // A degenerate point — a pole: look from a little way in.
            q += span * 1e-3;
        }
        DVec3::Z
    }

    /// Where on the surface `p` is, near `hint` when there is one.
    fn invert(&self, p: DVec3, hint: Option<DVec2>) -> DVec2 {
        match self {
            Surface::Plane(f) => {
                let l = f.local(p);
                DVec2::new(l.x, l.y)
            }
            Surface::Cylinder(f, _) | Surface::Cone(f, _, _) => {
                let l = f.local(p);
                DVec2::new(l.y.atan2(l.x), l.z)
            }
            Surface::Sphere(f, _) => {
                let l = f.local(p);
                DVec2::new(l.y.atan2(l.x), l.z.atan2(l.x.hypot(l.y)))
            }
            Surface::Torus(f, big, _) => {
                let l = f.local(p);
                DVec2::new(l.y.atan2(l.x), l.z.atan2(l.x.hypot(l.y) - big))
            }
            Surface::Spline(s, grid) => invert_spline(s, grid, p, hint),
            Surface::Revolution { curve, o, axis, e } => {
                let d = p - *o;
                let h = d.dot(*axis);
                let rad = d - *axis * h;
                let q = *o + *axis * h + *e * rad.length();
                let v = curve.closest(q, hint.map(|x| x.y));
                let u = e.cross(rad).dot(*axis).atan2(e.dot(rad));
                DVec2::new(u, v)
            }
            Surface::Extrusion { curve, dir } => {
                let len2 = dir.length_squared().max(1e-300);
                let mut u = curve.closest(p, hint.map(|x| x.x));
                let mut v = 0.0;
                for _ in 0..4 {
                    v = (p - curve.eval(u)).dot(*dir) / len2;
                    u = curve.closest(p - *dir * v, Some(u));
                }
                DVec2::new(u, v)
            }
            Surface::Offset(b, _) => b.invert(p, hint),
            Surface::Unknown => DVec2::ZERO,
        }
    }

    /// The period along each parameter, for a surface that closes on itself.
    fn periods(&self) -> [Option<f64>; 2] {
        match self {
            Surface::Cylinder(..) | Surface::Cone(..) | Surface::Sphere(..) => [Some(TAU), None],
            Surface::Torus(..) => [Some(TAU), Some(TAU)],
            Surface::Spline(s, _) => {
                let p = |a: usize| {
                    s.closed[a].then(|| {
                        let (lo, hi) = s.domain(a);
                        hi - lo
                    })
                };
                [p(0), p(1)]
            }
            Surface::Revolution { curve, .. } => [Some(TAU), curve.periodic()],
            Surface::Extrusion { curve, .. } => [curve.periodic(), None],
            Surface::Offset(b, _) => b.periods(),
            _ => [None, None],
        }
    }

    /// The ends of a parameter's range, where a face that winds once around
    /// the surface is closed off — a sphere's poles, a cone's apex.
    fn limits(&self, axis: usize) -> (Option<f64>, Option<f64>) {
        match (self, axis) {
            (Surface::Sphere(..), 1) => (Some(-PI / 2.0), Some(PI / 2.0)),
            (Surface::Cone(_, r, t), 1) if t.abs() > 1e-12 => {
                let apex = -r / t;
                if *t > 0.0 {
                    (Some(apex), None)
                } else {
                    (None, Some(apex))
                }
            }
            (Surface::Spline(s, _), a) => {
                let (lo, hi) = s.domain(a);
                (Some(lo), Some(hi))
            }
            (
                Surface::Revolution {
                    curve, o, axis: ax, ..
                },
                1,
            ) => match &**curve {
                Curve::Line { p, d } => {
                    // Where the line meets the axis.
                    let r0 = (*p - *o) - *ax * (*p - *o).dot(*ax);
                    let rd = *d - *ax * d.dot(*ax);
                    let t = -r0.dot(rd) / rd.length_squared().max(1e-300);
                    (Some(t), Some(t))
                }
                c => c.domain().map_or((None, None), |(a, b)| (Some(a), Some(b))),
            },
            (Surface::Offset(b, _), a) => b.limits(a),
            _ => (None, None),
        }
    }

    /// The parameter that means nothing at `p` (at `q`): the angle on a
    /// sphere's or a cone's axis, or either parameter along a side of a
    /// spline that is drawn together into one point.
    fn singular(&self, p: DVec3, q: DVec2, eps: f64) -> Option<usize> {
        match self {
            Surface::Cone(f, _, _) | Surface::Sphere(f, _) => {
                let l = f.local(p);
                (l.x.hypot(l.y) < eps).then_some(0)
            }
            Surface::Revolution { o, axis, .. } => {
                let d = p - *o;
                ((d - *axis * d.dot(*axis)).length() < eps).then_some(0)
            }
            Surface::Spline(n, _) => (0..2).find(|a| {
                let (lo, hi) = n.domain(*a);
                let at = |t: f64| {
                    if *a == 0 {
                        n.eval(t, q.y)
                    } else {
                        n.eval(q.x, t)
                    }
                };
                let first = at(lo);
                (1..=4).all(|k| at(lo + (hi - lo) * k as f64 / 4.0).distance(first) < eps)
            }),
            Surface::Offset(b, _) => b.singular(p, q, eps),
            _ => None,
        }
    }
}

fn invert_spline(s: &Nurbs, grid: &[(DVec2, DVec3)], p: DVec3, hint: Option<DVec2>) -> DVec2 {
    let (ulo, uhi) = s.domain(0);
    let (vlo, vhi) = s.domain(1);
    let span = DVec2::new(uhi - ulo, vhi - vlo);
    let newton = |mut q: DVec2| -> (f64, DVec2) {
        let h = span * 1e-6;
        for _ in 0..30 {
            let r = s.eval(q.x, q.y) - p;
            let su = (s.eval(q.x + h.x, q.y) - s.eval(q.x - h.x, q.y)) / (2.0 * h.x);
            let sv = (s.eval(q.x, q.y + h.y) - s.eval(q.x, q.y - h.y)) / (2.0 * h.y);
            let (a11, a12, a22) = (su.dot(su), su.dot(sv), sv.dot(sv));
            let (b1, b2) = (-r.dot(su), -r.dot(sv));
            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-300 {
                break;
            }
            let d = DVec2::new((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det);
            // A step no larger than a quarter of the domain: Newton far from
            // the answer overshoots.
            let d = d.clamp(-span * 0.25, span * 0.25);
            q += d;
            if !s.closed[0] {
                q.x = q.x.clamp(ulo, uhi);
            }
            if !s.closed[1] {
                q.y = q.y.clamp(vlo, vhi);
            }
            if d.abs().x < span.x * 1e-12 && d.abs().y < span.y * 1e-12 {
                break;
            }
        }
        let q = DVec2::new(s.fold(0, q.x), s.fold(1, q.y));
        (s.eval(q.x, q.y).distance_squared(p), q)
    };
    let start = grid
        .iter()
        .min_by(|a, b| a.1.distance_squared(p).total_cmp(&b.1.distance_squared(p)))
        .map(|g| g.0)
        .unwrap_or(DVec2::new(ulo, vlo));
    let mut best = newton(start);
    if let Some(h) = hint {
        let other = newton(DVec2::new(s.fold(0, h.x), s.fold(1, h.y)));
        if other.0 <= best.0 * (1.0 + 1e-9) {
            best = other;
        }
    }
    best.1
}

// ---------------------------------------------------------------------------
// Reading: geometry into meshes, representations into a hierarchy.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Vertex(u32),
    Point(u32),
    Edge(u32, u32),
}

/// An edge's vertices and points, from its first vertex to its second.
type SampledEdge = Rc<(Vec<u32>, Vec<DVec3>)>;

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<DVec3>,
    keys: HashMap<Key, u32>,
    edges: HashMap<u32, SampledEdge>,
    /// The surfaces each edge bounds whose parameters are not simply related
    /// to distance, for the edge to be cut finely enough for them.
    edge_surfaces: HashMap<u32, Vec<Rc<Surface>>>,
    /// The part's corners, for its size.
    lo: Option<(DVec3, DVec3)>,
    tris: Vec<[u32; 3]>,
    normals: Vec<[DVec3; 3]>,
    mats: Vec<i32>,
}

impl MeshBuilder {
    fn grow(&mut self, p: DVec3) {
        self.lo = Some(match self.lo {
            Some((lo, hi)) => (lo.min(p), hi.max(p)),
            None => (p, p),
        });
    }
    fn key(&mut self, k: Key, at: DVec3) -> u32 {
        if let Some(id) = self.keys.get(&k) {
            return *id;
        }
        let id = self.vertex(at);
        self.keys.insert(k, id);
        id
    }
    fn vertex(&mut self, at: DVec3) -> u32 {
        self.positions.push(at);
        (self.positions.len() - 1) as u32
    }
}

/// A face's boundary: its points in order, and each one's vertex.
struct Loop {
    ids: Vec<u32>,
    pts: Vec<DVec3>,
    /// The bound's orientation: whether the loop runs as the face's.
    orient: bool,
}

/// A loop on a surface's parameters.
struct Ring {
    uv: Vec<DVec2>,
    ids: Vec<Option<u32>>,
    pts: Vec<Option<DVec3>>,
    wind: [i64; 2],
    orient: bool,
}

const GEOMETRY: &[&str] = &[
    "MANIFOLD_SOLID_BREP",
    "BREP_WITH_VOIDS",
    "FACETED_BREP",
    "SHELL_BASED_SURFACE_MODEL",
    "FACE_BASED_SURFACE_MODEL",
    "CLOSED_SHELL",
    "OPEN_SHELL",
];

struct Reader<'m> {
    m: &'m Model,
    progress: &'m ReadProgress,
    /// Faces cut so far, for the progress bar.
    faces_done: u64,
    /// How far a facet may stray from the true surface, in file units.
    tol: f64,
    /// Radians per angle unit.
    angle: f64,
    colours: HashMap<u32, [f64; 3]>,
    materials: Vec<ImportMaterial>,
    material_of: HashMap<[u32; 3], i32>,
    surfaces: HashMap<u32, Rc<Surface>>,
    curves: HashMap<u32, Rc<Curve>>,
    mesh: MeshBuilder,
    rough: usize,
    unknown_surfaces: BTreeSet<String>,
}

impl<'m> Reader<'m> {
    fn new(m: &'m Model, angle: f64, progress: &'m ReadProgress) -> Self {
        // The tolerance from the model's size: a part a few millimetres
        // across and a building get the same look.
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        for e in m.ents.values() {
            if let Some(a) = e.part("CARTESIAN_POINT") {
                let c = arg(a, 1).list();
                if c.len() == 3 {
                    let p = DVec3::new(num(c, 0), num(c, 1), num(c, 2));
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        let diag = if hi.x >= lo.x {
            (hi - lo).length()
        } else {
            1.0
        };
        let tol = (diag * 4e-4).max(1e-9);
        let mut reader = Self {
            m,
            progress,
            faces_done: 0,
            tol,
            angle,
            colours: HashMap::new(),
            materials: Vec::new(),
            material_of: HashMap::new(),
            surfaces: HashMap::new(),
            curves: HashMap::new(),
            mesh: MeshBuilder::default(),
            rough: 0,
            unknown_surfaces: BTreeSet::new(),
        };
        reader.find_colours();
        reader
    }

    // -- colours -----------------------------------------------------------

    fn find_colours(&mut self) {
        let m = self.m;
        for pass in ["STYLED_ITEM", "OVER_RIDING_STYLED_ITEM"] {
            for id in &m.ids {
                let Some(a) = m.get(*id).and_then(|e| e.part(pass)) else {
                    continue;
                };
                let (Some(item), Some(c)) = (arg(a, 2).id(), self.colour_in(arg(a, 1), 0)) else {
                    continue;
                };
                self.colours.insert(item, c);
            }
        }
    }

    fn colour_in(&self, v: &Value, depth: u32) -> Option<[f64; 3]> {
        if depth > 12 {
            return None;
        }
        match v {
            Value::List(l) => l.iter().find_map(|x| self.colour_in(x, depth + 1)),
            Value::Ref(_) => {
                let e = self.m.at(v)?;
                if let Some(a) = e.part("COLOUR_RGB") {
                    return Some([num(a, 1), num(a, 2), num(a, 3)]);
                }
                if let Some(a) = e.part("DRAUGHTING_PRE_DEFINED_COLOUR") {
                    return named_colour(arg(a, 0).text());
                }
                let k = e.kind();
                if e.is("CURVE_STYLE") || e.is("POINT_STYLE") || k.contains("HATCHING") {
                    return None;
                }
                if !(k.contains("STYLE") || k.contains("COLOUR") || k.starts_with("PRESENTATION")) {
                    return None;
                }
                e.parts
                    .iter()
                    .flat_map(|p| p.1.iter())
                    .find_map(|x| self.colour_in(x, depth + 1))
            }
            _ => None,
        }
    }

    fn material(&mut self, colour: Option<[f64; 3]>) -> i32 {
        let Some(c) = colour else {
            return -1;
        };
        let key = c.map(|x| (x.clamp(0.0, 1.0) * 1000.0).round() as u32);
        if let Some(i) = self.material_of.get(&key) {
            return *i;
        }
        let hex: String = c
            .iter()
            .map(|x| format!("{:02X}", (x.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect();
        let linear = c.map(|x| kreate_math::srgb_to_linear(x.clamp(0.0, 1.0) as f32));
        self.materials.push(ImportMaterial {
            name: format!("Colour {hex}"),
            base_color: [linear[0], linear[1], linear[2], 1.0],
            ..ImportMaterial::default()
        });
        let i = (self.materials.len() - 1) as i32;
        self.material_of.insert(key, i);
        i
    }

    // -- curves and surfaces -------------------------------------------------

    fn curve(&mut self, v: &Value) -> Rc<Curve> {
        let Some(id) = v.id() else {
            return Rc::new(Curve::Unknown);
        };
        if let Some(c) = self.curves.get(&id) {
            return c.clone();
        }
        let c = Rc::new(self.make_curve(id, 0));
        self.curves.insert(id, c.clone());
        c
    }

    fn make_curve(&mut self, id: u32, depth: u32) -> Curve {
        let m = self.m;
        let Some(e) = m.get(id) else {
            return Curve::Unknown;
        };
        if depth > 8 {
            return Curve::Unknown;
        }
        if let Some(s) = self.spline(e, false) {
            return Curve::Spline(s);
        }
        let a = e.args();
        match e.kind() {
            "LINE" => match (m.point(arg(a, 1)), m.vector(arg(a, 2))) {
                (Some(p), Some(d)) if d.length_squared() > 0.0 => Curve::Line { p, d },
                _ => Curve::Unknown,
            },
            "CIRCLE" => Curve::Conic {
                f: m.frame(arg(a, 1)),
                a: num(a, 2),
                b: num(a, 2),
            },
            "ELLIPSE" => Curve::Conic {
                f: m.frame(arg(a, 1)),
                a: num(a, 2),
                b: num(a, 3),
            },
            "POLYLINE" => {
                let pts: Vec<DVec3> = arg(a, 1).list().iter().filter_map(|p| m.point(p)).collect();
                if pts.len() >= 2 {
                    Curve::Polyline(pts)
                } else {
                    Curve::Unknown
                }
            }
            "SURFACE_CURVE"
            | "SEAM_CURVE"
            | "BOUNDED_SURFACE_CURVE"
            | "INTERSECTION_CURVE"
            | "TRIMMED_CURVE" => match arg(a, 1).id() {
                Some(c) => self.make_curve(c, depth + 1),
                None => Curve::Unknown,
            },
            _ => Curve::Unknown,
        }
    }

    /// A B-spline curve or surface, simple or complex, rational or not.
    fn spline(&self, e: &Entity, surface: bool) -> Option<Nurbs> {
        let m = self.m;
        let (head, knots_part, rational) = if surface {
            (
                "B_SPLINE_SURFACE",
                "B_SPLINE_SURFACE_WITH_KNOTS",
                "RATIONAL_B_SPLINE_SURFACE",
            )
        } else {
            (
                "B_SPLINE_CURVE",
                "B_SPLINE_CURVE_WITH_KNOTS",
                "RATIONAL_B_SPLINE_CURVE",
            )
        };
        let simple_kinds: &[&str] = if surface {
            &[
                "B_SPLINE_SURFACE_WITH_KNOTS",
                "BEZIER_SURFACE",
                "UNIFORM_SURFACE",
                "QUASI_UNIFORM_SURFACE",
                "B_SPLINE_SURFACE",
            ]
        } else {
            &[
                "B_SPLINE_CURVE_WITH_KNOTS",
                "BEZIER_CURVE",
                "UNIFORM_CURVE",
                "QUASI_UNIFORM_CURVE",
                "B_SPLINE_CURVE",
            ]
        };
        // The head's fields (degree(s), points, form, closed…) and the knots.
        let (fields, knots, weights): (&[Value], Option<&[Value]>, Option<&[Value]>) =
            if !e.complex() {
                if !simple_kinds.contains(&e.kind()) {
                    return None;
                }
                let a = e.args();
                let n = if surface { 8 } else { 6 };
                let knots = (e.kind() == knots_part && a.len() > n).then(|| &a[n..]);
                (&a[1..], knots, None)
            } else {
                let head = e.part(head)?;
                (head, e.part(knots_part), e.part(rational))
            };
        let kind = spline_kind(e);
        if surface {
            let pu = num(fields, 0) as usize;
            let pv = num(fields, 1) as usize;
            let rows = arg(fields, 2).list();
            let nu = rows.len();
            let nv = rows.first().map(|r| r.list().len()).unwrap_or(0);
            let mut ctrl = Vec::with_capacity(nu * nv);
            for r in rows {
                if r.list().len() != nv {
                    return None;
                }
                for p in r.list() {
                    ctrl.push(m.point(p)?);
                }
            }
            let w: Vec<f64> = match weights {
                Some(w) => arg(w, 0)
                    .list()
                    .iter()
                    .flat_map(|r| r.list().iter().map(|x| x.num().unwrap_or(1.0)))
                    .collect(),
                None => vec![1.0; ctrl.len()],
            };
            let (ku, kv) = match knots {
                Some(k) => (
                    expand_knots(arg(k, 0).list(), arg(k, 2).list()),
                    expand_knots(arg(k, 1).list(), arg(k, 3).list()),
                ),
                None => (made_knots(kind, nu, pu), made_knots(kind, nv, pv)),
            };
            let tol = self.tol;
            let same =
                |a: &[DVec3], b: &[DVec3]| a.iter().zip(b).all(|(x, y)| x.distance(*y) < tol);
            let first_row = &ctrl[..nv.min(ctrl.len())];
            let last_row = &ctrl[ctrl.len().saturating_sub(nv)..];
            let first_col: Vec<DVec3> = (0..nu).map(|i| ctrl[i * nv]).collect();
            let last_col: Vec<DVec3> = (0..nu).map(|i| ctrl[i * nv + nv - 1]).collect();
            let closed = [
                nu > 1 && same(first_row, last_row),
                nv > 1 && same(&first_col, &last_col),
            ];
            Nurbs::new(pu, pv, ku, kv, nu, nv, ctrl, w, closed)
        } else {
            let p = num(fields, 0) as usize;
            let ctrl: Vec<DVec3> = arg(fields, 1)
                .list()
                .iter()
                .map(|x| m.point(x))
                .collect::<Option<_>>()?;
            let n = ctrl.len();
            let w = match weights {
                Some(w) => arg(w, 0)
                    .list()
                    .iter()
                    .map(|x| x.num().unwrap_or(1.0))
                    .collect(),
                None => vec![1.0; n],
            };
            let ku = match knots {
                Some(k) => expand_knots(arg(k, 0).list(), arg(k, 1).list()),
                None => made_knots(kind, n, p),
            };
            let closed = n > 2 && ctrl[0].distance(ctrl[n - 1]) < self.tol;
            Nurbs::new(p, 0, ku, vec![0.0, 1.0], n, 1, ctrl, w, [closed, false])
        }
    }

    fn surface(&mut self, v: &Value) -> Rc<Surface> {
        let Some(id) = v.id() else {
            return Rc::new(Surface::Unknown);
        };
        if let Some(s) = self.surfaces.get(&id) {
            return s.clone();
        }
        let s = Rc::new(self.make_surface(id, 0));
        self.surfaces.insert(id, s.clone());
        s
    }

    fn make_surface(&mut self, id: u32, depth: u32) -> Surface {
        let m = self.m;
        let Some(e) = m.get(id) else {
            return Surface::Unknown;
        };
        if depth > 6 {
            return Surface::Unknown;
        }
        if let Some(s) = self.spline(e, true) {
            let (ulo, uhi) = s.domain(0);
            let (vlo, vhi) = s.domain(1);
            let n = 24;
            let mut grid = Vec::with_capacity((n + 1) * (n + 1));
            for i in 0..=n {
                for j in 0..=n {
                    let q = DVec2::new(
                        ulo + (uhi - ulo) * i as f64 / n as f64,
                        vlo + (vhi - vlo) * j as f64 / n as f64,
                    );
                    grid.push((q, s.eval(q.x, q.y)));
                }
            }
            return Surface::Spline(Box::new(s), grid);
        }
        let a = e.args();
        match e.kind() {
            "PLANE" => Surface::Plane(m.frame(arg(a, 1))),
            "CYLINDRICAL_SURFACE" => Surface::Cylinder(m.frame(arg(a, 1)), num(a, 2)),
            "CONICAL_SURFACE" => {
                let mut half = num(a, 3) * self.angle;
                // Some writers give degrees whatever their unit says.
                if half.abs() >= PI / 2.0 {
                    half = half.to_radians() / self.angle;
                }
                Surface::Cone(m.frame(arg(a, 1)), num(a, 2), half.tan())
            }
            "SPHERICAL_SURFACE" => Surface::Sphere(m.frame(arg(a, 1)), num(a, 2)),
            "TOROIDAL_SURFACE" | "DEGENERATE_TOROIDAL_SURFACE" => {
                Surface::Torus(m.frame(arg(a, 1)), num(a, 2), num(a, 3))
            }
            "SURFACE_OF_REVOLUTION" => {
                let curve = self.curve(arg(a, 1));
                let ax = m.frame(arg(a, 2));
                let (o, axis) = (ax.o, ax.z);
                // The side of the axis the profile lies on.
                let samples: Vec<f64> = match curve.domain() {
                    Some((lo, hi)) => (0..=8).map(|i| lo + (hi - lo) * i as f64 / 8.0).collect(),
                    None => vec![0.0, 1.0, -1.0],
                };
                let e = samples
                    .iter()
                    .map(|t| {
                        let d = curve.eval(*t) - o;
                        d - axis * d.dot(axis)
                    })
                    .max_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
                    .and_then(DVec3::try_normalize)
                    .unwrap_or_else(|| Frame::new(o, axis, None).x);
                Surface::Revolution { curve, o, axis, e }
            }
            "SURFACE_OF_LINEAR_EXTRUSION" => match m.vector(arg(a, 2)) {
                Some(dir) if dir.length_squared() > 0.0 => Surface::Extrusion {
                    curve: self.curve(arg(a, 1)),
                    dir,
                },
                _ => Surface::Unknown,
            },
            "OFFSET_SURFACE" => match arg(a, 1).id() {
                Some(b) => Surface::Offset(Rc::new(self.make_surface(b, depth + 1)), num(a, 2)),
                None => Surface::Unknown,
            },
            "RECTANGULAR_TRIMMED_SURFACE" | "CURVE_BOUNDED_SURFACE" => match arg(a, 1).id() {
                Some(b) => self.make_surface(b, depth + 1),
                None => Surface::Unknown,
            },
            other => {
                self.unknown_surfaces.insert(other.to_string());
                Surface::Unknown
            }
        }
    }

    // -- edges and loops -----------------------------------------------------

    fn segments(&self, radius: f64, sweep: f64) -> usize {
        if radius <= 0.0 {
            return 1;
        }
        let step = if self.tol < radius {
            2.0 * (1.0 - self.tol / radius).acos()
        } else {
            PI / 2.0
        };
        let step = step.clamp(2f64.to_radians(), 12f64.to_radians());
        ((sweep.abs() / step).ceil() as usize).clamp(1, 2000)
    }

    /// Points along `f` from `a` to `b`, closer together where it bends.
    fn adapt(
        &self,
        f: &dyn Fn(f64) -> DVec3,
        a: f64,
        b: f64,
        n0: usize,
        out: &mut Vec<(f64, DVec3)>,
    ) {
        #[allow(clippy::too_many_arguments)]
        fn go(
            f: &dyn Fn(f64) -> DVec3,
            tol: f64,
            a: f64,
            pa: DVec3,
            b: f64,
            pb: DVec3,
            depth: u32,
            out: &mut Vec<(f64, DVec3)>,
        ) {
            let m = (a + b) * 0.5;
            let pm = f(m);
            if depth < 10 && pm.distance((pa + pb) * 0.5) > tol {
                go(f, tol, a, pa, m, pm, depth + 1, out);
                go(f, tol, m, pm, b, pb, depth + 1, out);
            } else {
                out.push((b, pb));
            }
        }
        let n0 = n0.max(1);
        let mut prev = (a, f(a));
        out.push(prev);
        for i in 1..=n0 {
            let t = a + (b - a) * i as f64 / n0 as f64;
            let p = f(t);
            go(f, self.tol, prev.0, prev.1, t, p, 0, out);
            prev = (t, p);
        }
    }

    /// An edge's points from its first vertex to its second, both included.
    fn sample_edge(&mut self, a: &[Value]) -> Option<Vec<DVec3>> {
        let m = self.m;
        let p1 = m.vertex(arg(a, 1))?;
        let p2 = m.vertex(arg(a, 2))?;
        let same = flag(a, 4);
        let curve = self.curve(arg(a, 3));
        let closed = p1.distance(p2) <= self.tol * 1e-2;
        let mut pts = match &*curve {
            Curve::Conic { f, a: ra, b: rb } => {
                let param = |p: DVec3| {
                    let l = f.local(p);
                    (l.y / rb).atan2(l.x / ra)
                };
                let t1 = param(p1);
                let mut t2 = param(p2);
                if closed {
                    t2 = if same { t1 + TAU } else { t1 - TAU };
                } else if same {
                    while t2 <= t1 {
                        t2 += TAU;
                    }
                } else {
                    while t2 >= t1 {
                        t2 -= TAU;
                    }
                }
                let n = self
                    .segments(ra.max(*rb), t2 - t1)
                    .max(if closed { 8 } else { 1 });
                (0..=n)
                    .map(|i| curve.eval(t1 + (t2 - t1) * i as f64 / n as f64))
                    .collect()
            }
            Curve::Spline(s) => {
                let (lo, hi) = s.domain(0);
                let range = hi - lo;
                let t1 = curve.closest(p1, None);
                let mut t2 = curve.closest(p2, None);
                if closed {
                    t2 = if same { t1 + range } else { t1 - range };
                } else if s.closed[0] {
                    if same && t2 < t1 {
                        t2 += range;
                    } else if !same && t2 > t1 {
                        t2 -= range;
                    }
                }
                let share = ((t2 - t1).abs() / range).clamp(0.0, 1.0);
                let n0 = ((s.nu as f64 * 2.0 * share).ceil() as usize).max(2);
                let mut out = Vec::new();
                self.adapt(&|t| curve.eval(t), t1, t2, n0, &mut out);
                out.into_iter().map(|x| x.1).collect()
            }
            Curve::Polyline(_) => {
                let t1 = curve.closest(p1, None);
                let t2 = curve.closest(p2, None);
                let mut out = vec![curve.eval(t1)];
                let (lo, hi) = (t1.min(t2), t1.max(t2));
                let mut inner: Vec<f64> = (lo.floor() as i64 + 1..=hi.ceil() as i64 - 1)
                    .map(|i| i as f64)
                    .filter(|t| *t > lo && *t < hi)
                    .collect();
                if t2 < t1 {
                    inner.reverse();
                }
                out.extend(inner.iter().map(|t| curve.eval(*t)));
                out.push(curve.eval(t2));
                out
            }
            _ => vec![p1, p2],
        };
        if pts.len() < 2 {
            pts = vec![p1, p2];
        }
        pts[0] = p1;
        let last = pts.len() - 1;
        pts[last] = p2;
        Some(pts)
    }

    fn edge(&mut self, id: u32) -> Option<SampledEdge> {
        if let Some(c) = self.mesh.edges.get(&id) {
            return Some(c.clone());
        }
        let m = self.m;
        let a = m.get(id)?.part("EDGE_CURVE")?;
        let pts = self.sample_edge(a)?;
        let pts = self.densify(id, pts);
        let (v1, v2) = (arg(a, 1).id()?, arg(a, 2).id()?);
        let last = pts.len() - 1;
        let ids: Vec<u32> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let key = if i == 0 {
                    Key::Vertex(v1)
                } else if i == last {
                    Key::Vertex(v2)
                } else {
                    Key::Edge(id, i as u32)
                };
                self.mesh.key(key, *p)
            })
            .collect();
        let c = Rc::new((ids, pts));
        self.mesh.edges.insert(id, c.clone());
        Some(c)
    }

    /// More points along an edge where a surface it bounds would see a
    /// straight step between two of them as a curve. A spline's parameters
    /// can run fast in one place and slow in another, so a segment that is
    /// straight in space may be bent on the surface's parameters, and a face
    /// triangulated there would never meet its own boundary. Both faces of
    /// the edge share what this adds, so they still meet without a crack.
    fn densify(&self, edge: u32, pts: Vec<DVec3>) -> Vec<DVec3> {
        let Some(surfaces) = self.mesh.edge_surfaces.get(&edge) else {
            return pts;
        };
        let mut out = Vec::with_capacity(pts.len());
        let bent = |a: DVec3, b: DVec3, ua: &[DVec2]| -> Option<Vec<DVec2>> {
            let mid = (a + b) * 0.5;
            let mut ubs = Vec::with_capacity(surfaces.len());
            let mut bad = false;
            for (s, ua) in surfaces.iter().zip(ua) {
                let mut ub = s.invert(b, Some(*ua));
                for (k, period) in s.periods().iter().enumerate() {
                    if let Some(p) = period {
                        ub[k] -= p * ((ub[k] - ua[k]) / p).round();
                    }
                }
                if s.eval((*ua + ub) * 0.5).distance(mid) > 2.0 * self.tol {
                    bad = true;
                }
                ubs.push(ub);
            }
            (!bad).then_some(ubs)
        };
        let first: Vec<DVec2> = surfaces.iter().map(|s| s.invert(pts[0], None)).collect();
        out.push(pts[0]);
        let mut ua = first;
        for w in pts.windows(2) {
            // A stack of the segment's pieces, nearest first.
            let mut todo = vec![(w[1], 0u32)];
            let mut a = w[0];
            while let Some((b, depth)) = todo.pop() {
                let check = (depth < 8 && a.distance(b) > self.tol).then(|| bent(a, b, &ua));
                match check {
                    Some(None) => {
                        todo.push((b, depth + 1));
                        todo.push(((a + b) * 0.5, depth + 1));
                    }
                    Some(Some(ub)) => {
                        out.push(b);
                        ua = ub;
                        a = b;
                    }
                    None => {
                        out.push(b);
                        ua = surfaces
                            .iter()
                            .zip(&ua)
                            .map(|(s, u)| s.invert(b, Some(*u)))
                            .collect();
                        a = b;
                    }
                }
            }
        }
        out
    }

    /// Notes, for each edge, the surfaces of the faces it bounds that
    /// [`Reader::densify`] must look at.
    fn gather(&mut self, v: &Value, depth: u32) {
        let m = self.m;
        let (Some(e), true) = (m.at(v), depth < 12) else {
            return;
        };
        let a = e.args();
        match e.kind() {
            "MANIFOLD_SOLID_BREP" | "FACETED_BREP" => self.gather(arg(a, 1), depth + 1),
            "BREP_WITH_VOIDS" => {
                self.gather(arg(a, 1), depth + 1);
                for x in arg(a, 2).list() {
                    self.gather(x, depth + 1);
                }
            }
            "SHELL_BASED_SURFACE_MODEL"
            | "FACE_BASED_SURFACE_MODEL"
            | "CLOSED_SHELL"
            | "OPEN_SHELL"
            | "CONNECTED_FACE_SET" => {
                for x in arg(a, 1).list() {
                    self.gather(x, depth + 1);
                }
            }
            "ORIENTED_CLOSED_SHELL" | "ORIENTED_OPEN_SHELL" | "ORIENTED_FACE" => {
                self.gather(arg(a, 2), depth + 1)
            }
            "ADVANCED_FACE" | "FACE_SURFACE" | "FACE" => {
                let s = (e.kind() != "FACE").then(|| self.surface(arg(a, 2)));
                let bent = s.as_deref().is_some_and(|s| {
                    matches!(
                        s,
                        Surface::Spline(..)
                            | Surface::Revolution { .. }
                            | Surface::Extrusion { .. }
                            | Surface::Offset(..)
                    )
                });
                for b in arg(a, 1).list() {
                    let Some(bound) = m
                        .part(b, "FACE_OUTER_BOUND")
                        .or_else(|| m.part(b, "FACE_BOUND"))
                    else {
                        continue;
                    };
                    if let Some(poly) = m.part(arg(bound, 1), "POLY_LOOP") {
                        for p in arg(poly, 1).list() {
                            if let Some(p) = m.point(p) {
                                self.mesh.grow(p);
                            }
                        }
                        continue;
                    }
                    let Some(edges) = m.part(arg(bound, 1), "EDGE_LOOP") else {
                        continue;
                    };
                    for oe in arg(edges, 1).list() {
                        let Some(edge) = m.part(oe, "ORIENTED_EDGE").and_then(|o| arg(o, 3).id())
                        else {
                            continue;
                        };
                        if let Some(ec) = m.get(edge).and_then(|e| e.part("EDGE_CURVE")) {
                            for v in [arg(ec, 1), arg(ec, 2)] {
                                if let Some(p) = m.vertex(v) {
                                    self.mesh.grow(p);
                                }
                            }
                        }
                        if let (true, Some(s)) = (bent, &s) {
                            let list = self.mesh.edge_surfaces.entry(edge).or_default();
                            if !list.iter().any(|x| Rc::ptr_eq(x, s)) {
                                list.push(s.clone());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn loop_points(&mut self, v: &Value, orient: bool) -> Option<Loop> {
        let m = self.m;
        let e = m.at(v)?;
        let a = e.args();
        let mut ids = Vec::new();
        let mut pts = Vec::new();
        match e.kind() {
            "EDGE_LOOP" => {
                for oe in arg(a, 1).list() {
                    let o = m.part(oe, "ORIENTED_EDGE")?;
                    let forward = flag(o, 4);
                    let edge = self.edge(arg(o, 3).id()?)?;
                    let n = edge.0.len();
                    for k in 0..n - 1 {
                        let i = if forward { k } else { n - 1 - k };
                        ids.push(edge.0[i]);
                        pts.push(edge.1[i]);
                    }
                }
            }
            "POLY_LOOP" => {
                for p in arg(a, 1).list() {
                    let at = m.point(p)?;
                    ids.push(self.mesh.key(Key::Point(p.id()?), at));
                    pts.push(at);
                }
                // A closing point repeating the first.
                if pts.len() > 1 && ids.first() == ids.last() {
                    ids.pop();
                    pts.pop();
                }
            }
            _ => return None,
        }
        (pts.len() >= 3).then_some(Loop { ids, pts, orient })
    }

    // -- faces -----------------------------------------------------------------

    fn solid(&mut self, v: &Value, flip: bool, colour: Option<[f64; 3]>, depth: u32) {
        let m = self.m;
        let (Some(id), true) = (v.id(), depth < 10) else {
            return;
        };
        let Some(e) = m.get(id) else {
            return;
        };
        let colour = self.colours.get(&id).copied().or(colour);
        let a = e.args();
        match e.kind() {
            "MANIFOLD_SOLID_BREP" | "FACETED_BREP" => {
                self.solid(arg(a, 1), flip, colour, depth + 1)
            }
            "BREP_WITH_VOIDS" => {
                self.solid(arg(a, 1), flip, colour, depth + 1);
                for s in arg(a, 2).list() {
                    self.solid(s, flip, colour, depth + 1);
                }
            }
            "SHELL_BASED_SURFACE_MODEL" | "FACE_BASED_SURFACE_MODEL" => {
                for s in arg(a, 1).list() {
                    self.solid(s, flip, colour, depth + 1);
                }
            }
            "CLOSED_SHELL" | "OPEN_SHELL" | "CONNECTED_FACE_SET" => {
                for f in arg(a, 1).list() {
                    self.face(f, flip, colour, 0);
                }
            }
            "ORIENTED_CLOSED_SHELL" | "ORIENTED_OPEN_SHELL" => {
                self.solid(arg(a, 2), flip ^ !flag(a, 3), colour, depth + 1)
            }
            "ADVANCED_FACE" | "FACE_SURFACE" | "FACE" | "ORIENTED_FACE" => {
                self.face(v, flip, colour, 0)
            }
            _ => {}
        }
    }

    fn face(&mut self, v: &Value, flip: bool, colour: Option<[f64; 3]>, depth: u32) {
        let m = self.m;
        let Some(id) = v.id() else {
            return;
        };
        let Some(e) = m.get(id) else {
            return;
        };
        let a = e.args();
        if e.kind() == "ORIENTED_FACE" {
            if depth < 4 {
                self.face(arg(a, 2), flip ^ !flag(a, 3), colour, depth + 1);
            }
            return;
        }
        self.faces_done += 1;
        if self.faces_done.is_multiple_of(64) {
            self.progress.advance(self.faces_done);
        }
        if self.progress.cancelled() {
            return;
        }
        let colour = self.colours.get(&id).copied().or(colour);
        let (bounds, surface, same) = match e.kind() {
            "ADVANCED_FACE" | "FACE_SURFACE" => (arg(a, 1).list(), Some(arg(a, 2)), flag(a, 3)),
            "FACE" => (arg(a, 1).list(), None, true),
            _ => return,
        };
        let mut loops: Vec<Loop> = Vec::new();
        for b in bounds {
            let Some(be) = m.at(b) else { continue };
            let Some(ba) = be.part("FACE_OUTER_BOUND").or(be.part("FACE_BOUND")) else {
                continue;
            };
            let outer = be.is("FACE_OUTER_BOUND");
            if let Some(l) = self.loop_points(arg(ba, 1), flag(ba, 2)) {
                if outer {
                    loops.insert(0, l);
                } else {
                    loops.push(l);
                }
            }
        }
        if loops.is_empty() {
            return;
        }
        let mat = self.material(colour);
        let surface = surface.map(|s| self.surface(s));
        let done = match surface.as_deref() {
            Some(Surface::Plane(f)) => {
                let sign = if same != flip { 1.0 } else { -1.0 };
                self.planar(*f, sign, &loops, mat);
                true
            }
            Some(Surface::Unknown) | None => false,
            Some(s) => self.curved(s, &loops, same, flip, mat).is_ok(),
        };
        if !done {
            if surface.is_some() && !matches!(surface.as_deref(), Some(Surface::Unknown)) {
                self.rough += 1;
            }
            // Flat over the edges, facing the way the outer loop runs.
            let outer = &loops[0];
            let mut n = newell(&outer.pts);
            if !outer.orient {
                n = -n;
            }
            if flip {
                n = -n;
            }
            if let Some(n) = n.try_normalize() {
                self.planar(Frame::new(outer.pts[0], n, None), 1.0, &loops, mat);
            }
        }
    }

    fn push_tri(&mut self, ids: [u32; 3], normals: [DVec3; 3], mat: i32) {
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
            return;
        }
        self.mesh.tris.push(ids);
        self.mesh.normals.push(normals);
        self.mesh.mats.push(mat);
    }

    /// A flat face: its loops in the plane, triangulated. `sign` turns the
    /// plane's normal to the face's.
    fn planar(&mut self, f: Frame, sign: f64, loops: &[Loop], mat: i32) {
        let rings: Vec<Vec<DVec2>> = loops
            .iter()
            .map(|l| {
                l.pts
                    .iter()
                    .map(|p| {
                        let q = f.local(*p);
                        DVec2::new(q.x, q.y)
                    })
                    .collect()
            })
            .collect();
        let ids: Vec<u32> = loops.iter().flat_map(|l| l.ids.iter().copied()).collect();
        let n = f.z * sign;
        for t in triangulate(&rings) {
            let mut c = t.map(|i| ids[i]);
            if sign < 0.0 {
                c.swap(1, 2);
            }
            self.push_tri(c, [n; 3], mat);
        }
    }

    /// A loop's points on the surface's parameters, unwrapped where the
    /// surface closes on itself.
    fn ring(&self, s: &Surface, l: &Loop) -> Result<Ring, ()> {
        let n = l.pts.len();
        let periods = s.periods();
        let mut uv = Vec::with_capacity(n);
        let mut hint = None;
        for p in &l.pts {
            let q = s.invert(*p, hint);
            if !q.is_finite() {
                return Err(());
            }
            uv.push(q);
            hint = Some(q);
        }
        let singular: Vec<Option<usize>> = l
            .pts
            .iter()
            .zip(&uv)
            .map(|(p, q)| s.singular(*p, *q, self.tol * 0.01))
            .collect();
        let Some(start) = singular.iter().position(|s| s.is_none()) else {
            return Err(());
        };
        let mut ids = l.ids.clone();
        let mut pts = l.pts.clone();
        let mut singular = singular;
        uv.rotate_left(start);
        ids.rotate_left(start);
        pts.rotate_left(start);
        singular.rotate_left(start);

        let mut wind = [0i64; 2];
        for (a, period) in periods.iter().enumerate() {
            let Some(p) = *period else { continue };
            let mut prev = uv[0][a];
            for i in 1..n {
                if singular[i] == Some(a) {
                    continue;
                }
                let x = uv[i][a] - p * ((uv[i][a] - prev) / p).round();
                uv[i][a] = x;
                prev = x;
            }
            let d = uv[0][a] - prev;
            let wrapped = d - p * (d / p).round();
            wind[a] = ((prev + wrapped - uv[0][a]) / p).round() as i64;
        }

        // A loop that passes through the axis can come back on the other side
        // of the seam: a cone's side from its rim up the seam to the apex and
        // down again. Its angle is free at the axis, so that is where it goes
        // round — not at the seam, where it would fold the face over itself.
        for (k, period) in periods.iter().enumerate() {
            let (Some(p), true) = (*period, wind[k] != 0) else {
                continue;
            };
            if let Some(first) = singular.iter().position(|s| *s == Some(k)) {
                for i in first + 1..n {
                    if singular[i] != Some(k) {
                        uv[i][k] -= wind[k] as f64 * p;
                    }
                }
                wind[k] = 0;
            }
        }

        // A loop that is only the seam, from pole to pole and back — a whole
        // sphere — has no width: the way back is the seam's other side.
        for (k, period) in periods.iter().enumerate() {
            let Some(p) = *period else { continue };
            let poles: Vec<usize> = (0..n).filter(|i| singular[*i] == Some(k)).collect();
            if poles.len() < 2 || wind[k] != 0 {
                continue;
            }
            let (lo, hi) = (0..n)
                .filter(|i| singular[*i] != Some(k))
                .fold((f64::MAX, f64::MIN), |(lo, hi), i| {
                    (lo.min(uv[i][k]), hi.max(uv[i][k]))
                });
            if hi - lo < p * 1e-6 {
                for q in &mut uv[poles[0] + 1..poles[1]] {
                    q[k] += p;
                }
            }
        }

        // A point on the axis is where the loop crosses it: its angle is that
        // of the way in and of the way out, so it is two points there.
        let mut out = Ring {
            uv: Vec::with_capacity(n + 4),
            ids: Vec::with_capacity(n + 4),
            pts: Vec::with_capacity(n + 4),
            wind,
            orient: l.orient,
        };
        for i in 0..n {
            if let Some(k) = singular[i] {
                let before = out.uv.last().map(|q| q[k]).unwrap_or(uv[0][k]);
                let after = (i + 1..n)
                    .find(|j| singular[*j] != Some(k))
                    .map(|j| uv[j][k])
                    .unwrap_or(uv[0][k] + wind[k] as f64 * periods[k].unwrap_or(0.0));
                for x in [before, after] {
                    let mut q = uv[i];
                    q[k] = x;
                    out.uv.push(q);
                    out.ids.push(Some(ids[i]));
                    out.pts.push(Some(pts[i]));
                }
            } else {
                out.uv.push(uv[i]);
                out.ids.push(Some(ids[i]));
                out.pts.push(Some(pts[i]));
            }
        }
        Ok(out)
    }

    /// Points along a cut across the surface from `a` to `b`, not including
    /// either: each a vertex both sides of the cut share.
    fn cut(&mut self, s: &Surface, a: DVec2, b: DVec2) -> Vec<(DVec2, u32)> {
        let mut pts = Vec::new();
        self.adapt(&|t| s.eval(a.lerp(b, t)), 0.0, 1.0, 4, &mut pts);
        let n = pts.len();
        pts[1..n.saturating_sub(1)]
            .iter()
            .map(|(t, p)| (a.lerp(b, *t), self.mesh.vertex(*p)))
            .collect()
    }

    fn curved(
        &mut self,
        s: &Surface,
        loops: &[Loop],
        same: bool,
        flip: bool,
        mat: i32,
    ) -> Result<(), ()> {
        let periods = s.periods();
        let mut rings: Vec<Ring> = Vec::with_capacity(loops.len());
        for l in loops {
            rings.push(self.ring(s, l)?);
        }
        let axis = (0..2).find(|a| rings.iter().any(|r| r.wind[*a] != 0));
        let mut polys: Vec<Ring> = Vec::new();
        match axis {
            None => {
                // The outside is the largest.
                let outer = (0..rings.len())
                    .max_by(|a, b| {
                        area(&rings[*a].uv)
                            .abs()
                            .total_cmp(&area(&rings[*b].uv).abs())
                    })
                    .unwrap_or(0);
                polys.push(rings.remove(outer));
            }
            Some(a) => {
                if rings.iter().any(|r| r.wind[1 - a] != 0) {
                    return Err(());
                }
                let winding: Vec<usize> = (0..rings.len())
                    .filter(|i| rings[*i].wind[a] != 0)
                    .collect();
                let p = periods[a].ok_or(())?;
                match winding.len() {
                    1 => {
                        let r = rings.remove(winding[0]);
                        polys.push(self.close_at_pole(s, r, a, p, same)?);
                    }
                    2 => {
                        let r2 = rings.remove(winding[1]);
                        let r1 = rings.remove(winding[0]);
                        polys.push(self.band(s, r1, r2, a, p));
                    }
                    _ => return Err(()),
                }
            }
        }
        // Holes: into the outside's range of each period.
        for mut r in rings {
            for (a, period) in periods.iter().enumerate() {
                let Some(p) = *period else { continue };
                let (lo, hi) = polys[0]
                    .uv
                    .iter()
                    .fold((f64::MAX, f64::MIN), |(lo, hi), q| {
                        (lo.min(q[a]), hi.max(q[a]))
                    });
                let mid = r.uv.iter().map(|q| q[a]).sum::<f64>() / r.uv.len() as f64;
                let k = ((lo + hi) * 0.5 - mid) / p;
                let shift = k.round() * p;
                for q in &mut r.uv {
                    q[a] += shift;
                }
            }
            polys.push(r);
        }

        // Parameters as lengths, so the triangulation sees the shape's shape.
        let scale = self.metric(s, &polys[0].uv);
        let xy: Vec<Vec<DVec2>> = polys
            .iter()
            .map(|r| r.uv.iter().map(|q| *q * scale).collect())
            .collect();
        let tris = triangulate(&xy);

        let mut uv: Vec<DVec2> = Vec::new();
        let mut ids: Vec<Option<u32>> = Vec::new();
        let mut pos: Vec<DVec3> = Vec::new();
        let mut fixed = HashSet::new();
        for r in &polys {
            let base = uv.len();
            let n = r.uv.len();
            for i in 0..n {
                uv.push(r.uv[i]);
                ids.push(r.ids[i]);
                pos.push(r.pts[i].unwrap_or_else(|| s.eval(r.uv[i])));
                fixed.insert(edge_key(base + i, base + (i + 1) % n));
            }
        }
        let on = uv.iter().map(|q| s.eval(*q)).collect();
        let mut refine = Refiner {
            s,
            uv,
            pos,
            on,
            tris,
            edges: HashMap::new(),
            fixed,
            tol: self.tol * 2.0,
            scale,
        };
        refine.run(20_000);
        let sense = same != flip;
        let sign = if sense { 1.0 } else { -1.0 };
        let mut made: Vec<Option<u32>> = ids;
        made.resize(refine.uv.len(), None);
        for t in refine.tris.clone() {
            let mut c = [0u32; 3];
            for k in 0..3 {
                let i = t[k];
                c[k] = match made[i] {
                    Some(id) => id,
                    None => {
                        let id = self.mesh.vertex(refine.pos[i]);
                        made[i] = Some(id);
                        id
                    }
                };
            }
            let mut n = t.map(|i| s.normal(refine.uv[i]) * sign);
            if !sense {
                c.swap(1, 2);
                n.swap(1, 2);
            }
            self.push_tri(c, n, mat);
        }
        Ok(())
    }

    /// Lengths per unit of each parameter, about the ring.
    fn metric(&self, s: &Surface, ring: &[DVec2]) -> DVec2 {
        let span = s.spans();
        let h = span * 1e-4;
        let step = (ring.len() / 16).max(1);
        let mut sum = DVec2::ZERO;
        let mut count = 0.0f64;
        for q in ring.iter().step_by(step) {
            let du = s
                .eval(*q + DVec2::new(h.x, 0.0))
                .distance(s.eval(*q - DVec2::new(h.x, 0.0)))
                / (2.0 * h.x);
            let dv = s
                .eval(*q + DVec2::new(0.0, h.y))
                .distance(s.eval(*q - DVec2::new(0.0, h.y)))
                / (2.0 * h.y);
            sum += DVec2::new(du, dv);
            count += 1.0;
        }
        let m = sum / count.max(1.0);
        DVec2::new(
            if m.x.is_normal() { m.x } else { 1.0 },
            if m.y.is_normal() { m.y } else { 1.0 },
        )
    }

    /// One loop that goes once around the surface — a sphere's cap, a cone's
    /// side — closed off at the pole on the face's side.
    fn close_at_pole(
        &mut self,
        s: &Surface,
        mut r: Ring,
        a: usize,
        p: f64,
        same: bool,
    ) -> Result<Ring, ()> {
        let b = 1 - a;
        let k = r.wind[a] as f64;
        // The face lies to the left of its loop, seen from its normal.
        let left = same == r.orient;
        let high = if a == 0 {
            (k > 0.0) == left
        } else {
            (k > 0.0) != left
        };
        let (lo, hi) = s.limits(b);
        let target = if high { hi } else { lo }.ok_or(())?;
        let first = r.uv[0];
        let mut shifted = first;
        shifted[a] += k * p;
        let mut pole_a = shifted;
        pole_a[b] = target;
        let mut pole_b = first;
        pole_b[b] = target;
        let cut = self.cut(s, shifted, pole_a);
        let pole = self.mesh.vertex(s.eval(pole_a));
        let first_id = r.ids[0];
        let first_pt = r.pts[0];
        r.uv.push(shifted);
        r.ids.push(first_id);
        r.pts.push(first_pt);
        for (q, id) in &cut {
            r.uv.push(*q);
            r.ids.push(Some(*id));
            r.pts.push(None);
        }
        for q in [pole_a, pole_b] {
            r.uv.push(q);
            r.ids.push(Some(pole));
            r.pts.push(None);
        }
        for (q, id) in cut.iter().rev() {
            let mut q = *q;
            q[a] -= k * p;
            r.uv.push(q);
            r.ids.push(Some(*id));
            r.pts.push(None);
        }
        Ok(r)
    }

    /// Two loops that each go once around — a cylinder's side between two
    /// circles — joined by a cut into one.
    fn band(&mut self, s: &Surface, mut r1: Ring, mut r2: Ring, a: usize, p: f64) -> Ring {
        let k1 = r1.wind[a];
        if r2.wind[a] == k1 {
            r2.uv.reverse();
            r2.ids.reverse();
            r2.pts.reverse();
            r2.wind[a] = -k1;
        }
        let k2 = r2.wind[a] as f64;
        let mut start = r1.uv[0];
        start[a] += k1 as f64 * p;
        let wrap = |d: f64| d - p * (d / p).round();
        let j = (0..r2.uv.len())
            .min_by(|x, y| {
                wrap(r2.uv[*x][a] - start[a])
                    .abs()
                    .total_cmp(&wrap(r2.uv[*y][a] - start[a]).abs())
            })
            .unwrap_or(0);
        let shift = ((start[a] - r2.uv[j][a]) / p).round() * p;
        let n2 = r2.uv.len();
        let mut seq: Vec<(DVec2, Option<u32>, Option<DVec3>)> = Vec::with_capacity(n2 + 1);
        for k in 0..n2 {
            let i = (j + k) % n2;
            let mut q = r2.uv[i];
            q[a] += shift + if i < j { k2 * p } else { 0.0 };
            seq.push((q, r2.ids[i], r2.pts[i]));
        }
        let (mut end, end_id, end_pt) = seq[0];
        end[a] += k2 * p;
        seq.push((end, end_id, end_pt));
        let cut = self.cut(s, start, seq[0].0);

        let first = (r1.ids[0], r1.pts[0]);
        r1.uv.push(start);
        r1.ids.push(first.0);
        r1.pts.push(first.1);
        for (q, id) in &cut {
            r1.uv.push(*q);
            r1.ids.push(Some(*id));
            r1.pts.push(None);
        }
        for (q, id, pt) in seq {
            r1.uv.push(q);
            r1.ids.push(id);
            r1.pts.push(pt);
        }
        for (q, id) in cut.iter().rev() {
            let mut q = *q;
            q[a] -= k1 as f64 * p;
            r1.uv.push(q);
            r1.ids.push(Some(*id));
            r1.pts.push(None);
        }
        r1.wind = [0, 0];
        r1
    }

    // -- the hierarchy -----------------------------------------------------------

    fn read_into(&mut self, scene: &mut ImportScene, file_name: &str) -> Result<(), StepError> {
        let m = self.m;
        // Representations, and which are the same shape.
        let mut reps: Vec<u32> = Vec::new();
        for id in &m.ids {
            if rep_items(m.get(*id).expect("listed")).is_some() {
                reps.push(*id);
            }
        }
        let index: HashMap<u32, usize> = reps.iter().enumerate().map(|(i, r)| (*r, i)).collect();
        let mut parent: Vec<usize> = (0..reps.len()).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }

        // Product definitions and their representations, for names and to
        // tell an assembly's parent from its child.
        let mut pd_rep: HashMap<u32, u32> = HashMap::new();
        let mut names: HashMap<u32, String> = HashMap::new();
        for id in &m.ids {
            let e = m.get(*id).expect("listed");
            let Some(a) = e.part("SHAPE_DEFINITION_REPRESENTATION") else {
                continue;
            };
            let Some(rep) = arg(a, 1).id() else { continue };
            let Some(pds) = m.part(arg(a, 0), "PRODUCT_DEFINITION_SHAPE") else {
                continue;
            };
            if let Some(pd) = arg(pds, 2).id() {
                pd_rep.insert(pd, rep);
                if let Some(n) = self.product_name(pd) {
                    names.entry(rep).or_insert(n);
                }
            }
        }

        // Links: a child shape placed in a parent.
        struct Link {
            child: u32,
            parent: u32,
            t: DAffine3,
        }
        let mut links: Vec<Link> = Vec::new();
        let mut placed_by: HashMap<u32, (u32, u32)> = HashMap::new(); // relationship → (parent pd, child pd)
        for id in &m.ids {
            let e = m.get(*id).expect("listed");
            if let Some(a) = e.part("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") {
                let (Some(rel), Some(pds)) = (
                    arg(a, 0).id(),
                    m.part(arg(a, 1), "PRODUCT_DEFINITION_SHAPE"),
                ) else {
                    continue;
                };
                if let Some(nauo) = m.at(arg(pds, 2)).and_then(|n| {
                    n.parts
                        .iter()
                        .find(|p| p.0.ends_with("USAGE_OCCURRENCE"))
                        .map(|p| p.1.as_slice())
                }) {
                    if let (Some(pp), Some(cp)) = (arg(nauo, 3).id(), arg(nauo, 4).id()) {
                        placed_by.insert(rel, (pp, cp));
                    }
                }
            }
        }
        for id in &m.ids {
            let e = m.get(*id).expect("listed");
            let rr = e
                .part("REPRESENTATION_RELATIONSHIP")
                .or(e.part("SHAPE_REPRESENTATION_RELATIONSHIP"));
            let Some(rr) = rr else { continue };
            let (Some(r1), Some(r2)) = (arg(rr, 2).id(), arg(rr, 3).id()) else {
                continue;
            };
            if !index.contains_key(&r1) || !index.contains_key(&r2) {
                continue;
            }
            match e.part("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION") {
                Some(t) => {
                    let Some(idt) = m.part(arg(t, 0), "ITEM_DEFINED_TRANSFORMATION") else {
                        continue;
                    };
                    // Takes the first item's frame, in rep 1, onto the
                    // second's, in rep 2.
                    let mut tr =
                        m.frame(arg(idt, 3)).matrix() * m.frame(arg(idt, 2)).matrix().inverse();
                    let (mut child, mut par) = (r1, r2);
                    // The assembly's own record of which is which, when it
                    // says the other way.
                    if let Some((pp, cp)) = placed_by.get(id) {
                        let mut same = |a: u32, b: Option<&u32>| match b.and_then(|b| index.get(b))
                        {
                            Some(&j) => root(&mut parent, index[&a]) == root(&mut parent, j),
                            None => false,
                        };
                        if same(r2, pd_rep.get(cp)) || same(r1, pd_rep.get(pp)) {
                            (child, par) = (r2, r1);
                            tr = tr.inverse();
                        }
                    }
                    links.push(Link {
                        child,
                        parent: par,
                        t: tr,
                    });
                }
                None => {
                    let (x, y) = (root(&mut parent, index[&r1]), root(&mut parent, index[&r2]));
                    parent[x] = y;
                }
            }
        }

        // Groups: representations that are one shape.
        let mut group_of: Vec<usize> = (0..reps.len()).map(|i| root(&mut parent, i)).collect();
        let mut group_ids: HashMap<usize, usize> = HashMap::new();
        for g in &mut group_of {
            let n = group_ids.len();
            *g = *group_ids.entry(*g).or_insert(n);
        }
        let groups = group_ids.len();
        let mut items: Vec<Vec<Value>> = vec![Vec::new(); groups];
        let mut name: Vec<Option<String>> = vec![None; groups];
        let mut children: Vec<Vec<(usize, DAffine3)>> = vec![Vec::new(); groups];
        let mut is_child = vec![false; groups];
        for (i, r) in reps.iter().enumerate() {
            let g = group_of[i];
            let e = m.get(*r).expect("listed");
            if name[g].is_none() {
                name[g] = names.get(r).cloned();
            }
            for item in rep_items(e).unwrap_or(&[]) {
                let Some(ie) = m.at(item) else { continue };
                if GEOMETRY.contains(&ie.kind()) {
                    items[g].push(item.clone());
                    if name[g].is_none() {
                        let n = arg(ie.args(), 0).text();
                        if !n.is_empty() {
                            name[g] = Some(n.to_string());
                        }
                    }
                } else if let Some(mi) = ie.part("MAPPED_ITEM") {
                    let Some(map) = m.part(arg(mi, 1), "REPRESENTATION_MAP") else {
                        continue;
                    };
                    let Some(child) = arg(map, 1).id().and_then(|c| index.get(&c)) else {
                        continue;
                    };
                    let t = m.frame(arg(mi, 2)).matrix() * m.frame(arg(map, 0)).matrix().inverse();
                    let c = group_of[*child];
                    if c != g {
                        children[g].push((c, t));
                        is_child[c] = true;
                    }
                }
            }
            if name[g].is_none() {
                let n = arg(e.args(), 0).text();
                if !n.is_empty() && !e.complex() {
                    name[g] = Some(n.to_string());
                }
            }
        }
        for l in &links {
            let (c, p) = (group_of[index[&l.child]], group_of[index[&l.parent]]);
            if c != p {
                children[p].push((c, l.t));
                is_child[c] = true;
            }
        }

        // What a group holds, all the way down: a root with nothing under it
        // is a context, not a part.
        let mut holds = vec![None; groups];
        fn holds_any(
            g: usize,
            items: &[Vec<Value>],
            children: &[Vec<(usize, DAffine3)>],
            memo: &mut [Option<bool>],
            depth: u32,
        ) -> bool {
            if let Some(h) = memo[g] {
                return h;
            }
            if depth > 64 {
                return false;
            }
            memo[g] = Some(false);
            let h = !items[g].is_empty()
                || children[g]
                    .iter()
                    .any(|(c, _)| holds_any(*c, items, children, memo, depth + 1));
            memo[g] = Some(h);
            h
        }
        let roots: Vec<usize> = (0..groups)
            .filter(|g| !is_child[*g] && holds_any(*g, &items, &children, &mut holds, 0))
            .collect();

        // Every face once, which is what is cut: a part placed twice is
        // shaped once.
        let faces = m
            .ids
            .iter()
            .filter(|id| {
                matches!(
                    m.get(**id).map(Entity::kind),
                    Some("ADVANCED_FACE" | "FACE_SURFACE" | "FACE")
                )
            })
            .count();
        self.progress.start(ReadStage::Shaping, faces as u64);
        let mut meshes: Vec<Option<Option<usize>>> = vec![None; groups];
        let fallback = file_name.to_string();
        let mut stack: Vec<(usize, Option<usize>, DAffine3, u32)> = roots
            .iter()
            .rev()
            .map(|g| (*g, None, DAffine3::IDENTITY, 0))
            .collect();
        while let Some((g, up, t, depth)) = stack.pop() {
            if self.progress.cancelled() {
                return Err(StepError::Cancelled);
            }
            if depth > 32 {
                continue;
            }
            let mesh = match meshes[g] {
                Some(made) => made,
                None => {
                    let made =
                        self.build(&items[g], name[g].as_deref().unwrap_or(&fallback), scene);
                    meshes[g] = Some(made);
                    made
                }
            };
            let node = scene.nodes.len();
            let mut n = ImportNode::new(name[g].clone().unwrap_or_else(|| fallback.clone()));
            n.parent = up;
            n.transform = import_transform(&t);
            n.data = match mesh {
                Some(i) => ImportNodeData::Mesh(i),
                None => ImportNodeData::Empty,
            };
            scene.nodes.push(n);
            for (c, ct) in children[g].iter().rev() {
                stack.push((*c, Some(node), *ct, depth + 1));
            }
        }

        // A file whose solids no representation names: every solid, as one part.
        if scene.meshes.is_empty() {
            let loose: Vec<Value> = m
                .ids
                .iter()
                .filter(|id| {
                    let k = m.get(**id).expect("listed").kind();
                    matches!(
                        k,
                        "MANIFOLD_SOLID_BREP"
                            | "BREP_WITH_VOIDS"
                            | "FACETED_BREP"
                            | "SHELL_BASED_SURFACE_MODEL"
                    )
                })
                .map(|id| Value::Ref(*id))
                .collect();
            scene.nodes.clear();
            if let Some(i) = self.build(&loose, file_name, scene) {
                let mut n = ImportNode::new(file_name);
                n.data = ImportNodeData::Mesh(i);
                scene.nodes.push(n);
            }
        }
        if scene.meshes.is_empty() {
            return Err(StepError::NoGeometry);
        }
        scene.materials = std::mem::take(&mut self.materials);
        Ok(())
    }

    fn product_name(&self, pd: u32) -> Option<String> {
        let m = self.m;
        let pda = m.get(pd)?.part("PRODUCT_DEFINITION")?;
        let fe = m.at(arg(pda, 2))?;
        let fa = fe
            .part("PRODUCT_DEFINITION_FORMATION")
            .or(fe.part("PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE"))?;
        let product = m.part(arg(fa, 2), "PRODUCT")?;
        [arg(product, 1).text(), arg(product, 0).text()]
            .into_iter()
            .find(|n| !n.trim().is_empty())
            .map(|n| n.trim().to_string())
    }

    /// One mesh from a part's solids, or `None` when they make no triangles.
    fn build(&mut self, items: &[Value], name: &str, scene: &mut ImportScene) -> Option<usize> {
        self.mesh = MeshBuilder::default();
        for item in items {
            self.gather(item, 0);
        }
        // The tolerance from the part's size: a screw and a building get the
        // same look.
        if let Some((lo, hi)) = self.mesh.lo {
            let diag = (hi - lo).length();
            if diag > 0.0 {
                self.tol = diag * 4e-4;
            }
        }
        for item in items {
            self.solid(item, false, None, 0);
        }
        let mesh = std::mem::take(&mut self.mesh);
        if mesh.tris.is_empty() {
            return None;
        }
        let coloured = mesh.mats.iter().any(|m| *m >= 0);
        let mut mats = mesh.mats;
        if coloured && mats.iter().any(|m| *m < 0) {
            let plain = self.material(Some([0.8, 0.8, 0.8]));
            for m in &mut mats {
                if *m < 0 {
                    *m = plain;
                }
            }
        }
        scene.meshes.push(ImportMesh {
            name: name.to_string(),
            positions: mesh.positions.iter().map(|p| p.as_vec3()).collect(),
            faces: mesh.tris.iter().map(|t| t.to_vec()).collect(),
            corner_uvs: None,
            corner_normals: Some(
                mesh.normals
                    .iter()
                    .flat_map(|n| n.iter().map(|x| x.as_vec3()))
                    .collect::<Vec<Vec3>>(),
            ),
            corner_colors: None,
            face_materials: coloured.then_some(mats),
            smoothing_groups: None,
        });
        Some(scene.meshes.len() - 1)
    }
}

/// A representation's items, for an entity that is one.
fn rep_items(e: &Entity) -> Option<&[Value]> {
    if e.complex() {
        return e.part("REPRESENTATION").map(|a| arg(a, 1).list());
    }
    let k = e.kind();
    let excluded = k.contains("RELATIONSHIP")
        || k.starts_with("SHAPE_DEFINITION")
        || k.starts_with("PROPERTY_DEFINITION")
        || k.starts_with("CONTEXT_DEPENDENT");
    (k.ends_with("REPRESENTATION") && !excluded).then(|| arg(e.args(), 1).list())
}

fn import_transform(t: &DAffine3) -> ImportTransform {
    let (scale, rotation, translation) = t.to_scale_rotation_translation();
    ImportTransform {
        translation: translation.as_vec3(),
        rotation: rotation.as_quat(),
        scale: scale.as_vec3(),
    }
}

fn named_colour(name: &str) -> Option<[f64; 3]> {
    Some(match name.to_ascii_lowercase().as_str() {
        "red" => [1.0, 0.0, 0.0],
        "green" => [0.0, 1.0, 0.0],
        "blue" => [0.0, 0.0, 1.0],
        "yellow" => [1.0, 1.0, 0.0],
        "magenta" => [1.0, 0.0, 1.0],
        "cyan" => [0.0, 1.0, 1.0],
        "black" => [0.0, 0.0, 0.0],
        "white" => [1.0, 1.0, 1.0],
        _ => return None,
    })
}

fn newell(pts: &[DVec3]) -> DVec3 {
    let mut n = DVec3::ZERO;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        n += DVec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    n
}

fn area(pts: &[DVec2]) -> f64 {
    let mut s = 0.0;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        s += a.perp_dot(b);
    }
    s * 0.5
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

// ---------------------------------------------------------------------------
// Triangulation: a polygon with holes, then refined until it follows the
// surface.

fn triangulate(rings: &[Vec<DVec2>]) -> Vec<[usize; 3]> {
    crate::earcut::triangulate(rings)
}

fn orient(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b - a).perp_dot(c - a)
}

/// Splits triangles until they lie within the tolerance of the surface. The
/// boundary is never split — its points are shared with the neighbouring
/// faces — so every new point is inside the face.
struct Refiner<'s> {
    s: &'s Surface,
    uv: Vec<DVec2>,
    /// Where each point goes in the mesh: the boundary's own points, which
    /// its neighbours share.
    pos: Vec<DVec3>,
    /// Where each point is on the surface. The two differ on the boundary by
    /// however far the file's edges are from its surfaces, and the bending is
    /// judged on this one, or a boundary point a little off the surface would
    /// look like a bend that splitting never straightens.
    on: Vec<DVec3>,
    tris: Vec<[usize; 3]>,
    edges: HashMap<(usize, usize), Vec<usize>>,
    fixed: HashSet<(usize, usize)>,
    tol: f64,
    /// Lengths per unit of each parameter: where the shape of a triangle is
    /// judged.
    scale: DVec2,
}

/// Positive when `d` is inside the circle through the anticlockwise `a b c`.
fn incircle(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> f64 {
    let (a, b, c) = (a - d, b - d, c - d);
    let (la, lb, lc) = (a.length_squared(), b.length_squared(), c.length_squared());
    a.x * (b.y * lc - lb * c.y) - a.y * (b.x * lc - lb * c.x) + la * (b.x * c.y - b.y * c.x)
}

impl Refiner<'_> {
    fn xy(&self, i: usize) -> DVec2 {
        self.uv[i] * self.scale
    }

    /// The triangle turned so it starts with the edge `a b` either way, or
    /// `None` when it does not have that edge.
    fn turned(tri: [usize; 3], a: usize, b: usize) -> Option<[usize; 3]> {
        (0..3)
            .find(|k| edge_key(tri[*k], tri[(k + 1) % 3]) == edge_key(a, b))
            .map(|k| [tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]])
    }

    /// Flips the inside edges until every pair of triangles is Delaunay on
    /// the parameters: no slivers between the points of a long boundary,
    /// and none left flat on a straight one.
    fn flip(&mut self, mut stack: Vec<(usize, usize)>) {
        // Lawson's flips can take the square of the triangles: an ear-clipped
        // fan turned into a strip does.
        let n = self.tris.len() + 32;
        let mut budget = (n * n).min(8_000_000);
        while let Some(e) = stack.pop() {
            if budget == 0 {
                return;
            }
            budget -= 1;
            if self.fixed.contains(&e) {
                continue;
            }
            let Some(l) = self.edges.get(&e) else {
                continue;
            };
            if l.len() != 2 {
                continue;
            }
            let (t1, t2) = (l[0], l[1]);
            let (Some([x, y, c]), Some([y2, x2, d])) = (
                Self::turned(self.tris[t1], e.0, e.1),
                Self::turned(self.tris[t2], e.0, e.1),
            ) else {
                continue;
            };
            if y2 != y || x2 != x || c == d {
                continue;
            }
            let (px, py, pc, pd) = (self.xy(x), self.xy(y), self.xy(c), self.xy(d));
            let size = (px - py).length_squared().max(1e-300);
            let valid = orient(px, pd, pc) > 1e-12 * size && orient(py, pc, pd) > 1e-12 * size;
            if !valid {
                continue;
            }
            let flat = orient(px, py, pc) <= 1e-12 * size || orient(py, px, pd) <= 1e-12 * size;
            if !flat && incircle(px, py, pc, pd) <= 0.0 {
                continue;
            }
            self.tris[t1] = [x, d, c];
            self.tris[t2] = [y, c, d];
            self.edges.remove(&e);
            self.edges.insert(edge_key(c, d), vec![t1, t2]);
            for (edge, from, to) in [((y, c), t1, t2), ((x, d), t2, t1)] {
                if let Some(l) = self.edges.get_mut(&edge_key(edge.0, edge.1)) {
                    for t in l.iter_mut() {
                        if *t == from {
                            *t = to;
                        }
                    }
                }
            }
            for (a, b) in [(x, c), (y, c), (x, d), (y, d)] {
                stack.push(edge_key(a, b));
            }
        }
    }

    fn point(&mut self, q: DVec2) -> usize {
        let at = self.s.eval(q);
        self.uv.push(q);
        self.pos.push(at);
        self.on.push(at);
        self.uv.len() - 1
    }

    fn needs(&self, a: usize, b: usize) -> bool {
        if self.fixed.contains(&edge_key(a, b)) {
            return false;
        }
        let (pa, pb) = (self.on[a], self.on[b]);
        if pa.distance(pb) < self.tol {
            return false;
        }
        self.s
            .eval((self.uv[a] + self.uv[b]) * 0.5)
            .distance((pa + pb) * 0.5)
            > self.tol
    }

    fn split(&mut self, a: usize, b: usize, out: &mut Vec<(usize, usize)>) {
        let Some(list) = self.edges.remove(&edge_key(a, b)) else {
            return;
        };
        let m = self.point((self.uv[a] + self.uv[b]) * 0.5);
        out.push((a, m));
        out.push((m, b));
        for t in list {
            let tri = self.tris[t];
            let Some(k) = (0..3).find(|k| edge_key(tri[*k], tri[(k + 1) % 3]) == edge_key(a, b))
            else {
                continue;
            };
            let (x, y, c) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            self.tris[t] = [x, m, c];
            let t2 = self.tris.len();
            self.tris.push([m, y, c]);
            if let Some(l) = self.edges.get_mut(&edge_key(y, c)) {
                for e in l.iter_mut() {
                    if *e == t {
                        *e = t2;
                    }
                }
            }
            self.edges.entry(edge_key(x, m)).or_default().push(t);
            self.edges.entry(edge_key(m, y)).or_default().push(t2);
            let mc = self.edges.entry(edge_key(m, c)).or_default();
            mc.push(t);
            mc.push(t2);
            out.push((m, c));
        }
    }

    fn split_centre(&mut self, t: usize) {
        let [a, b, c] = self.tris[t];
        let m = self.point((self.uv[a] + self.uv[b] + self.uv[c]) / 3.0);
        self.tris[t] = [a, b, m];
        let t2 = self.tris.len();
        self.tris.push([b, c, m]);
        let t3 = self.tris.len();
        self.tris.push([c, a, m]);
        for (e, to) in [((b, c), t2), ((c, a), t3)] {
            if let Some(l) = self.edges.get_mut(&edge_key(e.0, e.1)) {
                for x in l.iter_mut() {
                    if *x == t {
                        *x = to;
                    }
                }
            }
        }
        self.edges.insert(edge_key(a, m), vec![t, t3]);
        self.edges.insert(edge_key(b, m), vec![t, t2]);
        self.edges.insert(edge_key(c, m), vec![t2, t3]);
    }

    fn run(&mut self, max: usize) {
        for (t, tri) in self.tris.iter().enumerate() {
            for k in 0..3 {
                self.edges
                    .entry(edge_key(tri[k], tri[(k + 1) % 3]))
                    .or_default()
                    .push(t);
            }
        }
        for _ in 0..6 {
            let mut all: Vec<(usize, usize)> = self.edges.keys().copied().collect();
            all.sort_unstable();
            self.flip(all.clone());
            // Longest first, so a long edge is cut before the short ones
            // beside it, and the triangles stay shapely.
            let mut stack: Vec<(usize, usize)> = self.edges.keys().copied().collect();
            stack.sort_unstable();
            let mut fresh = Vec::new();
            let mut flips = Vec::new();
            while let Some((a, b)) = stack.pop() {
                if self.tris.len() > max {
                    return;
                }
                if self.edges.contains_key(&edge_key(a, b)) && self.needs(a, b) {
                    fresh.clear();
                    self.split(a, b, &mut fresh);
                    flips.clear();
                    for (x, y) in &fresh {
                        flips.push(edge_key(*x, *y));
                    }
                    // The edges around the new point, kept Delaunay.
                    let m = self.uv.len() - 1;
                    let around: Vec<usize> = [edge_key(a, m), edge_key(m, b)]
                        .iter()
                        .filter_map(|k| self.edges.get(k))
                        .flatten()
                        .copied()
                        .collect();
                    for t in around {
                        let [p, q, r] = self.tris[t];
                        for (u, v) in [(p, q), (q, r), (r, p)] {
                            if u != m && v != m {
                                flips.push(edge_key(u, v));
                            }
                        }
                    }
                    self.flip(flips.clone());
                    stack.extend(fresh.iter().map(|(x, y)| edge_key(*x, *y)));
                }
            }
            // A triangle whose edges all lie on the surface can still bulge
            // away from it in the middle.
            let mut any = false;
            for t in 0..self.tris.len() {
                if self.tris.len() > max {
                    return;
                }
                let [a, b, c] = self.tris[t];
                let (pa, pb, pc) = (self.on[a], self.on[b], self.on[c]);
                if (pb - pa).cross(pc - pa).length() < self.tol * self.tol {
                    continue;
                }
                let mid = self.s.eval((self.uv[a] + self.uv[b] + self.uv[c]) / 3.0);
                if mid.distance((pa + pb + pc) / 3.0) > self.tol {
                    self.split_centre(t);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('t','',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=CARTESIAN_POINT('',(0.,0.,20.));
#6=AXIS2_PLACEMENT_3D('',#5,#2,#3);
#7=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#8=( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );
#9=( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#7,#8)) REPRESENTATION_CONTEXT('','') );
#20=CIRCLE('',#4,10.);
#21=CIRCLE('',#6,10.);
#22=CARTESIAN_POINT('',(10.,0.,0.));
#23=CARTESIAN_POINT('',(10.,0.,20.));
#24=VERTEX_POINT('',#22);
#25=VERTEX_POINT('',#23);
#26=VECTOR('',#2,1.);
#27=LINE('',#22,#26);
#30=EDGE_CURVE('',#24,#24,#20,.T.);
#31=EDGE_CURVE('',#25,#25,#21,.T.);
#32=EDGE_CURVE('',#24,#25,#27,.T.);
#11=PLANE('',#4);
#12=PLANE('',#6);
#50=ORIENTED_EDGE('',*,*,#30,.F.);
#51=EDGE_LOOP('',(#50));
#52=FACE_OUTER_BOUND('',#51,.T.);
#53=ADVANCED_FACE('',(#52),#11,.F.);
";

    const TAIL: &str = "ENDSEC;
END-ISO-10303-21;
";

    /// A cylinder 10 mm in radius and 20 tall: two discs and a side.
    fn cylinder() -> String {
        format!(
            "{HEAD}#10=CYLINDRICAL_SURFACE('',#4,10.);
#40=ORIENTED_EDGE('',*,*,#30,.T.);
#41=ORIENTED_EDGE('',*,*,#32,.T.);
#42=ORIENTED_EDGE('',*,*,#31,.F.);
#43=ORIENTED_EDGE('',*,*,#32,.F.);
#44=EDGE_LOOP('',(#40,#41,#42,#43));
#45=FACE_OUTER_BOUND('',#44,.T.);
#46=ADVANCED_FACE('',(#45),#10,.T.);
#54=ORIENTED_EDGE('',*,*,#31,.T.);
#55=EDGE_LOOP('',(#54));
#56=FACE_OUTER_BOUND('',#55,.T.);
#57=ADVANCED_FACE('',(#56),#12,.T.);
#60=CLOSED_SHELL('',(#46,#53,#57));
#61=MANIFOLD_SOLID_BREP('Cylinder',#60);
#62=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#61,#4),#9);
#70=APPLICATION_CONTEXT('');
#71=PRODUCT_CONTEXT('',#70,'');
#72=PRODUCT('Can','Can','',(#71));
#73=PRODUCT_DEFINITION_FORMATION('','',#72);
#74=PRODUCT_DEFINITION_CONTEXT('',#70,'');
#75=PRODUCT_DEFINITION('','',#73,#74);
#76=PRODUCT_DEFINITION_SHAPE('','',#75);
#77=SHAPE_DEFINITION_REPRESENTATION(#76,#62);
{TAIL}"
        )
    }

    fn volume(mesh: &ImportMesh) -> f64 {
        mesh.faces
            .iter()
            .map(|f| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[f[k] as usize].as_dvec3());
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// Every edge between exactly two triangles, running opposite ways.
    fn watertight(mesh: &ImportMesh) -> bool {
        let mut seen: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &mesh.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *seen.entry((a, b)).or_default() += 1;
            }
        }
        seen.iter()
            .all(|((a, b), n)| *n == 1 && seen.get(&(*b, *a)) == Some(&1))
    }

    #[test]
    fn a_cylinder_reads_closed_round_and_in_millimetres() {
        let scene = parse_step(&cylinder(), "file").expect("reads");
        assert_eq!(scene.source.unit_scale, 1e-3);
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.nodes[0].name, "Can");
        let mesh = &scene.meshes[0];
        assert!(watertight(mesh), "the faces meet without a crack");
        assert!(
            mesh.faces.len() < 2000,
            "not more than it needs: {}",
            mesh.faces.len()
        );
        let v = volume(mesh);
        let exact = PI * 100.0 * 20.0;
        assert!(
            (v - exact).abs() < exact * 0.01,
            "volume {v} against {exact}"
        );
        // Every point on the cylinder.
        for p in &mesh.positions {
            let r = p.x.hypot(p.y) as f64;
            assert!(r <= 10.0 + 1e-3 && (p.z >= -1e-4 && p.z <= 20.0 + 1e-4));
        }
        // The side's normals point out, the caps' up and down.
        let normals = mesh.corner_normals.as_ref().unwrap();
        for (f, face) in mesh.faces.iter().enumerate() {
            for k in 0..3 {
                let p = mesh.positions[face[k] as usize];
                let n = normals[f * 3 + k];
                let out = (p.z.abs() < 1e-4 && n.z < -0.5)
                    || ((p.z - 20.0).abs() < 1e-4 && n.z > 0.5)
                    || Vec3::new(p.x, p.y, 0.0).dot(n) > 0.0;
                assert!(out, "{p:?} faces {n:?}");
            }
        }
    }

    #[test]
    fn a_hemisphere_closes_at_its_pole() {
        let text = format!(
            "{HEAD}#10=SPHERICAL_SURFACE('',#4,10.);
#40=ORIENTED_EDGE('',*,*,#30,.T.);
#44=EDGE_LOOP('',(#40));
#45=FACE_OUTER_BOUND('',#44,.T.);
#46=ADVANCED_FACE('',(#45),#10,.T.);
#60=CLOSED_SHELL('',(#46,#53));
#61=MANIFOLD_SOLID_BREP('Dome',#60);
#62=ADVANCED_BREP_SHAPE_REPRESENTATION('Dome',(#61,#4),#9);
{TAIL}"
        );
        let scene = parse_step(&text, "file").expect("reads");
        let mesh = &scene.meshes[0];
        assert!(watertight(mesh));
        let v = volume(mesh);
        let exact = 2.0 / 3.0 * PI * 1000.0;
        assert!(
            (v - exact).abs() < exact * 0.01,
            "volume {v} against {exact}"
        );
        let top = mesh.positions.iter().map(|p| p.z).fold(f32::MIN, f32::max);
        assert!((top - 10.0).abs() < 1e-3, "reaches the pole: {top}");
    }

    #[test]
    fn an_assembly_places_one_part_twice() {
        let text = cylinder().replace(
            TAIL,
            "#100=SHAPE_REPRESENTATION('Assembly',(#4),#9);
#101=CARTESIAN_POINT('',(50.,0.,0.));
#102=AXIS2_PLACEMENT_3D('',#101,#2,#3);
#103=ITEM_DEFINED_TRANSFORMATION('','',#4,#4);
#104=ITEM_DEFINED_TRANSFORMATION('','',#4,#102);
#105=( REPRESENTATION_RELATIONSHIP('','',#62,#100) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#103) SHAPE_REPRESENTATION_RELATIONSHIP() );
#106=( REPRESENTATION_RELATIONSHIP('','',#62,#100) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#104) SHAPE_REPRESENTATION_RELATIONSHIP() );
ENDSEC;
END-ISO-10303-21;
",
        );
        let scene = parse_step(&text, "file").expect("reads");
        assert_eq!(scene.meshes.len(), 1, "the part once");
        assert_eq!(scene.nodes.len(), 3, "an assembly and two placements");
        assert_eq!(scene.nodes[0].name, "Assembly");
        let placed: Vec<f32> = scene.nodes[1..]
            .iter()
            .map(|n| {
                assert_eq!(n.parent, Some(0));
                assert!(matches!(n.data, ImportNodeData::Mesh(0)));
                n.transform.translation.x
            })
            .collect();
        assert_eq!(placed, vec![0.0, 50.0]);
    }

    #[test]
    fn a_colour_becomes_a_material() {
        let text = cylinder().replace(
            TAIL,
            "#200=COLOUR_RGB('',1.,0.5,0.);
#201=FILL_AREA_STYLE_COLOUR('',#200);
#202=FILL_AREA_STYLE('',(#201));
#203=SURFACE_STYLE_FILL_AREA(#202);
#204=SURFACE_SIDE_STYLE('',(#203));
#205=SURFACE_STYLE_USAGE(.BOTH.,#204);
#206=PRESENTATION_STYLE_ASSIGNMENT((#205));
#207=STYLED_ITEM('',(#206),#61);
ENDSEC;
END-ISO-10303-21;
",
        );
        let scene = parse_step(&text, "file").expect("reads");
        assert_eq!(scene.materials.len(), 1);
        let c = scene.materials[0].base_color;
        assert!((c[0] - 1.0).abs() < 1e-6 && (c[1] - 0.214).abs() < 0.01 && c[2] == 0.0);
        assert!(scene.meshes[0]
            .face_materials
            .as_ref()
            .unwrap()
            .iter()
            .all(|m| *m == 0));
    }

    #[test]
    fn a_spline_patch_follows_its_curve() {
        // A quarter cylinder as a degree-2 rational spline in u, straight in v.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let text = format!(
            "{HEAD}#300=CARTESIAN_POINT('',(10.,0.,0.));
#301=CARTESIAN_POINT('',(10.,10.,0.));
#302=CARTESIAN_POINT('',(0.,10.,0.));
#303=CARTESIAN_POINT('',(10.,0.,20.));
#304=CARTESIAN_POINT('',(10.,10.,20.));
#305=CARTESIAN_POINT('',(0.,10.,20.));
#306=( BOUNDED_SURFACE() B_SPLINE_SURFACE(2,1,((#300,#303),(#301,#304),(#302,#305)),.UNSPECIFIED.,.F.,.F.,.F.) B_SPLINE_SURFACE_WITH_KNOTS((3,3),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.) GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(((1.,1.),({w},{w}),(1.,1.))) REPRESENTATION_ITEM('') SURFACE() );
#310=VERTEX_POINT('',#300);
#311=VERTEX_POINT('',#302);
#312=VERTEX_POINT('',#305);
#313=VERTEX_POINT('',#303);
#320=EDGE_CURVE('',#310,#311,#20,.T.);
#321=CARTESIAN_POINT('',(0.,10.,0.));
#322=LINE('',#321,#26);
#323=EDGE_CURVE('',#311,#312,#322,.T.);
#324=EDGE_CURVE('',#313,#312,#21,.T.);
#325=EDGE_CURVE('',#310,#313,#27,.T.);
#330=EDGE_LOOP('',(#331,#332,#333,#334));
#331=ORIENTED_EDGE('',*,*,#320,.T.);
#332=ORIENTED_EDGE('',*,*,#323,.T.);
#333=ORIENTED_EDGE('',*,*,#324,.F.);
#334=ORIENTED_EDGE('',*,*,#325,.F.);
#335=FACE_OUTER_BOUND('',#330,.T.);
#336=ADVANCED_FACE('',(#335),#306,.T.);
#340=OPEN_SHELL('',(#336));
#341=SHELL_BASED_SURFACE_MODEL('',(#340));
#342=MANIFOLD_SURFACE_SHAPE_REPRESENTATION('Quarter',(#341),#9);
{TAIL}"
        );
        let scene = parse_step(&text, "file").expect("reads");
        let mesh = &scene.meshes[0];
        assert!(mesh.faces.len() > 8, "curved, so more than two triangles");
        for p in &mesh.positions {
            let r = p.x.hypot(p.y);
            assert!((r - 10.0).abs() < 0.05, "on the cylinder: {p:?}");
        }
        let area: f64 = mesh
            .faces
            .iter()
            .map(|f| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[f[k] as usize].as_dvec3());
                (b - a).cross(c - a).length() / 2.0
            })
            .sum();
        let exact = PI / 2.0 * 10.0 * 20.0;
        assert!(
            (area - exact).abs() < exact * 0.01,
            "area {area} against {exact}"
        );
    }

    #[test]
    fn a_whole_sphere_bounded_by_its_seam_is_whole() {
        // The seam from pole to pole, used both ways: the whole sphere.
        let text = format!(
            "{HEAD}#10=SPHERICAL_SURFACE('',#4,10.);
#400=CARTESIAN_POINT('',(0.,0.,-10.));
#401=CARTESIAN_POINT('',(0.,0.,10.));
#402=VERTEX_POINT('',#400);
#403=VERTEX_POINT('',#401);
#404=DIRECTION('',(0.,-1.,0.));
#405=AXIS2_PLACEMENT_3D('',#1,#404,#3);
#406=CIRCLE('',#405,10.);
#407=EDGE_CURVE('',#402,#403,#406,.T.);
#408=ORIENTED_EDGE('',*,*,#407,.T.);
#409=ORIENTED_EDGE('',*,*,#407,.F.);
#410=EDGE_LOOP('',(#408,#409));
#411=FACE_OUTER_BOUND('',#410,.T.);
#412=ADVANCED_FACE('',(#411),#10,.T.);
#413=CLOSED_SHELL('',(#412));
#414=MANIFOLD_SOLID_BREP('Ball',#413);
#415=ADVANCED_BREP_SHAPE_REPRESENTATION('Ball',(#414),#9);
{TAIL}"
        );
        let scene = parse_step(&text, "file").expect("reads");
        let mesh = &scene.meshes[0];
        let v = volume(mesh).abs();
        let exact = 4.0 / 3.0 * PI * 1000.0;
        assert!(
            (v - exact).abs() < exact * 0.01,
            "volume {v} against {exact}"
        );
        assert!(watertight(mesh));
    }

    #[test]
    fn a_read_says_how_far_it_got_and_stops_when_asked() {
        let progress = ReadProgress::default();
        parse_step_with(&cylinder(), "file", &progress).expect("reads");
        let (stage, fraction) = progress.get();
        assert_eq!(stage, ReadStage::Shaping);
        assert!(fraction.is_some());
        let stopped = ReadProgress::default();
        stopped.cancel();
        assert!(matches!(
            parse_step_with(&cylinder(), "file", &stopped),
            Err(StepError::Cancelled)
        ));
    }

    #[test]
    fn kreates_own_step_comes_back() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let text = crate::step::write_step("t", &[("Box".into(), &cube, DAffine3::IDENTITY)], 1.0);
        let scene = parse_step(&text, "t").expect("reads");
        assert_eq!(scene.source.unit_scale, 1.0);
        let mesh = &scene.meshes[0];
        assert_eq!(mesh.positions.len(), 8, "corners shared");
        assert_eq!(mesh.faces.len(), 12);
        assert!(watertight(mesh));
        assert!((volume(mesh) - 8.0).abs() < 1e-6);
        assert_eq!(scene.nodes[0].name, "Box");
    }

    #[test]
    fn strings_decode_and_a_bad_entity_is_stepped_over() {
        assert_eq!(decode(b"caf\\X\\E9"), "caf\u{e9}");
        assert_eq!(decode(b"\\X2\\D55CAE00\\X0\\"), "\u{d55c}\u{ae00}");
        let text = cylinder().replace(
            "#71=PRODUCT_CONTEXT('',#70,'');",
            "#71=PRODUCT_CONTEXT('',#70,'';\n#999=JUNK(((;",
        );
        let scene = parse_step(&text, "file").expect("reads past it");
        assert_eq!(scene.meshes.len(), 1);
        assert!(matches!(
            parse_step("solid x", "f"),
            Err(StepError::NotStep)
        ));
    }
}

/// Reads the files `KREATE_STEP_FILES` names (separated by `;`) and reports
/// what came of each: for trying the reader on real CAD output.
#[cfg(test)]
#[test]
#[ignore]
fn read_real_files() {
    let Ok(list) = std::env::var("KREATE_STEP_FILES") else {
        return;
    };
    for path in list.split(';').filter(|p| !p.is_empty()) {
        let t = std::time::Instant::now();
        let bytes = std::fs::read(path).expect("readable");
        let text = String::from_utf8_lossy(&bytes);
        match parse_step(&text, "f") {
            Ok(scene) => {
                let tris: usize = scene.meshes.iter().map(|m| m.faces.len()).sum();
                eprintln!(
                    "{path}: {:.1} MB, {} ms, {} meshes, {} nodes, {} tris, {} materials, unit {}, notes {:?}",
                    bytes.len() as f64 / 1e6,
                    t.elapsed().as_millis(),
                    scene.meshes.len(),
                    scene.nodes.len(),
                    tris,
                    scene.materials.len(),
                    scene.source.unit_scale,
                    scene.source.notes
                );
            }
            Err(e) => eprintln!("{path}: error {e}"),
        }
    }
}
