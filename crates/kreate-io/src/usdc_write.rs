//! USD's binary layer, `.usdc`, written from a [`Layer`].
//!
//! The inverse of [`crate::usdc`]: the same six sections — tokens, strings,
//! fields, field sets, paths and specs — after the values they point into,
//! and a table of contents at the end. Version 0.8.0, which every USD since
//! 20.11 reads.
//!
//! The writer favours being plainly right over being small. Values are
//! stored where their encodings point rather than packed into them, except
//! the few kinds USD expects packed (a specifier, a variability, a bool);
//! arrays are stored whole, not integer-coded. The sections' own integer
//! lists are integer-coded, as the format requires, and every compressed
//! block is a single run of literals — valid LZ4 that any reader expands,
//! at the size it started.
//!
//! What it writes is what [`crate::usd`] makes: prims with their types,
//! API schemas and children; attributes with their values, time samples,
//! interpolation, element size, variability and connections; relationships
//! with their targets; and the stage's own metadata.

use std::collections::HashMap;

use crate::usd_read::{Layer, Prim, Property, Specifier, Val};

// Crate's type numbers (see `crate::usdc`).
const BOOL: u8 = 1;
const INT: u8 = 3;
const INT64: u8 = 5;
const HALF: u8 = 7;
const FLOAT: u8 = 8;
const DOUBLE: u8 = 9;
const STRING: u8 = 10;
const TOKEN: u8 = 11;
const ASSET: u8 = 12;
const MATRIX2D: u8 = 13;
const MATRIX3D: u8 = 14;
const MATRIX4D: u8 = 15;
const QUATD: u8 = 16;
const QUATF: u8 = 17;
const QUATH: u8 = 18;
const VEC2D: u8 = 19;
const TOKEN_LIST_OP: u8 = 32;
const PATH_LIST_OP: u8 = 34;
const PATH_VECTOR: u8 = 40;
const TOKEN_VECTOR: u8 = 41;
const SPECIFIER: u8 = 42;
const VARIABILITY: u8 = 44;
const TIME_SAMPLES: u8 = 46;
const DOUBLE_VECTOR: u8 = 48;

/// A value's 64-bit encoding.
fn rep(kind: u8, array: bool, inlined: bool, payload: u64) -> u64 {
    let mut r = ((kind as u64) << 48) | (payload & ((1 << 48) - 1));
    if array {
        r |= 1 << 63;
    }
    if inlined {
        r |= 1 << 62;
    }
    r
}

/// LZ4 as a single run of literals, in USD's wrapper (no chunks).
fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 255 + 16);
    out.push(0); // one block
    let n = data.len();
    if n < 15 {
        out.push((n as u8) << 4);
    } else {
        out.push(0xF0);
        let mut rest = n - 15;
        while rest >= 255 {
            out.push(255);
            rest -= 255;
        }
        out.push(rest as u8);
    }
    out.extend_from_slice(data);
    out
}

/// Integers as USD codes them (32-bit): a common difference, two bits a
/// number saying how each differs from the last, then those differences.
fn encode_ints(values: &[i64]) -> Vec<u8> {
    let mut deltas = Vec::with_capacity(values.len());
    let mut prev = 0i64;
    for v in values {
        deltas.push(v.wrapping_sub(prev) as i32);
        prev = *v;
    }
    // The most frequent difference is the common one.
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for d in &deltas {
        *counts.entry(*d).or_default() += 1;
    }
    let common = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(d, _)| *d)
        .unwrap_or(0);
    let mut out = common.to_le_bytes().to_vec();
    let mut codes = vec![0u8; (deltas.len() * 2).div_ceil(8)];
    let mut rest = Vec::new();
    for (i, d) in deltas.iter().enumerate() {
        let code = if *d == common {
            0
        } else if i8::try_from(*d).is_ok() {
            rest.push(*d as i8 as u8);
            1
        } else if let Ok(m) = i16::try_from(*d) {
            rest.extend(m.to_le_bytes());
            2
        } else {
            rest.extend(d.to_le_bytes());
            3
        };
        codes[i / 4] |= code << ((i % 4) * 2);
    }
    out.extend(codes);
    out.extend(rest);
    out
}

/// Builds the file.
struct Writer {
    body: Vec<u8>,
    tokens: Vec<String>,
    token_of: HashMap<String, u32>,
    strings: Vec<u32>,
    string_of: HashMap<String, u32>,
    /// Paths in the order they are numbered, and their numbers.
    paths: Vec<String>,
    path_of: HashMap<String, u32>,
    fields: Vec<(u32, u64)>,
    field_of: HashMap<(u32, u64), u32>,
    fieldsets: Vec<i64>,
    specs: Vec<(u32, u32, u32)>,
}

impl Writer {
    fn token(&mut self, t: &str) -> u32 {
        if let Some(i) = self.token_of.get(t) {
            return *i;
        }
        let i = self.tokens.len() as u32;
        self.tokens.push(t.to_string());
        self.token_of.insert(t.to_string(), i);
        i
    }

    fn string(&mut self, s: &str) -> u32 {
        if let Some(i) = self.string_of.get(s) {
            return *i;
        }
        let t = self.token(s);
        let i = self.strings.len() as u32;
        self.strings.push(t);
        self.string_of.insert(s.to_string(), i);
        i
    }

    fn path(&self, p: &str) -> u32 {
        self.path_of.get(p).copied().unwrap_or(0)
    }

    /// Where the next value goes, 8-byte aligned.
    fn here(&mut self) -> u64 {
        while !self.body.len().is_multiple_of(8) {
            self.body.push(0);
        }
        self.body.len() as u64
    }

    fn u32(&mut self, v: u32) {
        self.body.extend(v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.body.extend(v.to_le_bytes());
    }

    fn field(&mut self, name: &str, value: u64) -> u32 {
        let name = self.token(name);
        if let Some(i) = self.field_of.get(&(name, value)) {
            return *i;
        }
        let i = self.fields.len() as u32;
        self.fields.push((name, value));
        self.field_of.insert((name, value), i);
        i
    }

    fn token_value(&mut self, t: &str) -> u64 {
        let i = self.token(t);
        rep(TOKEN, false, true, i as u64)
    }

    fn token_vector(&mut self, list: &[String]) -> u64 {
        let ids: Vec<u32> = list.iter().map(|t| self.token(t)).collect();
        let at = self.here();
        self.u64(ids.len() as u64);
        for i in ids {
            self.u32(i);
        }
        rep(TOKEN_VECTOR, false, false, at)
    }

    fn token_list_op(&mut self, list: &[String]) -> u64 {
        let ids: Vec<u32> = list.iter().map(|t| self.token(t)).collect();
        let at = self.here();
        // Prepended items: what `prepend apiSchemas = [...]` says.
        self.body.push(1 << 5);
        self.u64(ids.len() as u64);
        for i in ids {
            self.u32(i);
        }
        rep(TOKEN_LIST_OP, false, false, at)
    }

    fn path_list_op(&mut self, list: &[String]) -> u64 {
        let ids: Vec<u32> = list.iter().map(|p| self.path(p)).collect();
        let at = self.here();
        // Explicit: the list is the list.
        self.body.push((1 << 0) | (1 << 1));
        self.u64(ids.len() as u64);
        for i in ids {
            self.u32(i);
        }
        rep(PATH_LIST_OP, false, false, at)
    }

    fn path_vector(&mut self, list: &[String]) -> u64 {
        let ids: Vec<u32> = list.iter().map(|p| self.path(p)).collect();
        let at = self.here();
        self.u64(ids.len() as u64);
        for i in ids {
            self.u32(i);
        }
        rep(PATH_VECTOR, false, false, at)
    }

    fn double_value(&mut self, v: f64) -> u64 {
        let at = self.here();
        self.body.extend(v.to_le_bytes());
        rep(DOUBLE, false, false, at)
    }

    /// An attribute's value, by its type's name.
    fn value(&mut self, type_name: &str, v: &Val) -> Option<u64> {
        let array = type_name.ends_with("[]");
        let base = type_name.trim_end_matches("[]");
        match v {
            Val::Texts(t) => {
                let kind = match base {
                    "string" => STRING,
                    "asset" => ASSET,
                    _ => TOKEN,
                };
                let ids: Vec<u32> = t
                    .iter()
                    .map(|s| {
                        if kind == STRING {
                            self.string(s)
                        } else {
                            self.token(s)
                        }
                    })
                    .collect();
                if array {
                    let at = self.here();
                    self.u64(ids.len() as u64);
                    for i in ids {
                        self.u32(i);
                    }
                    Some(rep(kind, true, false, at))
                } else {
                    Some(rep(kind, false, true, *ids.first()? as u64))
                }
            }
            Val::Nums(n) => {
                let (kind, width, components) = shape_of(base)?;
                if kind == BOOL && !array {
                    return Some(rep(
                        BOOL,
                        false,
                        true,
                        (n.first().copied().unwrap_or(0.0) != 0.0) as u64,
                    ));
                }
                // Quaternions are written real part last, as the file has
                // them; the text (and so the model) has it first.
                let mut n = n.clone();
                if matches!(kind, QUATD | QUATF | QUATH) {
                    for q in n.as_chunks_mut::<4>().0 {
                        q.rotate_left(1);
                    }
                }
                let at = self.here();
                if array {
                    self.u64((n.len() / components.max(1)) as u64);
                }
                for x in &n {
                    match (kind, width) {
                        (BOOL, _) => self.body.push((*x != 0.0) as u8),
                        (INT, _) => self.body.extend((*x as i32).to_le_bytes()),
                        (INT64, _) => self.body.extend((*x as i64).to_le_bytes()),
                        (_, 2) => self.body.extend(half::f16::from_f64(*x).to_le_bytes()),
                        (_, 4) => self.body.extend((*x as f32).to_le_bytes()),
                        _ => self.body.extend(x.to_le_bytes()),
                    }
                }
                Some(rep(kind, array, false, at))
            }
        }
    }

    /// Time samples: the times' data, then their encoding; the values'
    /// encodings after a jump — the layout `crate::usdc` reads.
    fn samples(&mut self, type_name: &str, samples: &[(f64, Val)]) -> Option<u64> {
        let reps: Vec<u64> = samples
            .iter()
            .map(|(_, v)| self.value(type_name, v))
            .collect::<Option<_>>()?;
        let at = self.here();
        // Jump over the times' own data to their encoding.
        let jump_at = self.body.len();
        self.u64(0);
        let times_at = self.body.len() as u64;
        self.u64(samples.len() as u64);
        for (t, _) in samples {
            self.body.extend(t.to_le_bytes());
        }
        let jump = self.body.len() - jump_at;
        self.body[jump_at..jump_at + 8].copy_from_slice(&(jump as i64).to_le_bytes());
        self.u64(rep(DOUBLE_VECTOR, false, false, times_at));
        // A jump of its own size, then the values.
        self.u64(8);
        self.u64(reps.len() as u64);
        for r in reps {
            self.u64(r);
        }
        Some(rep(TIME_SAMPLES, false, false, at))
    }

    fn spec(&mut self, path: &str, kind: u32, fields: Vec<u32>) {
        let set = self.fieldsets.len() as u32;
        self.fieldsets.extend(fields.into_iter().map(i64::from));
        self.fieldsets.push(-1);
        let path = self.path(path);
        self.specs.push((path, set, kind));
    }
}

/// A type's crate kind, its numbers' width in bytes and how many make one.
fn shape_of(base: &str) -> Option<(u8, usize, usize)> {
    let vec = |n: usize, suffix: char| -> Option<(u8, usize, usize)> {
        let (offset, width) = match suffix {
            'd' => (0, 8),
            'f' => (1, 4),
            'h' => (2, 2),
            'i' => (3, 4),
            _ => return None,
        };
        Some((VEC2D + (n as u8 - 2) * 4 + offset, width, n))
    };
    Some(match base {
        "bool" => (BOOL, 1, 1),
        "int" | "uint" => (INT, 4, 1),
        "int64" | "uint64" => (INT64, 8, 1),
        "half" => (HALF, 2, 1),
        "float" => (FLOAT, 4, 1),
        "double" | "timecode" => (DOUBLE, 8, 1),
        "matrix2d" => (MATRIX2D, 8, 4),
        "matrix3d" => (MATRIX3D, 8, 9),
        "matrix4d" | "frame4d" => (MATRIX4D, 8, 16),
        "quatd" => (QUATD, 8, 4),
        "quatf" => (QUATF, 4, 4),
        "quath" => (QUATH, 2, 4),
        "point3f" | "normal3f" | "vector3f" | "color3f" => vec(3, 'f')?,
        "point3d" | "normal3d" | "vector3d" | "color3d" => vec(3, 'd')?,
        "point3h" | "normal3h" | "vector3h" | "color3h" => vec(3, 'h')?,
        "color4f" => vec(4, 'f')?,
        "color4d" => vec(4, 'd')?,
        "texCoord2f" => vec(2, 'f')?,
        "texCoord2d" => vec(2, 'd')?,
        "texCoord2h" => vec(2, 'h')?,
        "texCoord3f" => vec(3, 'f')?,
        t if t.len() == 6 && t.starts_with("float") => vec((t.as_bytes()[5] - b'0') as usize, 'f')?,
        t if t.len() == 7 && t.starts_with("double") => {
            vec((t.as_bytes()[6] - b'0') as usize, 'd')?
        }
        t if t.len() == 5 && t.starts_with("half") => vec((t.as_bytes()[4] - b'0') as usize, 'h')?,
        t if t.len() == 4 && t.starts_with("int") => vec((t.as_bytes()[3] - b'0') as usize, 'i')?,
        _ => return None,
    })
}

/// The path tree, numbered depth first: each entry its path, its parent's
/// number, its own element and whether it is a property.
fn number_paths(layer: &Layer) -> Vec<(String, Option<usize>, String, bool)> {
    fn walk(
        prim: &Prim,
        at: &str,
        parent: usize,
        out: &mut Vec<(String, Option<usize>, String, bool)>,
    ) {
        let path = if at == "/" {
            format!("/{}", prim.name)
        } else {
            format!("{at}/{}", prim.name)
        };
        let me = out.len();
        out.push((path.clone(), Some(parent), prim.name.clone(), false));
        for p in &prim.properties {
            out.push((format!("{path}.{}", p.name), Some(me), p.name.clone(), true));
        }
        for c in &prim.children {
            walk(c, &path, me, out);
        }
    }
    let mut out = vec![("/".to_string(), None, String::new(), false)];
    for p in &layer.prims {
        walk(p, "/", 0, &mut out);
    }
    out
}

/// A layer as a `.usdc` file.
pub fn write_usdc(layer: &Layer) -> Vec<u8> {
    let mut w = Writer {
        body: vec![0; 88],
        tokens: Vec::new(),
        token_of: HashMap::new(),
        strings: Vec::new(),
        string_of: HashMap::new(),
        paths: Vec::new(),
        path_of: HashMap::new(),
        fields: Vec::new(),
        field_of: HashMap::new(),
        fieldsets: Vec::new(),
        specs: Vec::new(),
    };
    w.token("");
    let tree = number_paths(layer);
    for (i, (path, ..)) in tree.iter().enumerate() {
        w.paths.push(path.clone());
        w.path_of.insert(path.clone(), i as u32);
    }

    // The layer itself.
    let mut fields = Vec::new();
    if let Some(d) = &layer.default_prim {
        let v = w.token_value(d);
        fields.push(w.field("defaultPrim", v));
    }
    if let Some(a) = &layer.up_axis {
        let v = w.token_value(a);
        fields.push(w.field("upAxis", v));
    }
    for (name, value) in [
        ("metersPerUnit", layer.meters_per_unit),
        ("timeCodesPerSecond", layer.time_codes_per_second),
        ("framesPerSecond", layer.time_codes_per_second),
        ("startTimeCode", layer.start_time_code),
        ("endTimeCode", layer.end_time_code),
    ] {
        if let Some(v) = value {
            let v = w.double_value(v);
            fields.push(w.field(name, v));
        }
    }
    let roots: Vec<String> = layer.prims.iter().map(|p| p.name.clone()).collect();
    let v = w.token_vector(&roots);
    fields.push(w.field("primChildren", v));
    w.spec("/", 7, fields);

    fn prim_specs(w: &mut Writer, prim: &Prim, path: &str) {
        let mut fields = Vec::new();
        let spec = match prim.specifier {
            Specifier::Def => 0,
            Specifier::Over => 1,
            Specifier::Class => 2,
        };
        let f = w.field("specifier", rep(SPECIFIER, false, true, spec));
        fields.push(f);
        if !prim.type_name.is_empty() {
            let v = w.token_value(&prim.type_name);
            fields.push(w.field("typeName", v));
        }
        if !prim.api_schemas.is_empty() {
            let v = w.token_list_op(&prim.api_schemas);
            fields.push(w.field("apiSchemas", v));
        }
        if let Some(kind) = &prim.kind {
            let v = w.token_value(kind);
            fields.push(w.field("kind", v));
        }
        if let Some(active) = prim.active {
            fields.push(w.field("active", rep(BOOL, false, true, active as u64)));
        }
        let names: Vec<String> = prim.properties.iter().map(|p| p.name.clone()).collect();
        if !names.is_empty() {
            let v = w.token_vector(&names);
            fields.push(w.field("properties", v));
        }
        let children: Vec<String> = prim.children.iter().map(|c| c.name.clone()).collect();
        if !children.is_empty() {
            let v = w.token_vector(&children);
            fields.push(w.field("primChildren", v));
        }
        w.spec(path, 6, fields);
        for p in &prim.properties {
            property_spec(w, p, &format!("{path}.{}", p.name));
        }
        for c in &prim.children {
            prim_specs(w, c, &format!("{path}/{}", c.name));
        }
    }

    fn property_spec(w: &mut Writer, p: &Property, path: &str) {
        let mut fields = Vec::new();
        // Targets as the path table has them: `<Anim>` from the prim that
        // owns the property is `/…/Skel/Anim`.
        let owner = path.rsplit_once('.').map_or(path, |(prim, _)| prim);
        let resolved: Vec<String> = p
            .targets
            .iter()
            .map(|t| crate::usd_read::absolute(t, owner))
            .collect();
        let p = &Property {
            targets: resolved,
            ..p.clone()
        };
        if p.type_name == "rel" {
            let v = w.path_list_op(&p.targets);
            fields.push(w.field("targetPaths", v));
            let v = w.path_vector(&p.targets);
            fields.push(w.field("targetChildren", v));
            // A relationship is the same at every time.
            fields.push(w.field("variability", rep(VARIABILITY, false, true, 1)));
            w.spec(path, 8, fields);
            return;
        }
        let v = w.token_value(&p.type_name);
        fields.push(w.field("typeName", v));
        if p.uniform {
            fields.push(w.field("variability", rep(VARIABILITY, false, true, 1)));
        }
        if let Some(v) = &p.value {
            // A value that only exists as samples is not written twice.
            if p.samples.is_empty() {
                if let Some(r) = w.value(&p.type_name, v) {
                    fields.push(w.field("default", r));
                }
            }
        }
        if !p.samples.is_empty() {
            if let Some(r) = w.samples(&p.type_name, &p.samples) {
                fields.push(w.field("timeSamples", r));
            }
        }
        if let Some(i) = &p.interpolation {
            let v = w.token_value(i);
            fields.push(w.field("interpolation", v));
        }
        if let Some(e) = p.element_size {
            fields.push(w.field("elementSize", rep(INT, false, true, e as u32 as u64)));
        }
        if !p.targets.is_empty() {
            let v = w.path_list_op(&p.targets);
            fields.push(w.field("connectionPaths", v));
            let v = w.path_vector(&p.targets);
            fields.push(w.field("connectionChildren", v));
        }
        w.spec(path, 1, fields);
    }

    for p in &layer.prims {
        prim_specs(&mut w, p, &format!("/{}", p.name));
    }

    // The tokens every path element names, before the tokens are written.
    let elements: Vec<i64> = tree
        .iter()
        .map(|(_, _, e, prop)| {
            let t = w.token(e) as i64;
            if *prop {
                -t
            } else {
                t
            }
        })
        .collect();

    // --- the sections ------------------------------------------------------
    let mut sections: Vec<(&str, u64, u64)> = Vec::new();
    let begin = |w: &mut Writer| w.here();
    let compressed_ints = |w: &mut Writer, values: &[i64]| {
        let packed = compress(&encode_ints(values));
        w.u64(packed.len() as u64);
        w.body.extend(packed);
    };

    let start = begin(&mut w);
    let mut joined = Vec::new();
    for t in &w.tokens {
        joined.extend(t.as_bytes());
        joined.push(0);
    }
    let packed = compress(&joined);
    w.u64(w.tokens.len() as u64);
    w.u64(joined.len() as u64);
    w.u64(packed.len() as u64);
    w.body.extend(packed);
    sections.push(("TOKENS", start, w.body.len() as u64 - start));

    let start = begin(&mut w);
    w.u64(w.strings.len() as u64);
    for s in w.strings.clone() {
        w.u32(s);
    }
    sections.push(("STRINGS", start, w.body.len() as u64 - start));

    let start = begin(&mut w);
    w.u64(w.fields.len() as u64);
    let names: Vec<i64> = w.fields.iter().map(|(t, _)| *t as i64).collect();
    compressed_ints(&mut w, &names);
    let reps: Vec<u8> = w.fields.iter().flat_map(|(_, r)| r.to_le_bytes()).collect();
    let packed = compress(&reps);
    w.u64(packed.len() as u64);
    w.body.extend(packed);
    sections.push(("FIELDS", start, w.body.len() as u64 - start));

    let start = begin(&mut w);
    w.u64(w.fieldsets.len() as u64);
    let sets = w.fieldsets.clone();
    compressed_ints(&mut w, &sets);
    sections.push(("FIELDSETS", start, w.body.len() as u64 - start));

    // Paths: depth first, each with how far to its next sibling.
    let start = begin(&mut w);
    let n = tree.len();
    let mut subtree = vec![1usize; n];
    for i in (0..n).rev() {
        if let Some(p) = tree[i].1 {
            subtree[p] += subtree[i];
        }
    }
    let jumps: Vec<i64> = (0..n)
        .map(|i| {
            let has_child = i + 1 < n && tree[i + 1].1 == Some(i);
            let next = i + subtree[i];
            let has_sibling = next < n && tree[next].1 == tree[i].1 && tree[i].1.is_some();
            match (has_child, has_sibling) {
                (true, true) => subtree[i] as i64,
                (true, false) => -1,
                (false, true) => 0,
                (false, false) => -2,
            }
        })
        .collect();
    w.u64(n as u64);
    w.u64(n as u64);
    let indices: Vec<i64> = (0..n as i64).collect();
    compressed_ints(&mut w, &indices);
    compressed_ints(&mut w, &elements);
    compressed_ints(&mut w, &jumps);
    sections.push(("PATHS", start, w.body.len() as u64 - start));

    let start = begin(&mut w);
    w.u64(w.specs.len() as u64);
    let (paths, sets, kinds): (Vec<i64>, Vec<i64>, Vec<i64>) = w.specs.iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut a, mut b, mut c), (p, s, k)| {
            a.push(*p as i64);
            b.push(*s as i64);
            c.push(*k as i64);
            (a, b, c)
        },
    );
    compressed_ints(&mut w, &paths);
    compressed_ints(&mut w, &sets);
    compressed_ints(&mut w, &kinds);
    sections.push(("SPECS", start, w.body.len() as u64 - start));

    // The table of contents, and the header that points at it.
    let toc = begin(&mut w);
    w.u64(sections.len() as u64);
    for (name, start, size) in &sections {
        let mut label = [0u8; 16];
        label[..name.len()].copy_from_slice(name.as_bytes());
        w.body.extend(label);
        w.body.extend((*start as i64).to_le_bytes());
        w.body.extend((*size as i64).to_le_bytes());
    }
    w.body[..8].copy_from_slice(b"PXR-USDC");
    w.body[8..11].copy_from_slice(&[0, 8, 0]);
    w.body[16..24].copy_from_slice(&toc.to_le_bytes());
    w.body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_come_back_as_they_went() {
        let values = vec![0, 1, 2, 3, 300, -5, 70_000, -1, -1, 4];
        let coded = encode_ints(&values);
        // Read back through the reader's decoder, via a compressed block.
        let packed = compress(&coded);
        let mut file = (packed.len() as u64).to_le_bytes().to_vec();
        file.extend(packed);
        let back = crate::usdc::decode_for_test(&file, values.len()).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn a_stage_written_binary_reads_back_as_its_text() {
        let text = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    timeCodesPerSecond = 24
)
def Xform "World"
{
    matrix4d xformOp:transform.timeSamples = {
        0: ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) ),
        24: ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (2, 0, 0, 1) ),
    }
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Mesh "Tri" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
            interpolation = "faceVarying"
        )
        int[] primvars:skel:jointIndices = [0, 0, 0] (
            elementSize = 1
            interpolation = "vertex"
        )
        rel material:binding = </World/Looks/Red>
    }
    def Scope "Looks"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Looks/Red/S.outputs:surface>
            def Shader "S"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 0, 0)
                asset inputs:file = @tex.png@
                quatf[] rot = [(1, 0, 0, 0)]
                token outputs:surface
            }
        }
    }
}
"#;
        let layer = crate::usda::parse_usda(text).unwrap();
        let bytes = write_usdc(&layer);
        let back = crate::usdc::parse_usdc(&bytes).unwrap();
        assert_eq!(back.up_axis, layer.up_axis);
        assert_eq!(back.meters_per_unit, layer.meters_per_unit);
        assert_eq!(back.default_prim, layer.default_prim);
        // Prim by prim, property by property.
        fn same(a: &Prim, b: &Prim) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.type_name, b.type_name);
            assert_eq!(a.api_schemas, b.api_schemas, "{}", a.name);
            for p in &a.properties {
                let q = b
                    .property(&p.name)
                    .unwrap_or_else(|| panic!("{}.{}", a.name, p.name));
                assert_eq!(p.value, q.value, "{}.{}", a.name, p.name);
                assert_eq!(p.samples, q.samples, "{}.{}", a.name, p.name);
                assert_eq!(p.targets, q.targets, "{}.{}", a.name, p.name);
                assert_eq!(p.interpolation, q.interpolation, "{}.{}", a.name, p.name);
                assert_eq!(p.element_size, q.element_size, "{}.{}", a.name, p.name);
            }
            assert_eq!(a.children.len(), b.children.len());
            for (x, y) in a.children.iter().zip(&b.children) {
                same(x, y);
            }
        }
        same(&layer.prims[0], &back.prims[0]);
    }
}
