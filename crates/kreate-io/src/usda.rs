//! USD's text layer, `.usda`, read into a [`Layer`].
//!
//! The grammar as files are written: a header of layer metadata, then prims —
//! `def`, `over` or `class`, a type, a name, metadata in parentheses and a
//! body of properties, child prims and variant sets. A value is whatever the
//! property's type makes it, and here it is kept as numbers or as text,
//! flattened: a `point3f[]` is three numbers a point, a `matrix4d` sixteen,
//! a `token[]` the tokens. What a property means is the stage's business
//! ([`crate::usd_read`]), not the parser's.
//!
//! Animated values (`.timeSamples`) are read as their first sample: a stage
//! is opened at its start. Dictionaries (`customData`, `assetInfo`) and
//! anything else a reader of geometry has no use for are parsed and dropped.

use crate::usd_read::{Arc, Layer, Prim, Property, Specifier, Val};

/// How deep prims and values may nest.
const MAX_DEPTH: usize = 128;

/// Why a text layer could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsdaError(pub String);

impl std::fmt::Display for UsdaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "USDA: {}", self.0)
    }
}

impl std::error::Error for UsdaError {}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Punct(char),
    Word(String),
    Str(String),
    Num(f64),
    Path(String),
    Asset(String),
}

fn lex(text: &str) -> Result<Vec<(Tok, usize)>, UsdaError> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < b.len() {
        let c = b[i];
        match c {
            b'\n' => {
                line += 1;
                i += 1;
            }
            c if c.is_ascii_whitespace() => i += 1,
            b'#' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'=' | b',' | b':' | b';' => {
                out.push((Tok::Punct(c as char), line));
                i += 1;
            }
            b'"' | b'\'' => {
                let triple = b.len() >= i + 3 && b[i + 1] == c && b[i + 2] == c;
                let mut s = Vec::new();
                if triple {
                    i += 3;
                    while i < b.len()
                        && !(b[i] == c && b.get(i + 1) == Some(&c) && b.get(i + 2) == Some(&c))
                    {
                        if b[i] == b'\n' {
                            line += 1;
                        }
                        s.push(b[i]);
                        i += 1;
                    }
                    i += 3;
                } else {
                    i += 1;
                    while i < b.len() && b[i] != c {
                        if b[i] == b'\\' && i + 1 < b.len() {
                            i += 1;
                            s.push(match b[i] {
                                b'n' => b'\n',
                                b't' => b'\t',
                                other => other,
                            });
                        } else {
                            s.push(b[i]);
                        }
                        i += 1;
                    }
                    i += 1;
                }
                out.push((Tok::Str(String::from_utf8_lossy(&s).into_owned()), line));
            }
            b'<' => {
                let end = text[i..]
                    .find('>')
                    .ok_or_else(|| UsdaError(format!("line {line}: a path that never ends")))?;
                out.push((Tok::Path(text[i + 1..i + end].to_string()), line));
                i += end + 1;
            }
            b'@' => {
                // `@path@`, or `@@@path@@@` for a path with an @ in it.
                let (open, close) = if text[i..].starts_with("@@@") {
                    (3, "@@@")
                } else {
                    (1, "@")
                };
                let body = &text[i + open..];
                let end = body.find(close).ok_or_else(|| {
                    UsdaError(format!("line {line}: an asset path that never ends"))
                })?;
                out.push((Tok::Asset(body[..end].to_string()), line));
                i += open + end + close.len();
            }
            c if c == b'-' || c == b'+' || c == b'.' || c.is_ascii_digit() => {
                let start = i;
                i += 1;
                while i < b.len()
                    && (b[i].is_ascii_alphanumeric() || matches!(b[i], b'.' | b'-' | b'+'))
                {
                    i += 1;
                }
                let word = &text[start..i];
                match word.parse::<f64>() {
                    Ok(n) => out.push((Tok::Num(n), line)),
                    Err(_) => match word {
                        "-inf" => out.push((Tok::Num(f64::NEG_INFINITY), line)),
                        "inf" | "+inf" => out.push((Tok::Num(f64::INFINITY), line)),
                        "nan" => out.push((Tok::Num(f64::NAN), line)),
                        _ => {
                            return Err(UsdaError(format!("line {line}: `{word}` is not a number")))
                        }
                    },
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' || c >= 0x80 => {
                let start = i;
                while i < b.len()
                    && (b[i].is_ascii_alphanumeric()
                        || matches!(b[i], b'_' | b':' | b'.' | b'[' | b']')
                        || b[i] >= 0x80)
                {
                    // `[]` belongs to a type only right after it: `float3[]`.
                    if b[i] == b'[' && b.get(i + 1) != Some(&b']') {
                        break;
                    }
                    if b[i] == b']' && (i == start || b[i - 1] != b'[') {
                        break;
                    }
                    i += 1;
                }
                // A word never ends in `:` — that is the dictionary's colon.
                let mut word = &text[start..i];
                while word.ends_with(':') {
                    word = &word[..word.len() - 1];
                    i -= 1;
                }
                out.push((Tok::Word(word.to_string()), line));
            }
            _ => {
                return Err(UsdaError(format!(
                    "line {line}: `{}` is not something USD writes",
                    c as char
                )))
            }
        }
    }
    Ok(out)
}

/// A value as written, before the stage decides what it is.
#[derive(Clone, Debug, PartialEq)]
enum Raw {
    Num(f64),
    Text(String),
    Asset(String),
    Path(String),
    List(Vec<Raw>),
    /// A dictionary, or a time-sample map, as key and value.
    Dict(Vec<(String, Raw)>),
    None,
}

struct Parser {
    toks: Vec<(Tok, usize)>,
    at: usize,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at).map(|t| &t.0)
    }
    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.at + n).map(|t| &t.0)
    }
    fn line(&self) -> usize {
        self.toks
            .get(self.at.min(self.toks.len().saturating_sub(1)))
            .map(|t| t.1)
            .unwrap_or(0)
    }
    fn err<T>(&self, what: &str) -> Result<T, UsdaError> {
        Err(UsdaError(format!("line {}: {what}", self.line())))
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.at).map(|t| t.0.clone());
        self.at += 1;
        t
    }
    fn is(&self, c: char) -> bool {
        self.peek() == Some(&Tok::Punct(c))
    }
    fn eat(&mut self, c: char) -> bool {
        if self.is(c) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: char) -> Result<(), UsdaError> {
        if self.eat(c) {
            Ok(())
        } else {
            self.err(&format!("`{c}` expected, found {:?}", self.peek()))
        }
    }
    fn word(&mut self) -> Result<String, UsdaError> {
        match self.next() {
            Some(Tok::Word(w)) => Ok(w),
            other => self.err(&format!("a name expected, found {other:?}")),
        }
    }
    fn string(&mut self) -> Result<String, UsdaError> {
        match self.next() {
            Some(Tok::Str(s)) => Ok(s),
            other => self.err(&format!("a quoted name expected, found {other:?}")),
        }
    }
    fn deeper(&mut self) -> Result<(), UsdaError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.err("nested too deep");
        }
        Ok(())
    }

    fn value(&mut self) -> Result<Raw, UsdaError> {
        self.deeper()?;
        let v = match self.next() {
            Some(Tok::Num(n)) => Raw::Num(n),
            Some(Tok::Str(s)) => Raw::Text(s),
            Some(Tok::Asset(a)) => Raw::Asset(a),
            Some(Tok::Path(p)) => Raw::Path(p),
            Some(Tok::Word(w)) => match w.as_str() {
                "None" => Raw::None,
                "true" => Raw::Num(1.0),
                "false" => Raw::Num(0.0),
                _ => Raw::Text(w),
            },
            Some(Tok::Punct(open @ ('(' | '['))) => {
                let close = if open == '(' { ')' } else { ']' };
                let mut items = Vec::new();
                while !self.eat(close) {
                    items.push(self.value()?);
                    if let (Some(Raw::Asset(_)), Some(Tok::Path(p))) = (items.last(), self.peek()) {
                        let p = p.clone();
                        self.at += 1;
                        items.push(Raw::Path(p));
                    }
                    if !self.eat(',') && !self.is(close) {
                        return self.err(&format!("`,` or `{close}` expected"));
                    }
                }
                Raw::List(items)
            }
            Some(Tok::Punct('{')) => {
                let mut items = Vec::new();
                while !self.eat('}') {
                    // A dictionary entry is `type key = value`; a time sample
                    // `time: value`.
                    let key = match self.next() {
                        Some(Tok::Num(n)) => {
                            self.expect(':')?;
                            n.to_string()
                        }
                        Some(Tok::Word(_)) => {
                            let key = match self.next() {
                                Some(Tok::Word(k)) | Some(Tok::Str(k)) => k,
                                other => {
                                    return self.err(&format!("a key expected, found {other:?}"))
                                }
                            };
                            self.expect('=')?;
                            key
                        }
                        Some(Tok::Str(k)) => {
                            self.expect(':')?;
                            k
                        }
                        other => return self.err(&format!("a key expected, found {other:?}")),
                    };
                    let v = self.value()?;
                    items.push((key, v));
                    self.eat(',');
                    self.eat(';');
                }
                Raw::Dict(items)
            }
            other => return self.err(&format!("a value expected, found {other:?}")),
        };
        self.depth -= 1;
        Ok(v)
    }

    /// `( ... )` after a prim, a property or the layer: `key = value`
    /// entries, list-edited ones (`prepend references = ...`) and bare
    /// strings (documentation).
    fn metadata(&mut self) -> Result<Vec<(String, Option<String>, Raw)>, UsdaError> {
        let mut out = Vec::new();
        if !self.eat('(') {
            return Ok(out);
        }
        while !self.eat(')') {
            match self.peek() {
                Some(Tok::Str(_)) => {
                    let doc = self.string()?;
                    out.push(("doc".into(), None, Raw::Text(doc)));
                }
                Some(Tok::Word(_)) => {
                    let mut key = self.word()?;
                    let mut op = None;
                    if matches!(
                        key.as_str(),
                        "prepend" | "append" | "add" | "delete" | "reorder"
                    ) {
                        op = Some(key);
                        key = self.word()?;
                    }
                    if self.eat('=') {
                        let mut v = self.value()?;
                        // `@a.usda@</Prim>`: an asset and the prim in it.
                        if let (Raw::Asset(_), Some(Tok::Path(p))) = (&v, self.peek()) {
                            let p = p.clone();
                            self.at += 1;
                            v = Raw::List(vec![v, Raw::Path(p)]);
                        }
                        out.push((key, op, v));
                    } else {
                        out.push((key, op, Raw::None));
                    }
                }
                _ => return self.err("metadata expected"),
            }
            self.eat(';');
        }
        Ok(out)
    }

    fn prim(&mut self, specifier: Specifier) -> Result<Prim, UsdaError> {
        self.deeper()?;
        let mut prim = Prim {
            specifier,
            ..Prim::default()
        };
        if let Some(Tok::Word(_)) = self.peek() {
            prim.type_name = self.word()?;
        }
        prim.name = self.string()?;
        for (key, _op, value) in self.metadata()? {
            match key.as_str() {
                "references" | "payload" => {
                    let arcs = arcs_of(&value);
                    if key == "references" {
                        prim.references.extend(arcs);
                    } else {
                        prim.payloads.extend(arcs);
                    }
                }
                "inherits" | "specializes" => prim.inherits.extend(paths_of(&value)),
                "variants" => {
                    if let Raw::Dict(items) = value {
                        for (k, v) in items {
                            if let Raw::Text(s) = v {
                                prim.variant_selection.push((k, s));
                            }
                        }
                    }
                }
                "active" => prim.active = Some(!matches!(value, Raw::Num(n) if n == 0.0)),
                "kind" => {
                    if let Raw::Text(k) = value {
                        prim.kind = Some(k);
                    }
                }
                "apiSchemas" => {
                    if let Raw::List(items) = value {
                        prim.api_schemas
                            .extend(items.into_iter().filter_map(|i| match i {
                                Raw::Text(t) => Some(t),
                                _ => None,
                            }));
                    }
                }
                _ => {}
            }
        }
        self.expect('{')?;
        self.body(&mut prim)?;
        self.depth -= 1;
        Ok(prim)
    }

    /// A prim's body, or a variant's: up to and including its `}`.
    fn body(&mut self, prim: &mut Prim) -> Result<(), UsdaError> {
        loop {
            match self.peek() {
                None => return self.err("a prim that never closes"),
                Some(Tok::Punct('}')) => {
                    self.at += 1;
                    return Ok(());
                }
                Some(Tok::Punct(';')) => {
                    self.at += 1;
                }
                Some(Tok::Word(w)) => match w.as_str() {
                    "def" | "over" | "class" => {
                        let spec = match w.as_str() {
                            "def" => Specifier::Def,
                            "over" => Specifier::Over,
                            _ => Specifier::Class,
                        };
                        self.at += 1;
                        let child = self.prim(spec)?;
                        prim.children.push(child);
                    }
                    "variantSet" => {
                        self.at += 1;
                        let set = self.string()?;
                        self.expect('=')?;
                        self.expect('{')?;
                        let mut variants = Vec::new();
                        while !self.eat('}') {
                            let name = self.string()?;
                            let _ = self.metadata()?;
                            self.expect('{')?;
                            let mut body = Prim::default();
                            self.body(&mut body)?;
                            variants.push((name, body));
                        }
                        prim.variant_sets.push((set, variants));
                    }
                    "reorder"
                        if self.peek_at(1) == Some(&Tok::Word("nameChildren".into()))
                            || self.peek_at(1) == Some(&Tok::Word("properties".into())) =>
                    {
                        self.at += 2;
                        self.expect('=')?;
                        let _ = self.value()?;
                    }
                    _ => self.property(prim)?,
                },
                other => {
                    return self.err(&format!("a property or a prim expected, found {other:?}"))
                }
            }
        }
    }

    fn property(&mut self, prim: &mut Prim) -> Result<(), UsdaError> {
        let mut first = self.word()?;
        // List edits of a relationship: `prepend rel material:binding = ...`.
        if matches!(
            first.as_str(),
            "prepend" | "append" | "add" | "delete" | "reorder"
        ) {
            first = self.word()?;
        }
        let mut uniform = false;
        while matches!(first.as_str(), "custom" | "uniform" | "varying" | "config") {
            uniform |= first == "uniform";
            first = self.word()?;
        }
        if first == "rel" {
            let name = self.word()?;
            let mut targets = Vec::new();
            if self.eat('=') {
                targets = paths_of(&self.value()?);
            }
            let _ = self.metadata()?;
            prim.properties.push(Property {
                name,
                type_name: "rel".into(),
                targets,
                ..Property::default()
            });
            return Ok(());
        }
        let type_name = first;
        let full = self.word()?;
        let (name, suffix) = match full.rsplit_once('.') {
            Some((n, s)) if matches!(s, "timeSamples" | "connect" | "spline") => {
                (n.to_string(), Some(s.to_string()))
            }
            _ => (full, None),
        };
        let mut value = None;
        let mut targets = Vec::new();
        let mut samples = Vec::new();
        if self.eat('=') {
            let raw = self.value()?;
            match suffix.as_deref() {
                Some("connect") => targets = paths_of(&raw),
                Some("timeSamples") => {
                    if let Raw::Dict(raw) = raw {
                        let mut list: Vec<(f64, Val)> = raw
                            .into_iter()
                            .filter_map(|(t, v)| Some((t.parse::<f64>().ok()?, val_of(&v)?)))
                            .collect();
                        list.sort_by(|a, b| a.0.total_cmp(&b.0));
                        // At rest: the first sample in time.
                        value = list.first().map(|(_, v)| v.clone());
                        samples = list;
                    }
                }
                _ => value = val_of(&raw),
            }
        }
        let meta = self.metadata()?;
        // The same property may be written twice — a value and its
        // connection, or a default and its samples. One property.
        let property = match prim.properties.iter_mut().find(|p| p.name == name) {
            Some(p) => p,
            None => {
                prim.properties.push(Property {
                    name,
                    type_name,
                    uniform,
                    ..Property::default()
                });
                prim.properties.last_mut().expect("pushed")
            }
        };
        if value.is_some() && (property.value.is_none() || suffix.is_none()) {
            property.value = value;
        }
        property.targets.extend(targets);
        if !samples.is_empty() {
            property.samples = samples;
        }
        for (key, _, v) in meta {
            match (key.as_str(), v) {
                ("interpolation", Raw::Text(t)) => property.interpolation = Some(t),
                ("elementSize", Raw::Num(n)) => property.element_size = Some(n as i64),
                _ => {}
            }
        }
        Ok(())
    }
}

fn paths_of(raw: &Raw) -> Vec<String> {
    match raw {
        Raw::Path(p) => vec![p.clone()],
        Raw::List(items) => items.iter().flat_map(paths_of).collect(),
        _ => Vec::new(),
    }
}

/// `@a.usda@</Prim>`, a list of them, or an internal `</Prim>`.
fn arcs_of(raw: &Raw) -> Vec<Arc> {
    match raw {
        Raw::List(items) => {
            let mut out = Vec::new();
            let mut i = 0;
            while i < items.len() {
                match &items[i] {
                    Raw::Asset(a) => {
                        let path = match items.get(i + 1) {
                            Some(Raw::Path(p)) => {
                                i += 1;
                                Some(p.clone())
                            }
                            _ => None,
                        };
                        out.push(Arc {
                            asset: a.clone(),
                            prim_path: path,
                        });
                    }
                    Raw::Path(p) => out.push(Arc {
                        asset: String::new(),
                        prim_path: Some(p.clone()),
                    }),
                    other => out.extend(arcs_of(other)),
                }
                i += 1;
            }
            out
        }
        Raw::Asset(_) | Raw::Path(_) => arcs_of(&Raw::List(vec![raw.clone()])),
        _ => Vec::new(),
    }
}

/// A value as the stage keeps it: numbers, flattened, or text.
fn val_of(raw: &Raw) -> Option<Val> {
    fn nums(raw: &Raw, out: &mut Vec<f64>) -> bool {
        match raw {
            Raw::Num(n) => {
                out.push(*n);
                true
            }
            Raw::List(items) => items.iter().all(|i| nums(i, out)),
            _ => false,
        }
    }
    fn texts(raw: &Raw, out: &mut Vec<String>) -> bool {
        match raw {
            Raw::Text(t) | Raw::Asset(t) => {
                out.push(t.clone());
                true
            }
            Raw::List(items) => items.iter().all(|i| texts(i, out)),
            _ => false,
        }
    }
    let mut n = Vec::new();
    if nums(raw, &mut n) {
        return Some(Val::Nums(n));
    }
    let mut t = Vec::new();
    if texts(raw, &mut t) {
        return Some(Val::Texts(t));
    }
    None
}

/// A `.usda` layer.
pub fn parse_usda(text: &str) -> Result<Layer, UsdaError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if !text.trim_start().starts_with("#usda") {
        return Err(UsdaError("not a USD text layer (no `#usda` header)".into()));
    }
    let mut p = Parser {
        toks: lex(text)?,
        at: 0,
        depth: 0,
    };
    let mut layer = Layer::default();
    for (key, _, value) in p.metadata()? {
        match (key.as_str(), value) {
            ("upAxis", Raw::Text(a)) => layer.up_axis = Some(a),
            ("metersPerUnit", Raw::Num(m)) => layer.meters_per_unit = Some(m),
            ("defaultPrim", Raw::Text(d)) => layer.default_prim = Some(d),
            ("timeCodesPerSecond", Raw::Num(n)) => layer.time_codes_per_second = Some(n),
            ("startTimeCode", Raw::Num(n)) => layer.start_time_code = Some(n),
            ("endTimeCode", Raw::Num(n)) => layer.end_time_code = Some(n),
            ("framesPerSecond", Raw::Num(n)) if layer.time_codes_per_second.is_none() => {
                layer.time_codes_per_second = Some(n)
            }
            ("subLayers", v) => {
                let mut assets = Vec::new();
                collect_assets(&v, &mut assets);
                layer.sublayers = assets;
            }
            _ => {}
        }
    }
    while let Some(t) = p.peek() {
        match t {
            Tok::Word(w) if w == "def" || w == "over" || w == "class" => {
                let spec = match w.as_str() {
                    "def" => Specifier::Def,
                    "over" => Specifier::Over,
                    _ => Specifier::Class,
                };
                p.at += 1;
                let prim = p.prim(spec)?;
                layer.prims.push(prim);
            }
            Tok::Punct(';') => p.at += 1,
            other => return p.err(&format!("a prim expected at the top, found {other:?}")),
        }
    }
    Ok(layer)
}

fn collect_assets(raw: &Raw, out: &mut Vec<String>) {
    match raw {
        Raw::Asset(a) => out.push(a.clone()),
        Raw::List(items) => items.iter().for_each(|i| collect_assets(i, out)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
    doc = """A stage
over two lines"""
)

def Xform "World" (
    kind = "component"
    prepend references = @./part.usda@</Part>
    variants = {
        string shape = "round"
    }
)
{
    double3 xformOp:translate = (1, 2, 3)
    float3 xformOp:rotateXYZ.timeSamples = {
        0: (0, 90, 0),
        10: (0, 180, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

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
        rel material:binding = </World/Looks/Red>
        custom string note = "a # in a string"
    }

    variantSet "shape" = {
        "round" {
            def Sphere "Ball" { double radius = 2 }
        }
        "square" {
            def Cube "Box" {}
        }
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
                token outputs:surface
            }
        }
    }
}
"#;

    #[test]
    fn a_stage_reads_with_its_prims_values_and_arcs() {
        let layer = parse_usda(STAGE).unwrap();
        assert_eq!(layer.up_axis.as_deref(), Some("Y"));
        assert_eq!(layer.meters_per_unit, Some(0.01));
        assert_eq!(layer.default_prim.as_deref(), Some("World"));
        let world = &layer.prims[0];
        assert_eq!(
            (world.type_name.as_str(), world.name.as_str()),
            ("Xform", "World")
        );
        assert_eq!(world.references[0].asset, "./part.usda");
        assert_eq!(world.references[0].prim_path.as_deref(), Some("/Part"));
        assert_eq!(
            world.variant_selection,
            vec![("shape".into(), "round".into())]
        );
        assert_eq!(
            world.property("xformOp:translate").unwrap().nums(),
            Some(&[1.0, 2.0, 3.0][..])
        );
        // The first sample of an animated value.
        assert_eq!(
            world.property("xformOp:rotateXYZ").unwrap().nums(),
            Some(&[0.0, 90.0, 0.0][..])
        );
        assert_eq!(
            world
                .property("xformOpOrder")
                .unwrap()
                .texts()
                .unwrap()
                .len(),
            2
        );
        let tri = &world.children[0];
        assert_eq!(tri.property("points").unwrap().nums().unwrap().len(), 9);
        let st = tri.property("primvars:st").unwrap();
        assert_eq!(st.interpolation.as_deref(), Some("faceVarying"));
        assert_eq!(
            tri.property("material:binding").unwrap().targets,
            vec!["/World/Looks/Red".to_string()]
        );
        assert_eq!(world.variant_sets[0].1.len(), 2);
        let shader = &world.children[1].children[0].children[0];
        assert_eq!(
            shader.property("inputs:diffuseColor").unwrap().nums(),
            Some(&[1.0, 0.0, 0.0][..])
        );
        let material = &world.children[1].children[0];
        assert_eq!(
            material.property("outputs:surface").unwrap().targets,
            vec!["/World/Looks/Red/S.outputs:surface".to_string()]
        );
    }

    #[test]
    fn what_is_not_a_stage_is_said() {
        assert!(parse_usda("hello").is_err());
        assert!(parse_usda("#usda 1.0\ndef Xform \"A\" {").is_err());
        let deep = format!("#usda 1.0\n{}", "def \"a\" {".repeat(500));
        assert!(parse_usda(&deep).is_err());
    }
}
