//! DXF read into a drawing: the 2D file every drafting program writes.
//!
//! What a drawing holds is what DXF holds — lines, circles, arcs, ellipses,
//! polylines with bulges, splines, points and text, on layers with colours —
//! so each entity comes in as itself rather than as segments: a circle stays
//! a circle that a dimension can take hold of.
//!
//! A block inserted is drawn out where it is put: the drawing has no blocks,
//! and a title block or a bolt symbol drawn out is what the person sees in the
//! program it came from. Only the ASCII form is read; a binary DXF is refused.
//!
//! Units: `$INSUNITS` when the file says, millimetres when it does not — the
//! unit nearly every mechanical drawing is in, and the guess a drafting
//! program makes for a file that is silent.

use std::collections::HashMap;

use kreate_geom2d::{Curve2D, DVec2, Spline};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::drawing::{ColorRef, Drawing, EntityKind, LayerId, Text, TextAlign};
use kreate_scene::Document;

/// Reads a DXF into a document holding one drawing, named `name`, lying in
/// the ground plane.
pub fn parse_dxf(text: &str, name: &str) -> Result<Document, String> {
    if text.starts_with("AutoCAD Binary DXF") {
        return Err("a binary DXF: save it as ASCII DXF".into());
    }
    let pairs = pairs(text)?;
    let mut file = DxfFile::default();
    let mut i = 0;
    while i < pairs.len() {
        if pairs[i] == (0, "SECTION") && i + 1 < pairs.len() && pairs[i + 1].0 == 2 {
            let section = pairs[i + 1].1;
            let end = (i + 2..pairs.len())
                .find(|&k| pairs[k] == (0, "ENDSEC"))
                .unwrap_or(pairs.len());
            let body = &pairs[i + 2..end];
            match section {
                "HEADER" => file.header(body),
                "TABLES" => file.tables(body),
                "BLOCKS" => file.blocks(body),
                "ENTITIES" => file.entities = records(body),
                _ => {}
            }
            i = end + 1;
        } else {
            i += 1;
        }
    }

    let mut drawing = Drawing::new();
    let mut layers: HashMap<String, LayerId> = HashMap::new();
    for (lname, color, on) in &file.layers {
        let id = if lname == "0" {
            LayerId(0)
        } else {
            drawing.add_layer(lname)
        };
        if let Some(layer) = drawing.layers.iter_mut().find(|l| l.id == id) {
            if let Some(c) = color {
                layer.color = ColorRef::Index(*c);
            }
            layer.on = *on;
        }
        layers.insert(lname.clone(), id);
    }
    let mut out = Out {
        drawing: &mut drawing,
        layers: &mut layers,
        scale: file.metres_per_unit(),
        blocks: &file.blocks,
    };
    for record in &file.entities {
        out.entity(record, &Place::identity(), 0);
    }
    if drawing.entities.is_empty() {
        return Err("no drawing in the file: no lines, arcs or text it can read".into());
    }
    drawing.current_layer = LayerId(0);

    let mut doc = Document::new();
    let mut object = Object::new(name.to_string(), ObjectData::Empty);
    object.drawing = Some(Box::new(drawing));
    let id = doc.add_object(object);
    doc.selection.add(id);
    doc.selection.active = Some(id);
    Ok(doc)
}

/// The file as group-code and value pairs.
fn pairs(text: &str) -> Result<Vec<(i32, &str)>, String> {
    let mut lines = text.lines();
    let mut out = Vec::new();
    while let Some(code) = lines.next() {
        let code = code.trim();
        if code.is_empty() {
            continue;
        }
        let Some(value) = lines.next() else {
            break;
        };
        let code: i32 = code
            .parse()
            .map_err(|_| format!("not a DXF: `{code}` where a group code was expected"))?;
        out.push((code, value.trim()));
    }
    Ok(out)
}

/// One entity: its type and its pairs.
#[derive(Clone, Debug)]
struct Record<'a> {
    kind: &'a str,
    pairs: Vec<(i32, &'a str)>,
}

impl<'a> Record<'a> {
    fn get(&self, code: i32) -> Option<&'a str> {
        self.pairs.iter().find(|p| p.0 == code).map(|p| p.1)
    }
    fn f(&self, code: i32) -> Option<f64> {
        self.get(code).and_then(|v| v.parse().ok())
    }
    fn f0(&self, code: i32) -> f64 {
        self.f(code).unwrap_or(0.0)
    }
    fn i(&self, code: i32) -> i64 {
        self.get(code).and_then(|v| v.parse().ok()).unwrap_or(0)
    }
    fn all(&self, code: i32) -> impl Iterator<Item = f64> + '_ {
        self.pairs
            .iter()
            .filter(move |p| p.0 == code)
            .filter_map(|p| p.1.parse().ok())
    }
    /// Whether the entity's plane faces down: DXF's extrusion of (0, 0, -1),
    /// which a mirrored arc or circle is written with.
    fn flipped(&self) -> bool {
        self.f(230).is_some_and(|z| z < 0.0)
    }
}

/// The pairs split at each code 0. A POLYLINE's VERTEX records are folded
/// into it, as the entity they make up.
fn records<'a>(pairs: &[(i32, &'a str)]) -> Vec<Record<'a>> {
    let mut out: Vec<Record<'a>> = Vec::new();
    let mut in_polyline = false;
    for &(code, value) in pairs {
        if code == 0 {
            match value {
                "VERTEX" if in_polyline => {
                    if let Some(last) = out.last_mut() {
                        // A marker so the vertices can be told apart.
                        last.pairs.push((-1, "VERTEX"));
                    }
                    continue;
                }
                "SEQEND" if in_polyline => {
                    in_polyline = false;
                    continue;
                }
                _ => {}
            }
            in_polyline = value == "POLYLINE";
            out.push(Record {
                kind: value,
                pairs: Vec::new(),
            });
        } else if let Some(last) = out.last_mut() {
            last.pairs.push((code, value));
        }
    }
    out
}

#[derive(Default)]
struct DxfFile<'a> {
    insunits: i64,
    /// Name, colour, shown.
    layers: Vec<(String, Option<u8>, bool)>,
    /// Name to base point and entities.
    blocks: HashMap<String, (DVec2, Vec<Record<'a>>)>,
    entities: Vec<Record<'a>>,
}

impl<'a> DxfFile<'a> {
    fn header(&mut self, body: &[(i32, &'a str)]) {
        for k in 0..body.len() {
            if body[k] == (9, "$INSUNITS") {
                if let Some((70, v)) = body.get(k + 1) {
                    self.insunits = v.parse().unwrap_or(0);
                }
            }
        }
    }

    fn tables(&mut self, body: &[(i32, &'a str)]) {
        for r in records(body) {
            if r.kind != "LAYER" {
                continue;
            }
            let Some(name) = r.get(2) else {
                continue;
            };
            // A negative colour is the layer turned off.
            let color = r.i(62);
            let index = u8::try_from(color.unsigned_abs())
                .ok()
                .filter(|c| (1..=255).contains(c));
            self.layers.push((name.to_string(), index, color >= 0));
        }
    }

    fn blocks(&mut self, body: &[(i32, &'a str)]) {
        let mut current: Option<(String, DVec2, Vec<Record<'a>>)> = None;
        for r in records(body) {
            match r.kind {
                "BLOCK" => {
                    let name = r.get(2).unwrap_or("").to_string();
                    current = Some((name, DVec2::new(r.f0(10), r.f0(20)), Vec::new()));
                }
                "ENDBLK" => {
                    if let Some((name, base, items)) = current.take() {
                        self.blocks.insert(name, (base, items));
                    }
                }
                _ => {
                    if let Some((_, _, items)) = current.as_mut() {
                        items.push(r);
                    }
                }
            }
        }
    }

    /// Metres in one drawing unit. See the module notes.
    fn metres_per_unit(&self) -> f64 {
        match self.insunits {
            1 => 0.0254,
            2 => 0.3048,
            3 => 1609.344,
            4 => 0.001,
            5 => 0.01,
            6 => 1.0,
            7 => 1000.0,
            8 => 0.0254e-6,
            9 => 0.0254e-3,
            10 => 0.9144,
            14 => 0.1,
            _ => 0.001,
        }
    }
}

/// Where a block's contents land: a scale, a turn and a move, the turn
/// after the scale.
#[derive(Clone, Copy, Debug)]
struct Place {
    scale: DVec2,
    rotation: f64,
    offset: DVec2,
}

impl Place {
    fn identity() -> Place {
        Place {
            scale: DVec2::ONE,
            rotation: 0.0,
            offset: DVec2::ZERO,
        }
    }
    fn point(&self, p: DVec2) -> DVec2 {
        let s = p * self.scale;
        let (sin, cos) = self.rotation.sin_cos();
        DVec2::new(s.x * cos - s.y * sin, s.x * sin + s.y * cos) + self.offset
    }
    fn vector(&self, v: DVec2) -> DVec2 {
        self.point(v) - self.point(DVec2::ZERO)
    }
    /// Whether the place mirrors: arcs then run the other way.
    fn mirrors(&self) -> bool {
        self.scale.x * self.scale.y < 0.0
    }
    /// Whether it scales the same both ways, so a circle stays a circle.
    fn uniform(&self) -> bool {
        (self.scale.x.abs() - self.scale.y.abs()).abs() <= 1e-9 * self.scale.x.abs().max(1.0)
    }
    /// An angle in the block, as the drawing sees it.
    fn angle(&self, a: f64) -> f64 {
        let d = self.vector(DVec2::new(a.cos(), a.sin()));
        d.y.atan2(d.x)
    }
    /// This place, inside `outer`.
    fn within(&self, outer: &Place) -> Place {
        Place {
            scale: self.scale * outer.scale,
            rotation: self.rotation + outer.rotation,
            offset: outer.point(self.offset),
        }
    }
}

struct Out<'d, 'a> {
    drawing: &'d mut Drawing,
    layers: &'d mut HashMap<String, LayerId>,
    scale: f64,
    blocks: &'d HashMap<String, (DVec2, Vec<Record<'a>>)>,
}

/// Blocks inside blocks go this deep and no deeper: a block that inserts
/// itself would otherwise never end.
const MAX_NESTING: usize = 16;

impl Out<'_, '_> {
    fn add(&mut self, r: &Record, kind: EntityKind) {
        let layer_name = r.get(8).unwrap_or("0");
        let layer = match self.layers.get(layer_name) {
            Some(id) => *id,
            None => {
                let id = self.drawing.add_layer(layer_name);
                self.layers.insert(layer_name.to_string(), id);
                id
            }
        };
        let id = self.drawing.add(kind);
        if let Some(e) = self.drawing.entity_mut(id) {
            e.layer = layer;
            let c = r.i(62);
            if (1..=255).contains(&c) {
                e.color = ColorRef::Index(c as u8);
            }
        }
    }

    fn curve(&mut self, r: &Record, c: Curve2D) {
        self.add(r, EntityKind::Curve(c));
    }

    /// A point of the file, in metres in the drawing.
    fn at(&self, place: &Place, p: DVec2) -> DVec2 {
        place.point(p) * self.scale
    }

    fn entity(&mut self, r: &Record, outer: &Place, depth: usize) {
        // An entity facing down is drawn mirrored in X, which is what its
        // own coordinates mean seen from above.
        let mirror = Place {
            scale: DVec2::new(-1.0, 1.0),
            rotation: 0.0,
            offset: DVec2::ZERO,
        };
        let flip_place;
        let place = if r.flipped() {
            flip_place = mirror.within(outer);
            &flip_place
        } else {
            outer
        };
        let xy = |a: i32, b: i32| DVec2::new(r.f0(a), r.f0(b));
        match r.kind {
            "LINE" => {
                let (a, b) = (self.at(place, xy(10, 20)), self.at(place, xy(11, 21)));
                if a.distance(b) > 0.0 {
                    self.curve(r, Curve2D::Line { a, b });
                }
            }
            "POINT" => {
                let p = self.at(place, xy(10, 20));
                self.add(r, EntityKind::Point(p));
            }
            "CIRCLE" | "ARC" => {
                let c = xy(10, 20);
                let radius = r.f0(40);
                if radius <= 0.0 {
                    return;
                }
                let (start, end) = if r.kind == "ARC" {
                    (r.f0(50).to_radians(), r.f0(51).to_radians())
                } else {
                    (0.0, std::f64::consts::TAU)
                };
                let mut sweep = end - start;
                if r.kind == "ARC" {
                    sweep = sweep.rem_euclid(std::f64::consts::TAU);
                    if sweep == 0.0 {
                        sweep = std::f64::consts::TAU;
                    }
                }
                if place.uniform() {
                    let rr = radius * place.scale.x.abs() * self.scale;
                    let cc = self.at(place, c);
                    if r.kind == "CIRCLE" {
                        self.curve(r, Curve2D::Circle { c: cc, r: rr });
                    } else {
                        let s = place.angle(start);
                        let sweep = if place.mirrors() { -sweep } else { sweep };
                        self.curve(
                            r,
                            Curve2D::Arc {
                                c: cc,
                                r: rr,
                                start: s,
                                sweep,
                            },
                        );
                    }
                } else {
                    // Stretched unevenly a circle is an ellipse: its two
                    // half-axes carried across.
                    let major = place.vector(DVec2::new(radius, 0.0)) * self.scale;
                    let minor = place.vector(DVec2::new(0.0, radius)) * self.scale;
                    let (major, ratio) = if major.length() >= minor.length() {
                        (major, minor.length() / major.length())
                    } else {
                        (minor, major.length() / minor.length())
                    };
                    self.curve(
                        r,
                        Curve2D::Ellipse {
                            c: self.at(place, c),
                            major,
                            ratio,
                            start,
                            sweep,
                        },
                    );
                }
            }
            "ELLIPSE" => {
                let c = self.at(place, xy(10, 20));
                let major = place.vector(xy(11, 21)) * self.scale;
                let ratio = r.f(40).unwrap_or(1.0);
                let start = r.f0(41);
                let mut sweep = r.f(42).unwrap_or(std::f64::consts::TAU) - start;
                if sweep <= 0.0 {
                    sweep += std::f64::consts::TAU;
                }
                if place.mirrors() {
                    sweep = -sweep;
                }
                if major.length() > 0.0 {
                    self.curve(
                        r,
                        Curve2D::Ellipse {
                            c,
                            major,
                            ratio,
                            start,
                            sweep,
                        },
                    );
                }
            }
            "LWPOLYLINE" => {
                let closed = r.i(70) & 1 != 0;
                // Each 10 starts a vertex; a 42 after it is that vertex's
                // bulge.
                let mut verts: Vec<(DVec2, f64)> = Vec::new();
                let mut x = None;
                for &(code, value) in &r.pairs {
                    let v: f64 = value.parse().unwrap_or(0.0);
                    match code {
                        10 => x = Some(v),
                        20 => {
                            if let Some(x) = x.take() {
                                verts.push((self.at(place, DVec2::new(x, v)), 0.0));
                            }
                        }
                        42 => {
                            if let Some(last) = verts.last_mut() {
                                last.1 = if place.mirrors() { -v } else { v };
                            }
                        }
                        _ => {}
                    }
                }
                self.polyline(r, verts, closed);
            }
            "POLYLINE" => {
                let closed = r.i(70) & 1 != 0;
                let mut verts: Vec<(DVec2, f64)> = Vec::new();
                let mut seen_vertex = false;
                let (mut x, mut y, mut bulge) = (0.0, 0.0, 0.0);
                for &(code, value) in &r.pairs {
                    if code == -1 {
                        if seen_vertex {
                            verts.push((self.at(place, DVec2::new(x, y)), bulge));
                        }
                        seen_vertex = true;
                        (x, y, bulge) = (0.0, 0.0, 0.0);
                        continue;
                    }
                    if !seen_vertex {
                        continue;
                    }
                    let v: f64 = value.parse().unwrap_or(0.0);
                    match code {
                        10 => x = v,
                        20 => y = v,
                        42 => bulge = if place.mirrors() { -v } else { v },
                        _ => {}
                    }
                }
                if seen_vertex {
                    verts.push((self.at(place, DVec2::new(x, y)), bulge));
                }
                self.polyline(r, verts, closed);
            }
            "SPLINE" => {
                let degree = r.i(71).max(1) as usize;
                let xs: Vec<f64> = r.all(10).collect();
                let ys: Vec<f64> = r.all(20).collect();
                let ctrl: Vec<DVec2> = xs
                    .iter()
                    .zip(&ys)
                    .map(|(x, y)| self.at(place, DVec2::new(*x, *y)))
                    .collect();
                let fx: Vec<f64> = r.all(11).collect();
                let fy: Vec<f64> = r.all(21).collect();
                let fit: Vec<DVec2> = fx
                    .iter()
                    .zip(&fy)
                    .map(|(x, y)| self.at(place, DVec2::new(*x, *y)))
                    .collect();
                let knots: Vec<f64> = r.all(40).collect();
                let weights: Vec<f64> = r.all(41).collect();
                let spline = if ctrl.len() > degree && knots.len() == ctrl.len() + degree + 1 {
                    Spline {
                        degree,
                        weights: (weights.len() == ctrl.len()).then_some(weights),
                        ctrl,
                        knots,
                        fit,
                    }
                } else if fit.len() >= 2 {
                    Spline::through(&fit)
                } else if ctrl.len() >= 2 {
                    Spline::clamped(degree, ctrl)
                } else {
                    return;
                };
                self.curve(r, Curve2D::Spline(spline));
            }
            "TEXT" | "MTEXT" => {
                let mut content = String::new();
                // MTEXT runs long text over 3s before the last 1.
                for &(code, value) in &r.pairs {
                    if code == 3 || code == 1 {
                        content.push_str(value);
                    }
                }
                let content = if r.kind == "MTEXT" {
                    plain_mtext(&content)
                } else {
                    content
                };
                let content = unescape(&content);
                if content.trim().is_empty() {
                    return;
                }
                let height = r.f(40).unwrap_or(2.5) * place.scale.y.abs() * self.scale;
                // TEXT aligned other than left is placed by its second point.
                let halign = r.i(72);
                let at = if r.kind == "TEXT" && halign != 0 && r.get(11).is_some() {
                    xy(11, 21)
                } else {
                    xy(10, 20)
                };
                let align = match (r.kind, halign, r.i(71)) {
                    ("TEXT", 1 | 4, _) => TextAlign::Center,
                    ("TEXT", 2, _) => TextAlign::Right,
                    ("MTEXT", _, 2 | 5 | 8) => TextAlign::Center,
                    ("MTEXT", _, 3 | 6 | 9) => TextAlign::Right,
                    _ => TextAlign::Left,
                };
                let rotation = place.angle(r.f0(50).to_radians());
                let text = Text {
                    at: self.at(place, at),
                    height,
                    rotation,
                    content,
                    align,
                    font: String::new(),
                };
                self.add(r, EntityKind::Text(text));
            }
            "INSERT" => {
                if depth >= MAX_NESTING {
                    return;
                }
                let Some((base, items)) = r.get(2).and_then(|n| self.blocks.get(n)) else {
                    return;
                };
                let scale = DVec2::new(r.f(41).unwrap_or(1.0), r.f(42).unwrap_or(1.0));
                let rotation = r.f0(50).to_radians();
                let (cols, rows) = (r.i(70).max(1), r.i(71).max(1));
                let (dx, dy) = (r.f0(44), r.f0(45));
                for row in 0..rows {
                    for col in 0..cols {
                        // The block's base point lands on the insertion point;
                        // an array of them steps along the turned axes.
                        let (sin, cos) = rotation.sin_cos();
                        let step = DVec2::new(col as f64 * dx, row as f64 * dy);
                        let step =
                            DVec2::new(step.x * cos - step.y * sin, step.x * sin + step.y * cos);
                        let inner = Place {
                            scale,
                            rotation,
                            offset: xy(10, 20) + step,
                        };
                        let to_base = Place {
                            scale: DVec2::ONE,
                            rotation: 0.0,
                            offset: -*base,
                        };
                        let place = to_base.within(&inner).within(place);
                        for item in items.clone().iter() {
                            self.entity(item, &place, depth + 1);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn polyline(&mut self, r: &Record, verts: Vec<(DVec2, f64)>, closed: bool) {
        if verts.len() < 2 {
            return;
        }
        if verts.len() == 2 && verts[0].1 == 0.0 && !closed {
            self.curve(
                r,
                Curve2D::Line {
                    a: verts[0].0,
                    b: verts[1].0,
                },
            );
            return;
        }
        self.curve(r, Curve2D::Polyline { verts, closed });
    }
}

/// MTEXT's formatting codes taken out: `\P` is a new line, `{\fArial;…}`
/// and the like set a font or a size, `\~` is a space.
fn plain_mtext(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' | '}' => {}
            '\\' => match chars.next() {
                Some('P') | Some('X') => out.push('\n'),
                Some('~') => out.push(' '),
                Some('\\') => out.push('\\'),
                Some('{') => out.push('{'),
                Some('}') => out.push('}'),
                Some('U') => {
                    out.push('\\');
                    out.push('U');
                }
                // A code that runs to a semicolon: font, height, width,
                // colour, tracking, oblique, alignment.
                Some('f' | 'F' | 'H' | 'W' | 'C' | 'c' | 'T' | 'Q' | 'A' | 'p') => {
                    for d in chars.by_ref() {
                        if d == ';' {
                            break;
                        }
                    }
                }
                // Stacked text, `\S1/2;`: the parts as written.
                Some('S') => {
                    for d in chars.by_ref() {
                        if d == ';' {
                            break;
                        }
                        out.push(if d == '^' || d == '#' { '/' } else { d });
                    }
                }
                // Underline, overline, strike: on and off.
                Some('L' | 'l' | 'O' | 'o' | 'K' | 'k') => {}
                Some(other) => out.push(other),
                None => {}
            },
            _ => out.push(c),
        }
    }
    out
}

/// `\U+AC00` escapes, and `%%c`, `%%d`, `%%p` for Ø, ° and ± — Ø rather
/// than ⌀, which the bundled fonts lack.
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while !rest.is_empty() {
        if let Some(hex) = rest.strip_prefix("\\U+").filter(|h| h.len() >= 4) {
            if let Some(c) = u32::from_str_radix(&hex[..4], 16)
                .ok()
                .and_then(char::from_u32)
            {
                out.push(c);
                rest = &hex[4..];
                continue;
            }
        }
        if rest.len() >= 3 && rest[..2].eq_ignore_ascii_case("%%") {
            let sym = match rest.as_bytes()[2].to_ascii_lowercase() {
                b'c' => Some('Ø'),
                b'd' => Some('°'),
                b'p' => Some('±'),
                b'%' => Some('%'),
                _ => None,
            };
            if let Some(sym) = sym {
                out.push(sym);
                rest = &rest[3..];
                continue;
            }
        }
        let c = rest.chars().next().expect("not empty");
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dxf(header_units: Option<i32>, entities: &str, blocks: &str) -> String {
        let header = match header_units {
            Some(u) => format!("0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n{u}\n0\nENDSEC\n"),
            None => String::new(),
        };
        format!(
            "{header}0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n0\nLAYER\n2\nOutline\n62\n1\n0\nLAYER\n2\nHidden\n62\n-3\n0\nENDTAB\n0\nENDSEC\n\
             0\nSECTION\n2\nBLOCKS\n{blocks}0\nENDSEC\n\
             0\nSECTION\n2\nENTITIES\n{entities}0\nENDSEC\n0\nEOF\n"
        )
    }

    fn drawing(doc: &Document) -> &Drawing {
        doc.objects()
            .find_map(|(_, o)| o.drawing.as_deref())
            .expect("a drawing")
    }

    fn curves(d: &Drawing) -> Vec<Curve2D> {
        d.entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn each_entity_comes_in_as_itself_in_metres() {
        let text = dxf(
            Some(4),
            "0\nLINE\n8\nOutline\n10\n0\n20\n0\n11\n100\n21\n0\n\
             0\nCIRCLE\n8\nOutline\n10\n50\n20\n50\n40\n10\n\
             0\nARC\n10\n0\n20\n0\n40\n20\n50\n0\n51\n90\n\
             0\nLWPOLYLINE\n90\n3\n70\n1\n10\n0\n20\n0\n42\n1\n10\n10\n20\n0\n10\n10\n20\n10\n\
             0\nPOINT\n10\n5\n20\n5\n",
            "",
        );
        let doc = parse_dxf(&text, "Plate").unwrap();
        let d = drawing(&doc);
        let cs = curves(d);
        assert_eq!(cs.len(), 4, "{cs:?}");
        assert!(matches!(cs[0], Curve2D::Line { b, .. } if (b.x - 0.1).abs() < 1e-12));
        assert!(matches!(cs[1], Curve2D::Circle { r, .. } if (r - 0.01).abs() < 1e-12));
        assert!(
            matches!(cs[2], Curve2D::Arc { sweep, .. } if (sweep - std::f64::consts::FRAC_PI_2).abs() < 1e-12)
        );
        assert!(
            matches!(&cs[3], Curve2D::Polyline { verts, closed: true } if verts.len() == 3 && verts[0].1 == 1.0)
        );
        assert!(d
            .entities
            .iter()
            .any(|e| matches!(e.kind, EntityKind::Point(_))));
        // The layers and their state come across.
        let outline = d.layers.iter().find(|l| l.name == "Outline").unwrap();
        assert_eq!(outline.color, ColorRef::Index(1));
        assert!(!d.layers.iter().find(|l| l.name == "Hidden").unwrap().on);
        assert!(d.entities.iter().take(2).all(|e| e.layer == outline.id));
        assert_eq!(doc.objects().next().unwrap().1.name, "Plate");
    }

    #[test]
    fn a_silent_file_is_millimetres_and_a_declared_one_is_its_unit() {
        let line = "0\nLINE\n10\n0\n20\n0\n11\n1\n21\n0\n";
        let length =
            |units| match curves(drawing(&parse_dxf(&dxf(units, line, ""), "x").unwrap()))[0] {
                Curve2D::Line { a, b } => a.distance(b),
                _ => unreachable!(),
            };
        assert!((length(None) - 0.001).abs() < 1e-15);
        assert!((length(Some(1)) - 0.0254).abs() < 1e-15);
        assert!((length(Some(6)) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_block_is_drawn_out_where_it_is_inserted_turned_and_scaled() {
        let block = "0\nBLOCK\n2\nBOLT\n10\n1\n20\n0\n0\nCIRCLE\n10\n1\n20\n0\n40\n2\n\
                     0\nLINE\n10\n1\n20\n0\n11\n3\n21\n0\n0\nENDBLK\n";
        let insert = "0\nINSERT\n2\nBOLT\n10\n100\n20\n0\n41\n2\n42\n2\n50\n90\n";
        let doc = parse_dxf(&dxf(Some(6), insert, block), "x").unwrap();
        let cs = curves(drawing(&doc));
        // The base point lands on the insertion point.
        let Curve2D::Circle { c, r } = cs[0] else {
            panic!("{cs:?}")
        };
        assert!(c.distance(DVec2::new(100.0, 0.0)) < 1e-9 && (r - 4.0).abs() < 1e-12);
        // The line two units along X, doubled and turned a quarter: up Y.
        let Curve2D::Line { b, .. } = cs[1] else {
            panic!()
        };
        assert!(b.distance(DVec2::new(100.0, 4.0)) < 1e-9, "{b}");
    }

    #[test]
    fn text_is_read_plain() {
        let text = dxf(
            Some(4),
            "0\nTEXT\n10\n0\n20\n0\n40\n5\n1\n%%c10 \\U+AC00\n\
             0\nMTEXT\n10\n0\n20\n20\n40\n3\n1\n{\\fArial|b1;Line one}\\PLine two\n",
            "",
        );
        let doc = parse_dxf(&text, "x").unwrap();
        let d = drawing(&doc);
        let texts: Vec<&Text> = d
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Text(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(texts[0].content, "Ø10 가");
        assert!((texts[0].height - 0.005).abs() < 1e-12);
        assert_eq!(texts[1].content, "Line one\nLine two");
    }

    #[test]
    fn a_mirrored_arc_and_an_old_polyline_come_in_right() {
        let text = dxf(
            Some(6),
            // An arc facing down: its centre at x = 5 is at x = -5 seen
            // from above, and it runs clockwise.
            "0\nARC\n10\n5\n20\n0\n40\n1\n50\n0\n51\n90\n230\n-1\n\
             0\nPOLYLINE\n66\n1\n70\n1\n0\nVERTEX\n10\n0\n20\n0\n0\nVERTEX\n10\n2\n20\n0\n0\nVERTEX\n10\n2\n20\n2\n0\nSEQEND\n",
            "",
        );
        let cs = curves(drawing(&parse_dxf(&text, "x").unwrap()));
        let Curve2D::Arc { c, sweep, .. } = cs[0] else {
            panic!("{cs:?}")
        };
        assert!(c.distance(DVec2::new(-5.0, 0.0)) < 1e-12 && sweep < 0.0);
        assert!(matches!(&cs[1], Curve2D::Polyline { verts, closed: true } if verts.len() == 3));
    }

    #[test]
    fn what_kreate_writes_it_reads_back() {
        let lines = vec![crate::drawing::Line2 {
            a: [0.0, 0.0],
            b: [0.05, 0.02],
            hidden: false,
        }];
        let written = crate::drawing::write_dxf(&lines, 0.001);
        let doc = parse_dxf(&written, "x").unwrap();
        let Curve2D::Line { a, b } = curves(drawing(&doc))[0] else {
            panic!()
        };
        assert!(a.distance(DVec2::ZERO) < 1e-9 && b.distance(DVec2::new(0.05, 0.02)) < 1e-9);
    }

    #[test]
    fn a_binary_or_empty_file_is_refused() {
        assert!(parse_dxf("AutoCAD Binary DXF\r\n\x1a\0", "x").is_err());
        assert!(parse_dxf(&dxf(None, "", ""), "x").is_err());
        assert!(parse_dxf("hello\nworld\n", "x").is_err());
    }

    #[test]
    fn a_block_inserting_itself_stops() {
        let block = "0\nBLOCK\n2\nLOOP\n0\nLINE\n10\n0\n20\n0\n11\n1\n21\n0\n0\nINSERT\n2\nLOOP\n10\n1\n20\n0\n0\nENDBLK\n";
        let doc = parse_dxf(
            &dxf(Some(6), "0\nINSERT\n2\nLOOP\n10\n0\n20\n0\n", block),
            "x",
        )
        .unwrap();
        assert_eq!(curves(drawing(&doc)).len(), MAX_NESTING);
    }
}
