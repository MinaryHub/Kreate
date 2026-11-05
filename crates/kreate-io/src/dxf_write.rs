//! DXF write of a drawing: what was drawn in the drafting mode, out to the
//! file every drafting program reads. See design/2d-drafting.md §10.
//!
//! Written as R12 (AC1009): the version every drafting program, and nearly
//! every laser cutter's, plasma table's and router's software, reads — the
//! design's "for the machine shop" choice, taken first because it is the
//! one nothing refuses. What R12 has no entity for is drawn with what it
//! has, to the same hundredth of a millimetre a feature is built to:
//!
//! - lines, circles, arcs, points and text as themselves;
//! - polylines as POLYLINE with their bulges, so their arcs stay arcs;
//! - ellipses and splines as polylines through points on them;
//! - dimensions drawn out — extension lines, the dimension line and its
//!   number — on their own layer, as a program reading R12 shows them;
//! - construction geometry left out, as it is from a plot.
//!
//! Layers keep their names, colours, linetypes and whether they are on (a
//! layer turned off is written with its colour negative, as DXF has it). A
//! colour given as RGB, which R12 cannot hold, becomes the nearest of
//! AutoCAD's numbered colours.
//!
//! [`crate::dxf_read`] reads all of it back.

use std::fmt::Write as _;

use kreate_geom2d::{Curve2D, DVec2};
use kreate_scene::drawing::{
    ColorRef, DimKind, Dimension, Drawing, Entity, EntityKind, LinetypeRef, Text, TextAlign,
};

/// How far a curve R12 has no entity for may stray from its polyline: a
/// hundredth of a millimetre, as features are built to.
pub const CHORD: f64 = 1e-5;

/// The layer a dimension's lines and number go on when its own is "0":
/// where drafting programs keep them, so they can be turned off together.
const DIM_LAYER: &str = "DIM";

/// The drawing as an R12 DXF, its lengths in units of `metres_per_unit`
/// metres (`1e-3` for millimetres) — the unit the header names.
pub fn write(drawing: &Drawing, metres_per_unit: f64) -> String {
    let s = 1.0 / metres_per_unit;
    let mut w = Writer {
        out: String::new(),
        s,
    };
    w.pair(0, "SECTION");
    w.pair(2, "HEADER");
    w.pair(9, "$ACADVER");
    w.pair(1, "AC1009");
    w.pair(9, "$INSUNITS");
    w.pair(70, &insunits(metres_per_unit).to_string());
    if let Some((lo, hi)) = drawing.extents() {
        w.pair(9, "$EXTMIN");
        w.point(10, lo);
        w.pair(9, "$EXTMAX");
        w.point(10, hi);
    }
    w.pair(0, "ENDSEC");

    // Tables: the linetypes the layers and entities name, and the layers.
    let mut linetypes: Vec<String> = vec!["CONTINUOUS".into()];
    let mut note = |name: &str| {
        let name = name.trim().to_uppercase();
        if !name.is_empty() && !linetypes.contains(&name) {
            linetypes.push(name);
        }
    };
    for l in &drawing.layers {
        note(&l.linetype);
    }
    for e in &drawing.entities {
        if let LinetypeRef::Named(n) = &e.linetype {
            note(n);
        }
    }
    let has_dims = drawing
        .entities
        .iter()
        .any(|e| matches!(e.kind, EntityKind::Dimension(_)));
    w.pair(0, "SECTION");
    w.pair(2, "TABLES");
    w.pair(0, "TABLE");
    w.pair(2, "LTYPE");
    w.pair(70, &linetypes.len().to_string());
    for name in &linetypes {
        let (text, pattern) = linetype_pattern(name);
        w.pair(0, "LTYPE");
        w.pair(2, name);
        w.pair(70, "0");
        w.pair(3, text);
        w.pair(72, "65");
        w.pair(73, &pattern.len().to_string());
        // The pattern in the file's unit, from millimetres.
        let scale = 1e-3 * s;
        w.pair(
            40,
            &num(pattern.iter().map(|e| e.abs()).sum::<f64>() * scale),
        );
        for e in pattern {
            w.pair(49, &num(e * scale));
        }
    }
    w.pair(0, "ENDTAB");
    w.pair(0, "TABLE");
    w.pair(2, "LAYER");
    w.pair(
        70,
        &(drawing.layers.len() + usize::from(has_dims)).to_string(),
    );
    for l in &drawing.layers {
        let colour = aci(&l.color).unwrap_or(7) as i32;
        w.pair(0, "LAYER");
        w.pair(2, &layer_name(&l.name));
        w.pair(70, if l.frozen { "1" } else { "0" });
        w.pair(62, &(if l.on { colour } else { -colour }).to_string());
        let linetype = l.linetype.trim().to_uppercase();
        w.pair(
            6,
            if linetype.is_empty() {
                "CONTINUOUS"
            } else {
                &linetype
            },
        );
    }
    if has_dims && !drawing.layers.iter().any(|l| l.name == DIM_LAYER) {
        w.pair(0, "LAYER");
        w.pair(2, DIM_LAYER);
        w.pair(70, "0");
        w.pair(62, "3");
        w.pair(6, "CONTINUOUS");
    }
    w.pair(0, "ENDTAB");
    w.pair(0, "ENDSEC");

    w.pair(0, "SECTION");
    w.pair(2, "ENTITIES");
    for e in &drawing.entities {
        if e.construction {
            continue;
        }
        let layer = drawing
            .layer(e.layer)
            .map(|l| layer_name(&l.name))
            .unwrap_or_else(|| "0".into());
        match &e.kind {
            EntityKind::Point(p) => {
                w.start("POINT", &layer, e);
                w.point(10, *p);
            }
            EntityKind::Curve(c) => w.curve(c, &layer, e),
            EntityKind::Text(t) => w.text(t, &layer, e),
            EntityKind::Dimension(d) => {
                let layer = if layer == "0" {
                    DIM_LAYER.to_string()
                } else {
                    layer
                };
                w.dimension(drawing, d, &layer, e, metres_per_unit);
            }
        }
    }
    w.pair(0, "ENDSEC");
    w.pair(0, "EOF");
    w.out
}

/// AutoCAD's code for a length unit, for `$INSUNITS`; 0 for one it has
/// none for.
fn insunits(metres_per_unit: f64) -> u32 {
    [(0.0254, 1), (0.3048, 2), (1e-3, 4), (1e-2, 5), (1.0, 6)]
        .iter()
        .find(|(m, _)| (m - metres_per_unit).abs() < 1e-12)
        .map(|(_, c)| *c)
        .unwrap_or(0)
}

/// A layer's name as DXF allows it: no characters its tables reserve.
fn layer_name(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| {
            if "<>/\\\":;?*|=`".contains(c) || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    if clean.trim().is_empty() {
        "0".into()
    } else {
        clean
    }
}

/// A linetype's description and dash pattern in millimetres (dashes
/// positive, gaps negative, dots zero), for the ones drafting uses; an
/// unknown name is written solid under its own name.
fn linetype_pattern(name: &str) -> (&'static str, Vec<f64>) {
    match name {
        "DASHED" | "HIDDEN" => ("Dashed __ __ __", vec![5.0, -2.5]),
        "CENTER" => ("Center ____ _ ____ _", vec![12.5, -2.5, 2.5, -2.5]),
        "PHANTOM" => (
            "Phantom ____ _ _ ____",
            vec![12.5, -2.5, 2.5, -2.5, 2.5, -2.5],
        ),
        "DOT" => ("Dot . . . .", vec![0.0, -2.5]),
        "DASHDOT" => ("Dash dot __ . __ .", vec![5.0, -2.5, 0.0, -2.5]),
        _ => ("Solid line", Vec::new()),
    }
}

/// AutoCAD's colour number for a colour, `None` for "by layer"; "by block"
/// is 0. RGB, which R12 cannot hold, is the nearest numbered colour.
fn aci(c: &ColorRef) -> Option<u8> {
    match c {
        ColorRef::ByLayer => None,
        ColorRef::ByBlock => Some(0),
        ColorRef::Index(i) => Some(*i),
        ColorRef::Rgb(rgb) => {
            // The standard colours and greys, which every program shows the
            // same.
            const NAMED: [(u8, [u8; 3]); 9] = [
                (1, [255, 0, 0]),
                (2, [255, 255, 0]),
                (3, [0, 255, 0]),
                (4, [0, 255, 255]),
                (5, [0, 0, 255]),
                (6, [255, 0, 255]),
                (7, [255, 255, 255]),
                (8, [128, 128, 128]),
                (9, [192, 192, 192]),
            ];
            let d = |a: [u8; 3]| {
                (0..3)
                    .map(|i| (i32::from(a[i]) - i32::from(rgb[i])).pow(2))
                    .sum::<i32>()
            };
            NAMED.iter().min_by_key(|(_, c)| d(*c)).map(|(i, _)| *i)
        }
    }
}

/// A number as the file writes it: enough places for a micrometre in
/// metres, no trailing noise.
fn num(v: f64) -> String {
    let v = if v.abs() < 1e-12 { 0.0 } else { v };
    let s = format!("{v:.9}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".into()
    } else {
        s.into()
    }
}

struct Writer {
    out: String,
    /// File units per metre.
    s: f64,
}

impl Writer {
    fn pair(&mut self, code: i32, value: &str) {
        let _ = write!(self.out, "{code}\n{value}\n");
    }

    /// A point at `code`, `code + 10`, `code + 20`, in the file's unit.
    fn point(&mut self, code: i32, p: DVec2) {
        let (x, y) = (num(p.x * self.s), num(p.y * self.s));
        self.pair(code, &x);
        self.pair(code + 10, &y);
        self.pair(code + 20, "0");
    }

    fn length(&mut self, code: i32, v: f64) {
        let v = num(v * self.s);
        self.pair(code, &v);
    }

    /// An entity's first lines: its kind, layer, linetype and colour.
    fn start(&mut self, kind: &str, layer: &str, e: &Entity) {
        self.pair(0, kind);
        self.pair(8, layer);
        if let LinetypeRef::Named(n) = &e.linetype {
            let n = n.trim().to_uppercase();
            if !n.is_empty() {
                self.pair(6, &n);
            }
        }
        if let Some(c) = aci(&e.color) {
            self.pair(62, &c.to_string());
        }
    }

    fn curve(&mut self, c: &Curve2D, layer: &str, e: &Entity) {
        match c {
            Curve2D::Line { a, b } => {
                self.start("LINE", layer, e);
                self.point(10, *a);
                self.point(11, *b);
            }
            Curve2D::Circle { c, r } => {
                self.start("CIRCLE", layer, e);
                self.point(10, *c);
                self.length(40, *r);
            }
            Curve2D::Arc { c, r, start, sweep } => {
                // DXF's arcs run anticlockwise; a clockwise one is the
                // same arc from its other end.
                let (from, to) = if *sweep >= 0.0 {
                    (*start, start + sweep)
                } else {
                    (start + sweep, *start)
                };
                if (to - from).abs() >= std::f64::consts::TAU - 1e-12 {
                    self.start("CIRCLE", layer, e);
                    self.point(10, *c);
                    self.length(40, *r);
                    return;
                }
                self.start("ARC", layer, e);
                self.point(10, *c);
                self.length(40, *r);
                self.pair(50, &num(from.to_degrees().rem_euclid(360.0)));
                self.pair(51, &num(to.to_degrees().rem_euclid(360.0)));
            }
            Curve2D::Polyline { verts, closed } => {
                self.polyline(verts, *closed, layer, e);
            }
            Curve2D::Ellipse { .. } | Curve2D::Spline(_) => {
                let points = c.tessellate(CHORD);
                let closed = c.is_closed() && points.len() > 2;
                let verts: Vec<(DVec2, f64)> = if closed {
                    points[..points.len() - 1]
                        .iter()
                        .map(|p| (*p, 0.0))
                        .collect()
                } else {
                    points.iter().map(|p| (*p, 0.0)).collect()
                };
                self.polyline(&verts, closed, layer, e);
            }
        }
    }

    /// R12's polyline: a POLYLINE, its VERTEXes with their bulges, SEQEND.
    fn polyline(&mut self, verts: &[(DVec2, f64)], closed: bool, layer: &str, e: &Entity) {
        if verts.len() < 2 {
            return;
        }
        self.start("POLYLINE", layer, e);
        self.pair(66, "1");
        self.point(10, DVec2::ZERO);
        self.pair(70, if closed { "1" } else { "0" });
        for (p, bulge) in verts {
            self.pair(0, "VERTEX");
            self.pair(8, layer);
            self.point(10, *p);
            if *bulge != 0.0 {
                self.pair(42, &num(*bulge));
            }
        }
        self.pair(0, "SEQEND");
        self.pair(8, layer);
    }

    fn text(&mut self, t: &Text, layer: &str, e: &Entity) {
        if t.content.trim().is_empty() {
            return;
        }
        // One line a TEXT: R12 has no multi-line text.
        let step = t.height * 5.0 / 3.0;
        let down = DVec2::new(t.rotation.sin(), -t.rotation.cos()) * step;
        for (i, line) in t.content.lines().enumerate() {
            let at = t.at + down * i as f64;
            self.start("TEXT", layer, e);
            self.point(10, at);
            self.length(40, t.height);
            self.pair(1, &escape(line));
            if t.rotation != 0.0 {
                self.pair(50, &num(t.rotation.to_degrees()));
            }
            let h = match t.align {
                TextAlign::Left => 0,
                TextAlign::Center => 1,
                TextAlign::Right => 2,
            };
            if h != 0 {
                self.pair(72, &h.to_string());
                self.point(11, at);
            }
        }
    }

    /// A dimension drawn out: extension lines, its line, and its number
    /// over the middle — what a program reading R12 shows of one.
    fn dimension(
        &mut self,
        drawing: &Drawing,
        d: &Dimension,
        layer: &str,
        e: &Entity,
        metres_per_unit: f64,
    ) {
        let Some(shape) = dimension_shape(drawing, d) else {
            return;
        };
        for [a, b] in &shape.lines {
            self.start("LINE", layer, e);
            self.point(10, *a);
            self.point(11, *b);
        }
        if let Some((c, r, from, to)) = shape.arc {
            self.start("ARC", layer, e);
            self.point(10, c);
            self.length(40, r);
            self.pair(50, &num(from.to_degrees().rem_euclid(360.0)));
            self.pair(51, &num(to.to_degrees().rem_euclid(360.0)));
        }
        let text = dimension_text(d, metres_per_unit);
        let height = shape.height;
        self.start("TEXT", layer, e);
        self.point(10, shape.at);
        self.length(40, height);
        self.pair(1, &escape(&text));
        if shape.angle != 0.0 {
            self.pair(50, &num(shape.angle.to_degrees()));
        }
        self.pair(72, "1");
        self.point(11, shape.at);
    }
}

/// `%` is DXF's escape: `%%` writes one.
fn escape(s: &str) -> String {
    s.replace('%', "%%")
}

/// What a dimension says: its own text, or its value in the file's unit
/// (degrees for an angle), with its tolerance.
fn dimension_text(d: &Dimension, metres_per_unit: f64) -> String {
    use kreate_scene::drawing::Tolerance;
    let length = |v: f64| {
        let v = v / metres_per_unit;
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    let mut text = match &d.text {
        Some(t) if !t.trim().is_empty() => t.clone(),
        _ => match d.kind {
            DimKind::Angular => {
                let s = format!("{:.2}", d.value.to_degrees());
                format!("{}%%d", s.trim_end_matches('0').trim_end_matches('.'))
            }
            DimKind::Diameter => format!("%%c{}", length(d.value)),
            DimKind::Radius => format!("R{}", length(d.value)),
            _ => length(d.value),
        },
    };
    match &d.tolerance {
        Some(Tolerance::Symmetric(t)) => {
            let _ = write!(text, "%%p{}", length(*t));
        }
        Some(Tolerance::Deviation { upper, lower }) => {
            let _ = write!(text, " +{}/-{}", length(*upper), length(lower.abs()));
        }
        Some(Tolerance::Limits { upper, lower }) => {
            text = format!("{}/{}", length(*upper), length(*lower));
        }
        Some(Tolerance::Fit(fit)) => {
            let _ = write!(text, " {fit}");
        }
        None => {}
    }
    if !d.driving {
        text = format!("({text})");
    }
    text
}

/// A dimension's lines, an angle's arc (centre, radius, from, to in
/// radians, anticlockwise), and where its number goes, how tall and at what
/// angle.
struct DimShape {
    lines: Vec<[DVec2; 2]>,
    arc: Option<(DVec2, f64, f64, f64)>,
    at: DVec2,
    angle: f64,
    height: f64,
}

/// How tall a dimension's number is: 2.5 mm, ISO-25's, in metres.
const DIM_TEXT: f64 = 2.5e-3;

fn dimension_shape(drawing: &Drawing, d: &Dimension) -> Option<DimShape> {
    let point = |i: usize| d.refs.get(i).and_then(|r| drawing.point_of(*r));
    let curve = |i: usize| d.refs.get(i).and_then(|r| drawing.curve(r.entity));
    let height = DIM_TEXT;
    // A little past the dimension line, as ISO draws extension lines.
    let over = height * 0.8;
    match d.kind {
        DimKind::Horizontal
        | DimKind::Vertical
        | DimKind::Aligned
        | DimKind::OrdinateX
        | DimKind::OrdinateY => {
            let (a, b) = match (point(0), point(1)) {
                (Some(a), Some(b)) => (a, b),
                // One line measured end to end.
                _ => {
                    let c = curve(0)?;
                    (c.start(), c.end())
                }
            };
            let along = match d.kind {
                DimKind::Horizontal | DimKind::OrdinateX => DVec2::X,
                DimKind::Vertical | DimKind::OrdinateY => DVec2::Y,
                _ => (b - a).normalize_or(DVec2::X),
            };
            let across = along.perp();
            let text_at = if d.text_at == DVec2::ZERO {
                (a + b) * 0.5 + across * height * 3.0
            } else {
                d.text_at
            };
            // The dimension line through the number, along the measure.
            let level = across.dot(text_at);
            let foot = |p: DVec2| p + across * (level - across.dot(p));
            let (fa, fb) = (foot(a), foot(b));
            let mut lines = vec![[fa, fb]];
            for (p, f) in [(a, fa), (b, fb)] {
                let out = (f - p).normalize_or(across);
                if (f - p).length() > 1e-12 {
                    lines.push([p, f + out * over]);
                }
            }
            let angle = along.y.atan2(along.x);
            // Readable from the bottom or the right, as ISO has it.
            let angle = if angle > std::f64::consts::FRAC_PI_2 + 1e-9
                || angle <= -std::f64::consts::FRAC_PI_2 + 1e-9
            {
                angle + std::f64::consts::PI
            } else {
                angle
            };
            let at = foot(text_at) + across * height * 0.8;
            Some(DimShape {
                lines,
                arc: None,
                at,
                angle,
                height,
            })
        }
        DimKind::Radius | DimKind::Diameter | DimKind::ArcLength => {
            let (c, r) = match curve(0)? {
                Curve2D::Circle { c, r } | Curve2D::Arc { c, r, .. } => (*c, *r),
                _ => return None,
            };
            let toward = (d.text_at - c).normalize_or(DVec2::X);
            let far = c + toward * r;
            let near = if d.kind == DimKind::Diameter {
                c - toward * r
            } else {
                c
            };
            let at = if d.text_at == DVec2::ZERO || (d.text_at - c).length() < r {
                far + toward * height * 2.0
            } else {
                d.text_at
            };
            Some(DimShape {
                lines: vec![[near, far], [far, at]],
                arc: None,
                at: at + toward * height,
                angle: 0.0,
                height,
            })
        }
        DimKind::Angular => {
            // Between two lines, about where they meet.
            let (Some(Curve2D::Line { a: a0, b: b0 }), Some(Curve2D::Line { a: a1, b: b1 })) =
                (curve(0), curve(1))
            else {
                return None;
            };
            let (u, v) = (*b0 - *a0, *b1 - *a1);
            let cross = u.perp_dot(v);
            let centre = if cross.abs() < 1e-18 {
                (*a0 + *b1) * 0.5
            } else {
                *a0 + u * ((*a1 - *a0).perp_dot(v) / cross)
            };
            let r = if d.text_at == DVec2::ZERO {
                u.length().min(v.length()) * 0.5
            } else {
                (d.text_at - centre).length()
            }
            .max(height * 2.0);
            let (s, t) = (u.y.atan2(u.x), v.y.atan2(v.x));
            // The smaller way round, anticlockwise.
            let (from, to) = if (t - s).rem_euclid(std::f64::consts::TAU) <= std::f64::consts::PI {
                (s, s + (t - s).rem_euclid(std::f64::consts::TAU))
            } else {
                (t, t + (s - t).rem_euclid(std::f64::consts::TAU))
            };
            let mid = (from + to) * 0.5;
            Some(DimShape {
                lines: Vec::new(),
                arc: Some((centre, r, from, to)),
                at: centre + DVec2::new(mid.cos(), mid.sin()) * (r + height * 1.2),
                angle: 0.0,
                height,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::drawing::{EntityRef, Part};

    fn drawing() -> Drawing {
        let mut d = Drawing::new();
        let cut = d.add_layer("CUT");
        if let Some(l) = d.layers.iter_mut().find(|l| l.id == cut) {
            l.color = ColorRef::Index(1);
            l.linetype = "DASHED".into();
        }
        d.current_layer = cut;
        let line = d.add_curve(Curve2D::Line {
            a: DVec2::new(0.0, 0.0),
            b: DVec2::new(0.04, 0.0),
        });
        d.add_curve(Curve2D::Circle {
            c: DVec2::new(0.02, 0.01),
            r: 0.005,
        });
        // Clockwise: written from its other end.
        d.add_curve(Curve2D::Arc {
            c: DVec2::ZERO,
            r: 0.01,
            start: std::f64::consts::FRAC_PI_2,
            sweep: -std::f64::consts::FRAC_PI_2,
        });
        d.add_curve(Curve2D::Polyline {
            verts: vec![
                (DVec2::new(0.0, 0.02), 0.0),
                (DVec2::new(0.01, 0.02), 1.0),
                (DVec2::new(0.02, 0.02), 0.0),
            ],
            closed: false,
        });
        d.add_curve(Curve2D::Ellipse {
            c: DVec2::new(0.05, 0.05),
            major: DVec2::new(0.01, 0.0),
            ratio: 0.5,
            start: 0.0,
            sweep: std::f64::consts::TAU,
        });
        d.current_layer = kreate_scene::drawing::LayerId(0);
        d.add(EntityKind::Text(Text {
            at: DVec2::new(0.0, -0.01),
            height: 0.0025,
            rotation: 0.0,
            content: "50% PLATE".into(),
            align: TextAlign::Center,
            font: String::new(),
        }));
        d.add(EntityKind::Dimension(Dimension {
            kind: DimKind::Horizontal,
            refs: vec![
                EntityRef::at(line, Part::Start),
                EntityRef::at(line, Part::End),
            ],
            text_at: DVec2::new(0.02, -0.008),
            style: String::new(),
            driving: true,
            value: 0.04,
            text: None,
            tolerance: None,
        }));
        let guide = d.add_curve(Curve2D::Line {
            a: DVec2::new(-1.0, -1.0),
            b: DVec2::new(1.0, 1.0),
        });
        if let Some(e) = d.entity_mut(guide) {
            e.construction = true;
        }
        d
    }

    /// What is written in millimetres reads back as the same drawing, in
    /// metres: each curve as itself where R12 has one, the ellipse as a
    /// closed polyline on it, the dimension drawn out, construction left
    /// out.
    #[test]
    fn a_drawing_goes_out_and_comes_back_in_the_same_place() {
        let d = drawing();
        let text = write(&d, 1e-3);
        assert!(text.contains("AC1009") && text.contains("$INSUNITS\n70\n4\n"));
        assert!(text.contains("50%% PLATE"), "the per cent sign escaped");
        let doc = crate::dxf_read::parse_dxf(&text, "back").expect("reads");
        let back = doc
            .objects()
            .find_map(|(_, o)| o.drawing.as_deref())
            .expect("a drawing");
        let curves: Vec<&Curve2D> = back
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c) => Some(c),
                _ => None,
            })
            .collect();
        let near = |a: DVec2, b: DVec2| (a - b).length() < 1e-8;
        assert!(curves.iter().any(|c| matches!(c, Curve2D::Line { a, b }
            if near(*a, DVec2::ZERO) && near(*b, DVec2::new(0.04, 0.0)))));
        assert!(curves.iter().any(|c| matches!(c, Curve2D::Circle { c, r }
            if near(*c, DVec2::new(0.02, 0.01)) && (r - 0.005).abs() < 1e-9)));
        // The clockwise arc, anticlockwise from 0 to 90°.
        assert!(curves
            .iter()
            .any(|c| matches!(c, Curve2D::Arc { r, start, sweep, .. }
            if (r - 0.01).abs() < 1e-9 && start.abs() < 1e-6
                && (sweep - std::f64::consts::FRAC_PI_2).abs() < 1e-6)));
        // The bulged polyline keeps its bulge.
        assert!(curves
            .iter()
            .any(|c| matches!(c, Curve2D::Polyline { verts, .. }
            if verts.iter().any(|(_, b)| (b - 1.0).abs() < 1e-9))));
        // The ellipse, as a closed polyline every point of which is on it.
        let ellipse = curves
            .iter()
            .find_map(|c| match c {
                Curve2D::Polyline {
                    verts,
                    closed: true,
                } if verts.len() > 8 => Some(verts),
                _ => None,
            })
            .expect("the ellipse as a closed polyline");
        for (p, _) in ellipse {
            let q = *p - DVec2::new(0.05, 0.05);
            let on = (q.x / 0.01).powi(2) + (q.y / 0.005).powi(2);
            assert!((on - 1.0).abs() < 1e-3, "{p:?} is off the ellipse");
        }
        // The construction line is not there.
        assert!(!curves.iter().any(|c| matches!(c, Curve2D::Line { a, .. }
            if near(*a, DVec2::new(-1.0, -1.0)))));
        // The layer with its colour, and the dimensions' own.
        let cut = back.layers.iter().find(|l| l.name == "CUT").expect("CUT");
        assert_eq!(cut.color, ColorRef::Index(1));
        assert!(back.layers.iter().any(|l| l.name == DIM_LAYER));
        // The dimension's number, in millimetres.
        assert!(back.entities.iter().any(|e| matches!(&e.kind,
            EntityKind::Text(t) if t.content == "40")));
    }

    #[test]
    fn a_dimension_says_its_value_its_tolerance_and_whether_it_drives() {
        use kreate_scene::drawing::Tolerance;
        let mut d = Dimension {
            kind: DimKind::Diameter,
            refs: Vec::new(),
            text_at: DVec2::ZERO,
            style: String::new(),
            driving: true,
            value: 0.012,
            text: None,
            tolerance: Some(Tolerance::Fit("H7".into())),
        };
        assert_eq!(dimension_text(&d, 1e-3), "%%c12 H7");
        d.kind = DimKind::Horizontal;
        d.tolerance = Some(Tolerance::Symmetric(0.0001));
        d.driving = false;
        assert_eq!(dimension_text(&d, 1e-3), "(12%%p0.1)");
        d.kind = DimKind::Angular;
        d.value = std::f64::consts::FRAC_PI_4;
        d.tolerance = None;
        d.driving = true;
        assert_eq!(dimension_text(&d, 1e-3), "45%%d");
    }

    #[test]
    fn a_colour_r12_cannot_hold_is_the_nearest_it_can() {
        assert_eq!(aci(&ColorRef::Rgb([250, 10, 5])), Some(1));
        assert_eq!(aci(&ColorRef::Rgb([120, 130, 125])), Some(8));
        assert_eq!(aci(&ColorRef::ByLayer), None);
    }
}
