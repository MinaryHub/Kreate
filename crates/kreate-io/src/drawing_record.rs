//! A drawing as `scene.json` holds it: see `kreate_scene::drawing`.
//!
//! Entities, curves, constraints and dimensions are each an object with a
//! `type`, so a person reading a scene sees `"type": "arc"` and its centre,
//! and so a kind this build does not know is kept as it was written and
//! written back — never dropped, never guessed at.

use kreate_geom2d::{Curve2D, DVec2, Spline};
use kreate_scene::drawing::{
    ColorRef, Constraint, DimKind, Dimension, Drawing, Entity, EntityId, EntityKind, EntityRef,
    Layer, LayerId, LinetypeRef, Lineweight, Part, Text, TextAlign, Tolerance,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrawingRecord {
    pub entities: Vec<Value>,
    pub layers: Vec<LayerRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<Value>,
    #[serde(default)]
    pub current_layer: u32,
    /// The next entity id and layer id to issue.
    #[serde(default)]
    pub next: [u32; 2],
    /// Entries this build could not read when it loaded the drawing, as
    /// they were: see `Drawing::foreign`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub foreign: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRecord {
    pub id: u32,
    pub name: String,
    pub color: String,
    pub linetype: String,
    pub lineweight: String,
    #[serde(default = "yes")]
    pub on: bool,
    #[serde(default)]
    pub frozen: bool,
    #[serde(default)]
    pub locked: bool,
    #[serde(default = "yes")]
    pub plot: bool,
}

fn yes() -> bool {
    true
}

fn color_text(c: ColorRef) -> String {
    match c {
        ColorRef::ByLayer => "bylayer".into(),
        ColorRef::ByBlock => "byblock".into(),
        ColorRef::Index(i) => format!("aci {i}"),
        ColorRef::Rgb([r, g, b]) => format!("#{r:02x}{g:02x}{b:02x}"),
    }
}

fn color_from(s: &str) -> ColorRef {
    match s {
        "bylayer" => ColorRef::ByLayer,
        "byblock" => ColorRef::ByBlock,
        s if s.starts_with("aci ") => s[4..].parse().map(ColorRef::Index).unwrap_or_default(),
        s if s.len() == 7 && s.starts_with('#') => {
            let h = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).unwrap_or(0);
            ColorRef::Rgb([h(1), h(3), h(5)])
        }
        _ => ColorRef::ByLayer,
    }
}

fn weight_text(w: Lineweight) -> String {
    match w {
        Lineweight::ByLayer => "bylayer".into(),
        Lineweight::ByBlock => "byblock".into(),
        Lineweight::Mm(mm) => format!("{mm}"),
    }
}

fn weight_from(s: &str) -> Lineweight {
    match s {
        "bylayer" => Lineweight::ByLayer,
        "byblock" => Lineweight::ByBlock,
        s => s.parse().map(Lineweight::Mm).unwrap_or_default(),
    }
}

fn linetype_text(l: &LinetypeRef) -> String {
    match l {
        LinetypeRef::ByLayer => "bylayer".into(),
        LinetypeRef::ByBlock => "byblock".into(),
        LinetypeRef::Named(n) => n.clone(),
    }
}

fn linetype_from(s: &str) -> LinetypeRef {
    match s {
        "bylayer" => LinetypeRef::ByLayer,
        "byblock" => LinetypeRef::ByBlock,
        s => LinetypeRef::Named(s.to_string()),
    }
}

fn part_text(p: Part) -> String {
    match p {
        Part::Whole => "whole".into(),
        Part::Start => "start".into(),
        Part::End => "end".into(),
        Part::Center => "center".into(),
        Part::Mid => "mid".into(),
        Part::Vertex(i) => format!("v{i}"),
    }
}

fn part_from(s: &str) -> Option<Part> {
    Some(match s {
        "whole" => Part::Whole,
        "start" => Part::Start,
        "end" => Part::End,
        "center" => Part::Center,
        "mid" => Part::Mid,
        s if s.starts_with('v') => Part::Vertex(s[1..].parse().ok()?),
        _ => return None,
    })
}

fn xy(p: DVec2) -> Value {
    serde_json::json!([p.x, p.y])
}

fn from_xy(v: &Value) -> Option<DVec2> {
    let a = v.as_array()?;
    Some(DVec2::new(a.first()?.as_f64()?, a.get(1)?.as_f64()?))
}

fn r_text(r: EntityRef) -> Value {
    serde_json::json!([r.entity.0, part_text(r.part)])
}

fn r_from(v: &Value) -> Option<EntityRef> {
    let a = v.as_array()?;
    Some(EntityRef {
        entity: EntityId(a.first()?.as_u64()? as u32),
        part: part_from(a.get(1)?.as_str()?)?,
    })
}

fn curve_value(c: &Curve2D) -> Value {
    use serde_json::json;
    match c {
        Curve2D::Line { a, b } => json!({"type": "line", "a": xy(*a), "b": xy(*b)}),
        Curve2D::Circle { c, r } => json!({"type": "circle", "c": xy(*c), "r": r}),
        Curve2D::Arc { c, r, start, sweep } => {
            json!({"type": "arc", "c": xy(*c), "r": r, "start": start, "sweep": sweep})
        }
        Curve2D::Ellipse {
            c,
            major,
            ratio,
            start,
            sweep,
        } => json!({"type": "ellipse", "c": xy(*c), "major": xy(*major), "ratio": ratio,
                     "start": start, "sweep": sweep}),
        Curve2D::Polyline { verts, closed } => json!({
            "type": "polyline",
            "verts": verts.iter().map(|(p, b)| json!([p.x, p.y, b])).collect::<Vec<_>>(),
            "closed": closed,
        }),
        Curve2D::Spline(s) => json!({
            "type": "spline",
            "degree": s.degree,
            "ctrl": s.ctrl.iter().map(|p| xy(*p)).collect::<Vec<_>>(),
            "knots": s.knots,
            "weights": s.weights,
            "fit": s.fit.iter().map(|p| xy(*p)).collect::<Vec<_>>(),
        }),
    }
}

fn curve_from(v: &Value) -> Option<Curve2D> {
    let f = |k: &str| v.get(k).and_then(Value::as_f64);
    let p = |k: &str| v.get(k).and_then(from_xy);
    Some(match v.get("type")?.as_str()? {
        "line" => Curve2D::Line {
            a: p("a")?,
            b: p("b")?,
        },
        "circle" => Curve2D::Circle {
            c: p("c")?,
            r: f("r")?,
        },
        "arc" => Curve2D::Arc {
            c: p("c")?,
            r: f("r")?,
            start: f("start")?,
            sweep: f("sweep")?,
        },
        "ellipse" => Curve2D::Ellipse {
            c: p("c")?,
            major: p("major")?,
            ratio: f("ratio")?,
            start: f("start")?,
            sweep: f("sweep")?,
        },
        "polyline" => Curve2D::Polyline {
            verts: v
                .get("verts")?
                .as_array()?
                .iter()
                .map(|t| {
                    let a = t.as_array()?;
                    Some((
                        DVec2::new(a.first()?.as_f64()?, a.get(1)?.as_f64()?),
                        a.get(2).and_then(Value::as_f64).unwrap_or(0.0),
                    ))
                })
                .collect::<Option<Vec<_>>>()?,
            closed: v.get("closed").and_then(Value::as_bool).unwrap_or(false),
        },
        "spline" => {
            let pts = |k: &str| -> Option<Vec<DVec2>> {
                v.get(k)?.as_array()?.iter().map(from_xy).collect()
            };
            let degree = v.get("degree")?.as_u64()? as usize;
            let ctrl = pts("ctrl")?;
            let knots: Vec<f64> = v
                .get("knots")?
                .as_array()?
                .iter()
                .map(Value::as_f64)
                .collect::<Option<_>>()?;
            // A spline whose knots do not fit its control points would read
            // past them; it is not a spline.
            if degree == 0 || ctrl.len() <= degree || knots.len() != ctrl.len() + degree + 1 {
                return None;
            }
            let weights: Option<Vec<f64>> = match v.get("weights") {
                Some(Value::Array(w)) => {
                    let w: Vec<f64> = w.iter().map(Value::as_f64).collect::<Option<_>>()?;
                    (w.len() == ctrl.len()).then_some(w)
                }
                _ => None,
            };
            Curve2D::Spline(Spline {
                degree,
                ctrl,
                knots,
                weights,
                fit: pts("fit").unwrap_or_default(),
            })
        }
        _ => return None,
    })
}

fn dim_kind_text(k: DimKind) -> &'static str {
    match k {
        DimKind::Horizontal => "horizontal",
        DimKind::Vertical => "vertical",
        DimKind::Aligned => "aligned",
        DimKind::Angular => "angular",
        DimKind::Radius => "radius",
        DimKind::Diameter => "diameter",
        DimKind::ArcLength => "arc_length",
        DimKind::OrdinateX => "ordinate_x",
        DimKind::OrdinateY => "ordinate_y",
    }
}

fn dim_kind_from(s: &str) -> Option<DimKind> {
    Some(match s {
        "horizontal" => DimKind::Horizontal,
        "vertical" => DimKind::Vertical,
        "aligned" => DimKind::Aligned,
        "angular" => DimKind::Angular,
        "radius" => DimKind::Radius,
        "diameter" => DimKind::Diameter,
        "arc_length" => DimKind::ArcLength,
        "ordinate_x" => DimKind::OrdinateX,
        "ordinate_y" => DimKind::OrdinateY,
        _ => return None,
    })
}

fn tolerance_value(t: &Tolerance) -> Value {
    use serde_json::json;
    match t {
        Tolerance::Symmetric(v) => json!({"type": "symmetric", "value": v}),
        Tolerance::Deviation { upper, lower } => {
            json!({"type": "deviation", "upper": upper, "lower": lower})
        }
        Tolerance::Limits { upper, lower } => {
            json!({"type": "limits", "upper": upper, "lower": lower})
        }
        Tolerance::Fit(f) => json!({"type": "fit", "fit": f}),
    }
}

fn tolerance_from(v: &Value) -> Option<Tolerance> {
    let f = |k: &str| v.get(k).and_then(Value::as_f64);
    Some(match v.get("type")?.as_str()? {
        "symmetric" => Tolerance::Symmetric(f("value")?),
        "deviation" => Tolerance::Deviation {
            upper: f("upper")?,
            lower: f("lower")?,
        },
        "limits" => Tolerance::Limits {
            upper: f("upper")?,
            lower: f("lower")?,
        },
        "fit" => Tolerance::Fit(v.get("fit")?.as_str()?.to_string()),
        _ => return None,
    })
}

fn entity_value(e: &Entity) -> Value {
    use serde_json::json;
    let mut v = match &e.kind {
        EntityKind::Point(p) => json!({"type": "point", "at": xy(*p)}),
        EntityKind::Curve(c) => json!({"type": "curve", "curve": curve_value(c)}),
        EntityKind::Text(t) => json!({
            "type": "text", "at": xy(t.at), "height": t.height, "rotation": t.rotation,
            "content": t.content,
            "align": match t.align { TextAlign::Left => "left", TextAlign::Center => "center", TextAlign::Right => "right" },
            "font": t.font,
        }),
        EntityKind::Dimension(d) => json!({
            "type": "dimension",
            "kind": dim_kind_text(d.kind),
            "refs": d.refs.iter().map(|r| r_text(*r)).collect::<Vec<_>>(),
            "text_at": xy(d.text_at),
            "style": d.style,
            "driving": d.driving,
            "value": d.value,
            "text": d.text,
            "tolerance": d.tolerance.as_ref().map(tolerance_value),
        }),
    };
    let o = v.as_object_mut().expect("an object");
    o.insert("id".into(), json!(e.id.0));
    o.insert("layer".into(), json!(e.layer.0));
    if e.color != ColorRef::ByLayer {
        o.insert("color".into(), json!(color_text(e.color)));
    }
    if e.linetype != LinetypeRef::ByLayer {
        o.insert("linetype".into(), json!(linetype_text(&e.linetype)));
    }
    if e.lineweight != Lineweight::ByLayer {
        o.insert("lineweight".into(), json!(weight_text(e.lineweight)));
    }
    if e.construction {
        o.insert("construction".into(), json!(true));
    }
    v
}

fn entity_from(v: &Value) -> Option<Entity> {
    let s = |k: &str| v.get(k).and_then(Value::as_str);
    let kind = match s("type")? {
        "point" => EntityKind::Point(from_xy(v.get("at")?)?),
        "curve" => EntityKind::Curve(curve_from(v.get("curve")?)?),
        "text" => EntityKind::Text(Text {
            at: from_xy(v.get("at")?)?,
            height: v.get("height")?.as_f64()?,
            rotation: v.get("rotation").and_then(Value::as_f64).unwrap_or(0.0),
            content: s("content").unwrap_or_default().to_string(),
            align: match s("align") {
                Some("center") => TextAlign::Center,
                Some("right") => TextAlign::Right,
                _ => TextAlign::Left,
            },
            font: s("font").unwrap_or_default().to_string(),
        }),
        "dimension" => EntityKind::Dimension(Dimension {
            kind: dim_kind_from(s("kind")?)?,
            refs: v
                .get("refs")?
                .as_array()?
                .iter()
                .map(r_from)
                .collect::<Option<_>>()?,
            text_at: from_xy(v.get("text_at")?)?,
            style: s("style").unwrap_or_default().to_string(),
            driving: v.get("driving").and_then(Value::as_bool).unwrap_or(false),
            value: v.get("value").and_then(Value::as_f64).unwrap_or(0.0),
            text: s("text").map(str::to_string),
            tolerance: v.get("tolerance").and_then(tolerance_from),
        }),
        _ => return None,
    };
    Some(Entity {
        id: EntityId(v.get("id")?.as_u64()? as u32),
        kind,
        layer: LayerId(v.get("layer").and_then(Value::as_u64).unwrap_or(0) as u32),
        color: s("color").map(color_from).unwrap_or_default(),
        linetype: s("linetype").map(linetype_from).unwrap_or_default(),
        lineweight: s("lineweight").map(weight_from).unwrap_or_default(),
        construction: v
            .get("construction")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn constraint_value(c: &Constraint) -> Value {
    use serde_json::json;
    let id = |e: &EntityId| json!(e.0);
    match c {
        Constraint::Coincident(a, b) => {
            json!({"type": "coincident", "a": r_text(*a), "b": r_text(*b)})
        }
        Constraint::Horizontal(a, b) => {
            json!({"type": "horizontal", "a": r_text(*a), "b": b.map(r_text)})
        }
        Constraint::Vertical(a, b) => {
            json!({"type": "vertical", "a": r_text(*a), "b": b.map(r_text)})
        }
        Constraint::Parallel(a, b) => json!({"type": "parallel", "a": id(a), "b": id(b)}),
        Constraint::Perpendicular(a, b) => json!({"type": "perpendicular", "a": id(a), "b": id(b)}),
        Constraint::Tangent(a, b) => json!({"type": "tangent", "a": id(a), "b": id(b)}),
        Constraint::Equal(a, b) => json!({"type": "equal", "a": id(a), "b": id(b)}),
        Constraint::Concentric(a, b) => json!({"type": "concentric", "a": id(a), "b": id(b)}),
        Constraint::Smooth(a, b) => json!({"type": "smooth", "a": id(a), "b": id(b)}),
        Constraint::PointOn(p, c) => json!({"type": "point_on", "p": r_text(*p), "on": id(c)}),
        Constraint::Symmetric(a, b, l) => {
            json!({"type": "symmetric", "a": r_text(*a), "b": r_text(*b), "axis": id(l)})
        }
        Constraint::Midpoint(p, l) => json!({"type": "midpoint", "p": r_text(*p), "of": id(l)}),
        Constraint::Fixed(p, at) => json!({"type": "fixed", "p": r_text(*p), "at": xy(*at)}),
    }
}

fn constraint_from(v: &Value) -> Option<Constraint> {
    let r = |k: &str| v.get(k).and_then(r_from);
    let id = |k: &str| v.get(k).and_then(Value::as_u64).map(|i| EntityId(i as u32));
    Some(match v.get("type")?.as_str()? {
        "coincident" => Constraint::Coincident(r("a")?, r("b")?),
        "horizontal" => Constraint::Horizontal(r("a")?, r("b")),
        "vertical" => Constraint::Vertical(r("a")?, r("b")),
        "parallel" => Constraint::Parallel(id("a")?, id("b")?),
        "perpendicular" => Constraint::Perpendicular(id("a")?, id("b")?),
        "tangent" => Constraint::Tangent(id("a")?, id("b")?),
        "equal" => Constraint::Equal(id("a")?, id("b")?),
        "concentric" => Constraint::Concentric(id("a")?, id("b")?),
        "smooth" => Constraint::Smooth(id("a")?, id("b")?),
        "point_on" => Constraint::PointOn(r("p")?, id("on")?),
        "symmetric" => Constraint::Symmetric(r("a")?, r("b")?, id("axis")?),
        "midpoint" => Constraint::Midpoint(r("p")?, id("of")?),
        "fixed" => Constraint::Fixed(r("p")?, from_xy(v.get("at")?)?),
        _ => return None,
    })
}

/// A drawing as the file holds it.
pub fn record(d: &Drawing) -> DrawingRecord {
    DrawingRecord {
        entities: d.entities.iter().map(entity_value).collect(),
        layers: d
            .layers
            .iter()
            .map(|l| LayerRecord {
                id: l.id.0,
                name: l.name.clone(),
                color: color_text(l.color),
                linetype: l.linetype.clone(),
                lineweight: weight_text(l.lineweight),
                on: l.on,
                frozen: l.frozen,
                locked: l.locked,
                plot: l.plot,
            })
            .collect(),
        constraints: d.constraints.iter().map(constraint_value).collect(),
        current_layer: d.current_layer.0,
        next: {
            let (e, l) = d.counters();
            [e, l]
        },
        foreign: d
            .foreign
            .iter()
            .filter_map(|s| serde_json::from_str(s).ok())
            .collect(),
    }
}

/// What a file's record makes, and the entries it could not read. Those
/// are kept on the drawing too, as `foreign`, to write back.
pub fn drawing_from(r: &DrawingRecord) -> (Drawing, Vec<Value>) {
    let mut d = Drawing::new();
    let mut unread = Vec::new();
    d.layers = r
        .layers
        .iter()
        .map(|l| Layer {
            id: LayerId(l.id),
            name: l.name.clone(),
            color: color_from(&l.color),
            linetype: l.linetype.clone(),
            lineweight: weight_from(&l.lineweight),
            on: l.on,
            frozen: l.frozen,
            locked: l.locked,
            plot: l.plot,
        })
        .collect();
    if !d.layers.iter().any(|l| l.id == LayerId(0)) {
        d.layers.insert(0, Layer::new(LayerId(0), "0"));
    }
    for v in &r.entities {
        match entity_from(v) {
            // An id twice is the file's mistake; the first is kept.
            Some(e) if d.entity(e.id).is_none() => d.entities.push(e),
            _ => unread.push(v.clone()),
        }
    }
    for v in &r.constraints {
        match constraint_from(v) {
            Some(c) if c.entities().iter().all(|e| d.entity(*e).is_some()) => d.constraints.push(c),
            _ => unread.push(v.clone()),
        }
    }
    d.current_layer = if d.layer(LayerId(r.current_layer)).is_some() {
        LayerId(r.current_layer)
    } else {
        LayerId(0)
    };
    d.set_counters(r.next[0], r.next[1]);
    unread.extend(r.foreign.iter().cloned());
    d.foreign = unread.iter().map(Value::to_string).collect();
    (d, unread)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Everything a drawing can hold comes back from its record.
    #[test]
    fn a_drawing_survives_its_record() {
        let mut d = Drawing::new();
        let dims = d.add_layer("Dimensions");
        let a = d.add_curve(Curve2D::Line {
            a: DVec2::new(0.0, 0.0),
            b: DVec2::new(0.04, 0.0),
        });
        let c = d.add_curve(Curve2D::Arc {
            c: DVec2::new(0.02, 0.01),
            r: 0.005,
            start: 0.5,
            sweep: -2.0,
        });
        d.add_curve(Curve2D::Polyline {
            verts: vec![(DVec2::ZERO, 0.3), (DVec2::new(1.0, 0.0), 0.0)],
            closed: false,
        });
        d.add_curve(Curve2D::Spline(Spline::through(&[
            DVec2::ZERO,
            DVec2::new(1.0, 1.0),
            DVec2::new(2.0, 0.0),
        ])));
        d.add_curve(Curve2D::Ellipse {
            c: DVec2::ZERO,
            major: DVec2::new(2.0, 0.0),
            ratio: 0.5,
            start: 0.0,
            sweep: 1.0,
        });
        let p = d.add(EntityKind::Point(DVec2::new(0.1, 0.2)));
        d.entity_mut(p).unwrap().construction = true;
        d.entity_mut(p).unwrap().color = ColorRef::Rgb([255, 128, 0]);
        d.add(EntityKind::Text(Text {
            at: DVec2::new(0.0, 0.05),
            height: 0.0035,
            rotation: 0.1,
            content: "도면 A-1".into(),
            align: TextAlign::Center,
            font: String::new(),
        }));
        d.current_layer = dims;
        d.add(EntityKind::Dimension(Dimension {
            kind: DimKind::Radius,
            refs: vec![EntityRef::whole(c)],
            text_at: DVec2::new(0.03, 0.02),
            style: "ISO-25".into(),
            driving: true,
            value: 0.005,
            text: Some("R<>".into()),
            tolerance: Some(Tolerance::Fit("H7".into())),
        }));
        d.constraints
            .push(Constraint::Horizontal(EntityRef::whole(a), None));
        d.constraints.push(Constraint::Coincident(
            EntityRef::at(a, Part::End),
            EntityRef::at(c, Part::Start),
        ));
        d.constraints.push(Constraint::Fixed(
            EntityRef::at(a, Part::Start),
            DVec2::ZERO,
        ));
        let rec = record(&d);
        let text = serde_json::to_string(&rec).unwrap();
        let back: DrawingRecord = serde_json::from_str(&text).unwrap();
        let (got, unread) = drawing_from(&back);
        assert!(unread.is_empty(), "{unread:?}");
        assert_eq!(got, d);
    }

    /// A drawing on an object goes into a `.kreate` and comes back out.
    #[test]
    fn a_drawing_survives_a_save_and_a_load() {
        use kreate_scene::document::{Document, Object, ObjectData};
        let mut d = Drawing::new();
        let l = d.add_curve(Curve2D::Line {
            a: DVec2::ZERO,
            b: DVec2::new(0.05, 0.0),
        });
        d.constraints
            .push(Constraint::Horizontal(EntityRef::whole(l), None));
        d.foreign.push(r#"{"type":"hatch","id":9}"#.into());
        let mut doc = Document::new();
        let mut o = Object::new("Drawing", ObjectData::Empty);
        o.drawing = Some(Box::new(d.clone()));
        doc.add_object(o);
        let bytes = crate::container::write(&doc, &crate::container::Manifest::default(), &[])
            .expect("writes");
        let back = crate::container::read(std::io::Cursor::new(bytes))
            .expect("reads")
            .document;
        let (_, o) = back.objects().find(|(_, o)| o.name == "Drawing").unwrap();
        let got = o.drawing.as_deref().expect("its drawing");
        assert_eq!(got.entities, d.entities);
        assert_eq!(got.constraints, d.constraints);
        assert_eq!(got.foreign.len(), 1, "the hatch this build cannot read");
    }

    /// An entity of a kind this build does not know is kept, not lost.
    #[test]
    fn an_unknown_entity_is_kept_to_write_back() {
        let rec = DrawingRecord {
            entities: vec![
                serde_json::json!({"type": "hatch", "id": 7, "pattern": "ANSI31"}),
                serde_json::json!({"type": "point", "id": 3, "at": [1.0, 2.0]}),
            ],
            layers: Vec::new(),
            constraints: vec![serde_json::json!({"type": "parallel", "a": 3, "b": 99})],
            current_layer: 5,
            next: [1, 1],
            foreign: Vec::new(),
        };
        let (d, unread) = drawing_from(&rec);
        assert_eq!(d.entities.len(), 1);
        assert_eq!(
            unread.len(),
            2,
            "the hatch, and a constraint naming nothing"
        );
        assert_eq!(
            d.current_layer,
            LayerId(0),
            "a layer that is not there is not current"
        );
        assert_eq!(d.counters().0, 4, "never behind the ids it holds");
        // And written back as they came.
        let again = record(&d);
        assert_eq!(again.foreign.len(), 2);
        assert!(again.foreign.iter().any(|v| v["type"] == "hatch"));
    }
}
