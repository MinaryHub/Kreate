//! Text: a string's letters as a solid object, and as lines in a sketch.
//!
//! Both set it in a font named by its parameter: the two the program
//! carries (Pretendard, Latin and all of Hangul, the default; IBM Plex Mono)
//! or one of the computer's (`kreate_mesh::text::fonts`). A font is named,
//! not embedded, so a script replays wherever that font is installed — and
//! where it is not, the text comes out in the default, said so, rather than
//! not at all.

use kreate_math::{DQuat, DVec2, DVec3};
use kreate_mesh::text::{self, Align};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddText>();
    registry.register::<SketchText>();
    registry.register::<DraftTextOnPath>();
}

fn align_of(v: u32) -> Align {
    match v {
        1 => Align::Center,
        2 => Align::Right,
        _ => Align::Left,
    }
}

/// The layout, or why there is none; characters the font lacks are said.
fn laid_out(
    font: &str,
    body: &str,
    size: f64,
    align: u32,
    spacing: f64,
) -> Result<text::Layout, OpError> {
    if body.trim().is_empty() {
        return Err(OpError::Failed("there is no text to write".into()));
    }
    if !(size > 0.0 && size.is_finite()) {
        return Err(OpError::Failed("text needs a size above zero".into()));
    }
    let spacing = if spacing > 0.0 && spacing.is_finite() {
        spacing
    } else {
        1.0
    };
    let (bytes, index) = match text::font_data(font) {
        Some(found) => found,
        None => {
            crate::report::say(format!(
                "there is no font called {font:?} here: set in {}",
                text::DEFAULT_FONT_NAME
            ));
            text::font_data("").expect("the default is bundled")
        }
    };
    let l = text::layout_face(
        &bytes,
        index,
        body,
        size as f32,
        align_of(align),
        spacing as f32,
    )
    .map_err(OpError::Failed)?;
    if !l.missing.is_empty() {
        let list: String = l.missing.iter().collect();
        crate::report::say(format!("the font has no letters for: {list}"));
    }
    if l.contours.is_empty() {
        return Err(OpError::Failed(
            "none of these letters are in the font".into(),
        ));
    }
    Ok(l)
}

// --- a text object ---------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddTextParams {
    /// What it says; a line break starts a new line.
    pub text: String,
    /// The font, by name: the program's own or one this computer has.
    #[param(unit = "font")]
    pub font: String,
    /// The font's size (its em): a capital stands about seven tenths of it.
    #[param(min = 0.0001, soft_max = 1.0, unit = "length")]
    pub size: f64,
    /// How thick the letters are; zero for flat letters.
    #[param(min = 0.0, soft_max = 0.1, unit = "length")]
    pub depth: f64,
    /// How each line sits on the object's origin.
    #[param(variants("left", "center", "right"))]
    pub align: u32,
    /// Lines this many of the font's line heights apart.
    #[param(min = 0.1, soft_max = 3.0)]
    pub line_spacing: f64,
    /// Where the first line's baseline starts (or is centred, or ends).
    #[param(unit = "length")]
    pub location: [f64; 3],
    /// How it is turned, as Euler XYZ; unturned it lies on the ground,
    /// read from above.
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddTextParams {
    fn default() -> Self {
        Self {
            text: "Text".into(),
            font: text::DEFAULT_FONT_NAME.into(),
            size: 0.02,
            depth: 0.002,
            align: 0,
            line_spacing: 1.0,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// Adds the text as a solid mesh object, its letters `depth` thick from the
/// object's origin up its Z. A plain mesh: cut into a part with a boolean,
/// stood on a face, extruded further in edit mode.
pub struct AddText;

impl Operator for AddText {
    type Params = AddTextParams;
    fn id() -> &'static str {
        "object.add_text"
    }
    fn label() -> &'static str {
        "Add Text"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let l = laid_out(
            &params.font,
            &params.text,
            params.size,
            params.align,
            params.line_spacing,
        )?;
        let mesh = text::solid(&l, params.depth.max(0.0) as f32);
        if mesh.num_faces() == 0 {
            return Err(OpError::Failed("the letters made no faces".into()));
        }
        let angles = DVec3::from_array(params.rotation);
        let turn = if angles.is_finite() {
            kreate_scene::animation::from_euler(angles)
        } else {
            DQuat::IDENTITY
        };
        let d = doc.document_mut();
        // Named for what it says, as Blender names a text by its body.
        let first: String = params
            .text
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(24)
            .collect();
        let name = d.unique_object_name(&format!("Text {}", first.trim()));
        let id = d.add_mesh_object(name, mesh);
        let o = d.object_mut(id).expect("just added");
        o.transform.translation = DVec3::from_array(params.location);
        o.transform.rotation = turn;
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        Ok(OpResult::Finished)
    }
}

// --- text in a sketch ------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SketchTextParams {
    /// What it says; a line break starts a new line.
    pub text: String,
    /// The font, by name: the program's own or one this computer has.
    #[param(unit = "font")]
    pub font: String,
    /// The font's size (its em): a capital stands about seven tenths of it.
    #[param(min = 0.0001, soft_max = 1.0, unit = "length")]
    pub size: f64,
    /// Where the first line's baseline starts (or is centred, or ends), in
    /// the sketch's plane.
    #[param(unit = "length")]
    pub at: [f64; 3],
    /// How each line sits on `at`.
    #[param(variants("left", "center", "right"))]
    pub align: u32,
    /// Lines this many of the font's line heights apart.
    #[param(min = 0.1, soft_max = 3.0)]
    pub line_spacing: f64,
}

impl Default for SketchTextParams {
    fn default() -> Self {
        Self {
            text: "Text".into(),
            font: text::DEFAULT_FONT_NAME.into(),
            size: 0.01,
            at: [0.0; 3],
            align: 1,
            line_spacing: 1.0,
        }
    }
}

/// Writes the letters' outlines into the active sketch as lines: closed
/// regions, extruded or cut like any other — on a part's face, engraved
/// into it or standing out of it. Fusion's sketch text, exploded.
pub struct SketchText;

impl Operator for SketchText {
    type Params = SketchTextParams;
    fn id() -> &'static str {
        "sketch.text"
    }
    fn label() -> &'static str {
        "Sketch Text"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        crate::ops::sketch_draw::active_sketch(doc).is_some()
            || crate::ops::draft::active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if crate::ops::draft::active_drawing(doc.document()).is_some() {
            let l = laid_out(
                &params.font,
                &params.text,
                params.size,
                params.align,
                params.line_spacing,
            )?;
            let at = DVec2::new(params.at[0], params.at[1]);
            let rings = l
                .contours
                .iter()
                .map(|c| c.iter().map(|p| at + p.as_dvec2()).collect())
                .collect();
            return crate::ops::draft::add_outlines(doc, rings);
        }
        let id = crate::ops::sketch_draw::active_sketch(doc.document())
            .ok_or_else(|| OpError::Failed("no sketch is being drawn".into()))?;
        let l = laid_out(
            &params.font,
            &params.text,
            params.size,
            params.align,
            params.line_spacing,
        )?;
        let at = DVec2::new(params.at[0], params.at[1]);
        let mut pts = crate::ops::sketch_draw::points(doc.document(), id);
        let mut lines = Vec::new();
        for c in &l.contours {
            let start = pts.len() as u32;
            pts.extend(c.iter().map(|p| at + p.as_dvec2()));
            let n = c.len() as u32;
            lines.extend((0..n).map(|i| [start + i, start + (i + 1) % n]));
        }
        crate::ops::sketch_draw::add_drawing(doc, id, pts, lines)
    }
}

// --- text along a curve ----------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftTextOnPathParams {
    /// The curve it follows, by its number in the drawing.
    pub path: u32,
    pub text: String,
    #[param(unit = "font")]
    pub font: String,
    /// The font's size (its em).
    #[param(min = 0.0001, soft_max = 1.0, unit = "length")]
    pub size: f64,
    /// Where along the curve it sits: from its start, centred, or ending at
    /// its end — round a closed curve, centred on its top.
    #[param(variants("start", "center", "end"))]
    pub align: u32,
    /// The letters stand on the other side of the curve, reading the other
    /// way: inside a closed curve rather than round its outside.
    pub flip: bool,
    /// The letters this far off the curve, up from it.
    #[param(soft_min = -0.1, soft_max = 0.1, unit = "length")]
    pub offset: f64,
}

impl Default for DraftTextOnPathParams {
    fn default() -> Self {
        Self {
            path: 0,
            text: "Text".into(),
            font: text::DEFAULT_FONT_NAME.into(),
            size: 0.01,
            align: 1,
            flip: false,
            offset: 0.0,
        }
    }
}

/// A curve as a run of points with the distance along it to each.
struct Run {
    points: Vec<DVec2>,
    along: Vec<f64>,
    /// Round and round: past the end is the start again.
    closed: bool,
}

impl Run {
    fn new(points: Vec<DVec2>, closed: bool) -> Option<Self> {
        let mut along = vec![0.0];
        for w in points.windows(2) {
            along.push(along.last().unwrap() + w[0].distance(w[1]));
        }
        (points.len() >= 2 && *along.last().unwrap() > 0.0).then_some(Self {
            points,
            along,
            closed,
        })
    }

    fn length(&self) -> f64 {
        *self.along.last().unwrap()
    }

    /// The point `s` along and the way the run goes there; past either end,
    /// on along the end's direction.
    fn at(&self, s: f64) -> (DVec2, DVec2) {
        let s = if self.closed {
            s.rem_euclid(self.length())
        } else {
            s
        };
        let n = self.points.len();
        let i = match self.along.partition_point(|a| *a <= s) {
            0 => 1,
            k if k >= n => n - 1,
            k => k,
        };
        let (a, b) = (self.points[i - 1], self.points[i]);
        let span = self.along[i] - self.along[i - 1];
        let dir = (b - a) / span.max(1e-300);
        (a + dir * (s - self.along[i - 1]), dir)
    }
}

/// Letters laid along a curve of the drawing, each point of their outlines
/// carried from the straight baseline on to the curve: Fusion's text on a
/// path, as closed outlines to extrude or cut.
pub struct DraftTextOnPath;

impl Operator for DraftTextOnPath {
    type Params = DraftTextOnPathParams;
    fn id() -> &'static str {
        "draft.text_on_path"
    }
    fn label() -> &'static str {
        "Text on Path"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        crate::ops::draft::active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let curve = crate::ops::draft::active_drawing(doc.document())
            .and_then(|id| doc.document().object(id)?.drawing.as_deref())
            .and_then(|d| {
                d.curve(kreate_scene::drawing::EntityId(params.path))
                    .cloned()
            })
            .ok_or_else(|| OpError::Failed("there is no curve to set the text along".into()))?;
        let tol = (curve.length() * 1e-4).max(1e-9);
        let mut points = curve.tessellate(tol);
        let closed = curve.is_closed();
        // Round a closed curve the letters stand outside it, so read
        // clockwise; flipped, inside and anticlockwise.
        let anticlockwise = closed && {
            let n = points.len();
            (0..n)
                .map(|i| points[i].perp_dot(points[(i + 1) % n]))
                .sum::<f64>()
                > 0.0
        };
        if anticlockwise != params.flip && closed || params.flip && !closed {
            points.reverse();
        }
        let run = Run::new(points, closed)
            .ok_or_else(|| OpError::Failed("the curve has no length to set text along".into()))?;
        let l = laid_out(
            &params.font,
            params.text.lines().next().unwrap_or(""),
            params.size,
            0,
            1.0,
        )?;
        let width = l.width as f64;
        let from = if closed {
            // From, round, or to the curve's top (its bottom, flipped, where
            // text inside reads upright).
            let top = (0..run.points.len())
                .max_by(|a, b| {
                    let (ya, yb) = (run.points[*a].y, run.points[*b].y);
                    let (ya, yb) = if params.flip { (-ya, -yb) } else { (ya, yb) };
                    ya.total_cmp(&yb)
                })
                .map(|i| run.along[i])
                .unwrap_or(0.0);
            match params.align {
                0 => top,
                2 => top - width,
                _ => top - width * 0.5,
            }
        } else {
            match params.align {
                0 => 0.0,
                2 => run.length() - width,
                _ => (run.length() - width) * 0.5,
            }
        };
        if width > run.length() {
            crate::report::say(
                "the text is longer than the curve: it runs on past its ends".to_string(),
            );
        }
        let rings = l
            .contours
            .iter()
            .map(|c| {
                c.iter()
                    .map(|p| {
                        let (at, dir) = run.at(from + p.x as f64);
                        at + dir.perp() * (p.y as f64 + params.offset)
                    })
                    .collect()
            })
            .collect();
        crate::ops::draft::add_outlines(doc, rings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        doc: &mut Document,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(r, doc, &Context { mode: Mode::Object }, id, &p)
    }

    fn setup() -> (OperatorRegistry, History, Document) {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        (r, History::new(), Document::new())
    }

    /// A text object is a closed solid at its place, named for what it says,
    /// selected; and the line replays from the history.
    #[test]
    fn a_text_object_is_a_closed_solid_where_it_was_put() {
        let (r, mut h, mut doc) = setup();
        run(
            &r,
            &mut h,
            &mut doc,
            "object.add_text",
            &[
                ("text", PropertyValue::Text("Kreate 한글".into())),
                ("size", PropertyValue::Float(0.02)),
                ("depth", PropertyValue::Float(0.003)),
                ("location", PropertyValue::Vec3([0.1, 0.0, 0.0])),
            ],
        )
        .unwrap();
        let id = doc.selection.active.expect("selected");
        let o = doc.object(id).unwrap();
        assert_eq!(o.name, "Text Kreate 한글");
        let mesh = doc.mesh(o.mesh().unwrap()).unwrap();
        let mp = kreate_mesh::measure::mass_properties(mesh, |p| p);
        assert!(mp.closed && mp.volume > 0.0);
        let (lo, hi) = doc.object_world_bounds(id).unwrap();
        assert!(
            lo.x >= 0.1 - 1e-3 && (hi.z - 0.003).abs() < 1e-6,
            "{lo} {hi}"
        );
        // The history line reads back to the same call.
        let line = h.last().unwrap().to_script_call();
        assert!(
            line.contains("object.add_text") && line.contains("한글"),
            "{line}"
        );
        assert!(h.undo(&mut doc));
        assert_eq!(doc.object_count(), 0);
    }

    /// The font is a parameter: the mono face sets wider letters than the
    /// default, and a font this computer lacks falls back to the default.
    #[test]
    fn the_font_is_chosen_by_name() {
        let (r, mut h, mut doc) = setup();
        let width = |doc: &Document| {
            let id = doc.selection.active.unwrap();
            let (lo, hi) = doc.object_world_bounds(id).unwrap();
            hi.x - lo.x
        };
        let mut add = |doc: &mut Document, font: &str| {
            run(
                &r,
                &mut h,
                doc,
                "object.add_text",
                &[
                    ("text", PropertyValue::Text("iiii".into())),
                    ("font", PropertyValue::Text(font.into())),
                ],
            )
            .unwrap();
        };
        add(&mut doc, text::DEFAULT_FONT_NAME);
        let sans = width(&doc);
        add(&mut doc, text::MONO_FONT_NAME);
        let mono = width(&doc);
        assert!(
            mono > sans * 1.5,
            "a monospaced i is as wide as an m: {mono} {sans}"
        );
        add(&mut doc, "No Such Font 12345");
        assert!(
            (width(&doc) - sans).abs() < 1e-9,
            "the default, when the font is missing"
        );
    }

    /// Empty text is refused with a reason rather than making nothing.
    #[test]
    fn empty_text_is_refused() {
        let (r, mut h, mut doc) = setup();
        let err = run(
            &r,
            &mut h,
            &mut doc,
            "object.add_text",
            &[("text", PropertyValue::Text("  ".into()))],
        );
        assert!(err.is_err());
        assert_eq!(doc.object_count(), 0);
    }

    /// Text on a part's face, extruded from the sketch, stands on the face:
    /// the letters as regions the extrude finds without being told where.
    #[test]
    fn sketch_text_on_a_face_extrudes_into_raised_letters() {
        let (r, mut h, mut doc) = setup();
        let part = doc.add_mesh_object("Part", kreate_mesh::primitives::cube(0.04));
        doc.selection.add(part);
        doc.selection.active = Some(part);
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[
                ("plane", PropertyValue::Enum(4)),
                ("at", PropertyValue::Vec3([0.0, 0.0, 0.02])),
            ],
        )
        .unwrap();
        let sketch = doc.selection.active.unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.text",
            &[
                ("text", PropertyValue::Text("A".into())),
                ("size", PropertyValue::Float(0.02)),
                ("at", PropertyValue::Vec3([0.0, -0.005, 0.0])),
            ],
        )
        .unwrap();
        let regions = kreate_scene::feature::sketch_regions(&doc, sketch);
        let picked = kreate_scene::feature::profile_seeds(&doc, sketch);
        // The face's frame, the A's body and its counter: only the body.
        assert!(regions.len() >= 3, "{}", regions.len());
        assert_eq!(picked.len(), 1, "{picked:?}");
        let seeds: Vec<f64> = picked.iter().flat_map(|s| [s[0], s[1]]).collect();
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(seeds)),
                ("distance", PropertyValue::Float(0.002)),
                ("operation", PropertyValue::Enum(1)),
                ("target", PropertyValue::Text("Part".into())),
            ],
        )
        .unwrap();
        let mut cache = kreate_scene::modifier::ModifierCache::new();
        let ev = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
        let (_, joined) = ev.iter().find(|(i, _)| *i == part).unwrap();
        let mp = kreate_mesh::measure::mass_properties(joined, |p| p);
        assert!(
            mp.volume > 0.04f64.powi(3) * 1.0001,
            "the letter stands out: {}",
            mp.volume
        );
        assert!(
            mp.volume < 0.04f64.powi(3) * 1.01,
            "a letter, not a slab: {}",
            mp.volume
        );
    }

    fn path_text_points(doc: &Document, first: u32) -> Vec<DVec2> {
        let d = doc
            .object(crate::ops::draft::active_drawing(doc).unwrap())
            .unwrap()
            .drawing
            .as_deref()
            .unwrap();
        d.entities
            .iter()
            .filter(|e| e.id.0 >= first)
            .filter_map(|e| d.curve(e.id))
            .flat_map(|c| match c {
                kreate_geom2d::Curve2D::Polyline { verts, .. } => {
                    verts.iter().map(|v| v.0).collect()
                }
                _ => Vec::new(),
            })
            .collect()
    }

    /// Text on a circle stands on it, its letters' height out from it — or
    /// in from it, flipped; on a line it is the text as written, centred.
    #[test]
    fn text_on_a_path_follows_the_curve() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "draft.new", &[]).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.circle",
            &[
                ("centre", PropertyValue::Vec3([0.0; 3])),
                ("radius", PropertyValue::Float(0.05)),
            ],
        )
        .unwrap();
        let circle = path_text_points(&doc, 0).len();
        assert_eq!(circle, 0);
        let first = |doc: &Document| {
            doc.object(crate::ops::draft::active_drawing(doc).unwrap())
                .unwrap()
                .drawing
                .as_deref()
                .unwrap()
                .counters()
                .0
        };
        let circle_id = first(&doc) - 1;
        for flip in [false, true] {
            let from = first(&doc);
            run(
                &r,
                &mut h,
                &mut doc,
                "draft.text_on_path",
                &[
                    ("path", PropertyValue::Int(circle_id as i64)),
                    ("text", PropertyValue::Text("KREATE".into())),
                    ("size", PropertyValue::Float(0.01)),
                    ("flip", PropertyValue::Bool(flip)),
                ],
            )
            .unwrap();
            let pts = path_text_points(&doc, from);
            assert!(pts.len() > 20);
            // Centred on the top, or flipped on the bottom, reading upright.
            let mean = pts.iter().copied().sum::<DVec2>() / pts.len() as f64;
            assert!(
                mean.x.abs() < 0.005 && (mean.y > 0.04) != flip && (mean.y < -0.04) == flip,
                "{mean}"
            );
            let (left, right) = (
                pts.iter().map(|p| p.x).fold(f64::MAX, f64::min),
                pts.iter().map(|p| p.x).fold(f64::MIN, f64::max),
            );
            assert!(left < -0.01 && right > 0.01);
            for p in &pts {
                let off = p.length() - 0.05;
                if flip {
                    assert!((-0.0101..=0.0001).contains(&off), "{off}");
                } else {
                    assert!((-0.0001..=0.0101).contains(&off), "{off}");
                }
            }
        }
        // Along a straight line: the letters as written, centred on it.
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.line",
            &[("points", PropertyValue::Floats(vec![0.2, 0.0, 0.4, 0.0]))],
        )
        .unwrap();
        let line_id = first(&doc) - 1;
        let from = first(&doc);
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.text_on_path",
            &[
                ("path", PropertyValue::Int(line_id as i64)),
                ("text", PropertyValue::Text("KREATE".into())),
                ("size", PropertyValue::Float(0.01)),
            ],
        )
        .unwrap();
        let pts = path_text_points(&doc, from);
        let l = laid_out(text::DEFAULT_FONT_NAME, "KREATE", 0.01, 0, 1.0).unwrap();
        let start = 0.3 - l.width as f64 * 0.5;
        let want: Vec<DVec2> = l
            .contours
            .iter()
            .flatten()
            .map(|p| DVec2::new(start + p.x as f64, p.y as f64))
            .collect();
        let (min_x, max_x) = pts
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.x), b.max(p.x)));
        let (want_min, want_max) = want
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.x), b.max(p.x)));
        assert!((min_x - want_min).abs() < 1e-6 && (max_x - want_max).abs() < 1e-6);
        // No curve there, no text.
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "draft.text_on_path",
            &[("path", PropertyValue::Int(9999))],
        )
        .is_err());
    }
}
