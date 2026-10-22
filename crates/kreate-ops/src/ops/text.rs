//! Text: a string's letters as a solid object, and as lines in a sketch.
//!
//! Both read the bundled font (`kreate_mesh::text::DEFAULT_FONT`, Latin and
//! all of Hangul). Neither reads a font from a file: an operator that
//! touches the disk does not replay, and the history is a script.

use kreate_math::{DQuat, DVec2, DVec3};
use kreate_mesh::text::{self, Align};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddText>();
    registry.register::<SketchText>();
}

fn align_of(v: u32) -> Align {
    match v {
        1 => Align::Center,
        2 => Align::Right,
        _ => Align::Left,
    }
}

/// The layout, or why there is none; characters the font lacks are said.
fn laid_out(body: &str, size: f64, align: u32, spacing: f64) -> Result<text::Layout, OpError> {
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
    let l = text::layout(
        text::DEFAULT_FONT,
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
        let l = laid_out(&params.text, params.size, params.align, params.line_spacing)?;
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
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = crate::ops::sketch_draw::active_sketch(doc.document())
            .ok_or_else(|| OpError::Failed("no sketch is being drawn".into()))?;
        let l = laid_out(&params.text, params.size, params.align, params.line_spacing)?;
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
}
