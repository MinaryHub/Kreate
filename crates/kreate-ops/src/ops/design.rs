//! The scene's design settings: the unit it is drawn in, and the dimensions
//! left on it. See `kreate_scene::design`.

use kreate_math::DVec3;
use kreate_scene::design::{LengthUnit, Measurement, Section};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<UnitsSet>();
    registry.register::<DensitySet>();
    registry.register::<SectionSet>();
    registry.register::<Mate>();
    registry.register::<MeasureAdd>();
    registry.register::<MeasureRemove>();
    registry.register::<ParameterSet>();
    registry.register::<ParameterRemove>();
    registry.register::<Bind>();
    registry.register::<Unbind>();
}

// --- parameters --------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParameterSetParams {
    /// The parameter's name: letters, digits and underscores, starting
    /// with a letter.
    pub name: String,
    /// What it is: a number, or an expression of other parameters —
    /// `40`, `width / 2`, `2 * pi`. Lengths in the scene's unit, angles in
    /// degrees.
    pub expression: String,
}

/// Makes or changes a named parameter, and puts everything bound to the
/// parameters where they now have it. Fusion's Change Parameters.
pub struct ParameterSet;

impl Operator for ParameterSet {
    type Params = ParameterSetParams;
    fn id() -> &'static str {
        "design.parameter_set"
    }
    fn label() -> &'static str {
        "Set Parameter"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let name = params.name.trim();
        if !kreate_scene::design::valid_parameter_name(name) {
            return Err(OpError::Failed(format!(
                "{name:?} cannot name a parameter: letters, digits and underscores, starting with a letter"
            )));
        }
        let expression = params.expression.trim();
        if expression.is_empty() {
            return Err(OpError::Failed(
                "a parameter needs a value or an expression".into(),
            ));
        }
        let d = doc.document_mut();
        // Read now, with the others, so a bad expression is refused rather
        // than kept and complained about on every frame.
        let mut trial = d.design.clone();
        match trial.parameters.iter_mut().find(|p| p.name == name) {
            Some(p) => p.expression = expression.to_string(),
            None => trial.parameters.push(kreate_scene::design::Parameter {
                name: name.to_string(),
                expression: expression.to_string(),
                value: 0.0,
            }),
        }
        let problems = trial.evaluate_parameters();
        if let Some((_, why)) = problems.iter().find(|(n, _)| n == name) {
            return Err(OpError::Failed(format!("{name} = {expression}: {why}")));
        }
        d.design = trial;
        d.apply_parameters();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParameterRemoveParams {
    pub name: String,
}

/// Takes a parameter away. Refused while another parameter reads it, as
/// Fusion refuses; a dimension bound to it alone is let go and keeps its
/// last value.
pub struct ParameterRemove;

impl Operator for ParameterRemove {
    type Params = ParameterRemoveParams;
    fn id() -> &'static str {
        "design.parameter_remove"
    }
    fn label() -> &'static str {
        "Remove Parameter"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let name = params.name.trim();
        let d = doc.document();
        if !d.design.parameters.iter().any(|p| p.name == name) {
            return Err(OpError::Failed(format!("no parameter called {name:?}")));
        }
        // Another parameter that reads this one would be left reading
        // nothing: refused, and said which.
        let mut without = d.design.clone();
        without.parameters.retain(|p| p.name != name);
        let readers: Vec<String> = without
            .parameters
            .iter()
            .filter(|p| without.evaluate(&p.expression).is_err())
            .map(|p| p.name.clone())
            .collect();
        if !readers.is_empty() {
            return Err(OpError::Failed(format!(
                "{name} is read by {}: change or remove that first",
                readers.join(", ")
            )));
        }
        // A binding that only this parameter could read is let go: the
        // dimension stays as it is, free again.
        let kept: Vec<bool> = without
            .bindings
            .iter()
            .map(|b| without.evaluate(&b.expression).is_ok())
            .collect();
        let mut keep = kept.into_iter();
        without.bindings.retain(|_| keep.next().unwrap_or(false));
        let d = doc.document_mut();
        d.design = without;
        d.apply_parameters();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct BindParams {
    /// A sketch's constraint or a body's feature.
    #[param(variants("sketch", "feature", "drawing"))]
    pub kind: u32,
    /// The sketch or body, by name; empty for the active object.
    pub object: String,
    /// The constraint's place in the sketch's list, or the feature's order.
    pub index: u32,
    /// What holds it: an expression of the parameters, or a number.
    pub expression: String,
}

/// Holds a dimension or a feature's amount to an expression of the
/// parameters: typing `width / 2` into a dimension.
pub struct Bind;

impl Operator for Bind {
    type Params = BindParams;
    fn id() -> &'static str {
        "design.bind"
    }
    fn label() -> &'static str {
        "Bind to Parameters"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let expression = params.expression.trim().to_string();
        d.design
            .evaluate(&expression)
            .map_err(|e| OpError::Failed(format!("{expression}: {e}")))?;
        let name = params.object.trim();
        let id = if name.is_empty() {
            d.selection.active
        } else {
            d.objects().find(|(_, o)| o.name == name).map(|(id, _)| id)
        }
        .ok_or_else(|| OpError::Failed("no such object".into()))?;
        let uid = d.object_uid(id).expect("an object");
        let o = d.object(id).expect("an object");
        let target = match params.kind {
            0 => {
                let c = o
                    .sketch
                    .get(params.index as usize)
                    .ok_or_else(|| OpError::Failed("no such constraint".into()))?;
                if !matches!(
                    c,
                    kreate_scene::sketch::Constraint::Distance { .. }
                        | kreate_scene::sketch::Constraint::Angle { .. }
                ) {
                    return Err(OpError::Failed("only a dimension can be bound".into()));
                }
                kreate_scene::design::Bound::SketchDimension {
                    sketch: uid,
                    index: params.index,
                }
            }
            2 => {
                let is_dim = o
                    .drawing
                    .as_deref()
                    .and_then(|d| d.entity(kreate_scene::drawing::EntityId(params.index)))
                    .is_some_and(|e| {
                        matches!(e.kind, kreate_scene::drawing::EntityKind::Dimension(_))
                    });
                if !is_dim {
                    return Err(OpError::Failed("no such dimension in the drawing".into()));
                }
                kreate_scene::design::Bound::DrawingDimension {
                    drawing: uid,
                    dim: params.index,
                }
            }
            _ => {
                let order = u64::from(params.index);
                if !o.features.iter().any(|f| f.order == order) {
                    return Err(OpError::Failed("no such feature".into()));
                }
                kreate_scene::design::Bound::FeatureAmount { body: uid, order }
            }
        };
        let d = doc.document_mut();
        d.design.bindings.retain(|b| b.target != target);
        d.design
            .bindings
            .push(kreate_scene::design::Binding { target, expression });
        d.apply_parameters();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct UnbindParams {
    #[param(variants("sketch", "feature", "drawing"))]
    pub kind: u32,
    pub object: String,
    pub index: u32,
}

/// Lets a dimension go: it keeps its value and can be changed again.
pub struct Unbind;

impl Operator for Unbind {
    type Params = UnbindParams;
    fn id() -> &'static str {
        "design.unbind"
    }
    fn label() -> &'static str {
        "Unbind from Parameters"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let name = params.object.trim();
        let id = if name.is_empty() {
            d.selection.active
        } else {
            d.objects().find(|(_, o)| o.name == name).map(|(id, _)| id)
        }
        .ok_or_else(|| OpError::Failed("no such object".into()))?;
        let uid = d.object_uid(id).expect("an object");
        let target = match params.kind {
            0 => kreate_scene::design::Bound::SketchDimension {
                sketch: uid,
                index: params.index,
            },
            2 => kreate_scene::design::Bound::DrawingDimension {
                drawing: uid,
                dim: params.index,
            },
            _ => kreate_scene::design::Bound::FeatureAmount {
                body: uid,
                order: u64::from(params.index),
            },
        };
        let d = doc.document_mut();
        let before = d.design.bindings.len();
        d.design.bindings.retain(|b| b.target != target);
        Ok(if d.design.bindings.len() < before {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MateParams {
    /// The part that moves, by its uid.
    pub uid: String,
    /// A point on its face and the face's outward normal, in the world.
    #[param(unit = "length")]
    pub from: [f64; 3],
    pub from_normal: [f64; 3],
    /// A point on the face it goes against, and that face's normal.
    #[param(unit = "length")]
    pub to: [f64; 3],
    pub to_normal: [f64; 3],
    /// A gap left between the faces.
    #[param(unit = "length")]
    pub offset: f64,
    /// The part it goes against, by its uid. Given, the moving part is
    /// attached to it — parented, where it now sits — so the two stay
    /// together when that part moves: the mate holds, as an assembly's does.
    pub target: String,
}

/// Puts one part's face flat against another's: turned so the faces look at
/// each other, moved so the two points meet — and, given the other part,
/// attaches it there, so the mate holds when that part moves.
pub struct Mate;

impl Operator for Mate {
    type Params = MateParams;
    fn id() -> &'static str {
        "object.mate"
    }
    fn label() -> &'static str {
        "Align Faces"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Ok(raw) = params.uid.trim().parse::<u64>() else {
            return Err(OpError::Failed("no part named".into()));
        };
        let d = doc.document();
        let Some(id) = d.object_by_uid(kreate_scene::Uid::from_raw(raw)) else {
            return Err(OpError::Failed("no such part".into()));
        };
        let (from, to) = (DVec3::from_array(params.from), DVec3::from_array(params.to));
        let nf = DVec3::from_array(params.from_normal).normalize_or_zero();
        let nt = DVec3::from_array(params.to_normal).normalize_or_zero();
        if nf == DVec3::ZERO || nt == DVec3::ZERO || !from.is_finite() || !to.is_finite() {
            return Err(OpError::Failed("a face needs a direction".into()));
        }
        // Turn the moving face to look back at the other, about its point,
        // then carry its point onto the other's, the gap away.
        let turn = kreate_math::DQuat::from_rotation_arc(nf, -nt);
        let to = to + nt * params.offset;
        let motion = kreate_math::DAffine3::from_translation(to)
            * kreate_math::DAffine3::from_quat(turn)
            * kreate_math::DAffine3::from_translation(-from);
        let object = d.object(id).expect("found");
        let above = d.world_transform(id) * object.transform.to_affine().inverse();
        let local = above.inverse() * motion * above * object.transform.to_affine();
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        if !translation.is_finite() || !rotation.is_finite() {
            return Err(OpError::Failed(
                "the parts cannot be put together that way".into(),
            ));
        }
        let target = params
            .target
            .trim()
            .parse::<u64>()
            .ok()
            .and_then(|raw| d.object_by_uid(kreate_scene::Uid::from_raw(raw)))
            .filter(|t| *t != id);
        let d = doc.document_mut();
        let o = d.object_mut(id).expect("found");
        o.transform.translation = translation;
        o.transform.rotation = rotation.normalize();
        o.transform.scale = scale;
        if let Some(target) = target {
            // Where it now sits, kept: parenting does not move it. A part
            // that would end up its own ancestor stays unattached.
            let _ = d.set_parent(id, Some(target));
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SectionSetParams {
    /// Cut the view.
    pub enabled: bool,
    /// Across which axis, or the work plane (the 3D cursor's Z).
    #[param(variants("x", "y", "z", "cursor"))]
    pub axis: u32,
    /// How far along the axis the plane is.
    #[param(unit = "length", soft_min = -10.0, soft_max = 10.0)]
    pub offset: f64,
    /// Keep the other side.
    pub flip: bool,
    /// Close the cut, so walls read as solid.
    pub cap: bool,
}

impl Default for SectionSetParams {
    fn default() -> Self {
        Self {
            enabled: false,
            axis: 0,
            offset: 0.0,
            flip: false,
            cap: true,
        }
    }
}

/// The section as the operator's parameters, for the panel.
pub fn section_params(doc: &Document) -> SectionSetParams {
    match doc.design.section {
        Some(s) => SectionSetParams {
            enabled: true,
            axis: s.axis,
            offset: s.offset,
            flip: s.flip,
            cap: s.cap,
        },
        None => SectionSetParams::default(),
    }
}

/// Turns the section view on or off, or moves its plane.
pub struct SectionSet;

impl Operator for SectionSet {
    type Params = SectionSetParams;
    fn id() -> &'static str {
        "scene.section_set"
    }
    fn label() -> &'static str {
        "Section View"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let section = (params.enabled && params.offset.is_finite()).then_some(Section {
            axis: params.axis.min(3),
            offset: params.offset,
            flip: params.flip,
            cap: params.cap,
        });
        if doc.document().design.section == section {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().design.section = section;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DensitySetParams {
    /// Kilograms per cubic metre: PLA about 1240, aluminium 2700, steel 7850.
    #[param(min = 0.0, soft_max = 20000.0)]
    pub density: f64,
}

/// What the parts are made of, for their mass.
pub struct DensitySet;

impl Operator for DensitySet {
    type Params = DensitySetParams;
    fn id() -> &'static str {
        "scene.density_set"
    }
    fn label() -> &'static str {
        "Set Density"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.density.is_finite()
            || params.density < 0.0
            || doc.document().design.density == params.density
        {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().design.density = params.density;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MeasureAddParams {
    /// Two points for a distance, three for the angle at the middle one:
    /// x, y, z each, in the world.
    pub points: Vec<f64>,
}

/// Leaves a dimension on the scene — what the Measure tool's clicks make.
pub struct MeasureAdd;

impl Operator for MeasureAdd {
    type Params = MeasureAddParams;
    fn id() -> &'static str {
        "measure.add"
    }
    fn label() -> &'static str {
        "Add Measurement"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let (chunks, rest) = params.points.as_chunks::<3>();
        let m = Measurement {
            points: chunks.iter().map(|c| DVec3::from_array(*c)).collect(),
        };
        if !rest.is_empty() || !m.is_valid() {
            return Err(OpError::Failed(
                "a measurement is two points, or three for an angle".into(),
            ));
        }
        doc.document_mut().design.measurements.push(m);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MeasureRemoveParams {
    /// Which one, in the order they were taken.
    pub index: u32,
    /// Every one of them instead.
    pub all: bool,
}

/// Takes a dimension off the scene, or all of them.
pub struct MeasureRemove;

impl Operator for MeasureRemove {
    type Params = MeasureRemoveParams;
    fn id() -> &'static str {
        "measure.remove"
    }
    fn label() -> &'static str {
        "Remove Measurement"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.design.measurements.is_empty()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let list = &doc.document().design.measurements;
        if list.is_empty() || (!params.all && params.index as usize >= list.len()) {
            return Ok(OpResult::Cancelled);
        }
        let list = &mut doc.document_mut().design.measurements;
        if params.all {
            list.clear();
        } else {
            list.remove(params.index as usize);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct UnitsSetParams {
    /// The unit lengths are shown and typed in.
    #[param(variants("metres", "centimetres", "millimetres", "inches", "feet"))]
    pub length: u32,
}

/// The scene's units as the operator's parameters, for the panel.
pub fn units_params(doc: &Document) -> UnitsSetParams {
    UnitsSetParams {
        length: doc.design.length_unit.index(),
    }
}

/// Sets the unit lengths are shown and typed in. Nothing in the model
/// changes: lengths are metres underneath whatever this says.
pub struct UnitsSet;

impl Operator for UnitsSet {
    type Params = UnitsSetParams;
    fn id() -> &'static str {
        "scene.units_set"
    }
    fn label() -> &'static str {
        "Set Units"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let unit = LengthUnit::from_index(params.length);
        if doc.document().design.length_unit == unit {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().design.length_unit = unit;
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};

    /// `width = 40` holds a sketch's side and an extrude's height; change
    /// it and both follow, through the file as well.
    #[test]
    fn parameters_hold_dimensions_and_features_and_follow_a_change() {
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        let mut doc = Document::new();
        doc.design.length_unit = kreate_scene::design::LengthUnit::Millimetres;
        let history = std::cell::RefCell::new(History::new());
        let ctx = Context::default();
        let run = |doc: &mut Document, id: &str, pairs: &[(&'static str, PropertyValue)]| {
            let mut p = PropertyMap::new();
            for (k, v) in pairs {
                p.set(k, v.clone());
            }
            history.borrow_mut().dispatch(&registry, doc, &ctx, id, &p)
        };
        let v = |x: f64| PropertyValue::Float(x);
        let t = |x: &str| PropertyValue::Text(x.to_string());
        // A 30 x 20 mm plate, its width dimensioned.
        run(&mut doc, "sketch.new", &[]).unwrap();
        let sketch = doc.selection.active.unwrap();
        run(
            &mut doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.03, 0.02, 0.0])),
                ("join", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        run(
            &mut doc,
            "sketch.constrain",
            &[
                ("kind", PropertyValue::Enum(0)),
                ("points", PropertyValue::Ints(vec![0, 1])),
                ("value", v(0.03)),
            ],
        )
        .unwrap();
        let dim = doc.object(sketch).unwrap().sketch.len() as i64 - 1;
        run(
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.01, 0.01])),
                ("distance", v(0.01)),
            ],
        )
        .unwrap();
        let body = doc
            .objects()
            .find(|(_, o)| !o.features.is_empty())
            .map(|(id, _)| id)
            .unwrap();
        let body_name = doc.object(body).unwrap().name.clone();
        let order = doc.object(body).unwrap().features[0].order;

        // The parameters, and what is held to them.
        run(
            &mut doc,
            "design.parameter_set",
            &[("name", t("width")), ("expression", t("40"))],
        )
        .unwrap();
        run(
            &mut doc,
            "design.parameter_set",
            &[("name", t("height")), ("expression", t("width / 4"))],
        )
        .unwrap();
        assert!(run(
            &mut doc,
            "design.parameter_set",
            &[("name", t("2bad")), ("expression", t("1"))]
        )
        .is_err());
        assert!(run(
            &mut doc,
            "design.parameter_set",
            &[("name", t("loop")), ("expression", t("nope + 1"))]
        )
        .is_err());
        doc.selection.clear();
        doc.selection.add(sketch);
        doc.selection.active = Some(sketch);
        run(
            &mut doc,
            "design.bind",
            &[
                ("kind", PropertyValue::Enum(0)),
                ("index", PropertyValue::Int(dim)),
                ("expression", t("width")),
            ],
        )
        .unwrap();
        run(
            &mut doc,
            "design.bind",
            &[
                ("kind", PropertyValue::Enum(1)),
                ("object", t(&body_name)),
                ("index", PropertyValue::Int(order as i64)),
                ("expression", t("height")),
            ],
        )
        .unwrap();
        let width = |d: &Document| {
            let p = crate::ops::sketch_draw::points(d, sketch);
            (p[1] - p[0]).length()
        };
        let height = |d: &Document| match d.object(body).unwrap().features[0].shape {
            kreate_scene::feature::Shape::Extrude { distance, .. } => distance,
            _ => unreachable!(),
        };
        assert!((width(&doc) - 0.04).abs() < 1e-6, "40 mm: {}", width(&doc));
        assert!(
            (height(&doc) - 0.01).abs() < 1e-9,
            "10 mm: {}",
            height(&doc)
        );

        // Changed: both follow.
        run(
            &mut doc,
            "design.parameter_set",
            &[("name", t("width")), ("expression", t("60"))],
        )
        .unwrap();
        assert!((width(&doc) - 0.06).abs() < 1e-6, "{}", width(&doc));
        assert!((height(&doc) - 0.015).abs() < 1e-9, "{}", height(&doc));
        // Undone: back, as one step.
        assert!(history.borrow_mut().undo(&mut doc));
        doc.apply_parameters();
        assert!((width(&doc) - 0.04).abs() < 1e-6, "{}", width(&doc));

        // Through the file.
        let bytes = kreate_io::container::write(&doc, &Default::default(), &[]).unwrap();
        let back = kreate_io::container::read(std::io::Cursor::new(bytes))
            .unwrap()
            .document;
        assert_eq!(back.design.parameters.len(), 2);
        assert_eq!(back.design.bindings.len(), 2);
        assert!((back.design.parameters[1].value - 10.0).abs() < 1e-9);

        // Let go: a number typed sets it and it no longer follows.
        run(
            &mut doc,
            "design.unbind",
            &[
                ("kind", PropertyValue::Enum(0)),
                ("index", PropertyValue::Int(dim)),
            ],
        )
        .unwrap();
        run(
            &mut doc,
            "design.parameter_set",
            &[("name", t("width")), ("expression", t("80"))],
        )
        .unwrap();
        assert!(
            (width(&doc) - 0.04).abs() < 1e-6,
            "left alone: {}",
            width(&doc)
        );
        // Removed: refused while `height` reads it; `height` gone, the
        // extrude is let go with its last height, and then `width` may go.
        assert!(run(&mut doc, "design.parameter_remove", &[("name", t("width"))]).is_err());
        run(
            &mut doc,
            "design.parameter_remove",
            &[("name", t("height"))],
        )
        .unwrap();
        assert!(
            doc.design.bindings.is_empty(),
            "the extrude read height, so it was let go"
        );
        assert!((height(&doc) - 0.02).abs() < 1e-9, "{}", height(&doc));
        run(&mut doc, "design.parameter_remove", &[("name", t("width"))]).unwrap();
        assert!(doc.design.parameters.is_empty());
    }

    #[test]
    fn measurements_are_added_and_removed() {
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        let mut doc = Document::new();
        let mut history = History::new();
        let ctx = Context::default();
        let mut run = |doc: &mut Document, id: &str, pairs: &[(&'static str, PropertyValue)]| {
            let mut p = PropertyMap::new();
            for (k, v) in pairs {
                p.set(k, v.clone());
            }
            history.dispatch(&registry, doc, &ctx, id, &p)
        };
        let two = PropertyValue::Floats(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.02]);
        run(&mut doc, "measure.add", &[("points", two)]).unwrap();
        assert_eq!(doc.design.measurements[0].length(), Some(0.02));
        let bad = PropertyValue::Floats(vec![1.0, 2.0]);
        assert!(run(&mut doc, "measure.add", &[("points", bad)]).is_err());
        let three = PropertyValue::Floats(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        run(&mut doc, "measure.add", &[("points", three)]).unwrap();
        assert_eq!(doc.design.measurements.len(), 2);
        run(
            &mut doc,
            "measure.remove",
            &[("index", PropertyValue::Int(0))],
        )
        .unwrap();
        assert!(doc.design.measurements[0].angle().is_some());
        run(
            &mut doc,
            "measure.remove",
            &[("all", PropertyValue::Bool(true))],
        )
        .unwrap();
        assert!(doc.design.measurements.is_empty());
    }

    #[test]
    fn a_face_goes_flat_against_another() {
        use kreate_scene::document::{Object, ObjectData};
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let a = doc.add_object(Object::new("A", ObjectData::Mesh(mesh)));
        doc.object_mut(a).unwrap().transform.translation = DVec3::new(5.0, 3.0, 0.0);
        let uid = doc.object_uid(a).unwrap().raw().to_string();
        let mut history = History::new();
        let mut p = PropertyMap::new();
        // A's +X face (its middle at (6, 3, 0)) onto the top of something at
        // the origin, two up: A ends standing on it, its +X face down.
        p.set("uid", PropertyValue::Text(uid));
        p.set("from", PropertyValue::Vec3([6.0, 3.0, 0.0]));
        p.set("from_normal", PropertyValue::Vec3([1.0, 0.0, 0.0]));
        p.set("to", PropertyValue::Vec3([0.0, 0.0, 2.0]));
        p.set("to_normal", PropertyValue::Vec3([0.0, 0.0, 1.0]));
        history
            .dispatch(&registry, &mut doc, &Context::default(), "object.mate", &p)
            .unwrap();
        let world = doc.world_transform(a);
        let centre = world.transform_point3(DVec3::ZERO);
        assert!(
            (centre - DVec3::new(0.0, 0.0, 3.0)).length() < 1e-9,
            "{centre}"
        );
        let face = world.transform_vector3(DVec3::X);
        assert!(
            (face - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-9,
            "{face}"
        );
    }

    #[test]
    fn the_unit_is_set_and_undone() {
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        let mut doc = Document::new();
        let mut history = History::new();
        let mut p = PropertyMap::new();
        p.set("length", PropertyValue::Enum(2));
        let ctx = Context::default();
        history
            .dispatch(&registry, &mut doc, &ctx, "scene.units_set", &p)
            .unwrap();
        assert_eq!(doc.design.length_unit, LengthUnit::Millimetres);
        assert!(history.undo(&mut doc));
        assert_eq!(doc.design.length_unit, LengthUnit::Metres);
    }
}
