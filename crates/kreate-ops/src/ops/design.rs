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
    registry.register::<MeasureAdd>();
    registry.register::<MeasureRemove>();
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
        run(&mut doc, "measure.remove", &[("index", PropertyValue::Int(0))]).unwrap();
        assert!(doc.design.measurements[0].angle().is_some());
        run(&mut doc, "measure.remove", &[("all", PropertyValue::Bool(true))]).unwrap();
        assert!(doc.design.measurements.is_empty());
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
