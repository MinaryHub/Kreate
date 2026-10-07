//! The scene's design settings: the unit it is drawn in. See
//! `kreate_scene::design`.

use kreate_scene::design::LengthUnit;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<UnitsSet>();
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
