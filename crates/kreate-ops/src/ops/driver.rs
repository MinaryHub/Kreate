//! Adding, editing and removing drivers.
//!
//! Every change to a driver is one of these, so a driver undoes and replays as
//! a key does. A driver is named by its object's identity and the component it
//! drives — `12/location/2` — the text [`driver_target`] writes, because an
//! arena id means nothing to a replay and a name changes when somebody renames.

use kreate_ops_derive::OpParams;
use kreate_scene::driver::{Driver, DriverVariable};
use kreate_scene::{DocMut, Document, Uid};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddDriver>();
    registry.register::<RemoveDriver>();
    registry.register::<SetDriverExpression>();
    registry.register::<AddDriverVariable>();
    registry.register::<SetDriverVariable>();
    registry.register::<RemoveDriverVariable>();
}

/// The properties a driver can drive, and a variable read.
pub const DRIVABLE: [&str; 3] = ["location", "rotation_euler", "scale"];

/// One driven component, as text.
pub fn driver_target(object_uid: u64, property: &str, index: u8) -> String {
    super::anim::curve_target(object_uid, property, index)
}

/// Where a target's driver is: the object, and the driver's place on it —
/// `None` for a component nothing drives yet.
fn find(doc: &Document, target: &str) -> Option<(kreate_scene::ObjectId, String, u8, Option<usize>)> {
    let (uid, property, index) = super::anim::parse_curve(target)?;
    if !DRIVABLE.contains(&property.as_str()) || index > 2 {
        return None;
    }
    let object = doc.object_by_uid(Uid::from_raw(uid)).filter(|_| uid != 0)?;
    let at = doc
        .object(object)?
        .drivers
        .iter()
        .position(|d| d.property.as_ref() == property && d.index == index);
    Some((object, property, index, at))
}

/// The driver a target names, to change.
fn driver_mut<'a>(doc: &'a mut DocMut<'_>, target: &str) -> Option<&'a mut Driver> {
    let (object, _, _, at) = find(doc.document(), target)?;
    let at = at?;
    doc.document_mut().object_mut(object)?.drivers.get_mut(at)
}

fn any_object(doc: &Document) -> bool {
    doc.objects().next().is_some()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TargetParams {
    /// The driven component, as [`driver_target`] writes it.
    pub target: String,
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddDriverParams {
    pub target: String,
    /// Empty for the property's value now, so adding a driver moves nothing.
    pub expression: String,
}

/// Puts a driver on one component of an object's transform.
pub struct AddDriver;

impl Operator for AddDriver {
    type Params = AddDriverParams;
    fn id() -> &'static str {
        "anim.driver_add"
    }
    fn label() -> &'static str {
        "Add Driver"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_object(doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((object, property, index, None)) = find(doc.document(), &params.target) else {
            return Ok(OpResult::Cancelled);
        };
        let expression = if params.expression.trim().is_empty() {
            // The value now, written so that it reads back as the same number:
            // `{}` on an f64 is the shortest text that parses to it exactly.
            let now = doc
                .document()
                .object(object)
                .and_then(|o| kreate_scene::animation::read_object(&o.transform, &property))
                .map(|v| v[index as usize])
                .unwrap_or(0.0);
            format!("{now}")
        } else {
            params.expression.clone()
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        o.drivers.push(Driver::new(&property, index, expression));
        Ok(OpResult::Finished)
    }
}

/// Takes a driver off. The property keeps the value the driver last gave it.
pub struct RemoveDriver;

impl Operator for RemoveDriver {
    type Params = TargetParams;
    fn id() -> &'static str {
        "anim.driver_remove"
    }
    fn label() -> &'static str {
        "Remove Driver"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_object(doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((object, _, _, Some(at))) = find(doc.document(), &params.target) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        o.drivers.remove(at);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ExpressionParams {
    pub target: String,
    pub expression: String,
}

/// Rewrites a driver's expression.
///
/// Stored whether or not it parses: a half-written expression is where
/// somebody is, and the panel says what is wrong with it rather than refusing
/// to keep it.
pub struct SetDriverExpression;

impl Operator for SetDriverExpression {
    type Params = ExpressionParams;
    fn id() -> &'static str {
        "anim.driver_expression"
    }
    fn label() -> &'static str {
        "Set Driver Expression"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_object(doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let unchanged = find(doc.document(), &params.target)
            .and_then(|(o, _, _, at)| Some(doc.document().object(o)?.drivers.get(at?)?.expression == params.expression));
        if unchanged != Some(false) {
            return Ok(OpResult::Cancelled);
        }
        let Some(d) = driver_mut(doc, &params.target) else {
            return Ok(OpResult::Cancelled);
        };
        d.expression = params.expression.clone();
        Ok(OpResult::Finished)
    }
}

/// Gives a driver a new variable, named `var`, then `var_1` and on.
pub struct AddDriverVariable;

impl Operator for AddDriverVariable {
    type Params = TargetParams;
    fn id() -> &'static str {
        "anim.driver_variable_add"
    }
    fn label() -> &'static str {
        "Add Driver Variable"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_object(doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(d) = driver_mut(doc, &params.target) else {
            return Ok(OpResult::Cancelled);
        };
        let taken = |n: &str| d.variables.iter().any(|v| v.name == n);
        let name = std::iter::once("var".to_string())
            .chain((1..).map(|i| format!("var_{i}")))
            .find(|n| !taken(n))
            .expect("the names run out after the integers do");
        d.variables.push(DriverVariable::new(name));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct VariableParams {
    pub target: String,
    /// Which of the driver's variables, counting from zero.
    #[param(min = 0, unit = "count")]
    pub variable: u32,
    pub name: String,
    /// The object it reads, by identity; empty for none.
    pub object: String,
    #[param(variants("location", "rotation_euler", "scale"))]
    pub property: u32,
    #[param(min = 0, max = 2, unit = "count")]
    pub index: u32,
}

/// Sets everything about one variable at once.
///
/// One operator for the four fields rather than four, because the panel shows
/// them as one row and a person changes one at a time: each change is one
/// step either way, and one operator is one line in a script.
pub struct SetDriverVariable;

impl Operator for SetDriverVariable {
    type Params = VariableParams;
    fn id() -> &'static str {
        "anim.driver_variable_set"
    }
    fn label() -> &'static str {
        "Set Driver Variable"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_object(doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let target = match params.object.trim() {
            "" => Uid::NONE,
            text => match text.parse::<u64>() {
                Ok(raw) => Uid::from_raw(raw),
                Err(_) => return Ok(OpResult::Cancelled),
            },
        };
        let value = DriverVariable {
            name: params.name.clone(),
            target,
            property: DRIVABLE[(params.property as usize).min(2)].into(),
            index: params.index.min(2) as u8,
        };
        let Some(d) = driver_mut(doc, &params.target) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(v) = d.variables.get_mut(params.variable as usize) else {
            return Ok(OpResult::Cancelled);
        };
        if *v == value {
            return Ok(OpResult::Cancelled);
        }
        *v = value;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct VariableIndexParams {
    pub target: String,
    #[param(min = 0, unit = "count")]
    pub variable: u32,
}

/// Takes one variable off a driver.
pub struct RemoveDriverVariable;

impl Operator for RemoveDriverVariable {
    type Params = VariableIndexParams;
    fn id() -> &'static str {
        "anim.driver_variable_remove"
    }
    fn label() -> &'static str {
        "Remove Driver Variable"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_object(doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(d) = driver_mut(doc, &params.target) else {
            return Ok(OpResult::Cancelled);
        };
        if (params.variable as usize) >= d.variables.len() {
            return Ok(OpResult::Cancelled);
        }
        d.variables.remove(params.variable as usize);
        Ok(OpResult::Finished)
    }
}
