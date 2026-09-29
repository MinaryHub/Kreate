//! Drivers: a property whose value is an expression of other values.
//!
//! Blender's model, pared down to what an object's transform needs. A driver
//! sits on the object whose property it drives, names the property and its
//! component, and holds an expression over *variables*, each of which reads
//! one component of another object's transform. `frame` is always a variable.
//!
//! # Where they live, and why there
//!
//! On the object, as its action is, rather than in a list on the document. A
//! duplicate then carries its drivers along, a pasted object arrives with its
//! own, and deleting an object cannot leave a driver behind driving nothing.
//!
//! # When they run
//!
//! After the actions, on every frame, and after every edit — so a driven
//! property follows its variables while somebody drags them, not only when the
//! frame moves. A driver wins over a key on the same property, as it does in
//! Blender: the key is what the curve says and the driver is what the property
//! *is*.
//!
//! # Order
//!
//! A driver can read a property another driver writes, so the objects are
//! evaluated with each one's variables' targets first. A loop — A reads B and
//! B reads A — has no right order; the objects in it are evaluated in the
//! document's order after everything else, each reading what the other had,
//! and [`Document::driver_cycles`] says which they are.

use crate::document::{Document, ObjectId};
use crate::expression::{self, ExprError};
use kreate_reflect::Uid;

/// One driven component.
#[derive(Clone, Debug, PartialEq)]
pub struct Driver {
    /// Which property of its object: `location`, `rotation_euler` or `scale`.
    pub property: Box<str>,
    /// Which component of it.
    pub index: u8,
    /// See [`crate::expression`].
    pub expression: String,
    pub variables: Vec<DriverVariable>,
}

impl Driver {
    pub fn new(property: &str, index: u8, expression: impl Into<String>) -> Self {
        Self {
            property: property.into(),
            index,
            expression: expression.into(),
            variables: Vec::new(),
        }
    }
}

/// A name in a driver's expression, and the value it reads.
///
/// One component of one object's own transform — Blender's Single Property on
/// a transform, which is what nearly every rig reads.
#[derive(Clone, Debug, PartialEq)]
pub struct DriverVariable {
    pub name: String,
    /// The object read, by identity, so renaming it does not break the driver.
    /// `Uid::NONE` is a variable that has not been pointed at anything yet.
    pub target: Uid,
    pub property: Box<str>,
    pub index: u8,
}

impl DriverVariable {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), target: Uid::NONE, property: "location".into(), index: 0 }
    }
}

/// Why a driver could not be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub enum DriverError {
    Expression(ExprError),
    /// A variable points at nothing, or at an object that is gone.
    NoTarget { variable: String },
    /// A variable reads a property that is not a transform's.
    BadProperty { variable: String },
    /// A variable's name is not one an expression can say.
    BadName { variable: String },
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriverError::Expression(e) => write!(f, "{e}"),
            DriverError::NoTarget { variable } => write!(f, "'{variable}' reads nothing"),
            DriverError::BadProperty { variable } => write!(f, "'{variable}' reads no such property"),
            DriverError::BadName { variable } => write!(f, "'{variable}' is not a name"),
        }
    }
}

/// Whether `name` can appear in an expression as a variable.
pub fn is_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !expression::FUNCTIONS.iter().any(|(f, _)| *f == name)
}

impl Document {
    /// The value `object`'s driver number `index` gives, without writing it.
    pub fn evaluate_driver(&self, object: ObjectId, index: usize) -> Result<f64, DriverError> {
        let driver = self
            .object(object)
            .and_then(|o| o.drivers.get(index))
            .ok_or(DriverError::Expression(ExprError::Empty))?;
        let mut values = Vec::with_capacity(driver.variables.len());
        for v in &driver.variables {
            if !is_name(&v.name) {
                return Err(DriverError::BadName { variable: v.name.clone() });
            }
            let source = self
                .object_by_uid(v.target)
                .filter(|_| v.target != Uid::NONE)
                .and_then(|id| self.object(id))
                .ok_or_else(|| DriverError::NoTarget { variable: v.name.clone() })?;
            let value = crate::animation::read_object(&source.transform, &v.property)
                .and_then(|x| (v.index < 3).then(|| x[v.index as usize]))
                .ok_or_else(|| DriverError::BadProperty { variable: v.name.clone() })?;
            values.push((v.name.as_str(), value));
        }
        let frame = self.playback.frame;
        expression::evaluate(&driver.expression, &|n| {
            values.iter().find(|(name, _)| *name == n).map(|(_, v)| *v).or((n == "frame").then_some(frame))
        })
        .map_err(DriverError::Expression)
    }

    /// Runs every driver, in dependency order, writing what each gives.
    ///
    /// A driver that cannot be evaluated leaves its property as it was. Not an
    /// edit, as [`Document::animate`] is not: it is what the drivers say the
    /// properties are, and undoing it would only have it run again. Returns
    /// how many objects moved.
    pub fn drive(&mut self) -> usize {
        let order = self.driver_order().0;
        let mut moved = 0;
        for object in order {
            let count = self.object(object).map(|o| o.drivers.len()).unwrap_or(0);
            for i in 0..count {
                let Ok(value) = self.evaluate_driver(object, i) else { continue };
                let Some(o) = self.object_untouched_mut(object) else { continue };
                let (property, index) = (o.drivers[i].property.clone(), o.drivers[i].index as usize);
                if index > 2 {
                    continue;
                }
                let before = o.transform;
                match property.as_ref() {
                    "location" => o.transform.translation[index] = value,
                    "scale" => o.transform.scale[index] = value,
                    "rotation_euler" => {
                        let mut e = crate::animation::to_euler(o.transform.rotation);
                        e[index] = value;
                        let q = crate::animation::from_euler(e);
                        // Compared as rotations: the angles come back wrapped
                        // into ±π, so comparing them would call every driven
                        // spin past half a turn a change, every run.
                        if q.dot(o.transform.rotation).abs() < 1.0 - 1e-15 {
                            o.transform.rotation = q;
                        }
                    }
                    _ => {}
                }
                moved += usize::from(o.transform != before);
            }
        }
        if moved > 0 {
            self.touch_version();
        }
        moved
    }

    /// The objects in a loop of drivers reading each other, which have no
    /// order that is right.
    pub fn driver_cycles(&self) -> Vec<ObjectId> {
        self.driver_order().1
    }

    /// Driven objects with each after the objects it reads, then those in
    /// loops; and the ones in loops.
    fn driver_order(&self) -> (Vec<ObjectId>, Vec<ObjectId>) {
        let driven: Vec<ObjectId> =
            self.objects().filter(|(_, o)| !o.drivers.is_empty()).map(|(id, _)| id).collect();
        // What each driven object reads that is itself driven: the only edges
        // that constrain the order.
        let reads = |id: ObjectId| -> Vec<ObjectId> {
            let Some(o) = self.object(id) else { return Vec::new() };
            let mut out: Vec<ObjectId> = o
                .drivers
                .iter()
                .flat_map(|d| d.variables.iter())
                .filter_map(|v| self.object_by_uid(v.target).filter(|_| v.target != Uid::NONE))
                .filter(|t| *t != id && driven.contains(t))
                .collect();
            out.sort_unstable_by_key(|t| driven.iter().position(|d| d == t));
            out.dedup();
            out
        };
        let edges: Vec<Vec<ObjectId>> = driven.iter().map(|id| reads(*id)).collect();
        let mut done: Vec<ObjectId> = Vec::with_capacity(driven.len());
        // Kahn's, over a list small enough that the quadratic scan is the
        // cheap part.
        loop {
            let before = done.len();
            for (i, id) in driven.iter().enumerate() {
                if !done.contains(id) && edges[i].iter().all(|t| done.contains(t)) {
                    done.push(*id);
                }
            }
            if done.len() == before {
                break;
            }
        }
        let cycles: Vec<ObjectId> = driven.iter().copied().filter(|id| !done.contains(id)).collect();
        done.extend(cycles.iter().copied());
        (done, cycles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};

    fn doc_with(names: &[&str]) -> (Document, Vec<ObjectId>) {
        let mut doc = Document::new();
        let ids = names.iter().map(|n| doc.add_object(Object::new(*n, ObjectData::Empty))).collect();
        (doc, ids)
    }

    fn var(doc: &Document, name: &str, target: ObjectId, property: &str, index: u8) -> DriverVariable {
        DriverVariable { name: name.into(), target: doc.object_uid(target).unwrap(), property: property.into(), index }
    }

    #[test]
    fn a_driver_follows_its_variable_and_the_frame() {
        let (mut doc, ids) = doc_with(&["Handle", "Door"]);
        doc.object_mut(ids[0]).unwrap().transform.translation.x = 2.0;
        let v = var(&doc, "h", ids[0], "location", 0);
        let mut d = Driver::new("rotation_euler", 2, "h * 0.5 + frame / 100");
        d.variables.push(v);
        doc.object_mut(ids[1]).unwrap().drivers.push(d);
        doc.playback.frame = 10.0;
        assert_eq!(doc.drive(), 1);
        let z = crate::animation::to_euler(doc.object(ids[1]).unwrap().transform.rotation).z;
        assert!((z - 1.1).abs() < 1e-9, "the door turned to {z}");
        assert_eq!(doc.drive(), 0, "a second run with nothing changed moved something");
    }

    #[test]
    fn a_chain_is_evaluated_in_order_whatever_order_the_objects_are_in() {
        // C reads B reads A; C is first in the document. Evaluated in document
        // order, C would read B's value from before B's driver ran.
        let (mut doc, ids) = doc_with(&["C", "B", "A"]);
        doc.object_mut(ids[2]).unwrap().transform.translation.x = 3.0;
        let mut b = Driver::new("location", 0, "a * 2");
        b.variables.push(var(&doc, "a", ids[2], "location", 0));
        let mut c = Driver::new("location", 0, "b + 1");
        c.variables.push(var(&doc, "b", ids[1], "location", 0));
        doc.object_mut(ids[0]).unwrap().drivers.push(c);
        doc.object_mut(ids[1]).unwrap().drivers.push(b);
        doc.drive();
        assert_eq!(doc.object(ids[0]).unwrap().transform.translation.x, 7.0);
        assert!(doc.driver_cycles().is_empty());
    }

    #[test]
    fn a_loop_is_named_and_does_not_hang() {
        let (mut doc, ids) = doc_with(&["A", "B"]);
        let mut a = Driver::new("location", 0, "b + 1");
        a.variables.push(var(&doc, "b", ids[1], "location", 0));
        let mut b = Driver::new("location", 0, "a + 1");
        b.variables.push(var(&doc, "a", ids[0], "location", 0));
        doc.object_mut(ids[0]).unwrap().drivers.push(a);
        doc.object_mut(ids[1]).unwrap().drivers.push(b);
        doc.drive();
        assert_eq!(doc.driver_cycles().len(), 2);
    }

    #[test]
    fn a_broken_driver_leaves_its_property_alone_and_says_why() {
        let (mut doc, ids) = doc_with(&["Thing"]);
        doc.object_mut(ids[0]).unwrap().transform.translation.y = 4.0;
        let mut d = Driver::new("location", 1, "x + 1");
        d.variables.push(DriverVariable::new("x"));
        doc.object_mut(ids[0]).unwrap().drivers.push(d);
        assert_eq!(doc.drive(), 0);
        assert_eq!(doc.object(ids[0]).unwrap().transform.translation.y, 4.0);
        assert_eq!(doc.evaluate_driver(ids[0], 0), Err(DriverError::NoTarget { variable: "x".into() }));
        doc.object_mut(ids[0]).unwrap().drivers[0].variables.clear();
        assert_eq!(
            doc.evaluate_driver(ids[0], 0),
            Err(DriverError::Expression(ExprError::UnknownName("x".into())))
        );
    }

    #[test]
    fn a_name_is_a_name() {
        assert!(is_name("var"));
        assert!(is_name("_x2"));
        assert!(!is_name("2x"));
        assert!(!is_name("a b"));
        assert!(!is_name(""));
        assert!(!is_name("sin"), "a function's name would be shadowed by the call");
    }
}
