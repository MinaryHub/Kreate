//! What a scene keeps for being used as a design rather than a picture: the
//! unit it is drawn in, and — as they arrive — the measurements left on it,
//! the material it is made of, the plane it is cut by.
//!
//! The scene's, as Blender keeps its units: saved with the file, set by
//! operators so a change is one undo step.

pub use kreate_reflect::LengthUnit;

use crate::Document;
use kreate_math::DVec2;
use kreate_math::{DQuat, DVec3};
use kreate_reflect::Uid;

/// A section view: the scene cut by a plane, the far side taken away, and
/// the cut capped so walls read as solid. What is drawn, not what is saved:
/// the meshes are untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section {
    /// Which plane: across X, Y or Z, or 3 for the work plane (the 3D
    /// cursor's Z).
    pub axis: u32,
    /// How far along the plane's normal from the origin (or the cursor).
    pub offset: f64,
    /// Keep the other side.
    pub flip: bool,
    /// Close the cut.
    pub cap: bool,
}

impl Section {
    /// The plane in the world: a point on it, and the normal pointing at the
    /// side taken away.
    pub fn plane(&self, cursor: DVec3, cursor_rotation: DQuat) -> (DVec3, DVec3) {
        let (origin, normal) = match self.axis {
            0 => (DVec3::ZERO, DVec3::X),
            1 => (DVec3::ZERO, DVec3::Y),
            3 => (cursor, cursor_rotation * DVec3::Z),
            _ => (DVec3::ZERO, DVec3::Z),
        };
        let normal = if self.flip { -normal } else { normal };
        let along = if self.flip { -self.offset } else { self.offset };
        (origin + normal * along, normal)
    }
}

/// A dimension left on the scene: two points and the distance between
/// them, or three and the angle at the middle one. In the world, so it stays
/// where it was measured when what it measured moves — as a ruler laid on a
/// drawing does.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub points: Vec<DVec3>,
}

impl Measurement {
    /// The distance from the first point to the last, for two.
    pub fn length(&self) -> Option<f64> {
        match self.points.as_slice() {
            [a, b] => Some(a.distance(*b)),
            _ => None,
        }
    }

    /// The angle at the middle point, in radians, for three.
    pub fn angle(&self) -> Option<f64> {
        match self.points.as_slice() {
            [a, b, c] => {
                let (u, v) = (*a - *b, *c - *b);
                (u.length() > 0.0 && v.length() > 0.0).then(|| u.angle_between(v))
            }
            _ => None,
        }
    }

    /// Two or three finite points: anything else is not a measurement.
    pub fn is_valid(&self) -> bool {
        matches!(self.points.len(), 2 | 3) && self.points.iter().all(|p| p.is_finite())
    }
}

/// Water's density, kilograms per cubic metre: what a new scene assumes a
/// part is made of until it is told.
pub const DEFAULT_DENSITY: f64 = 1000.0;

/// The scene's design settings.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignSettings {
    /// The unit lengths are shown and typed in. Lengths are stored in metres
    /// whatever this says; see [`LengthUnit`].
    pub length_unit: LengthUnit,
    /// The dimensions left on the scene, in the order they were taken.
    pub measurements: Vec<Measurement>,
    /// What the parts are made of, in kilograms per cubic metre, for their
    /// mass: PLA is about 1240, aluminium 2700, steel 7850.
    pub density: f64,
    /// The section view, when it is on.
    pub section: Option<Section>,
    /// Named numbers a design is set up by — `width = 40`, `height = width /
    /// 2` — in the scene's length unit for lengths and degrees for angles,
    /// each an expression of the ones before it. Fusion's user parameters.
    pub parameters: Vec<Parameter>,
    /// Dimensions and feature amounts held to expressions of the parameters:
    /// change a parameter and they follow.
    pub bindings: Vec<Binding>,
}

/// One named number.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub expression: String,
    /// What the expression came to when it was last evaluated.
    pub value: f64,
}

/// What a binding holds to an expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Bound {
    /// A sketch's constraint, by the sketch's identity and the constraint's
    /// place in its list: a distance or an angle.
    SketchDimension { sketch: Uid, index: u32 },
    /// A body's feature, by the body's identity and the feature's order:
    /// an extrude's distance, a revolve's angle, a round's radius.
    FeatureAmount { body: Uid, order: u64 },
}

/// A dimension held to an expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub target: Bound,
    pub expression: String,
}

/// Whether `name` can name a parameter: a letter or underscore, then
/// letters, digits and underscores — what the expression parser reads as a
/// name, and not one it keeps for itself.
pub fn valid_parameter_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(name, "pi" | "e" | "frame")
}

impl DesignSettings {
    /// Every parameter's value, each expression read with the others' values
    /// — in list order, then again until nothing changes, so one may name
    /// another listed after it. A parameter whose expression does not read
    /// keeps the value it had.
    pub fn evaluate_parameters(&mut self) -> Vec<(String, String)> {
        let mut problems = Vec::new();
        for _ in 0..self.parameters.len().max(1) {
            let mut changed = false;
            problems.clear();
            for i in 0..self.parameters.len() {
                let values: Vec<(String, f64)> = self
                    .parameters
                    .iter()
                    .filter(|p| p.name != self.parameters[i].name)
                    .map(|p| (p.name.clone(), p.value))
                    .collect();
                let p = &self.parameters[i];
                match crate::expression::evaluate(&p.expression, &|n| {
                    values.iter().find(|(name, _)| name == n).map(|(_, v)| *v)
                }) {
                    Ok(v) => {
                        if (v - p.value).abs() > 1e-12 || !p.value.is_finite() {
                            self.parameters[i].value = v;
                            changed = true;
                        }
                    }
                    Err(e) => problems.push((p.name.clone(), e.to_string())),
                }
            }
            if !changed {
                break;
            }
        }
        problems
    }

    /// What `expression` comes to with the parameters as they are.
    pub fn evaluate(&self, expression: &str) -> Result<f64, crate::expression::ExprError> {
        crate::expression::evaluate(expression, &|n| {
            self.parameters
                .iter()
                .find(|p| p.name == n)
                .map(|p| p.value)
        })
    }

    /// The binding on `target`, if there is one.
    pub fn binding(&self, target: &Bound) -> Option<&Binding> {
        self.bindings.iter().find(|b| &b.target == target)
    }
}

impl Document {
    /// Puts every bound dimension and feature where the parameters have it,
    /// and solves the sketches that moved. Like the drivers, not an edit:
    /// what the parameters say the dimensions are. Returns how many changed.
    pub fn apply_parameters(&mut self) -> usize {
        if self.design.bindings.is_empty() {
            return 0;
        }
        self.design.evaluate_parameters();
        let unit = self.design.length_unit.metres();
        let mut changed = 0;
        let mut solve: Vec<crate::document::ObjectId> = Vec::new();
        let bindings = self.design.bindings.clone();
        for b in &bindings {
            let Ok(v) = self.design.evaluate(&b.expression) else {
                continue;
            };
            match b.target {
                Bound::SketchDimension { sketch, index } => {
                    let Some(id) = self.object_by_uid(sketch) else {
                        continue;
                    };
                    let Some(o) = self.object_untouched_mut(id) else {
                        continue;
                    };
                    let Some(c) = o.sketch.get_mut(index as usize) else {
                        continue;
                    };
                    let want = match c {
                        crate::sketch::Constraint::Distance { .. } => v * unit,
                        crate::sketch::Constraint::Angle { .. } => v.to_radians(),
                        _ => continue,
                    };
                    let value = match c {
                        crate::sketch::Constraint::Distance { value, .. }
                        | crate::sketch::Constraint::Angle { value, .. } => value,
                        _ => continue,
                    };
                    if (*value - want).abs() > 1e-12 {
                        *value = want;
                        changed += 1;
                        if !solve.contains(&id) {
                            solve.push(id);
                        }
                    }
                }
                Bound::FeatureAmount { body, order } => {
                    let Some(id) = self.object_by_uid(body) else {
                        continue;
                    };
                    let Some(o) = self.object_untouched_mut(id) else {
                        continue;
                    };
                    let Some(f) = o.features.iter_mut().find(|f| f.order == order) else {
                        continue;
                    };
                    use crate::feature::Shape;
                    let slot = match &mut f.shape {
                        Shape::Extrude { distance, .. } => (distance, v * unit),
                        Shape::Revolve { angle, .. } => (angle, v.to_radians()),
                        Shape::Round { radius, .. } => (radius, v * unit),
                        Shape::Shell { thickness, .. } => (thickness, v * unit),
                        _ => continue,
                    };
                    if (*slot.0 - slot.1).abs() > 1e-12 {
                        *slot.0 = slot.1;
                        changed += 1;
                    }
                }
            }
        }
        for id in solve {
            self.solve_sketch(id);
        }
        if changed > 0 {
            self.touch_version();
        }
        changed
    }

    /// Solves a sketch's constraints and writes its points where they put
    /// them, as the sketch tools do after an edit.
    pub fn solve_sketch(&mut self, id: crate::document::ObjectId) -> Option<crate::sketch::Solve> {
        let (mesh_id, constraints) = {
            let o = self.object(id)?;
            (o.mesh()?, o.sketch.clone())
        };
        let mesh = self.mesh(mesh_id)?;
        let mut points: Vec<DVec2> = mesh
            .positions()
            .iter()
            .map(|p| DVec2::new(p.x as f64, p.y as f64))
            .collect();
        let faces: Vec<Vec<u32>> = mesh.faces().map(|f| f.to_vec()).collect();
        let result = crate::sketch::solve(&mut points, &constraints);
        let positions: Vec<kreate_math::Vec3> = points
            .iter()
            .map(|p| kreate_math::Vec3::new(p.x as f32, p.y as f32, 0.0))
            .collect();
        *self.mesh_mut(mesh_id)? = kreate_mesh::MeshData::from_faces(positions, &faces);
        Some(result)
    }
}

impl Default for DesignSettings {
    fn default() -> Self {
        Self {
            length_unit: LengthUnit::default(),
            measurements: Vec::new(),
            density: DEFAULT_DENSITY,
            section: None,
            parameters: Vec::new(),
            bindings: Vec::new(),
        }
    }
}

impl DesignSettings {
    /// Nothing set: what a file without design settings reads as.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measurement_reads_a_length_or_an_angle() {
        let d = Measurement {
            points: vec![DVec3::ZERO, DVec3::new(3.0, 4.0, 0.0)],
        };
        assert_eq!(d.length(), Some(5.0));
        assert_eq!(d.angle(), None);
        let a = Measurement {
            points: vec![DVec3::X, DVec3::ZERO, DVec3::Y],
        };
        assert!((a.angle().unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(!Measurement {
            points: vec![DVec3::ZERO]
        }
        .is_valid());
    }
}
