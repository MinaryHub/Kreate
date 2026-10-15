//! Sketching: constraints on a sketch's points, the solve that makes them
//! hold, and the solid a sketch becomes. See `kreate_scene::sketch` and
//! `kreate_mesh::profile`.
//!
//! A sketch is a mesh object of flat faces in its own XY plane. A constraint
//! is added from what is selected in edit mode — two points or an edge for a
//! distance, two edges for an angle — and every change to the constraints
//! solves the sketch at once and writes the points where they now must be,
//! in the same undo step. The solid is a geometry node graph on the sketch
//! (extrude or revolve its profile), so changing a dimension changes the
//! part.

use kreate_math::DVec2;
use kreate_mesh::attribute::Domain;
use kreate_mesh::selection;
use kreate_scene::document::ObjectId;
use kreate_scene::geonodes::{GeoGraph, GeoNode, GeoNodeKind};
use kreate_scene::sketch::{self, Constraint, KINDS};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<ConstraintAdd>();
    registry.register::<ConstraintSet>();
    registry.register::<ConstraintRemove>();
    registry.register::<ConstraintFree>();
    registry.register::<Solve>();
    registry.register::<MakeSolid>();
}

/// The active object, when it is a mesh: the sketch.
fn sketch_object(doc: &Document) -> Option<ObjectId> {
    let id = doc.selection.active?;
    doc.object(id)?.mesh()?;
    Some(id)
}

/// Solves the sketch and writes its points back, their Z left as it was.
pub fn solve_sketch(doc: &mut DocMut<'_>, id: ObjectId) -> Option<sketch::Solve> {
    let d = doc.document();
    let constraints = d.object(id)?.sketch.clone();
    let mesh_id = d.object(id)?.mesh()?;
    let mut points: Vec<DVec2> = d
        .mesh(mesh_id)?
        .positions()
        .iter()
        .map(|p| DVec2::new(p.x as f64, p.y as f64))
        .collect();
    let result = sketch::solve(&mut points, &constraints);
    let mesh = doc.document_mut().mesh_mut(mesh_id)?;
    for (p, q) in mesh.positions_mut().iter_mut().zip(&points) {
        p.x = q.x as f32;
        p.y = q.y as f32;
    }
    Some(result)
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ConstraintAddParams {
    /// What kind. See `kreate_scene::sketch::KINDS`.
    #[param(variants(
        "distance",
        "horizontal",
        "vertical",
        "fixed",
        "angle",
        "parallel",
        "perpendicular",
        "equal"
    ))]
    pub kind: u32,
    /// The dimension, for a distance or an angle; zero takes it as it is now.
    pub value: f64,
}

/// Adds a constraint on what is selected, and solves.
pub struct ConstraintAdd;

impl Operator for ConstraintAdd {
    type Params = ConstraintAddParams;
    fn id() -> &'static str {
        "sketch.constraint_add"
    }
    fn label() -> &'static str {
        "Add Sketch Constraint"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        ctx.mode.edits_elements() && sketch_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let id = sketch_object(d).ok_or_else(|| OpError::Failed("no sketch".into()))?;
        let mesh = d
            .object(id)
            .and_then(|o| o.mesh())
            .and_then(|m| d.mesh(m))
            .ok_or_else(|| OpError::Failed("no sketch".into()))?;
        let verts = selection::selected(mesh, Domain::Point);
        let edges: Vec<[u32; 2]> = selection::selected(mesh, Domain::Edge)
            .iter()
            .filter_map(|e| mesh.edges().get(*e as usize).copied())
            .collect();
        let p = |v: u32| {
            let q = mesh.positions()[v as usize];
            DVec2::new(q.x as f64, q.y as f64)
        };
        // Two points: one selected edge's, or the two selected.
        let pair = match (edges.as_slice(), verts.as_slice()) {
            ([e], _) => Some((e[0], e[1])),
            (_, [a, b]) => Some((*a, *b)),
            _ => None,
        };
        let lines = match edges.as_slice() {
            [e, f] => Some((e[0], e[1], f[0], f[1])),
            _ => None,
        };
        let need_pair = || OpError::Failed("select two points, or one edge".into());
        let need_lines = || OpError::Failed("select two edges".into());
        let given = (params.value != 0.0 && params.value.is_finite()).then_some(params.value);
        let mut added: Vec<Constraint> = Vec::new();
        match KINDS
            .get(params.kind as usize)
            .copied()
            .unwrap_or("distance")
        {
            "distance" => {
                let (a, b) = pair.ok_or_else(need_pair)?;
                let value = given.unwrap_or_else(|| p(a).distance(p(b)));
                added.push(Constraint::Distance { a, b, value });
            }
            "horizontal" => {
                let (a, b) = pair.ok_or_else(need_pair)?;
                added.push(Constraint::Horizontal { a, b });
            }
            "vertical" => {
                let (a, b) = pair.ok_or_else(need_pair)?;
                added.push(Constraint::Vertical { a, b });
            }
            "fixed" => {
                if verts.is_empty() {
                    return Err(OpError::Failed("select points to fix".into()));
                }
                added.extend(verts.iter().map(|v| Constraint::Fixed {
                    a: *v,
                    at: p(*v).to_array(),
                }));
            }
            "angle" => {
                let (a, b, c, d) = lines.ok_or_else(need_lines)?;
                let value = given.unwrap_or_else(|| {
                    let (u, v) = (p(b) - p(a), p(d) - p(c));
                    u.perp_dot(v).atan2(u.dot(v))
                });
                added.push(Constraint::Angle { a, b, c, d, value });
            }
            kind => {
                let (a, b, c, d) = lines.ok_or_else(need_lines)?;
                added.push(match kind {
                    "parallel" => Constraint::Parallel { a, b, c, d },
                    "perpendicular" => Constraint::Perpendicular { a, b, c, d },
                    _ => Constraint::Equal { a, b, c, d },
                });
            }
        }
        doc.document_mut()
            .object_mut(id)
            .expect("the sketch")
            .sketch
            .extend(added);
        solve_sketch(doc, id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ConstraintSetParams {
    /// Which, in the order they were added.
    pub index: u32,
    /// Its new dimension: a length, or an angle in radians.
    pub value: f64,
}

/// Changes a constraint's dimension, and solves.
pub struct ConstraintSet;

impl Operator for ConstraintSet {
    type Params = ConstraintSetParams;
    fn id() -> &'static str {
        "sketch.constraint_set"
    }
    fn label() -> &'static str {
        "Set Sketch Dimension"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        sketch_object(doc).is_some_and(|id| !doc.object(id).expect("active").sketch.is_empty())
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = sketch_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let o = doc.document_mut().object_mut(id).expect("active");
        let Some(c) = o.sketch.get_mut(params.index as usize) else {
            return Ok(OpResult::Cancelled);
        };
        if c.value().is_none() || c.value() == Some(params.value) || !params.value.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        *c = c.with_value(params.value);
        solve_sketch(doc, id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ConstraintRemoveParams {
    /// Which, in the order they were added.
    pub index: u32,
}

/// Takes a constraint off. The points stay where they are.
pub struct ConstraintRemove;

impl Operator for ConstraintRemove {
    type Params = ConstraintRemoveParams;
    fn id() -> &'static str {
        "sketch.constraint_remove"
    }
    fn label() -> &'static str {
        "Remove Sketch Constraint"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        sketch_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = sketch_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let list = &mut doc.document_mut().object_mut(id).expect("active").sketch;
        if params.index as usize >= list.len() {
            return Ok(OpResult::Cancelled);
        }
        list.remove(params.index as usize);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ConstraintFreeParams {
    /// The points freed, by index.
    pub points: Vec<u32>,
}

/// Takes off every constraint on the given points: what they were held by
/// goes, and they stay where they are. A circle's or an arc's own shape is
/// kept — it is what makes the points round one.
pub struct ConstraintFree;

impl Operator for ConstraintFree {
    type Params = ConstraintFreeParams;
    fn id() -> &'static str {
        "sketch.constraint_free"
    }
    fn label() -> &'static str {
        "Remove Constraints"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        sketch_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = sketch_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let list = &mut doc.document_mut().object_mut(id).expect("active").sketch;
        let before = list.len();
        list.retain(|c| {
            matches!(
                c,
                kreate_scene::sketch::Constraint::Circle { .. }
                    | kreate_scene::sketch::Constraint::Arc { .. }
            ) || !c.points().iter().any(|p| params.points.contains(p))
        });
        Ok(if list.len() < before {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SolveParams {}

/// Puts the sketch's points back where its constraints say — after the
/// points were moved by hand.
pub struct Solve;

impl Operator for Solve {
    type Params = SolveParams;
    fn id() -> &'static str {
        "sketch.solve"
    }
    fn label() -> &'static str {
        "Solve Sketch"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        sketch_object(doc).is_some_and(|id| !doc.object(id).expect("active").sketch.is_empty())
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = sketch_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        match solve_sketch(doc, id) {
            Some(s) if !s.converged => Err(OpError::Failed(
                "the constraints contradict each other; the sketch is as close as it gets".into(),
            )),
            Some(_) => Ok(OpResult::Finished),
            None => Ok(OpResult::Cancelled),
        }
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct MakeSolidParams {
    /// Push the profile out, or turn it about the sketch's Y axis.
    #[param(variants("extrude", "revolve"))]
    pub kind: u32,
    /// How far an extrude goes, along the sketch's Z.
    #[param(unit = "length")]
    pub height: f64,
    /// Steps round a revolve.
    #[param(min = 3, max = 512, unit = "count")]
    pub segments: u32,
}

impl Default for MakeSolidParams {
    fn default() -> Self {
        Self {
            kind: 0,
            height: 1.0,
            segments: 64,
        }
    }
}

/// Makes the sketch a solid: a geometry node graph that extrudes or revolves
/// its profile. The sketch stays editable; the solid follows it.
pub struct MakeSolid;

impl Operator for MakeSolid {
    type Params = MakeSolidParams;
    fn id() -> &'static str {
        "sketch.make_solid"
    }
    fn label() -> &'static str {
        "Make Solid from Sketch"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        sketch_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = sketch_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let (name, kind) = if params.kind == 1 {
            (
                "Revolve Profile",
                GeoNodeKind::RevolveProfile {
                    segments: params.segments.clamp(3, 512),
                },
            )
        } else {
            (
                "Extrude Profile",
                GeoNodeKind::ExtrudeProfile {
                    height: params.height as f32,
                },
            )
        };
        let mut graph = GeoGraph::passthrough();
        graph.nodes.push(GeoNode {
            name: name.into(),
            kind,
            position: [0.0, 0.0],
        });
        graph.link(0, 2, 0).expect("input into the feature");
        graph.link(2, 1, 0).expect("the feature into the output");
        doc.document_mut()
            .object_mut(id)
            .expect("active")
            .geometry_nodes = Some(graph);
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {

    /// Freeing points takes off what holds them and nothing else: a
    /// constraint on other points stays, and so does a circle's shape.
    #[test]
    fn freeing_points_removes_only_their_constraints() {
        use kreate_scene::sketch::Constraint as C;
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        let mut history = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::MeshData::from_faces(
            vec![kreate_math::Vec3::ZERO; 8],
            &[],
        ));
        let mut o = Object::new("Sketch", ObjectData::Mesh(mesh));
        o.sketch = vec![
            C::Horizontal { a: 0, b: 1 },
            C::Distance {
                a: 1,
                b: 2,
                value: 1.0,
            },
            C::Vertical { a: 3, b: 4 },
            C::Circle {
                centre: 5,
                ring: vec![1, 6, 7],
            },
        ];
        let id = doc.add_object(o);
        doc.selection.add(id);
        run(
            &registry,
            &mut history,
            &mut doc,
            &Context::default(),
            "sketch.constraint_free",
            &[("points", PropertyValue::Ints(vec![1]))],
        )
        .unwrap();
        let left: Vec<&str> = doc
            .object(id)
            .unwrap()
            .sketch
            .iter()
            .map(|c| c.tag())
            .collect();
        assert_eq!(left, vec!["vertical", "circle"]);
        assert!(history.undo(&mut doc));
        assert_eq!(doc.object(id).unwrap().sketch.len(), 4);
    }

    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::document::{Object, ObjectData};

    fn run(
        registry: &OperatorRegistry,
        history: &mut History,
        doc: &mut Document,
        ctx: &Context,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        history.dispatch(registry, doc, ctx, id, &p)
    }

    #[test]
    fn a_plane_sketch_gets_a_width_and_becomes_a_block() {
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        let mut history = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::plane(2.0));
        let id = doc.add_object(Object::new("Sketch", ObjectData::Mesh(mesh)));
        doc.selection.add(id);
        let edit = Context {
            mode: Mode::EditMesh,
        };
        // Select one edge of the plane: its first.
        {
            let m = doc.mesh_mut(mesh).unwrap();
            selection::ensure_layers(m);
            selection::select_none(m);
            selection::set_selected(m, Domain::Edge, 0, true);
            selection::flush(m, kreate_mesh::SelectMode::Edge);
        }
        run(
            &registry,
            &mut history,
            &mut doc,
            &edit,
            "sketch.constraint_add",
            &[("kind", PropertyValue::Enum(0))],
        )
        .unwrap();
        assert_eq!(doc.object(id).unwrap().sketch.len(), 1);
        // The edge's length was taken as it is: 2.
        assert_eq!(doc.object(id).unwrap().sketch[0].value(), Some(2.0));
        run(
            &registry,
            &mut history,
            &mut doc,
            &edit,
            "sketch.constraint_set",
            &[
                ("index", PropertyValue::Int(0)),
                ("value", PropertyValue::Float(5.0)),
            ],
        )
        .unwrap();
        let [a, b] = doc.mesh(mesh).unwrap().edges()[0];
        let p = doc.mesh(mesh).unwrap().positions();
        let length = (p[a as usize] - p[b as usize]).length();
        assert!((length - 5.0).abs() < 1e-5, "{length}");

        run(
            &registry,
            &mut history,
            &mut doc,
            &Context::default(),
            "sketch.make_solid",
            &[("height", PropertyValue::Float(0.5))],
        )
        .unwrap();
        let graph = doc.object(id).unwrap().geometry_nodes.clone().unwrap();
        let solid = graph.evaluate(doc.mesh(mesh).unwrap()).unwrap();
        assert!(kreate_mesh::measure::is_closed(&solid));
        assert!((solid.bounds().max.z - 0.5).abs() < 1e-5);
    }
}
