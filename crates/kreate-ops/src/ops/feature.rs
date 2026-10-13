//! Features: a sketch's regions made solid — extruded or turned — as a new
//! body, or joined to one, or cut from one. See `kreate_scene::feature`.
//!
//! The sketch is the active object; the regions are named by a point in
//! each, in the sketch's plane, so the body follows the sketch when it is
//! edited. A feature can be changed or taken out later, from the feature
//! list, and the body is built again from its features as they then are.

use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::feature::{self, Feature, Operation, Shape};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::ops::sketch_draw::active_sketch;
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Extrude>();
    registry.register::<Revolve>();
    registry.register::<Set>();
    registry.register::<Remove>();
    registry.register::<ToMesh>();
}

/// A body, by name; the most recently made when the name is empty.
fn body_named(doc: &Document, name: &str) -> Option<ObjectId> {
    let name = name.trim();
    doc.objects()
        .filter(|(_, o)| !o.features.is_empty())
        .filter(|(_, o)| name.is_empty() || o.name == name)
        .max_by_key(|(_, o)| o.features.iter().map(|f| f.order).max().unwrap_or(0))
        .map(|(id, _)| id)
}

fn seeds_of(v: &[f64]) -> Vec<[f64; 2]> {
    v.as_chunks::<2>()
        .0
        .iter()
        .copied()
        .filter(|s| s[0].is_finite() && s[1].is_finite())
        .collect()
}

/// Adds a feature from the active sketch: to a new body, or to `target`.
fn add(
    doc: &mut DocMut<'_>,
    shape: Shape,
    seeds: &[f64],
    operation: u32,
    target: &str,
) -> Result<OpResult, OpError> {
    let d = doc.document();
    let sketch = active_sketch(d).ok_or_else(|| OpError::Failed("no sketch is active".into()))?;
    let sketch_uid = d.object_uid(sketch).expect("an object");
    let seeds = seeds_of(seeds);
    let regions = feature::sketch_regions(d, sketch);
    if feature::picked(&regions, &seeds).is_empty() {
        return Err(OpError::Failed("pick a closed region of the sketch".into()));
    }
    let order = feature::next_order(d);
    // 0 a new body, 1 join, 2 cut.
    let into = match operation {
        0 => None,
        _ => Some(
            body_named(d, target)
                .ok_or_else(|| OpError::Failed("there is no body to join or cut".into()))?,
        ),
    };
    let f = Feature {
        shape,
        sketch: sketch_uid,
        seeds,
        operation: if operation == 2 {
            Operation::Cut
        } else {
            Operation::Join
        },
        order,
    };
    let d = doc.document_mut();
    match into {
        Some(body) => d.object_mut(body).expect("a body").features.push(f),
        None => {
            let mesh = d.add_mesh(kreate_mesh::MeshData::empty());
            let name = d.unique_object_name("Body");
            let mut object = Object::new(name, ObjectData::Mesh(mesh));
            object.features.push(f);
            d.add_object(object);
        }
    }
    Ok(OpResult::Finished)
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ExtrudeParams {
    /// A point in each region, as x, y pairs in the sketch's plane.
    #[param(hidden)]
    pub seeds: Vec<f64>,
    /// How far, out of the sketch's plane; negative the other way.
    #[param(soft_min = -1.0, soft_max = 1.0, unit = "length")]
    pub distance: f64,
    /// Half each way.
    pub symmetric: bool,
    #[param(variants("new_body", "join", "cut"))]
    pub operation: u32,
    /// The body joined or cut, by name; the latest when empty.
    pub target: String,
}

impl Default for ExtrudeParams {
    fn default() -> Self {
        Self {
            seeds: Vec::new(),
            distance: 0.01,
            symmetric: false,
            operation: 0,
            target: String::new(),
        }
    }
}

/// Pushes the picked regions out of the sketch.
pub struct Extrude;

impl Operator for Extrude {
    type Params = ExtrudeParams;
    fn id() -> &'static str {
        "feature.extrude"
    }
    fn label() -> &'static str {
        "Extrude"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.distance.is_finite() || params.distance == 0.0 {
            return Err(OpError::Failed("an extrude needs a distance".into()));
        }
        add(
            doc,
            Shape::Extrude {
                distance: params.distance,
                symmetric: params.symmetric,
            },
            &params.seeds,
            params.operation,
            &params.target,
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct RevolveParams {
    #[param(hidden)]
    pub seeds: Vec<f64>,
    /// The two points of the sketch the axis runs through.
    #[param(hidden)]
    pub axis: Vec<u32>,
    /// How far round, in radians: a whole turn is a ring.
    #[param(unit = "angle")]
    pub angle: f64,
    #[param(variants("new_body", "join", "cut"))]
    pub operation: u32,
    pub target: String,
}

impl Default for RevolveParams {
    fn default() -> Self {
        Self {
            seeds: Vec::new(),
            axis: Vec::new(),
            angle: std::f64::consts::TAU,
            operation: 0,
            target: String::new(),
        }
    }
}

/// Turns the picked regions about a line of the sketch.
pub struct Revolve;

impl Operator for Revolve {
    type Params = RevolveParams;
    fn id() -> &'static str {
        "feature.revolve"
    }
    fn label() -> &'static str {
        "Revolve"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let [a, b] = params.axis[..] else {
            return Err(OpError::Failed("pick the line to turn about".into()));
        };
        if a == b || !params.angle.is_finite() || params.angle == 0.0 {
            return Err(OpError::Failed(
                "a revolve needs an axis and an angle".into(),
            ));
        }
        add(
            doc,
            Shape::Revolve {
                axis: [a, b],
                angle: params.angle,
            },
            &params.seeds,
            params.operation,
            &params.target,
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetParams {
    /// The body, by name.
    pub body: String,
    /// Which of its features.
    pub index: u32,
    /// The extrude's distance, or the revolve's angle.
    pub amount: f64,
    pub symmetric: bool,
    #[param(variants("join", "cut"))]
    pub operation: u32,
    /// New regions, as x, y pairs; empty keeps them.
    #[param(hidden)]
    pub seeds: Vec<f64>,
}

/// Changes a feature: the body is built again with it, and whatever comes
/// after it with it.
pub struct Set;

impl Operator for Set {
    type Params = SetParams;
    fn id() -> &'static str {
        "feature.set"
    }
    fn label() -> &'static str {
        "Edit Feature"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = body_named(doc.document(), &params.body)
            .filter(|_| !params.body.trim().is_empty())
            .ok_or_else(|| OpError::Failed("no such body".into()))?;
        let o = doc.document_mut().object_mut(body).expect("a body");
        let Some(f) = o.features.get_mut(params.index as usize) else {
            return Ok(OpResult::Cancelled);
        };
        let before = f.clone();
        if params.amount.is_finite() && params.amount != 0.0 {
            match &mut f.shape {
                Shape::Extrude {
                    distance,
                    symmetric,
                } => {
                    *distance = params.amount;
                    *symmetric = params.symmetric;
                }
                Shape::Revolve { angle, .. } => *angle = params.amount,
            }
        }
        f.operation = if params.operation == 1 {
            Operation::Cut
        } else {
            Operation::Join
        };
        let seeds = seeds_of(&params.seeds);
        if !seeds.is_empty() {
            f.seeds = seeds;
        }
        if *f == before {
            return Ok(OpResult::Cancelled);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RemoveParams {
    pub body: String,
    pub index: u32,
}

/// Takes a feature out of a body; a body left with none is taken out too.
pub struct Remove;

impl Operator for Remove {
    type Params = RemoveParams;
    fn id() -> &'static str {
        "feature.remove"
    }
    fn label() -> &'static str {
        "Delete Feature"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = body_named(doc.document(), &params.body)
            .filter(|_| !params.body.trim().is_empty())
            .ok_or_else(|| OpError::Failed("no such body".into()))?;
        let d = doc.document_mut();
        let o = d.object_mut(body).expect("a body");
        if params.index as usize >= o.features.len() {
            return Ok(OpResult::Cancelled);
        }
        o.features.remove(params.index as usize);
        if o.features.is_empty() {
            d.remove_object(body);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ToMeshParams {
    /// The body by name; empty for the active object.
    pub body: String,
}

/// A body made into the plain mesh its features build, the features
/// dropped: Fusion's "Do not capture design history" for one body. What
/// edit mode needs, because a body holds no mesh of its own to edit — it is
/// built again from its sketches every time — and moving its vertices by
/// hand is a different kind of model from the parametric one.
pub struct ToMesh;

impl Operator for ToMesh {
    type Params = ToMeshParams;
    fn id() -> &'static str {
        "feature.to_mesh"
    }
    fn label() -> &'static str {
        "Convert Body to Mesh"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let body = if params.body.trim().is_empty() {
            d.selection.active
        } else {
            body_named(d, &params.body)
        };
        let Some(body) = body.filter(|id| d.object(*id).is_some_and(|o| !o.features.is_empty()))
        else {
            return Ok(OpResult::Cancelled);
        };
        let object = d.object(body).expect("a body");
        let Some(mesh) = object.mesh() else {
            return Ok(OpResult::Cancelled);
        };
        let built = feature::body(d, &object.features, d.world_transform(body));
        let d = doc.document_mut();
        if let Some(m) = d.mesh_mut(mesh) {
            *m = built;
        }
        d.object_mut(body).expect("a body").features.clear();
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::modifier::{EvalCtx, ModifierCache};

    fn run(
        registry: &OperatorRegistry,
        history: &mut History,
        doc: &mut Document,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        history.dispatch(registry, doc, &Context { mode: Mode::Object }, id, &p)
    }

    fn volume(doc: &Document, id: ObjectId) -> f64 {
        let mut cache = ModifierCache::new();
        let mesh = cache.evaluated(doc, id, &EvalCtx::still()).expect("a body");
        kreate_mesh::measure::mass_properties(&mesh, |p| p).volume
    }

    fn body(doc: &Document) -> ObjectId {
        body_named(doc, "").expect("a body")
    }

    /// The flow the user asked for, as operators: a sketch on the ground, a
    /// 40 × 20 mm rectangle from the origin, a hole, 10 mm up with the hole
    /// cut, then the width changed and the body following.
    #[test]
    fn a_plate_with_a_hole_follows_its_sketch_and_comes_back_from_the_file_way() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let sketch = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.035, 0.018, 0.0])),
                ("join", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        for (pts, value) in [(vec![0u32, 1], 0.04), (vec![1u32, 2], 0.02)] {
            run(
                &r,
                &mut h,
                &mut doc,
                "sketch.constrain",
                &[
                    ("kind", PropertyValue::Enum(0)),
                    ("points", PropertyValue::Ints(pts)),
                    ("value", PropertyValue::Float(value)),
                ],
            )
            .unwrap();
        }
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.02, 0.01, 0.0])),
                ("radius", PropertyValue::Float(0.005)),
            ],
        )
        .unwrap();
        // The plate (the outer region, which goes round the hole) up 10 mm.
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.002, 0.002])),
                ("distance", PropertyValue::Float(0.01)),
            ],
        )
        .unwrap();
        let disc = std::f64::consts::PI * 0.005f64.powi(2) * 0.01;
        let v = volume(&doc, body(&doc));
        let want = 0.04 * 0.02 * 0.01 - disc;
        assert!((v - want).abs() < want * 5e-3, "{v} against {want}");

        // Wider: the dimension changed, the body follows.
        let index = doc
            .object(sketch)
            .unwrap()
            .sketch
            .iter()
            .position(|c| c.value() == Some(0.04))
            .unwrap();
        doc.selection.active = Some(sketch);
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.constraint_set",
            &[
                ("index", PropertyValue::Int(index as i64)),
                ("value", PropertyValue::Float(0.06)),
            ],
        )
        .unwrap();
        let v = volume(&doc, body(&doc));
        let want = 0.06 * 0.02 * 0.01 - disc;
        assert!((v - want).abs() < want * 5e-3, "{v} against {want}");

        // Thicker, through the feature.
        let name = doc.object(body(&doc)).unwrap().name.clone();
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.set",
            &[
                ("body", PropertyValue::Text(name)),
                ("index", PropertyValue::Int(0)),
                ("amount", PropertyValue::Float(0.02)),
            ],
        )
        .unwrap();
        let v = volume(&doc, body(&doc));
        assert!((v - want * 2.0).abs() < want * 1e-2, "{v}");

        // And all of it undoes, one step each.
        while h.can_undo() {
            h.undo(&mut doc);
        }
        assert!(body_named(&doc, "").is_none());
    }

    #[test]
    fn a_body_made_a_mesh_keeps_its_shape_and_undoes_back_to_its_features() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.04, 0.02, 0.0])),
                ("join", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.002, 0.002])),
                ("distance", PropertyValue::Float(0.01)),
            ],
        )
        .unwrap();
        let id = body(&doc);
        let before = volume(&doc, id);
        let mesh = doc.object(id).unwrap().mesh().unwrap();
        assert_eq!(
            doc.mesh(mesh).unwrap().num_faces(),
            0,
            "a body holds no mesh"
        );

        doc.selection.active = Some(id);
        run(&r, &mut h, &mut doc, "feature.to_mesh", &[]).unwrap();
        assert!(doc.object(id).unwrap().features.is_empty());
        // Its own mesh now, the shape it had: something edit mode can select.
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6);
        assert!((volume(&doc, id) - before).abs() < before * 1e-6);

        h.undo(&mut doc);
        assert_eq!(doc.object(id).unwrap().features.len(), 1);
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 0);
    }

    #[test]
    fn a_cut_takes_a_region_out_of_the_body_it_names() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.04, 0.02, 0.0])),
            ],
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.02, 0.01, 0.0])),
                ("radius", PropertyValue::Float(0.005)),
            ],
        )
        .unwrap();
        // The plate and the disc together: a solid block.
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                (
                    "seeds",
                    PropertyValue::Floats(vec![0.002, 0.002, 0.02, 0.01]),
                ),
                ("distance", PropertyValue::Float(0.01)),
            ],
        )
        .unwrap();
        let block = volume(&doc, body(&doc));
        assert!((block - 0.04 * 0.02 * 0.01).abs() < 1e-9, "{block}");
        // Then the disc cut through, both ways.
        let sketch = doc
            .objects()
            .find(|(_, o)| o.name.starts_with("Sketch"))
            .map(|(id, _)| id)
            .unwrap();
        doc.selection.active = Some(sketch);
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.02, 0.01])),
                ("distance", PropertyValue::Float(0.05)),
                ("symmetric", PropertyValue::Bool(true)),
                ("operation", PropertyValue::Enum(2)),
            ],
        )
        .unwrap();
        let v = volume(&doc, body(&doc));
        let disc = std::f64::consts::PI * 0.005f64.powi(2) * 0.01;
        assert!((v - (block - disc)).abs() < block * 5e-3, "{v}");
        assert_eq!(doc.object(body(&doc)).unwrap().features.len(), 2);
    }
}
