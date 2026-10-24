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
    registry.register::<Round>();
    registry.register::<ShellFeature>();
    registry.register::<PatternFeature>();
    registry.register::<CircularFeature>();
    registry.register::<MirrorFeature>();
}

/// The body a body-level feature goes on: by name, or the active body.
fn body_for(doc: &Document, name: &str) -> Result<ObjectId, OpError> {
    if name.trim().is_empty() {
        doc.selection
            .active
            .filter(|id| doc.object(*id).is_some_and(|o| !o.features.is_empty()))
    } else {
        body_named(doc, name)
    }
    .ok_or_else(|| OpError::Failed("pick a body made from a sketch".into()))
}

/// Adds a feature that works on the body so far, as the next step of its
/// timeline, doing what the step before it did — joining or cutting.
fn add_on_body(doc: &mut DocMut<'_>, body: ObjectId, shape: Shape) -> Result<OpResult, OpError> {
    let order = feature::next_order(doc.document());
    let operation = doc
        .document()
        .object(body)
        .and_then(|o| o.features.last())
        .map(|f| f.operation)
        .unwrap_or_default();
    let f = Feature {
        shape,
        sketch: kreate_scene::Uid::NONE,
        seeds: Vec::new(),
        operation,
        order,
    };
    doc.document_mut()
        .object_mut(body)
        .expect("a body")
        .features
        .push(f);
    Ok(OpResult::Finished)
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ShellParams {
    /// The body by name; empty for the active one.
    pub body: String,
    /// How thick the wall is left.
    #[param(min = 0.0, soft_max = 0.02, unit = "length")]
    pub thickness: f64,
    /// The side taken off, so the body is a tray rather than hollow.
    #[param(variants("none", "+x", "-x", "+y", "-y", "+z", "-z"))]
    pub open: u32,
}

impl Default for ShellParams {
    fn default() -> Self {
        Self {
            body: String::new(),
            thickness: 0.002,
            open: 5,
        }
    }
}

/// Hollows the body to a wall, its top taken off: Fusion's Shell, as a
/// step of the timeline.
pub struct ShellFeature;

impl Operator for ShellFeature {
    type Params = ShellParams;
    fn id() -> &'static str {
        "feature.shell"
    }
    fn label() -> &'static str {
        "Shell"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = body_for(doc.document(), &params.body)?;
        if !(params.thickness > 0.0 && params.thickness.is_finite()) {
            return Ok(OpResult::Cancelled);
        }
        add_on_body(
            doc,
            body,
            Shape::Shell {
                thickness: params.thickness,
                open: params.open.min(6),
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PatternParams {
    pub body: String,
    /// How many in all, the original counted.
    #[param(min = 1, max = 1000, unit = "count")]
    pub count: u32,
    /// From one copy to the next, in the body's space.
    #[param(unit = "length")]
    pub step: [f64; 3],
}

impl Default for PatternParams {
    fn default() -> Self {
        Self {
            body: String::new(),
            count: 3,
            step: [0.02, 0.0, 0.0],
        }
    }
}

/// Repeats the body's last feature along a line: Fusion's Rectangular
/// Pattern, along one direction.
pub struct PatternFeature;

impl Operator for PatternFeature {
    type Params = PatternParams;
    fn id() -> &'static str {
        "feature.pattern"
    }
    fn label() -> &'static str {
        "Linear Pattern"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = body_for(doc.document(), &params.body)?;
        if params.count < 2 || params.step.iter().any(|v| !v.is_finite()) {
            return Ok(OpResult::Cancelled);
        }
        add_on_body(
            doc,
            body,
            Shape::Pattern {
                count: params.count,
                step: params.step,
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct CircularParams {
    pub body: String,
    #[param(min = 1, max = 1000, unit = "count")]
    pub count: u32,
    /// The body's axis they go round, through its origin.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
}

impl Default for CircularParams {
    fn default() -> Self {
        Self {
            body: String::new(),
            count: 6,
            axis: 2,
        }
    }
}

/// Repeats the body's last feature round an axis: Fusion's Circular
/// Pattern.
pub struct CircularFeature;

impl Operator for CircularFeature {
    type Params = CircularParams;
    fn id() -> &'static str {
        "feature.circular"
    }
    fn label() -> &'static str {
        "Circular Pattern"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = body_for(doc.document(), &params.body)?;
        if params.count < 2 {
            return Ok(OpResult::Cancelled);
        }
        add_on_body(
            doc,
            body,
            Shape::Circular {
                count: params.count,
                axis: params.axis.min(2),
            },
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MirrorParams {
    pub body: String,
    /// Square to this axis of the body, through its origin.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
}

/// Reflects the body's last feature across a plane: Fusion's Mirror.
pub struct MirrorFeature;

impl Operator for MirrorFeature {
    type Params = MirrorParams;
    fn id() -> &'static str {
        "feature.mirror"
    }
    fn label() -> &'static str {
        "Mirror Feature"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = body_for(doc.document(), &params.body)?;
        add_on_body(
            doc,
            body,
            Shape::Mirror {
                axis: params.axis.min(2),
            },
        )
    }
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

/// What a feature joins or cuts: a body by name — the latest when the name
/// is empty — or, named, any other mesh object that is not a sketch.
fn target_of(doc: &Document, name: &str) -> Option<ObjectId> {
    if let Some(body) = body_named(doc, name) {
        return Some(body);
    }
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    doc.objects()
        .find(|(_, o)| {
            o.name == name && o.mesh().is_some() && o.sketch.is_empty() && o.sketch_lines.is_empty()
        })
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
    let regions = feature::shape_regions(d, sketch, &shape);
    if feature::picked(&regions, &seeds).is_empty() {
        return Err(OpError::Failed("pick a closed region of the sketch".into()));
    }
    let order = feature::next_order(d);
    // 0 a new body, 1 join, 2 cut.
    let into = match operation {
        0 => None,
        _ => Some(
            target_of(d, target)
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
    let plain = into.filter(|o| d.object(*o).is_some_and(|o| o.features.is_empty()));
    if let Some(target) = plain {
        return join_or_cut_mesh(doc, f, target);
    }
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

/// A feature joined to or cut from a mesh that is not a body: what it makes
/// becomes a hidden object of its own, and the mesh gains a boolean against
/// it — so the cut can be moved, changed or taken off later, as Blender's
/// boolean cutters are.
fn join_or_cut_mesh(
    doc: &mut DocMut<'_>,
    f: Feature,
    target: ObjectId,
) -> Result<OpResult, OpError> {
    use kreate_scene::modifier::{BooleanModifier, ModifierInstance, ModifierKind};
    let d = doc.document();
    let mut piece = feature::solid(d, &f);
    if piece.num_faces() == 0 {
        return Err(OpError::Failed("the regions make no solid".into()));
    }
    let sketch_world = d
        .object_by_uid(f.sketch)
        .map(|id| d.world_transform(id))
        .unwrap_or_default();
    for p in piece.positions_mut() {
        *p = sketch_world.transform_point3(p.as_dvec3()).as_vec3();
    }
    let cut = f.operation == Operation::Cut;
    let target_name = d.object(target).map(|o| o.name.clone()).unwrap_or_default();
    let d = doc.document_mut();
    let mesh = d.add_mesh(piece);
    let name = d.unique_object_name(&format!(
        "{target_name} {}",
        if cut { "Cut" } else { "Join" }
    ));
    let mut tool = Object::new(name, ObjectData::Mesh(mesh));
    tool.visible = false;
    let tool = d.add_object(tool);
    let operand = d.object_uid(tool).expect("just added");
    let op = if cut {
        kreate_mesh::modifiers::BooleanOp::Difference
    } else {
        kreate_mesh::modifiers::BooleanOp::Union
    };
    d.object_mut(target)
        .expect("the target")
        .modifiers
        .push(ModifierInstance::new(ModifierKind::Boolean(
            BooleanModifier { operand, op },
        )));
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
    /// What it turns about: the line through `axis`, or the sketch's own
    /// horizontal or vertical through its origin.
    #[param(variants("line", "sketch_x", "sketch_y"))]
    pub about: u32,
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
            about: 0,
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
        let [a, b] = match params.about {
            1 => [feature::AXIS_X; 2],
            2 => [feature::AXIS_Y; 2],
            _ => match params.axis[..] {
                [a, b] => [a, b],
                _ => return Err(OpError::Failed("pick the line to turn about".into())),
            },
        };
        if (a == b && params.about == 0) || !params.angle.is_finite() || params.angle == 0.0 {
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
                Shape::Round { radius, .. } => *radius = params.amount.abs(),
                Shape::Shell { thickness, .. } => *thickness = params.amount.abs(),
                Shape::Pattern { count, .. } | Shape::Circular { count, .. } => {
                    *count = (params.amount.round() as i64).clamp(1, 1000) as u32;
                }
                Shape::Mirror { .. } => {}
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

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct RoundParams {
    /// The body by name; empty for the active one.
    pub body: String,
    /// The fillet's radius, or how far a chamfer cuts back along each face.
    #[param(min = 0.0, soft_max = 0.02, unit = "length")]
    pub radius: f64,
    /// A flat cut instead of a round.
    pub chamfer: bool,
}

impl Default for RoundParams {
    fn default() -> Self {
        Self {
            body: String::new(),
            radius: 0.001,
            chamfer: false,
        }
    }
}

/// A fillet or a chamfer on every sharp edge of a body, as a step of its
/// timeline: the sketches change, the body is built again, and its edges
/// are broken again.
pub struct Round;

impl Operator for Round {
    type Params = RoundParams;
    fn id() -> &'static str {
        "feature.round"
    }
    fn label() -> &'static str {
        "Fillet or Chamfer Edges"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let body = if params.body.trim().is_empty() {
            d.selection
                .active
                .filter(|id| d.object(*id).is_some_and(|o| !o.features.is_empty()))
        } else {
            body_named(d, &params.body)
        }
        .ok_or_else(|| {
            OpError::Failed(
                "pick a body made from a sketch; on a mesh, round its edges in edit mode".into(),
            )
        })?;
        if params.radius.is_nan() || params.radius <= 0.0 {
            return Ok(OpResult::Cancelled);
        }
        let order = feature::next_order(d);
        let f = Feature {
            shape: Shape::Round {
                radius: params.radius,
                chamfer: params.chamfer,
            },
            sketch: kreate_scene::Uid::NONE,
            seeds: Vec::new(),
            operation: Operation::Join,
            order,
        };
        doc.document_mut()
            .object_mut(body)
            .expect("a body")
            .features
            .push(f);
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

    /// A profile drawn as one open run of lines beside the sketch's vertical
    /// — a cup's wall, from the axis out along the bottom and up — turned a
    /// whole way round about that axis is a closed solid of the right size.
    #[test]
    fn an_open_profile_turned_about_the_sketchs_axis_is_a_solid() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.line",
            &[(
                "points",
                PropertyValue::Floats(vec![0.0, 0.0, 0.03, 0.0, 0.03, 0.05]),
            )],
        )
        .unwrap();
        let sketch = active_sketch(&doc).unwrap();
        let shape = Shape::Revolve {
            axis: [feature::AXIS_Y; 2],
            angle: std::f64::consts::TAU,
        };
        let seeds = feature::shape_profile_seeds(&doc, sketch, &shape);
        assert_eq!(seeds.len(), 1, "the closed profile");
        // Without saying the axis, there is nothing closed to turn.
        assert!(feature::profile_seeds(&doc, sketch).is_empty());
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.revolve",
            &[
                ("seeds", PropertyValue::Floats(seeds[0].to_vec())),
                ("about", PropertyValue::Enum(2)),
            ],
        )
        .unwrap();
        let mut cache = ModifierCache::new();
        let mesh = cache
            .evaluated(&doc, body(&doc), &EvalCtx::still())
            .expect("a body");
        assert!(kreate_mesh::measure::is_closed(&mesh));
        let v = kreate_mesh::measure::mass_properties(&mesh, |p| p).volume;
        let want = std::f64::consts::PI * 0.03f64.powi(2) * 0.05;
        assert!((v - want).abs() < want * 3e-3, "{v} against {want}");
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

    /// A cut into a mesh that is not a body: the extrude becomes a hidden
    /// cutter and the mesh a boolean against it — one step.
    #[test]
    fn a_cut_into_a_plain_mesh_is_a_boolean_against_a_hidden_cutter() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let cube = doc.add_mesh_object("Box", kreate_mesh::primitives::cube(2.0));
        // A square on the ground, cut upward into the box.
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([-0.5, -0.5, 0.0])),
                ("b", PropertyValue::Vec3([0.5, 0.5, 0.0])),
            ],
        )
        .unwrap();
        let steps = h.entries().len();
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.0, 0.0])),
                ("distance", PropertyValue::Float(0.5)),
                ("operation", PropertyValue::Enum(2)),
                ("target", PropertyValue::Text("Box".into())),
            ],
        )
        .unwrap();
        assert_eq!(h.entries().len(), steps + 1, "one step");
        let o = doc.object(cube).unwrap();
        assert_eq!(o.modifiers.len(), 1);
        let kreate_scene::modifier::ModifierKind::Boolean(b) = o.modifiers[0].kind else {
            panic!("a boolean");
        };
        let cutter = doc.object_by_uid(b.operand).expect("the cutter");
        assert!(!doc.object(cutter).unwrap().visible, "hidden");
        let cut = kreate_scene::modifier::apply_one(&doc, cube, 0).unwrap();
        let volume = kreate_mesh::measure::mass_properties(&cut, |p| p).volume;
        // 8 less a 1 x 1 x 0.5 pocket.
        assert!((volume - 7.5).abs() < 1e-3, "{volume}");
    }

    /// Where a vertical line through `(x, y)` meets the mesh's faces, top
    /// first.
    fn heights_at(mesh: &kreate_mesh::MeshData, x: f32, y: f32) -> Vec<f32> {
        let p = mesh.positions();
        let tri = mesh.triangulate();
        let mut out = Vec::new();
        for t in tri.indices.chunks(3) {
            let [a, b, c] = [p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]];
            let cross = |u: kreate_math::Vec3, v: kreate_math::Vec3| {
                (v.x - u.x) * (y - u.y) - (v.y - u.y) * (x - u.x)
            };
            let (d1, d2, d3) = (cross(a, b), cross(b, c), cross(c, a));
            let inside =
                (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0) || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0);
            let area = (b - a).truncate().perp_dot((c - a).truncate());
            if !inside || area.abs() < 1e-12 {
                continue;
            }
            // Barycentric height.
            let w1 = (b - x_y(x, y, a))
                .truncate()
                .perp_dot((c - x_y(x, y, a)).truncate())
                / area;
            let w2 = (c - x_y(x, y, a))
                .truncate()
                .perp_dot((a - x_y(x, y, a)).truncate())
                / area;
            let _ = (w1, w2);
            let n = (b - a).cross(c - a);
            if n.z.abs() < 1e-12 {
                continue;
            }
            let z = a.z - (n.x * (x - a.x) + n.y * (y - a.y)) / n.z;
            out.push(z);
        }
        out.sort_by(|a, b| b.total_cmp(a));
        out.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
        out
    }

    fn x_y(x: f32, y: f32, like: kreate_math::Vec3) -> kreate_math::Vec3 {
        kreate_math::Vec3::new(x, y, like.z)
    }

    /// Fusion's commonest cut: a sketch on a body's top face, extruded back
    /// into the body. The cutter's top lies in the body's top face; the hole
    /// must open there, not stay covered by a skin of the old face.
    #[test]
    fn a_hole_sketched_on_a_bodys_face_and_extruded_in_opens_through_it() {
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
        let plate = body(&doc);
        let name = doc.object(plate).unwrap().name.clone();
        // On the top face, through the cursor, as the app puts a sketch on
        // a face.
        doc.cursor = kreate_math::DVec3::new(0.0, 0.0, 0.01);
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[("plane", PropertyValue::Enum(3))],
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
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.02, 0.01])),
                ("distance", PropertyValue::Float(-0.004)),
                ("operation", PropertyValue::Enum(2)),
                ("target", PropertyValue::Text(name)),
            ],
        )
        .unwrap();
        let mut cache = ModifierCache::new();
        let mesh = cache.evaluated(&doc, plate, &EvalCtx::still()).unwrap();
        let v = kreate_mesh::measure::mass_properties(&mesh, |p| p).volume;
        let want = 0.04 * 0.02 * 0.01 - std::f64::consts::PI * 0.005f64.powi(2) * 0.004;
        assert!((v - want).abs() < want * 5e-3, "{v} against {want}");
        // Down through the hole: the pocket's floor first, no lid above it.
        let at = heights_at(&mesh, 0.02, 0.01);
        assert!(
            (at[0] - 0.006).abs() < 1e-6,
            "the first face down through the hole is at {:?}, not the floor at 0.006",
            at
        );
        // Beside the hole, the top face is where it was.
        let beside = heights_at(&mesh, 0.003, 0.003);
        assert!((beside[0] - 0.01).abs() < 1e-6, "{beside:?}");
    }

    /// The same into a plain mesh — a cube from the Add menu — whose top
    /// the cutter's top lies in.
    #[test]
    fn a_hole_sketched_on_a_plain_meshs_face_and_extruded_in_opens_through_it() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let cube = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        doc.cursor = kreate_math::DVec3::new(0.0, 0.0, 1.0);
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[("plane", PropertyValue::Enum(3))],
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("radius", PropertyValue::Float(0.4)),
            ],
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.0, 0.0])),
                ("distance", PropertyValue::Float(-0.5)),
                ("operation", PropertyValue::Enum(2)),
                ("target", PropertyValue::Text("Cube".into())),
            ],
        )
        .unwrap();
        let mut cache = ModifierCache::new();
        let mesh = cache.evaluated(&doc, cube, &EvalCtx::still()).unwrap();
        let at = heights_at(&mesh, 0.05, 0.03);
        let beside = heights_at(&mesh, 0.8, 0.8);
        let v = kreate_mesh::measure::mass_properties(&mesh, |p| p).volume;
        assert!(
            (at[0] - 0.5).abs() < 1e-5 && (beside[0] - 1.0).abs() < 1e-5,
            "through the hole {at:?}, beside it {beside:?}, volume {v}"
        );
    }

    /// A 40 × 20 × 10 mm block, made from a sketch.
    fn block(r: &OperatorRegistry, h: &mut History, doc: &mut Document) -> ObjectId {
        run(r, h, doc, "sketch.new", &[]).unwrap();
        run(
            r,
            h,
            doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.04, 0.02, 0.0])),
                ("join", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        run(
            r,
            h,
            doc,
            "feature.extrude",
            &[
                ("seeds", PropertyValue::Floats(vec![0.002, 0.002])),
                ("distance", PropertyValue::Float(0.01)),
            ],
        )
        .unwrap();
        body(doc)
    }

    #[test]
    fn a_fillet_and_a_chamfer_take_what_they_should_off_every_edge() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        // Twelve edges, 280 mm of them, each losing (1 − π/4) r² rounded or
        // r²/2 cut, less a little where three meet at a corner.
        let whole = 0.04 * 0.02 * 0.01;
        for (chamfer, per_mm2) in [(false, 1.0 - std::f64::consts::PI / 4.0), (true, 0.5)] {
            let mut h = History::new();
            let mut doc = Document::new();
            let id = block(&r, &mut h, &mut doc);
            doc.selection.active = Some(id);
            run(
                &r,
                &mut h,
                &mut doc,
                "feature.round",
                &[
                    ("radius", PropertyValue::Float(0.001)),
                    ("chamfer", PropertyValue::Bool(chamfer)),
                ],
            )
            .unwrap();
            let v = volume(&doc, id);
            let gone = (whole - v) * 1e9;
            let want = per_mm2 * 280.0;
            assert!(
                (gone - want).abs() < want * 0.15,
                "chamfer {chamfer}: {gone} mm³ off, against about {want}"
            );
            assert_eq!(doc.object(id).unwrap().features.len(), 2);
            // The radius is a step to change, like any other.
            let name = doc.object(id).unwrap().name.clone();
            run(
                &r,
                &mut h,
                &mut doc,
                "feature.set",
                &[
                    ("body", PropertyValue::Text(name)),
                    ("index", PropertyValue::Int(1)),
                    ("amount", PropertyValue::Float(0.002)),
                ],
            )
            .unwrap();
            assert!(
                whole - volume(&doc, id) > (whole - v) * 3.0,
                "doubling the radius"
            );
        }
    }

    /// A plate shelled, patterned, mirrored and turned round: each a step
    /// of the timeline, each read back from the file.
    #[test]
    fn body_features_shell_pattern_mirror_and_circular() {
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
                ("seeds", PropertyValue::Floats(vec![0.01, 0.01])),
                ("distance", PropertyValue::Float(0.01)),
            ],
        )
        .unwrap();
        let id = body(&doc);
        let plate = volume(&doc, id);
        assert!((plate - 8e-6).abs() < 1e-8, "{plate}");
        doc.selection.clear();
        doc.selection.add(id);
        doc.selection.active = Some(id);

        // A linear pattern of three: three plates.
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.pattern",
            &[
                ("count", PropertyValue::Int(3)),
                ("step", PropertyValue::Vec3([0.05, 0.0, 0.0])),
            ],
        )
        .unwrap();
        assert!(
            (volume(&doc, id) - 3.0 * plate).abs() < 1e-7,
            "{}",
            volume(&doc, id)
        );
        assert!(h.undo(&mut doc));

        // Mirrored across the YZ plane: a plate from -40 to 40 mm.
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.mirror",
            &[("axis", PropertyValue::Enum(0))],
        )
        .unwrap();
        assert!(
            (volume(&doc, id) - 2.0 * plate).abs() < 1e-7,
            "{}",
            volume(&doc, id)
        );
        assert!(h.undo(&mut doc));

        // Round Z, four times: four plates, each a quarter turn on.
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.circular",
            &[
                ("count", PropertyValue::Int(4)),
                ("axis", PropertyValue::Enum(2)),
            ],
        )
        .unwrap();
        let four = volume(&doc, id);
        assert!(four > 3.5 * plate && four < 4.1 * plate, "{four}");
        assert!(h.undo(&mut doc));

        // Shelled, 2 mm, open at the top: a tray.
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.shell",
            &[
                ("thickness", PropertyValue::Float(0.002)),
                ("open", PropertyValue::Enum(5)),
            ],
        )
        .unwrap();
        let tray = volume(&doc, id);
        // The walls and floor: 8 - (36 x 16 x 8) mm³.
        let want = 8e-6 - 0.036 * 0.016 * 0.008;
        assert!((tray - want).abs() < want * 0.05, "{tray} vs {want}");
        assert_eq!(doc.object(id).unwrap().features.len(), 2);

        // Through the file.
        let bytes = kreate_io::container::write(&doc, &Default::default(), &[]).unwrap();
        let back = kreate_io::container::read(std::io::Cursor::new(bytes))
            .unwrap()
            .document;
        let (_, o) = back
            .objects()
            .find(|(_, o)| !o.features.is_empty())
            .unwrap();
        assert_eq!(
            o.features[1].shape,
            Shape::Shell {
                thickness: 0.002,
                open: 5
            }
        );
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
