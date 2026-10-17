//! Curve objects: adding one, shaping how it is shown, and its points.
//! See `kreate_scene::curve`.
//!
//! The curve is on the active object, which is what every operator here
//! acts on; one that is not a curve object is said so.

use kreate_math::DVec3;
use kreate_scene::curve::{Curve, CurvePoint, Spline, SplineKind};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddCurve>();
    registry.register::<Set>();
    registry.register::<PointAdd>();
    registry.register::<PointSet>();
    registry.register::<PointRemove>();
    registry.register::<SplineSet>();
    registry.register::<ToMesh>();
}

/// The active object, when it is a curve object.
pub fn active_curve(doc: &Document) -> Option<ObjectId> {
    let id = doc.selection.active?;
    doc.object(id)?.curve.as_ref()?;
    Some(id)
}

fn need_curve(doc: &Document) -> Result<ObjectId, OpError> {
    active_curve(doc).ok_or_else(|| OpError::Failed("the active object is not a curve".into()))
}

fn curve_mut<'a>(doc: &'a mut DocMut<'_>, id: ObjectId) -> &'a mut Curve {
    doc.document_mut()
        .object_mut(id)
        .and_then(|o| o.curve.as_mut())
        .expect("checked to be a curve")
}

/// The shapes `object.add_curve` starts from.
pub const CURVE_SHAPES: [&str; 4] = ["bezier", "circle", "path", "nurbs"];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddCurveParams {
    /// A Bézier arc, a Bézier circle, a straight polyline path, or a NURBS
    /// run.
    #[param(variants("bezier", "circle", "path", "nurbs"))]
    pub shape: u32,
    /// How big: a circle's radius, the others' length.
    #[param(min = 0.001, soft_max = 10.0, unit = "length")]
    pub size: f64,
    #[param(unit = "length")]
    pub location: [f64; 3],
}

impl Default for AddCurveParams {
    fn default() -> Self {
        Self {
            shape: 0,
            size: 1.0,
            location: [0.0; 3],
        }
    }
}

/// Adds a curve object.
pub struct AddCurve;

impl Operator for AddCurve {
    type Params = AddCurveParams;
    fn id() -> &'static str {
        "object.add_curve"
    }
    fn label() -> &'static str {
        "Add Curve"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let s = params.size;
        if !(s > 0.0 && s.is_finite()) {
            return Err(OpError::Failed("a curve needs a size".into()));
        }
        let (name, curve) = match params.shape {
            1 => ("Circle", Curve::circle(s)),
            2 => (
                "Path",
                Curve::poly(
                    &[
                        DVec3::new(-s, 0.0, 0.0),
                        DVec3::new(-s * 0.5, 0.0, 0.0),
                        DVec3::ZERO,
                        DVec3::new(s * 0.5, 0.0, 0.0),
                        DVec3::new(s, 0.0, 0.0),
                    ],
                    false,
                ),
            ),
            3 => {
                let mut c = Curve::poly(
                    &[
                        DVec3::new(-s, 0.0, 0.0),
                        DVec3::new(-s * 0.5, s * 0.5, 0.0),
                        DVec3::new(s * 0.5, -s * 0.5, 0.0),
                        DVec3::new(s, 0.0, 0.0),
                    ],
                    false,
                );
                c.splines[0].kind = SplineKind::Nurbs;
                ("NurbsCurve", c)
            }
            _ => {
                // Blender's Bézier: two points, an S through them.
                let pts = vec![
                    CurvePoint {
                        position: DVec3::new(-s, 0.0, 0.0),
                        left: DVec3::new(-s * 1.5, -s * 0.5, 0.0),
                        right: DVec3::new(-s * 0.5, s * 0.5, 0.0),
                        radius: 1.0,
                        tilt: 0.0,
                    },
                    CurvePoint {
                        position: DVec3::new(s, 0.0, 0.0),
                        left: DVec3::new(s * 0.5, -s * 0.5, 0.0),
                        right: DVec3::new(s * 1.5, s * 0.5, 0.0),
                        radius: 1.0,
                        tilt: 0.0,
                    },
                ];
                (
                    "BezierCurve",
                    Curve {
                        splines: vec![Spline::new(SplineKind::Bezier, pts, false)],
                        ..Default::default()
                    },
                )
            }
        };
        let d = doc.document_mut();
        let mesh = d.add_mesh(curve.to_mesh());
        let name = d.unique_object_name(name);
        let mut object = Object::new(name, ObjectData::Mesh(mesh));
        object.transform.translation = DVec3::from_array(params.location);
        object.curve = Some(curve);
        let id = d.add_object(object);
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        Ok(OpResult::Finished)
    }
}

/// The settings `curve.set` takes, as a curve has them now: what the panel
/// starts from, so an edit of one leaves the rest alone.
pub fn curve_params(curve: &Curve) -> SetParams {
    SetParams {
        resolution: curve.resolution,
        bevel_depth: curve.bevel_depth,
        bevel_resolution: curve.bevel_resolution,
        extrude: curve.extrude,
        fill_caps: curve.fill_caps,
        fill: curve.fill,
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetParams {
    /// Steps a segment when the curve is sampled.
    #[param(min = 1, max = 64, unit = "count")]
    pub resolution: u32,
    /// A round profile of this radius swept along the curve; 0 is a wire.
    #[param(min = 0.0, soft_max = 1.0, unit = "length")]
    pub bevel_depth: f64,
    /// Segments round the bevel.
    #[param(min = 3, max = 256, unit = "count")]
    pub bevel_resolution: u32,
    /// A ribbon this wide across the curve; with a bevel, a rounded slab.
    #[param(min = 0.0, soft_max = 1.0, unit = "length")]
    pub extrude: f64,
    /// A tube's ends closed.
    pub fill_caps: bool,
    /// A closed curve's inside filled — a face, or a slab when extruded.
    pub fill: bool,
}

impl Default for SetParams {
    fn default() -> Self {
        curve_params(&Curve::default())
    }
}

/// Sets how the active curve is shown.
pub struct Set;

impl Operator for Set {
    type Params = SetParams;
    fn id() -> &'static str {
        "curve.set"
    }
    fn label() -> &'static str {
        "Curve Settings"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_curve(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_curve(doc.document())?;
        let c = curve_mut(doc, id);
        c.resolution = params.resolution.clamp(1, 64);
        c.bevel_depth = params.bevel_depth.max(0.0);
        c.bevel_resolution = params.bevel_resolution.clamp(3, 256);
        c.extrude = params.extrude.max(0.0);
        c.fill_caps = params.fill_caps;
        c.fill = params.fill;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PointAddParams {
    /// Which spline; a new one past the last.
    pub spline: u32,
    /// Where, in the object's space.
    #[param(unit = "length")]
    pub position: [f64; 3],
    /// After this point; past the end, at the end.
    pub after: u32,
}

impl Default for PointAddParams {
    fn default() -> Self {
        Self {
            spline: 0,
            position: [0.0; 3],
            after: u32::MAX,
        }
    }
}

/// Adds a point to a spline of the active curve, its handles continuing
/// the run.
pub struct PointAdd;

impl Operator for PointAdd {
    type Params = PointAddParams;
    fn id() -> &'static str {
        "curve.point_add"
    }
    fn label() -> &'static str {
        "Add Curve Point"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_curve(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_curve(doc.document())?;
        let position = DVec3::from_array(params.position);
        if !position.is_finite() {
            return Err(OpError::Failed("a point needs a place".into()));
        }
        let c = curve_mut(doc, id);
        let s = params.spline as usize;
        if s >= c.splines.len() {
            c.splines.push(Spline::new(
                SplineKind::Bezier,
                vec![CurvePoint::at(position)],
                false,
            ));
            return Ok(OpResult::Finished);
        }
        let spline = &mut c.splines[s];
        let at = (params.after as usize)
            .saturating_add(1)
            .min(spline.points.len());
        // Handles a third of the way back towards the point before it.
        let tangent = at
            .checked_sub(1)
            .and_then(|i| spline.points.get(i))
            .map(|p| (position - p.position) / 3.0)
            .unwrap_or(DVec3::ZERO);
        spline
            .points
            .insert(at, CurvePoint::with_handles(position, tangent));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PointSetParams {
    pub spline: u32,
    pub index: u32,
    #[param(unit = "length")]
    pub position: [f64; 3],
    /// The handles, as positions in the object's space.
    #[param(unit = "length")]
    pub left: [f64; 3],
    #[param(unit = "length")]
    pub right: [f64; 3],
    #[param(min = 0.0, soft_max = 10.0)]
    pub radius: f64,
    #[param(unit = "angle")]
    pub tilt: f64,
}

impl Default for PointSetParams {
    fn default() -> Self {
        Self {
            spline: 0,
            index: 0,
            position: [0.0; 3],
            left: [0.0; 3],
            right: [0.0; 3],
            radius: 1.0,
            tilt: 0.0,
        }
    }
}

/// The parameters that leave a point as it is.
pub fn point_params(spline: u32, index: u32, p: &CurvePoint) -> PointSetParams {
    PointSetParams {
        spline,
        index,
        position: p.position.to_array(),
        left: p.left.to_array(),
        right: p.right.to_array(),
        radius: p.radius,
        tilt: p.tilt,
    }
}

/// Sets a point of the active curve: where it and its handles are.
pub struct PointSet;

impl Operator for PointSet {
    type Params = PointSetParams;
    fn id() -> &'static str {
        "curve.point_set"
    }
    fn label() -> &'static str {
        "Set Curve Point"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_curve(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_curve(doc.document())?;
        let c = curve_mut(doc, id);
        let point = c
            .splines
            .get_mut(params.spline as usize)
            .and_then(|s| s.points.get_mut(params.index as usize))
            .ok_or_else(|| OpError::Failed("no such point".into()))?;
        let v = |a: [f64; 3]| DVec3::from_array(a);
        if !(v(params.position).is_finite()
            && v(params.left).is_finite()
            && v(params.right).is_finite())
        {
            return Err(OpError::Failed("a point needs a place".into()));
        }
        point.position = v(params.position);
        point.left = v(params.left);
        point.right = v(params.right);
        point.radius = params.radius.max(0.0);
        point.tilt = params.tilt;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PointRemoveParams {
    pub spline: u32,
    pub index: u32,
}

/// Takes a point out of the active curve; a spline left with none goes.
pub struct PointRemove;

impl Operator for PointRemove {
    type Params = PointRemoveParams;
    fn id() -> &'static str {
        "curve.point_remove"
    }
    fn label() -> &'static str {
        "Remove Curve Point"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_curve(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_curve(doc.document())?;
        let c = curve_mut(doc, id);
        let s = params.spline as usize;
        let Some(spline) = c.splines.get_mut(s) else {
            return Err(OpError::Failed("no such spline".into()));
        };
        if params.index as usize >= spline.points.len() {
            return Err(OpError::Failed("no such point".into()));
        }
        spline.points.remove(params.index as usize);
        if spline.points.is_empty() {
            c.splines.remove(s);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SplineSetParams {
    pub spline: u32,
    #[param(variants("bezier", "poly", "nurbs"))]
    pub kind: u32,
    pub closed: bool,
}

/// Sets a spline's kind and whether it closes.
pub struct SplineSet;

impl Operator for SplineSet {
    type Params = SplineSetParams;
    fn id() -> &'static str {
        "curve.spline_set"
    }
    fn label() -> &'static str {
        "Set Spline"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_curve(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_curve(doc.document())?;
        let c = curve_mut(doc, id);
        let spline = c
            .splines
            .get_mut(params.spline as usize)
            .ok_or_else(|| OpError::Failed("no such spline".into()))?;
        let kind = SplineKind::ALL
            .get(params.kind as usize)
            .copied()
            .unwrap_or(SplineKind::Bezier);
        // Into Bézier from a kind without handles: handles laid along the
        // run, so the curve keeps its shape rather than turning to corners.
        if kind == SplineKind::Bezier && spline.kind != SplineKind::Bezier {
            let n = spline.points.len();
            let positions: Vec<DVec3> = spline.points.iter().map(|p| p.position).collect();
            for (i, p) in spline.points.iter_mut().enumerate() {
                let before = if i > 0 {
                    positions[i - 1]
                } else if spline.closed && n > 1 {
                    positions[n - 1]
                } else {
                    positions[i]
                };
                let after = if i + 1 < n {
                    positions[i + 1]
                } else if spline.closed && n > 1 {
                    positions[0]
                } else {
                    positions[i]
                };
                let tangent = (after - before) / 6.0;
                p.left = p.position - tangent;
                p.right = p.position + tangent;
            }
        }
        spline.kind = kind;
        spline.closed = params.closed;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ToMeshParams {}

/// Makes the active curve object a plain mesh of what it shows: the curve
/// is let go.
pub struct ToMesh;

impl Operator for ToMesh {
    type Params = ToMeshParams;
    fn id() -> &'static str {
        "curve.to_mesh"
    }
    fn label() -> &'static str {
        "Convert Curve to Mesh"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_curve(doc).is_some()
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_curve(doc.document())?;
        let d = doc.document();
        let o = d.object(id).expect("checked");
        let curve = o.curve.clone().expect("checked");
        let mesh = curve.to_mesh();
        if mesh.num_faces() == 0 {
            return Err(OpError::Failed(
                "a wire makes no mesh: give the curve a bevel, an extrude or a fill first".into(),
            ));
        }
        let mesh_id = o.mesh().expect("a curve object is a mesh object");
        let d = doc.document_mut();
        *d.mesh_mut(mesh_id).expect("the curve's mesh") = mesh;
        d.object_mut(id).expect("checked").curve = None;
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};

    fn bench() -> (OperatorRegistry, History, Document) {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        (r, History::new(), Document::new())
    }

    fn run(
        (r, h, doc): &mut (OperatorRegistry, History, Document),
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(r, doc, &Context { mode: Mode::Object }, id, &p)
    }

    #[test]
    fn a_circle_is_added_bevelled_and_made_a_mesh() {
        let mut b = bench();
        run(
            &mut b,
            "object.add_curve",
            &[("shape", PropertyValue::Enum(1))],
        )
        .unwrap();
        let id = active_curve(&b.2).expect("a curve object");
        assert_eq!(b.2.object(id).unwrap().name, "Circle");
        assert!(b.2.object(id).unwrap().curve.as_ref().unwrap().is_wire());
        // Settings from the curve, one changed: the rest as they were.
        let mut p = kreate_reflect::OpParams::to_map(&curve_params(
            b.2.object(id).unwrap().curve.as_ref().unwrap(),
        ));
        p.set("bevel_depth", PropertyValue::Float(0.1));
        b.1.dispatch(
            &b.0,
            &mut b.2,
            &Context { mode: Mode::Object },
            "curve.set",
            &p,
        )
        .unwrap();
        let c = b.2.object(id).unwrap().curve.as_ref().unwrap();
        assert_eq!(c.bevel_depth, 0.1);
        assert_eq!(c.resolution, 12, "left alone");
        // A mesh now, and no curve.
        run(&mut b, "curve.to_mesh", &[]).unwrap();
        let o = b.2.object(id).unwrap();
        assert!(o.curve.is_none());
        assert!(b.2.mesh(o.mesh().unwrap()).unwrap().num_faces() > 0);
        // And one step each, undone in turn.
        assert!(b.1.undo(&mut b.2));
        assert!(b.2.object(id).unwrap().curve.is_some());
    }

    #[test]
    fn points_are_added_moved_and_removed() {
        let mut b = bench();
        run(
            &mut b,
            "object.add_curve",
            &[("shape", PropertyValue::Enum(2))],
        )
        .unwrap();
        let id = active_curve(&b.2).unwrap();
        let count = |d: &Document| {
            d.object(id).unwrap().curve.as_ref().unwrap().splines[0]
                .points
                .len()
        };
        assert_eq!(count(&b.2), 5);
        run(
            &mut b,
            "curve.point_add",
            &[("position", PropertyValue::Vec3([2.0, 1.0, 0.0]))],
        )
        .unwrap();
        assert_eq!(count(&b.2), 6);
        run(
            &mut b,
            "curve.point_set",
            &[
                ("index", PropertyValue::Int(5)),
                ("position", PropertyValue::Vec3([2.0, 2.0, 0.0])),
                ("left", PropertyValue::Vec3([1.5, 2.0, 0.0])),
                ("right", PropertyValue::Vec3([2.5, 2.0, 0.0])),
            ],
        )
        .unwrap();
        let p = b.2.object(id).unwrap().curve.as_ref().unwrap().splines[0].points[5];
        assert_eq!(p.position, DVec3::new(2.0, 2.0, 0.0));
        run(
            &mut b,
            "curve.point_remove",
            &[("index", PropertyValue::Int(0))],
        )
        .unwrap();
        assert_eq!(count(&b.2), 5);
        assert!(run(
            &mut b,
            "curve.point_remove",
            &[("index", PropertyValue::Int(99))]
        )
        .is_err());
    }

    #[test]
    fn a_path_made_bezier_keeps_its_shape_and_can_close() {
        let mut b = bench();
        run(
            &mut b,
            "object.add_curve",
            &[("shape", PropertyValue::Enum(2))],
        )
        .unwrap();
        let id = active_curve(&b.2).unwrap();
        run(
            &mut b,
            "curve.spline_set",
            &[
                ("kind", PropertyValue::Enum(0)),
                ("closed", PropertyValue::Bool(true)),
            ],
        )
        .unwrap();
        let s = &b.2.object(id).unwrap().curve.as_ref().unwrap().splines[0];
        assert_eq!(s.kind, SplineKind::Bezier);
        assert!(s.closed);
        assert!(
            s.points.iter().all(|p| p.left != p.position),
            "handles laid out"
        );
    }

    #[test]
    fn a_curve_object_is_evaluated_as_its_curve() {
        let mut b = bench();
        run(
            &mut b,
            "object.add_curve",
            &[("shape", PropertyValue::Enum(1))],
        )
        .unwrap();
        let id = active_curve(&b.2).unwrap();
        let mut p = PropertyMap::new();
        p.set("bevel_depth", PropertyValue::Float(0.2));
        p.set("resolution", PropertyValue::Int(12));
        p.set("bevel_resolution", PropertyValue::Int(8));
        p.set("fill_caps", PropertyValue::Bool(true));
        b.1.dispatch(
            &b.0,
            &mut b.2,
            &Context { mode: Mode::Object },
            "curve.set",
            &p,
        )
        .unwrap();
        let mut cache = kreate_scene::ModifierCache::new();
        let ctx = kreate_scene::modifier::EvalCtx::default();
        let shown = cache.evaluated(&b.2, id, &ctx).expect("evaluated");
        assert!(
            shown.num_faces() > 0,
            "the tube, not the empty mesh it holds"
        );
        assert!(kreate_mesh::measure::is_closed(&shown));
    }

    #[test]
    fn a_curve_survives_the_file() {
        let mut b = bench();
        run(
            &mut b,
            "object.add_curve",
            &[("shape", PropertyValue::Enum(0))],
        )
        .unwrap();
        let id = active_curve(&b.2).unwrap();
        let bytes = kreate_io::container::write(&b.2, &Default::default(), &[]).unwrap();
        let back = kreate_io::container::read(std::io::Cursor::new(bytes))
            .unwrap()
            .document;
        let (_, o) = back.objects().next().unwrap();
        assert_eq!(o.curve, b.2.object(id).unwrap().curve);
    }
}
