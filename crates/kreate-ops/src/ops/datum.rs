//! Construction geometry, as Fusion's Construct menu makes it: planes and
//! axes to draw on and turn about, where no face of a part is.
//!
//! A plane is an empty whose XY is the plane and whose Z its normal; an
//! axis, an empty whose Z is the axis. Each is marked by its `generator` —
//! `datum:plane <size>` or `datum:axis <length>` — which says what to draw it
//! as and how big. A sketch goes on a plane by `draft.new(plane="object")`.

use kreate_math::{DQuat, DVec3};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Plane>();
    registry.register::<PlaneThroughPoints>();
    registry.register::<Axis>();
}

/// What a construction object is, by its `generator`: a plane or an axis,
/// and how big it is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Datum {
    Plane { size: f64 },
    Axis { length: f64 },
}

/// What `object` is, if it is construction geometry.
pub fn datum_of(object: &Object) -> Option<Datum> {
    let line = object.generator.as_deref()?.strip_prefix("datum:")?;
    let (kind, size) = line.split_once(' ').unwrap_or((line, "0.05"));
    let size = size
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|s| *s > 0.0)
        .unwrap_or(0.05);
    match kind {
        "plane" => Some(Datum::Plane { size }),
        "axis" => Some(Datum::Axis { length: size }),
        _ => None,
    }
}

/// How big a new plane is drawn: a little more than the parts are wide, so
/// it reads as theirs; a tenth of a metre with nothing there.
fn scale_of(doc: &Document) -> f64 {
    let mut lo = DVec3::splat(f64::MAX);
    let mut hi = DVec3::splat(f64::MIN);
    for (id, o) in doc.objects() {
        let Some(mesh) = o.mesh().and_then(|m| doc.mesh(m)) else {
            continue;
        };
        if mesh.num_verts() == 0 || !doc.is_shown(id) {
            continue;
        }
        let b = mesh.bounds();
        let w = doc.world_transform(id);
        for p in [b.min, b.max] {
            let p = w.transform_point3(p.as_dvec3());
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if lo.x > hi.x {
        return 0.05;
    }
    ((hi - lo).max_element() * 0.6).clamp(0.005, 50.0)
}

/// An empty placed at `at`, its Z along `z`, marked as `line`; selected.
fn add_datum(doc: &mut Document, name: &str, at: DVec3, z: DVec3, line: String) -> ObjectId {
    let name = doc.unique_object_name(name);
    let mut o = Object::new(name, ObjectData::Empty);
    o.transform.translation = at;
    o.transform.rotation = DQuat::from_rotation_arc(DVec3::Z, z.normalize());
    o.generator = Some(line);
    let id = doc.add_object(o);
    doc.selection.clear();
    doc.selection.add(id);
    doc.selection.active = Some(id);
    id
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PlaneParams {
    /// A point the plane goes from, and the way it faces: a face's middle
    /// and its normal, or the origin and a main axis.
    #[param(unit = "length")]
    pub origin: [f64; 3],
    pub normal: [f64; 3],
    /// How far it is moved along its normal: Fusion's Offset Plane.
    #[param(unit = "length", soft_min = -0.1, soft_max = 0.1)]
    pub offset: f64,
    /// How far it is turned about `hinge` — a direction in it, through
    /// its origin — Fusion's Plane at Angle; nought for none.
    #[param(unit = "angle")]
    pub angle: f64,
    pub hinge: [f64; 3],
}

impl Default for PlaneParams {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            offset: 0.01,
            angle: 0.0,
            hinge: [1.0, 0.0, 0.0],
        }
    }
}

/// A construction plane: off a face or a main plane by a distance, turned
/// by an angle if asked. What a sketch is drawn on where no face is.
pub struct Plane;

impl Operator for Plane {
    type Params = PlaneParams;
    fn id() -> &'static str {
        "datum.plane"
    }
    fn label() -> &'static str {
        "Construction Plane"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let origin = DVec3::from_array(params.origin);
        let mut normal = DVec3::from_array(params.normal).normalize_or_zero();
        if normal == DVec3::ZERO || !origin.is_finite() || !params.offset.is_finite() {
            return Err(OpError::Failed(
                "a plane needs a point and a direction".into(),
            ));
        }
        let mut at = origin;
        if params.angle != 0.0 && params.angle.is_finite() {
            // The hinge, laid in the plane.
            let hinge = DVec3::from_array(params.hinge);
            let hinge = (hinge - normal * hinge.dot(normal)).normalize_or_zero();
            if hinge == DVec3::ZERO {
                return Err(OpError::Failed(
                    "the plane turns about a direction that lies in it".into(),
                ));
            }
            normal = DQuat::from_axis_angle(hinge, params.angle) * normal;
        }
        at += normal * params.offset;
        let size = scale_of(doc.document());
        add_datum(
            doc.document_mut(),
            "Plane",
            at,
            normal,
            format!("datum:plane {size}"),
        );
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PlaneThroughPointsParams {
    /// Three points, x, y, z each, in the world; or four for a midplane:
    /// two faces, each a point and its normal.
    #[param(unit = "length")]
    pub points: Vec<f64>,
    /// The four numbers are two faces: the plane half way between them.
    pub midplane: bool,
}

/// A construction plane through three points — Fusion's Plane Through
/// Three Points — or half way between two faces, Fusion's Midplane.
pub struct PlaneThroughPoints;

impl Operator for PlaneThroughPoints {
    type Params = PlaneThroughPointsParams;
    fn id() -> &'static str {
        "datum.plane_points"
    }
    fn label() -> &'static str {
        "Plane Through Points"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let p: Vec<DVec3> = params
            .points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| DVec3::from_array(*c))
            .collect();
        let (at, normal, name) = if params.midplane {
            let [a, na, b, nb] = p[..] else {
                return Err(OpError::Failed(
                    "a midplane takes two faces: a point and a normal each".into(),
                ));
            };
            // Half way along, square to the first face's normal; faces
            // that look the same way or opposite ways alike.
            let n = na.normalize_or_zero();
            let nb = nb.normalize_or_zero();
            if n == DVec3::ZERO || n.dot(nb).abs() < 0.999 {
                return Err(OpError::Failed(
                    "a midplane is between two parallel faces".into(),
                ));
            }
            let middle = a + n * ((b - a).dot(n) * 0.5);
            (middle, n, "Midplane")
        } else {
            let [a, b, c] = p[..] else {
                return Err(OpError::Failed("a plane through points takes three".into()));
            };
            let n = (b - a).cross(c - a).normalize_or_zero();
            if n == DVec3::ZERO {
                return Err(OpError::Failed(
                    "the three points are on one line: no plane goes through them alone".into(),
                ));
            }
            ((a + b + c) / 3.0, n, "Plane")
        };
        if !at.is_finite() {
            return Err(OpError::Failed("no plane there".into()));
        }
        let size = scale_of(doc.document());
        add_datum(
            doc.document_mut(),
            name,
            at,
            normal,
            format!("datum:plane {size}"),
        );
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AxisParams {
    /// Two points it goes through, in the world.
    #[param(unit = "length")]
    pub a: [f64; 3],
    #[param(unit = "length")]
    pub b: [f64; 3],
}

impl Default for AxisParams {
    fn default() -> Self {
        Self {
            a: [0.0; 3],
            b: [0.0, 0.0, 0.1],
        }
    }
}

/// A construction axis through two points — a hole's two ends, two
/// corners — to turn a revolve or a joint about. Fusion's Axis Through Two
/// Points.
pub struct Axis;

impl Operator for Axis {
    type Params = AxisParams;
    fn id() -> &'static str {
        "datum.axis"
    }
    fn label() -> &'static str {
        "Construction Axis"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let (a, b) = (DVec3::from_array(params.a), DVec3::from_array(params.b));
        let length = a.distance(b);
        if !length.is_finite() || length < 1e-9 {
            return Err(OpError::Failed("an axis needs two points apart".into()));
        }
        // Drawn past its ends, as an axis is.
        let shown = length.max(scale_of(doc.document()));
        add_datum(
            doc.document_mut(),
            "Axis",
            (a + b) * 0.5,
            b - a,
            format!("datum:axis {}", shown * 0.75),
        );
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::OperatorRegistry;
    use crate::History;
    use kreate_reflect::{PropertyMap, PropertyValue as V};

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        doc: &mut Document,
        id: &str,
        pairs: &[(&'static str, V)],
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(r, doc, &Context::default(), id, &p)
    }

    /// An offset plane sits off its face; one through three points lies on
    /// them; a midplane half way; an axis through two points along them;
    /// and a sketch made on a plane lies in it.
    #[test]
    fn planes_and_axes_are_where_they_are_asked_for() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(
            &r,
            &mut h,
            &mut doc,
            "datum.plane",
            &[
                ("origin", V::Vec3([0.0, 0.0, 0.01])),
                ("normal", V::Vec3([0.0, 0.0, 1.0])),
                ("offset", V::Float(0.02)),
            ],
        )
        .unwrap();
        let plane = doc.selection.active.unwrap();
        let o = doc.object(plane).unwrap();
        assert!(matches!(datum_of(o), Some(Datum::Plane { .. })));
        assert!((o.transform.translation - DVec3::new(0.0, 0.0, 0.03)).length() < 1e-12);

        run(
            &r,
            &mut h,
            &mut doc,
            "datum.plane_points",
            &[(
                "points",
                V::Floats(vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            )],
        )
        .unwrap();
        let tilted = doc.object(doc.selection.active.unwrap()).unwrap();
        let n = tilted.transform.rotation * DVec3::Z;
        // Square to both the points' directions.
        assert!(n.dot(DVec3::new(1.0, 0.0, 1.0)).abs() < 1e-9);
        assert!(n.dot(DVec3::Y).abs() < 1e-9);

        run(
            &r,
            &mut h,
            &mut doc,
            "datum.plane_points",
            &[
                (
                    "points",
                    V::Floats(vec![
                        0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.04, 0.0, 0.0, 1.0,
                    ]),
                ),
                ("midplane", V::Bool(true)),
            ],
        )
        .unwrap();
        let mid = doc.object(doc.selection.active.unwrap()).unwrap();
        assert!((mid.transform.translation.z - 0.02).abs() < 1e-12);

        run(
            &r,
            &mut h,
            &mut doc,
            "datum.axis",
            &[
                ("a", V::Vec3([0.0, 0.0, 0.0])),
                ("b", V::Vec3([0.0, 0.1, 0.0])),
            ],
        )
        .unwrap();
        let axis = doc.object(doc.selection.active.unwrap()).unwrap();
        assert!(matches!(datum_of(axis), Some(Datum::Axis { .. })));
        assert!((axis.transform.rotation * DVec3::Z - DVec3::Y).length() < 1e-9);

        // A sketch on the first plane: its plane is the plane's.
        let name = doc.object(plane).unwrap().name.clone();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.new",
            &[("plane", V::Enum(5)), ("object", V::Text(name))],
        )
        .unwrap();
        let sketch = doc.selection.active.unwrap();
        let w = doc.world_transform(sketch);
        assert!((w.translation - DVec3::new(0.0, 0.0, 0.03)).length() < 1e-12);
    }
}
