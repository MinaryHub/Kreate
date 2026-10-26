//! Drawing in 2D: the operators of the drafting mode. See
//! design/2d-drafting.md and `kreate_scene::drawing`.
//!
//! Every operator acts on the active object's drawing, takes places on the
//! drawing's plane (`[x, y, _]`, the third ignored, as the sketch tools
//! took them), and ends by solving the drawing, so what it leaves is held.
//!
//! An entity is named by its id, and a part of one as `id.part` — `3`,
//! `3.start`, `3.end`, `3.center`, `3.mid`, `3.v2` — in a list separated by
//! spaces or commas: the form a script and the connector write.

use kreate_geom2d::{Curve2D, DVec2, Spline};
use kreate_math::{DMat3, DQuat, DVec3};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::drawing::{
    Constraint, DimKind, Dimension, Drawing, EntityId, EntityKind, EntityRef, Part,
};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::ops::sketch_draw::NewSketchParams;
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<NewDrawing>();
    registry.register::<DraftLine>();
    registry.register::<DraftRectangle>();
    registry.register::<DraftCircle>();
    registry.register::<DraftArc>();
    registry.register::<DraftSpline>();
    registry.register::<DraftPoint>();
    registry.register::<DraftConstrain>();
    registry.register::<DraftDimension>();
    registry.register::<DraftDimensionSet>();
    registry.register::<DraftDelete>();
    registry.register::<DraftDrag>();
    registry.register::<DraftConstruction>();
    registry.register::<DraftTrim>();
    registry.register::<DraftOffset>();
    registry.register::<DraftFillet>();
}

/// The active object, when it is a drawing.
pub fn active_drawing(doc: &Document) -> Option<ObjectId> {
    let id = doc.selection.active?;
    doc.object(id)?.drawing.as_ref()?;
    Some(id)
}

fn need_drawing(doc: &Document) -> Result<ObjectId, OpError> {
    active_drawing(doc).ok_or_else(|| OpError::Failed("no drawing is being drawn".into()))
}

fn xy(p: [f64; 3]) -> DVec2 {
    DVec2::new(p[0], p[1])
}

/// Changes the active drawing, then solves it. Refused, with nothing
/// changed, when what is left cannot be held.
fn edit(
    doc: &mut DocMut<'_>,
    change: impl FnOnce(&mut Drawing) -> Result<(), OpError>,
) -> Result<OpResult, OpError> {
    let id = need_drawing(doc.document())?;
    let mut d = doc
        .document()
        .object(id)
        .and_then(|o| o.drawing.as_deref())
        .cloned()
        .expect("a drawing");
    change(&mut d)?;
    let out = d.solve();
    if out.iter().any(|p| !p.converged) {
        return Err(OpError::Failed(
            "that contradicts what the drawing already holds".into(),
        ));
    }
    doc.document_mut()
        .object_mut(id)
        .expect("the drawing")
        .drawing = Some(Box::new(d));
    Ok(OpResult::Finished)
}

/// `3`, `3.start`, `3.v2`: an entity and the part of it.
pub fn parse_ref(text: &str) -> Option<EntityRef> {
    let (id, part) = match text.split_once('.') {
        Some((id, part)) => (id, part),
        None => (text, "whole"),
    };
    let entity = EntityId(id.trim().parse().ok()?);
    let part = match part.trim() {
        "whole" | "" => Part::Whole,
        "start" => Part::Start,
        "end" => Part::End,
        "center" | "centre" => Part::Center,
        "mid" => Part::Mid,
        v if v.starts_with('v') => Part::Vertex(v[1..].parse().ok()?),
        _ => return None,
    };
    Some(EntityRef { entity, part })
}

fn parse_refs(text: &str) -> Result<Vec<EntityRef>, OpError> {
    text.split([' ', ','])
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse_ref(s).ok_or_else(|| OpError::Failed(format!("`{s}` names no entity"))))
        .collect()
}

fn known(d: &Drawing, refs: &[EntityRef]) -> Result<(), OpError> {
    for r in refs {
        if d.entity(r.entity).is_none() {
            return Err(OpError::Failed(format!(
                "no entity {} in the drawing",
                r.entity.0
            )));
        }
    }
    Ok(())
}

/// A line level or upright when it was drawn so: the hold a person meant
/// when they drew along the grid.
fn square_up(d: &mut Drawing, id: EntityId, a: DVec2, b: DVec2) {
    let dir = b - a;
    let len = dir.length();
    if len < 1e-12 {
        return;
    }
    if (dir.y / len).abs() < 1e-9 {
        d.constraints
            .push(Constraint::Horizontal(EntityRef::whole(id), None));
    } else if (dir.x / len).abs() < 1e-9 {
        d.constraints
            .push(Constraint::Vertical(EntityRef::whole(id), None));
    }
}

// --- a new drawing -------------------------------------------------------------

/// Starts a drawing on a plane: the ground, the front, the side, the work
/// plane at the 3D cursor, or a part's flat face — whose outline comes in
/// held, to draw against, as a face's edges do in Fusion.
pub struct NewDrawing;

impl Operator for NewDrawing {
    type Params = NewSketchParams;
    fn id() -> &'static str {
        "draft.new"
    }
    fn label() -> &'static str {
        "New Drawing"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mut outlines: Vec<Vec<DVec2>> = Vec::new();
        let mut host: Option<ObjectId> = None;
        let (location, rotation) = match params.plane {
            1 => (
                DVec3::ZERO,
                DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
            ),
            2 => (
                DVec3::ZERO,
                DQuat::from_mat3(&DMat3::from_cols(DVec3::Y, DVec3::Z, DVec3::X)),
            ),
            3 => {
                let d = doc.document();
                (d.cursor, d.cursor_rotation)
            }
            4 => {
                let frame = crate::ops::sketch_draw::face_plane(doc.document(), params)?;
                host = crate::ops::sketch_draw::face_host(doc.document(), params);
                outlines = frame.loops;
                (frame.origin, frame.rotation)
            }
            _ => (DVec3::ZERO, DQuat::IDENTITY),
        };
        let mut drawing = Drawing::new();
        // The face's outlines, held where the face is.
        for ring in &outlines {
            if ring.len() < 3 {
                continue;
            }
            let id = drawing.add_curve(Curve2D::Polyline {
                verts: ring.iter().map(|p| (*p, 0.0)).collect(),
                closed: true,
            });
            for (k, p) in ring.iter().enumerate() {
                drawing.constraints.push(Constraint::Fixed(
                    EntityRef::at(id, Part::Vertex(k as u32)),
                    *p,
                ));
            }
        }
        let d = doc.document_mut();
        let name = d.unique_object_name("Drawing");
        let mut object = Object::new(name, ObjectData::Empty);
        object.transform.translation = location;
        object.transform.rotation = rotation;
        object.drawing = Some(Box::new(drawing));
        let id = d.add_object(object);
        if let Some(h) = host {
            let _ = d.set_parent(id, Some(h));
        }
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        Ok(OpResult::Finished)
    }
}

// --- drawing ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftLineParams {
    /// x, y pairs on the drawing's plane: a line between each and the next.
    #[param(unit = "length")]
    pub points: Vec<f64>,
    /// Back from the last to the first.
    pub closed: bool,
}

/// Lines from point to point, each one's end held to the next one's start,
/// and level or upright where they were drawn so.
pub struct DraftLine;

impl Operator for DraftLine {
    type Params = DraftLineParams;
    fn id() -> &'static str {
        "draft.line"
    }
    fn label() -> &'static str {
        "Line"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let pts: Vec<DVec2> = params
            .points
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| DVec2::new(c[0], c[1]))
            .collect();
        if pts.len() < 2 {
            return Err(OpError::Failed("a line needs two points".into()));
        }
        edit(doc, |d| {
            chain(d, &pts, params.closed);
            Ok(())
        })
    }
}

/// Lines through `pts`, joined end to start.
fn chain(d: &mut Drawing, pts: &[DVec2], closed: bool) -> Vec<EntityId> {
    let n = pts.len();
    let segments = if closed { n } else { n - 1 };
    let mut ids = Vec::new();
    for k in 0..segments {
        let (a, b) = (pts[k], pts[(k + 1) % n]);
        let id = d.add_curve(Curve2D::Line { a, b });
        square_up(d, id, a, b);
        ids.push(id);
    }
    for k in 0..ids.len() {
        let next = if k + 1 < ids.len() {
            ids[k + 1]
        } else if closed && ids.len() > 1 {
            ids[0]
        } else {
            continue;
        };
        d.constraints.push(Constraint::Coincident(
            EntityRef::at(ids[k], Part::End),
            EntityRef::at(next, Part::Start),
        ));
    }
    ids
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftRectangleParams {
    #[param(unit = "length")]
    pub a: [f64; 3],
    #[param(unit = "length")]
    pub b: [f64; 3],
    /// `a` is the centre and `b` a corner, instead of two corners.
    pub centred: bool,
}

/// Four lines, square: two level, two upright, their corners held.
pub struct DraftRectangle;

impl Operator for DraftRectangle {
    type Params = DraftRectangleParams;
    fn id() -> &'static str {
        "draft.rectangle"
    }
    fn label() -> &'static str {
        "Rectangle"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let (a, b) = if params.centred {
            let c = xy(params.a);
            let h = xy(params.b) - c;
            (c - h, c + h)
        } else {
            (xy(params.a), xy(params.b))
        };
        if (a.x - b.x).abs() < 1e-12 || (a.y - b.y).abs() < 1e-12 {
            return Err(OpError::Failed(
                "a rectangle needs a width and a height".into(),
            ));
        }
        let corners = [a, DVec2::new(b.x, a.y), b, DVec2::new(a.x, b.y)];
        edit(doc, |d| {
            chain(d, &corners, true);
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftCircleParams {
    #[param(unit = "length")]
    pub centre: [f64; 3],
    #[param(unit = "length", min = 0.0)]
    pub radius: f64,
}

pub struct DraftCircle;

impl Operator for DraftCircle {
    type Params = DraftCircleParams;
    fn id() -> &'static str {
        "draft.circle"
    }
    fn label() -> &'static str {
        "Circle"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if params.radius <= 0.0 || !params.radius.is_finite() {
            return Err(OpError::Failed("a circle needs a radius".into()));
        }
        edit(doc, |d| {
            d.add_curve(Curve2D::Circle {
                c: xy(params.centre),
                r: params.radius,
            });
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftArcParams {
    #[param(unit = "length")]
    pub start: [f64; 3],
    /// A point the arc passes on its way.
    #[param(unit = "length")]
    pub through: [f64; 3],
    #[param(unit = "length")]
    pub end: [f64; 3],
}

/// An arc through three points: from the first, by the second, to the third.
pub struct DraftArc;

impl Operator for DraftArc {
    type Params = DraftArcParams;
    fn id() -> &'static str {
        "draft.arc"
    }
    fn label() -> &'static str {
        "Arc"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let arc = arc_through(xy(params.start), xy(params.through), xy(params.end))
            .ok_or_else(|| OpError::Failed("three points in a line make no arc".into()))?;
        edit(doc, |d| {
            d.add_curve(arc);
            Ok(())
        })
    }
}

/// The arc from `a` by `m` to `b`, or none when they are in a line.
pub fn arc_through(a: DVec2, m: DVec2, b: DVec2) -> Option<Curve2D> {
    // The centre: where the perpendicular bisectors of am and mb meet.
    let (d1, d2) = (m - a, b - m);
    let den = 2.0 * d1.perp_dot(d2);
    if den.abs() < 1e-18 {
        return None;
    }
    let (s1, s2) = (
        a.length_squared() - m.length_squared(),
        m.length_squared() - b.length_squared(),
    );
    let c = DVec2::new(
        (s1 * (m.y - b.y) - s2 * (a.y - m.y)) / -den,
        (s2 * (a.x - m.x) - s1 * (m.x - b.x)) / -den,
    );
    let r = c.distance(a);
    let (ta, tm, tb) = ((a - c).to_angle(), (m - c).to_angle(), (b - c).to_angle());
    let tau = std::f64::consts::TAU;
    // Anticlockwise from a: does it pass m before b? Otherwise it runs the
    // other way.
    let ccw_b = (tb - ta).rem_euclid(tau);
    let ccw_m = (tm - ta).rem_euclid(tau);
    let sweep = if ccw_m <= ccw_b { ccw_b } else { ccw_b - tau };
    Some(Curve2D::Arc {
        c,
        r,
        start: ta,
        sweep,
    })
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftSplineParams {
    /// x, y pairs it passes through, in order.
    #[param(unit = "length")]
    pub points: Vec<f64>,
    /// Back round to the first.
    pub closed: bool,
}

pub struct DraftSpline;

impl Operator for DraftSpline {
    type Params = DraftSplineParams;
    fn id() -> &'static str {
        "draft.spline"
    }
    fn label() -> &'static str {
        "Spline"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mut pts: Vec<DVec2> = params
            .points
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| DVec2::new(c[0], c[1]))
            .collect();
        if pts.len() < 2 {
            return Err(OpError::Failed("a spline needs two points".into()));
        }
        if params.closed {
            pts.push(pts[0]);
        }
        edit(doc, |d| {
            d.add_curve(Curve2D::Spline(Spline::through(&pts)));
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftPointParams {
    #[param(unit = "length")]
    pub at: [f64; 3],
}

pub struct DraftPoint;

impl Operator for DraftPoint {
    type Params = DraftPointParams;
    fn id() -> &'static str {
        "draft.point"
    }
    fn label() -> &'static str {
        "Point"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        edit(doc, |d| {
            d.add(EntityKind::Point(xy(params.at)));
            Ok(())
        })
    }
}

// --- holding ----------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftConstrainParams {
    #[param(variants(
        "coincident",
        "horizontal",
        "vertical",
        "parallel",
        "perpendicular",
        "tangent",
        "equal",
        "concentric",
        "point_on",
        "symmetric",
        "midpoint",
        "fixed",
        "smooth"
    ))]
    pub kind: u32,
    /// What it holds, as `id` or `id.part`, in the order the kind names
    /// them: two points for coincident; a line, or two points, for level
    /// and upright; two curves for the rest; a point then a curve for on
    /// and midpoint; two points then the line for symmetric.
    pub refs: String,
}

/// Holds entities in a relation, and solves.
pub struct DraftConstrain;

impl Operator for DraftConstrain {
    type Params = DraftConstrainParams;
    fn id() -> &'static str {
        "draft.constrain"
    }
    fn label() -> &'static str {
        "Constrain"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let refs = parse_refs(&params.refs)?;
        edit(doc, |d| {
            known(d, &refs)?;
            let need = |n: usize| -> Result<(), OpError> {
                if refs.len() < n {
                    Err(OpError::Failed(format!("that needs {n} entities")))
                } else {
                    Ok(())
                }
            };
            let id = |k: usize| refs[k].entity;
            let c = match params.kind {
                0 => {
                    need(2)?;
                    Constraint::Coincident(refs[0], refs[1])
                }
                1 | 2 => {
                    need(1)?;
                    let other = refs.get(1).copied();
                    if params.kind == 1 {
                        Constraint::Horizontal(refs[0], other)
                    } else {
                        Constraint::Vertical(refs[0], other)
                    }
                }
                3..=7 | 12 => {
                    need(2)?;
                    match params.kind {
                        3 => Constraint::Parallel(id(0), id(1)),
                        4 => Constraint::Perpendicular(id(0), id(1)),
                        5 => Constraint::Tangent(id(0), id(1)),
                        6 => Constraint::Equal(id(0), id(1)),
                        7 => Constraint::Concentric(id(0), id(1)),
                        _ => Constraint::Smooth(id(0), id(1)),
                    }
                }
                8 => {
                    need(2)?;
                    Constraint::PointOn(refs[0], id(1))
                }
                9 => {
                    need(3)?;
                    Constraint::Symmetric(refs[0], refs[1], id(2))
                }
                10 => {
                    need(2)?;
                    Constraint::Midpoint(refs[0], id(1))
                }
                _ => {
                    need(1)?;
                    let at = d
                        .point_of(refs[0])
                        .ok_or_else(|| OpError::Failed("that is not a point".into()))?;
                    Constraint::Fixed(refs[0], at)
                }
            };
            d.constraints.push(c);
            Ok(())
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftDimensionParams {
    #[param(variants(
        "horizontal",
        "vertical",
        "aligned",
        "angular",
        "radius",
        "diameter",
        "arc_length",
        "ordinate_x",
        "ordinate_y"
    ))]
    pub kind: u32,
    /// What it measures: two points, or a line, for a distance; two lines
    /// for an angle; a circle or an arc for a radius.
    pub refs: String,
    /// What it holds to, in metres or radians; zero takes it as it is now.
    pub value: f64,
    /// Holds the geometry to its value. Off, it only measures.
    pub driving: bool,
    /// Where its text goes.
    #[param(unit = "length")]
    pub at: [f64; 3],
}

impl Default for DraftDimensionParams {
    fn default() -> Self {
        Self {
            kind: 2,
            refs: String::new(),
            value: 0.0,
            driving: true,
            at: [0.0; 3],
        }
    }
}

fn dim_kind(k: u32) -> DimKind {
    [
        DimKind::Horizontal,
        DimKind::Vertical,
        DimKind::Aligned,
        DimKind::Angular,
        DimKind::Radius,
        DimKind::Diameter,
        DimKind::ArcLength,
        DimKind::OrdinateX,
        DimKind::OrdinateY,
    ][(k as usize).min(8)]
}

/// A dimension: driving, it holds the geometry to its value; driven, it
/// shows what the geometry measures. One that would contradict what is
/// already held is refused — make it driven instead.
pub struct DraftDimension;

impl Operator for DraftDimension {
    type Params = DraftDimensionParams;
    fn id() -> &'static str {
        "draft.dimension"
    }
    fn label() -> &'static str {
        "Dimension"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let refs = parse_refs(&params.refs)?;
        let kind = dim_kind(params.kind);
        edit(doc, |d| {
            known(d, &refs)?;
            let mut dim = Dimension {
                kind,
                refs: refs.clone(),
                text_at: xy(params.at),
                style: String::new(),
                driving: params.driving,
                value: params.value,
                text: None,
                tolerance: None,
            };
            let now = kreate_scene::drawing_solve::measure(&dim, &|r| d.point_of(r), &|id| {
                d.curve(id).cloned()
            })
            .ok_or_else(|| OpError::Failed("that dimension does not fit what it names".into()))?;
            if dim.value == 0.0 || !dim.driving {
                dim.value = now.abs();
            }
            d.add(EntityKind::Dimension(dim));
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftDimensionSetParams {
    /// The dimension's id.
    pub id: u32,
    pub value: f64,
    /// Whether it holds the geometry (driving) or only measures (driven).
    pub driving: bool,
}

/// A dimension's new value, and the geometry following it.
pub struct DraftDimensionSet;

impl Operator for DraftDimensionSet {
    type Params = DraftDimensionSetParams;
    fn id() -> &'static str {
        "draft.dimension_set"
    }
    fn label() -> &'static str {
        "Set Dimension"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.value.is_finite() {
            return Err(OpError::Failed("not a number".into()));
        }
        edit(doc, |d| {
            match d.entity_mut(EntityId(params.id)).map(|e| &mut e.kind) {
                Some(EntityKind::Dimension(dim)) => {
                    dim.value = params.value;
                    dim.driving = params.driving;
                    Ok(())
                }
                _ => Err(OpError::Failed(format!("{} is not a dimension", params.id))),
            }
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftDeleteParams {
    /// The entities, by id.
    pub ids: Vec<u32>,
}

/// Removes entities, with the constraints and dimensions on them.
pub struct DraftDelete;

impl Operator for DraftDelete {
    type Params = DraftDeleteParams;
    fn id() -> &'static str {
        "draft.delete"
    }
    fn label() -> &'static str {
        "Delete"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let ids: Vec<EntityId> = params.ids.iter().map(|i| EntityId(*i)).collect();
        edit(doc, |d| {
            d.remove(&ids);
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftDragParams {
    /// The point dragged, as `id.part`.
    pub at: String,
    /// Where to.
    #[param(unit = "length")]
    pub to: [f64; 3],
}

/// Drags a point: it goes as near the place as what holds it lets it, and
/// what is joined to it follows.
pub struct DraftDrag;

impl Operator for DraftDrag {
    type Params = DraftDragParams;
    fn id() -> &'static str {
        "draft.drag"
    }
    fn label() -> &'static str {
        "Drag"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let at = parse_ref(&params.at)
            .ok_or_else(|| OpError::Failed(format!("`{}` names no point", params.at)))?;
        let id = need_drawing(doc.document())?;
        let mut d = doc
            .document()
            .object(id)
            .and_then(|o| o.drawing.as_deref())
            .cloned()
            .expect("a drawing");
        known(&d, &[at])?;
        // Free geometry follows the drag directly; held geometry through
        // the solver.
        let to = xy(params.to);
        let held = d.constraints.iter().any(|c| c.entities().contains(&at.entity))
            || d.entities.iter().any(|e| {
                matches!(&e.kind, EntityKind::Dimension(x) if x.driving && x.refs.iter().any(|r| r.entity == at.entity))
            });
        if held {
            d.solve_dragging(&[kreate_scene::drawing_solve::Drag { at, to }]);
        } else if let Some(from) = d.point_of(at) {
            translate_part(&mut d, at, to - from);
        }
        doc.document_mut()
            .object_mut(id)
            .expect("the drawing")
            .drawing = Some(Box::new(d));
        Ok(OpResult::Finished)
    }
}

/// Moves one part of a free entity: a line's end, a circle's centre (the
/// whole circle), a polyline's vertex.
fn translate_part(d: &mut Drawing, at: EntityRef, by: DVec2) {
    let Some(e) = d.entity_mut(at.entity) else {
        return;
    };
    match (&mut e.kind, at.part) {
        (EntityKind::Point(p), _) => *p += by,
        (EntityKind::Text(t), _) => t.at += by,
        (EntityKind::Curve(c), part) => match (c, part) {
            (Curve2D::Line { a, .. }, Part::Start) => *a += by,
            (Curve2D::Line { b, .. }, Part::End) => *b += by,
            (Curve2D::Line { a, b }, _) => {
                *a += by;
                *b += by;
            }
            (Curve2D::Circle { c, .. }, _)
            | (Curve2D::Arc { c, .. }, _)
            | (Curve2D::Ellipse { c, .. }, _) => *c += by,
            (Curve2D::Polyline { verts, .. }, Part::Vertex(i)) => {
                if let Some(v) = verts.get_mut(i as usize) {
                    v.0 += by;
                }
            }
            (Curve2D::Polyline { verts, .. }, _) => {
                for v in verts {
                    v.0 += by;
                }
            }
            (Curve2D::Spline(s), _) => {
                let fit: Vec<DVec2> = s.fit.iter().map(|p| *p + by).collect();
                if fit.is_empty() {
                    for p in &mut s.ctrl {
                        *p += by;
                    }
                } else {
                    *s = Spline::through(&fit);
                }
            }
        },
        _ => {}
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftConstructionParams {
    pub ids: Vec<u32>,
    /// Construction (guides, bounding nothing) or not.
    pub on: bool,
}

impl Default for DraftConstructionParams {
    fn default() -> Self {
        Self {
            ids: Vec::new(),
            on: true,
        }
    }
}

/// Makes entities construction — guides that bound no region and are not
/// printed — or ordinary again.
pub struct DraftConstruction;

impl Operator for DraftConstruction {
    type Params = DraftConstructionParams;
    fn id() -> &'static str {
        "draft.construction"
    }
    fn label() -> &'static str {
        "Construction"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        edit(doc, |d| {
            for i in &params.ids {
                if let Some(e) = d.entity_mut(EntityId(*i)) {
                    e.construction = params.on;
                }
            }
            Ok(())
        })
    }
}

// --- editing ----------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftTrimParams {
    /// The curve to trim.
    pub id: u32,
    /// A point on the piece to take away: between the crossings either
    /// side of it, or to the curve's end when there is none that side.
    #[param(unit = "length")]
    pub at: [f64; 3],
}

/// Cuts away the piece of a curve between the crossings either side of a
/// point, as AutoCAD's TRIM does where it is clicked. What is left keeps
/// the entity's id (its first piece) and the rest are new.
pub struct DraftTrim;

impl Operator for DraftTrim {
    type Params = DraftTrimParams;
    fn id() -> &'static str {
        "draft.trim"
    }
    fn label() -> &'static str {
        "Trim"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let target = EntityId(params.id);
        let at = xy(params.at);
        edit(doc, |d| {
            let curve = d
                .curve(target)
                .cloned()
                .ok_or_else(|| OpError::Failed(format!("{} is not a curve", params.id)))?;
            let (lo, hi) = curve.domain();
            // Where everything else crosses it.
            let mut cuts: Vec<f64> = Vec::new();
            for e in &d.entities {
                if e.id == target {
                    continue;
                }
                if let EntityKind::Curve(other) = &e.kind {
                    for hit in kreate_geom2d::intersect(&curve, other) {
                        match hit {
                            kreate_geom2d::Intersection::Point { ta, .. } => cuts.push(ta),
                            kreate_geom2d::Intersection::Overlap { a, .. } => {
                                cuts.extend([a.0, a.1])
                            }
                        }
                    }
                }
            }
            let (t, _) = curve.closest(at);
            let below = cuts
                .iter()
                .copied()
                .filter(|c| *c < t - 1e-9)
                .fold(None, |m: Option<f64>, c| Some(m.map_or(c, |m| m.max(c))));
            let above = cuts
                .iter()
                .copied()
                .filter(|c| *c > t + 1e-9)
                .fold(None, |m: Option<f64>, c| Some(m.map_or(c, |m| m.min(c))));
            let closed = curve.is_closed();
            let keep: Vec<Curve2D> = match (below, above, closed) {
                // A closed curve cut at least twice: the rest of it, the
                // long way round.
                (Some(b), Some(a), true) => vec![curve.sub(a, b)],
                (None, Some(a), true) | (Some(a), None, true) => {
                    // Crossed on one side only of the seam: the other side's
                    // crossing is the outermost one.
                    let other = if below.is_none() {
                        cuts.iter().copied().fold(f64::MIN, f64::max)
                    } else {
                        cuts.iter().copied().fold(f64::MAX, f64::min)
                    };
                    let (from, to) = if below.is_none() {
                        (a, other)
                    } else {
                        (other, a)
                    };
                    if (from - to).abs() < 1e-12 {
                        return Err(OpError::Failed(
                            "a closed curve is trimmed between two crossings".into(),
                        ));
                    }
                    vec![curve.sub(from, to)]
                }
                (None, None, _) => {
                    return Err(OpError::Failed(
                        "nothing crosses it: delete it instead".into(),
                    ))
                }
                (b, a, false) => {
                    let mut keep = Vec::new();
                    if let Some(b) = b {
                        keep.push(curve.sub(lo, b));
                    }
                    if let Some(a) = a {
                        keep.push(curve.sub(a, hi));
                    }
                    keep
                }
            };
            // The constraints on the old curve no longer mean what they
            // did; the pieces are free, and their ends lie on what cut them.
            let layer = d.entity(target).map(|e| e.layer);
            d.remove(&[target]);
            for piece in keep {
                let id = d.add_curve(piece);
                if let (Some(l), Some(e)) = (layer, d.entity_mut(id)) {
                    e.layer = l;
                }
            }
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftOffsetParams {
    pub id: u32,
    /// How far, in metres.
    #[param(unit = "length")]
    pub distance: f64,
    /// A point on the side to offset to.
    #[param(unit = "length")]
    pub toward: [f64; 3],
}

/// A copy of a curve a distance away, on the side of a point.
pub struct DraftOffset;

impl Operator for DraftOffset {
    type Params = DraftOffsetParams;
    fn id() -> &'static str {
        "draft.offset"
    }
    fn label() -> &'static str {
        "Offset"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let toward = xy(params.toward);
        edit(doc, |d| {
            let curve = d
                .curve(EntityId(params.id))
                .cloned()
                .ok_or_else(|| OpError::Failed(format!("{} is not a curve", params.id)))?;
            // Left of travel or right: whichever is nearer the point.
            let tol = params.distance.abs() * 1e-3;
            let left = curve.offset(params.distance.abs(), tol);
            let right = curve.offset(-params.distance.abs(), tol);
            let near = |c: &Option<Curve2D>| {
                c.as_ref()
                    .map_or(f64::MAX, |c| c.closest(toward).1.distance(toward))
            };
            let chosen = if near(&left) <= near(&right) {
                left
            } else {
                right
            };
            let made =
                chosen.ok_or_else(|| OpError::Failed("nothing is left that far in".into()))?;
            d.add_curve(made);
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftFilletParams {
    pub a: u32,
    pub b: u32,
    #[param(unit = "length", min = 0.0)]
    pub radius: f64,
    /// Where each was picked: which corner is rounded.
    #[param(unit = "length")]
    pub near_a: [f64; 3],
    #[param(unit = "length")]
    pub near_b: [f64; 3],
}

/// Rounds the corner between two lines, arcs or circles: an arc tangent to
/// both, each trimmed back to where it touches.
pub struct DraftFillet;

impl Operator for DraftFillet {
    type Params = DraftFilletParams;
    fn id() -> &'static str {
        "draft.fillet"
    }
    fn label() -> &'static str {
        "Fillet"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let (ia, ib) = (EntityId(params.a), EntityId(params.b));
        let (pa, pb) = (xy(params.near_a), xy(params.near_b));
        edit(doc, |d| {
            let (Some(a), Some(b)) = (d.curve(ia).cloned(), d.curve(ib).cloned()) else {
                return Err(OpError::Failed("a fillet joins two curves".into()));
            };
            let f = kreate_geom2d::fillet(&a, pa, &b, pb, params.radius)
                .ok_or_else(|| OpError::Failed("that radius does not fit the corner".into()))?;
            // Each kept from its picked side to where the arc touches it.
            let trim = |c: &Curve2D, t: f64, near: DVec2| -> Curve2D {
                let (lo, hi) = c.domain();
                let (tn, _) = c.closest(near);
                if c.is_closed() {
                    return c.clone();
                }
                if tn < t {
                    c.sub(lo, t)
                } else {
                    c.sub(t, hi)
                }
            };
            let (na, nb) = (trim(&a, f.ta, pa), trim(&b, f.tb, pb));
            if let Some(e) = d.entity_mut(ia) {
                e.kind = EntityKind::Curve(na);
            }
            if let Some(e) = d.entity_mut(ib) {
                e.kind = EntityKind::Curve(nb);
            }
            // The corner's old meeting no longer holds: what joined the two
            // there now joins each to the arc.
            d.constraints.retain(|c| {
                !matches!(c, Constraint::Coincident(x, y)
                    if (x.entity == ia && y.entity == ib) || (x.entity == ib && y.entity == ia))
            });
            let arc = d.add_curve(f.arc);
            d.constraints.push(Constraint::Tangent(ia, arc));
            d.constraints.push(Constraint::Tangent(ib, arc));
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::{Mode, OperatorRegistry};
    use kreate_reflect::{PropertyMap, PropertyValue};

    struct Bench {
        registry: OperatorRegistry,
        history: History,
        doc: Document,
    }

    impl Bench {
        fn new() -> Bench {
            let mut registry = OperatorRegistry::new();
            crate::register_builtin(&mut registry);
            Bench {
                registry,
                history: History::new(),
                doc: Document::new(),
            }
        }
        fn run(&mut self, line: &str) -> Result<OpResult, OpError> {
            let call = crate::script::parse_call(&self.registry, line)
                .unwrap_or_else(|e| panic!("{line}: {e}"));
            self.history.dispatch(
                &self.registry,
                &mut self.doc,
                &Context { mode: Mode::Object },
                call.op_id,
                &call.params,
            )
        }
        fn drawing(&self) -> &Drawing {
            let id = active_drawing(&self.doc).expect("a drawing");
            self.doc.object(id).unwrap().drawing.as_deref().unwrap()
        }
    }

    #[test]
    fn a_rectangle_is_four_held_lines_and_one_region() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.rectangle(a=(0, 0, 0), b=(0.04, 0.02, 0))")
            .unwrap();
        let d = b.drawing();
        assert_eq!(d.profile_curves().len(), 4);
        let rs = d.regions(1e-9);
        assert_eq!(rs.len(), 1);
        assert!((rs[0].area - 0.0008).abs() < 1e-12);
        // Dimensioned, it follows.
        b.run("draft.dimension(kind=\"aligned\", refs=\"1\", value=0.06)")
            .unwrap();
        b.run("draft.constrain(kind=\"fixed\", refs=\"1.start\")")
            .unwrap();
        let rs = b.drawing().regions(1e-9);
        assert!((rs[0].area - 0.0012).abs() < 1e-9, "{}", rs[0].area);
    }

    /// The width set on its dimension, and the drawing following; then
    /// undone, one step.
    #[test]
    fn a_dimension_set_moves_the_drawing_and_undoes() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.rectangle(a=(0, 0, 0), b=(0.04, 0.02, 0))")
            .unwrap();
        b.run("draft.constrain(kind=\"fixed\", refs=\"1.start\")")
            .unwrap();
        b.run("draft.dimension(kind=\"aligned\", refs=\"1\", value=0.04)")
            .unwrap();
        let dim = b
            .drawing()
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::Dimension(_)))
            .unwrap()
            .id;
        b.run(&format!(
            "draft.dimension_set(id={}, value=0.05, driving=true)",
            dim.0
        ))
        .unwrap();
        let area = b.drawing().regions(1e-9)[0].area;
        assert!((area - 0.001).abs() < 1e-9, "{area}");
        b.history.undo(&mut b.doc);
        let area = b.drawing().regions(1e-9)[0].area;
        assert!((area - 0.0008).abs() < 1e-9, "{area}");
    }

    #[test]
    fn a_contradiction_is_refused_and_changes_nothing() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.line(points=[0, 0, 0.03, 0.01])").unwrap();
        b.run("draft.constrain(kind=\"horizontal\", refs=\"1\")")
            .unwrap();
        b.run("draft.dimension(kind=\"aligned\", refs=\"1\")")
            .unwrap();
        let before = b.drawing().clone();
        assert!(b
            .run("draft.constrain(kind=\"vertical\", refs=\"1\")")
            .is_err());
        assert_eq!(b.drawing(), &before);
    }

    /// A circle across a line, the line trimmed inside it: two pieces left,
    /// each ending on the circle.
    #[test]
    fn trimming_takes_the_piece_between_crossings() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.line(points=[-0.03, 0, 0.03, 0])").unwrap();
        b.run("draft.circle(centre=(0, 0, 0), radius=0.01)")
            .unwrap();
        b.run("draft.trim(id=1, at=(0.002, 0, 0))").unwrap();
        let d = b.drawing();
        let lines: Vec<&Curve2D> = d
            .profile_curves()
            .into_iter()
            .map(|(_, c)| c)
            .filter(|c| matches!(c, Curve2D::Line { .. }))
            .collect();
        assert_eq!(lines.len(), 2);
        for l in lines {
            let near = l.start().length().min(l.end().length());
            assert!((near - 0.01).abs() < 1e-12, "ends on the circle: {l:?}");
        }
        // And the circle trimmed by the pieces: the half above taken away.
        let circle = d
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::Curve(Curve2D::Circle { .. })))
            .unwrap()
            .id;
        b.run(&format!("draft.trim(id={}, at=(0, 0.01, 0))", circle.0))
            .unwrap();
        let arcs: Vec<Curve2D> = b
            .drawing()
            .profile_curves()
            .into_iter()
            .map(|(_, c)| c.clone())
            .filter(|c| matches!(c, Curve2D::Arc { .. }))
            .collect();
        assert_eq!(arcs.len(), 1);
        let mid = arcs[0].point(0.5);
        assert!(mid.y < 0.0, "the lower half is left: {mid}");
    }

    #[test]
    fn a_rounded_corner_is_tangent_and_trims_both() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.line(points=[0, 0, 0.04, 0, 0.04, 0.03])")
            .unwrap();
        b.run("draft.fillet(a=1, b=2, radius=0.005, near_a=(0.02, 0, 0), near_b=(0.04, 0.02, 0))")
            .unwrap();
        let d = b.drawing();
        let arc = d
            .profile_curves()
            .into_iter()
            .find_map(|(_, c)| match c {
                Curve2D::Arc { c, r, .. } => Some((*c, *r)),
                _ => None,
            })
            .expect("an arc");
        assert!(
            arc.0.distance(DVec2::new(0.035, 0.005)) < 1e-9,
            "{:?}",
            arc.0
        );
        let Some(Curve2D::Line { b: end, .. }) = d.curve(EntityId(1)).cloned() else {
            panic!()
        };
        assert!(
            (end.x - 0.035).abs() < 1e-9,
            "trimmed back to the arc: {end}"
        );
    }

    #[test]
    fn an_offset_goes_to_the_side_asked() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.circle(centre=(0, 0, 0), radius=0.01)")
            .unwrap();
        b.run("draft.offset(id=1, distance=0.002, toward=(0.02, 0, 0))")
            .unwrap();
        b.run("draft.offset(id=1, distance=0.002, toward=(0.001, 0, 0))")
            .unwrap();
        let mut radii: Vec<f64> = b
            .drawing()
            .profile_curves()
            .into_iter()
            .filter_map(|(_, c)| match c {
                Curve2D::Circle { r, .. } => Some(*r),
                _ => None,
            })
            .collect();
        radii.sort_by(f64::total_cmp);
        assert_eq!(radii.len(), 3);
        assert!(
            (radii[0] - 0.008).abs() < 1e-12 && (radii[2] - 0.012).abs() < 1e-12,
            "{radii:?}"
        );
    }

    /// A free line's end goes where it is dragged; a held one only as far
    /// as what holds it lets it.
    #[test]
    fn dragging_moves_free_ends_freely_and_held_ends_held() {
        let mut b = Bench::new();
        b.run("draft.new()").unwrap();
        b.run("draft.line(points=[0, 0, 0.03, 0.01])").unwrap();
        b.run("draft.drag(at=\"1.end\", to=(0.05, 0.05, 0))")
            .unwrap();
        let Some(Curve2D::Line { b: end, .. }) = b.drawing().curve(EntityId(1)).cloned() else {
            panic!()
        };
        assert!(end.distance(DVec2::new(0.05, 0.05)) < 1e-12);
        // The start held first, where it is, then the line made level: it
        // turns about its start.
        b.run("draft.constrain(kind=\"fixed\", refs=\"1.start\")")
            .unwrap();
        b.run("draft.constrain(kind=\"horizontal\", refs=\"1\")")
            .unwrap();
        b.run("draft.drag(at=\"1.end\", to=(0.06, 0.02, 0))")
            .unwrap();
        let Some(Curve2D::Line { b: end, .. }) = b.drawing().curve(EntityId(1)).cloned() else {
            panic!()
        };
        assert!(end.y.abs() < 1e-9 && (end.x - 0.06).abs() < 1e-6, "{end}");
    }

    #[test]
    fn three_points_make_the_arc_through_them() {
        let arc = arc_through(
            DVec2::new(1.0, 0.0),
            DVec2::new(0.0, 1.0),
            DVec2::new(-1.0, 0.0),
        )
        .unwrap();
        assert!(arc.point(0.5).distance(DVec2::new(0.0, 1.0)) < 1e-12);
        let back = arc_through(
            DVec2::new(1.0, 0.0),
            DVec2::new(0.0, -1.0),
            DVec2::new(-1.0, 0.0),
        )
        .unwrap();
        assert!(
            back.point(0.5).distance(DVec2::new(0.0, -1.0)) < 1e-12,
            "the other way round"
        );
        assert!(arc_through(DVec2::ZERO, DVec2::X, DVec2::X * 2.0).is_none());
        let _ = (PropertyMap::new(), PropertyValue::Bool(true));
    }
}

#[cfg(test)]
mod feature_tests {
    use super::*;
    use crate::history::History;
    use crate::operator::{Mode, OperatorRegistry};
    use kreate_scene::modifier::{EvalCtx, ModifierCache};

    fn run(r: &OperatorRegistry, h: &mut History, doc: &mut Document, line: &str) {
        let call = crate::script::parse_call(r, line).unwrap_or_else(|e| panic!("{line}: {e}"));
        h.dispatch(
            r,
            doc,
            &Context { mode: Mode::Object },
            call.op_id,
            &call.params,
        )
        .unwrap_or_else(|e| panic!("{line}: {e}"));
    }

    fn body_volume(doc: &Document) -> f64 {
        let (id, _) = doc
            .objects()
            .find(|(_, o)| !o.features.is_empty())
            .expect("a body");
        let mesh = ModifierCache::new()
            .evaluated(doc, id, &EvalCtx::still())
            .expect("its solid");
        assert!(kreate_mesh::measure::is_closed(&mesh), "a closed solid");
        kreate_mesh::measure::mass_properties(&mesh, |p| p).volume
    }

    /// The flow the sketch had, on a drawing: a plate with a hole, 10 mm
    /// up; its width changed on its dimension, and the body following.
    #[test]
    fn a_plate_with_a_hole_drawn_and_extruded_follows_its_dimension() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "draft.new()");
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.rectangle(a=(0, 0, 0), b=(0.04, 0.02, 0))",
        );
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.constrain(kind=\"fixed\", refs=\"1.start\")",
        );
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.dimension(kind=\"aligned\", refs=\"1\", value=0.04)",
        );
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.circle(centre=(0.02, 0.01, 0), radius=0.005)",
        );
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.extrude(seeds=[0.002, 0.002], distance=0.01)",
        );
        let disc = std::f64::consts::PI * 0.005f64.powi(2) * 0.01;
        let want = 0.04 * 0.02 * 0.01 - disc;
        let v = body_volume(&doc);
        assert!((v - want).abs() < want * 1e-3, "{v} against {want}");
        // Wider, by its dimension: the body follows.
        let drawing = doc
            .objects()
            .find(|(_, o)| o.drawing.is_some())
            .map(|(id, _)| id)
            .unwrap();
        let dim = doc
            .object(drawing)
            .unwrap()
            .drawing
            .as_deref()
            .unwrap()
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::Dimension(_)))
            .unwrap()
            .id;
        doc.selection.active = Some(drawing);
        run(
            &r,
            &mut h,
            &mut doc,
            &format!(
                "draft.dimension_set(id={}, value=0.06, driving=true)",
                dim.0
            ),
        );
        let want = 0.06 * 0.02 * 0.01 - disc;
        let v = body_volume(&doc);
        assert!((v - want).abs() < want * 1e-3, "{v} against {want}");
    }

    /// An open profile turned about one of the drawing's lines.
    #[test]
    fn an_open_profile_turned_about_a_drawn_line_is_a_solid() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "draft.new()");
        // The axis first (entity 1), then the profile beside it.
        run(&r, &mut h, &mut doc, "draft.line(points=[0, 0, 0, 0.05])");
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.line(points=[0, 0, 0.03, 0, 0.03, 0.05, 0, 0.05])",
        );
        run(
            &r,
            &mut h,
            &mut doc,
            "feature.revolve(seeds=[0.015, 0.025], axis=[1, 1], about=\"line\", angle=6.283185307179586)",
        );
        let v = body_volume(&doc);
        let want = std::f64::consts::PI * 0.03f64.powi(2) * 0.05;
        assert!((v - want).abs() < want * 3e-3, "{v} against {want}");
    }
}
