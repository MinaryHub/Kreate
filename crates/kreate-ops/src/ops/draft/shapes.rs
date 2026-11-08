//! More ways to draw, as Fusion's sketch Create menu has them: a rectangle
//! by three points, an ellipse, a slot round an arc, circles tangent to
//! two or three lines, and a spline by its control points.

use kreate_geom2d::{Curve2D, DVec2, Spline};
use kreate_scene::drawing::{Constraint, Drawing, EntityId, EntityRef, Part};
use kreate_scene::{DocMut, Document};

use super::{active_drawing, chain, draw, xy};
use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<DraftRectangle3>();
    registry.register::<DraftEllipse>();
    registry.register::<DraftArcSlot>();
    registry.register::<DraftTangentCircle>();
    registry.register::<DraftControlSpline>();
}

// --- three-point rectangle ----------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftRectangle3Params {
    /// One side, from `a` to `b`.
    #[param(unit = "length")]
    pub a: [f64; 3],
    #[param(unit = "length")]
    pub b: [f64; 3],
    /// How far, and which way, the other sides reach from it.
    #[param(unit = "length")]
    pub c: [f64; 3],
}

/// The four corners of the rectangle with side `a`–`b` reaching as far
/// square to it as `c` is.
pub fn rectangle3(a: DVec2, b: DVec2, c: DVec2) -> Option<[DVec2; 4]> {
    let side = b - a;
    let across = side.perp().normalize_or_zero();
    let h = (c - b).dot(across);
    if side.length() < 1e-12 || h.abs() < 1e-12 {
        return None;
    }
    let up = across * h;
    Some([a, b, b + up, a + up])
}

/// A rectangle at any angle: one side by its two ends, then how far the
/// other reaches — Fusion's 3-Point Rectangle. Its corners held square, its
/// sides parallel in pairs.
pub struct DraftRectangle3;

impl Operator for DraftRectangle3 {
    type Params = DraftRectangle3Params;
    fn id() -> &'static str {
        "draft.rectangle3"
    }
    fn label() -> &'static str {
        "3-Point Rectangle"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let corners = rectangle3(xy(params.a), xy(params.b), xy(params.c))
            .ok_or_else(|| OpError::Failed("a rectangle needs a side and a width".into()))?;
        draw(doc, |d| {
            let s = chain(d, &corners, true);
            d.constraints.push(Constraint::Perpendicular(s[0], s[1]));
            d.constraints.push(Constraint::Parallel(s[0], s[2]));
            d.constraints.push(Constraint::Parallel(s[1], s[3]));
            Ok(())
        })
    }
}

// --- ellipse ------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftEllipseParams {
    #[param(unit = "length")]
    pub centre: [f64; 3],
    /// The end of its major axis.
    #[param(unit = "length")]
    pub major: [f64; 3],
    /// A point its minor axis reaches, square to the major.
    #[param(unit = "length")]
    pub minor: [f64; 3],
}

/// An ellipse: its centre, the end of one axis, and how far the other
/// reaches — Fusion's Ellipse.
pub struct DraftEllipse;

impl Operator for DraftEllipse {
    type Params = DraftEllipseParams;
    fn id() -> &'static str {
        "draft.ellipse"
    }
    fn label() -> &'static str {
        "Ellipse"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let c = xy(params.centre);
        let mut major = xy(params.major) - c;
        let across = major.perp().normalize_or_zero();
        let mut minor = ((xy(params.minor) - c).dot(across)).abs();
        if major.length() < 1e-12 || minor < 1e-12 {
            return Err(OpError::Failed("an ellipse needs both its axes".into()));
        }
        // The longer axis is the major one, whichever was clicked first.
        if minor > major.length() {
            let m = major.length();
            major = across * minor;
            minor = m;
        }
        let ratio = minor / major.length();
        draw(doc, |d| {
            d.add_curve(Curve2D::Ellipse {
                c,
                major,
                ratio,
                start: 0.0,
                sweep: std::f64::consts::TAU,
            });
            Ok(())
        })
    }
}

// --- arc slot -----------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftArcSlotParams {
    /// The centre of the arc the slot runs round.
    #[param(unit = "length")]
    pub centre: [f64; 3],
    /// Where the slot's middle line starts and ends, anticlockwise.
    #[param(unit = "length")]
    pub start: [f64; 3],
    #[param(unit = "length")]
    pub end: [f64; 3],
    #[param(unit = "length", min = 0.0)]
    pub width: f64,
}

/// A slot that runs round an arc: two arcs about one centre and a half
/// circle at each end, tangent where they meet — Fusion's Center Point Arc
/// Slot.
pub struct DraftArcSlot;

impl Operator for DraftArcSlot {
    type Params = DraftArcSlotParams;
    fn id() -> &'static str {
        "draft.arc_slot"
    }
    fn label() -> &'static str {
        "Arc Slot"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let c = xy(params.centre);
        let (s, e) = (xy(params.start), xy(params.end));
        let r = s.distance(c);
        let half = params.width * 0.5;
        if !(r > 0.0 && half > 0.0 && half < r && params.width.is_finite()) {
            return Err(OpError::Failed(
                "the slot needs a width less than twice its radius".into(),
            ));
        }
        let a0 = (s - c).y.atan2((s - c).x);
        let a1 = (e - c).y.atan2((e - c).x);
        let sweep = (a1 - a0).rem_euclid(std::f64::consts::TAU);
        if sweep < 1e-9 {
            return Err(OpError::Failed("the slot's ends are in one place".into()));
        }
        let at = |radius: f64, a: f64| c + DVec2::from_angle(a) * radius;
        draw(doc, |d| {
            let middle = d.add_curve(Curve2D::Arc {
                c,
                r,
                start: a0,
                sweep,
            });
            if let Some(m) = d.entity_mut(middle) {
                m.construction = true;
            }
            // Round the outside one way, the inside back, a half circle at
            // each end: one closed outline.
            let outer = d.add_curve(Curve2D::Arc {
                c,
                r: r + half,
                start: a0,
                sweep,
            });
            let end_cap = d.add_curve(Curve2D::Arc {
                c: at(r, a1),
                r: half,
                start: a1,
                sweep: std::f64::consts::PI,
            });
            let inner = d.add_curve(Curve2D::Arc {
                c,
                r: r - half,
                start: a0,
                sweep,
            });
            let start_cap = d.add_curve(Curve2D::Arc {
                c: at(r, a0),
                r: half,
                start: a0 + std::f64::consts::PI,
                sweep: std::f64::consts::PI,
            });
            let join = |d: &mut Drawing, x: EntityRef, y: EntityRef| {
                d.constraints.push(Constraint::Coincident(x, y));
            };
            join(
                d,
                EntityRef::at(outer, Part::End),
                EntityRef::at(end_cap, Part::Start),
            );
            join(
                d,
                EntityRef::at(end_cap, Part::End),
                EntityRef::at(inner, Part::End),
            );
            join(
                d,
                EntityRef::at(inner, Part::Start),
                EntityRef::at(start_cap, Part::Start),
            );
            join(
                d,
                EntityRef::at(start_cap, Part::End),
                EntityRef::at(outer, Part::Start),
            );
            for cap in [end_cap, start_cap] {
                d.constraints.push(Constraint::Tangent(cap, outer));
                d.constraints.push(Constraint::Tangent(cap, inner));
            }
            d.constraints.push(Constraint::Concentric(outer, middle));
            d.constraints.push(Constraint::Concentric(inner, middle));
            join(
                d,
                EntityRef::at(end_cap, Part::Center),
                EntityRef::at(middle, Part::End),
            );
            join(
                d,
                EntityRef::at(start_cap, Part::Center),
                EntityRef::at(middle, Part::Start),
            );
            Ok(())
        })
    }
}

// --- tangent circles ----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftTangentCircleParams {
    /// Two or three lines, by id.
    #[param(hidden)]
    pub lines: Vec<u32>,
    /// With two lines, the radius.
    #[param(unit = "length", min = 0.0)]
    pub radius: f64,
    /// Near where the circle goes: which side of each line it is on.
    #[param(unit = "length", hidden)]
    pub near: [f64; 3],
}

impl Default for DraftTangentCircleParams {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            radius: 0.005,
            near: [0.0; 3],
        }
    }
}

/// The circle tangent to the given lines (carried on as far as they need)
/// on the side of each that `near` is: of radius `radius` for two lines,
/// the one circle there is for three. Its centre and radius.
pub fn tangent_circle(lines: &[(DVec2, DVec2)], radius: f64, near: DVec2) -> Option<(DVec2, f64)> {
    // Each line as n · x = d with n its unit normal turned toward `near`.
    let rows: Vec<(DVec2, f64)> = lines
        .iter()
        .map(|(a, b)| {
            let mut n = (*b - *a).perp().normalize_or_zero();
            if n.dot(near - *a) < 0.0 {
                n = -n;
            }
            (n, n.dot(*a))
        })
        .collect();
    if rows.iter().any(|(n, _)| *n == DVec2::ZERO) {
        return None;
    }
    match rows.as_slice() {
        // n_i · c = d_i + r.
        [(n1, d1), (n2, d2)] => {
            let det = n1.perp_dot(*n2);
            if det.abs() < 1e-12 || radius <= 0.0 {
                return None;
            }
            let (e1, e2) = (d1 + radius, d2 + radius);
            let c = DVec2::new(e1 * n2.y - e2 * n1.y, n1.x * e2 - n2.x * e1) / det;
            Some((c, radius))
        }
        // n_i · c - r = d_i, three unknowns.
        [(n1, d1), (n2, d2), (n3, d3)] => {
            let m = kreate_math::DMat3::from_cols(
                kreate_math::DVec3::new(n1.x, n2.x, n3.x),
                kreate_math::DVec3::new(n1.y, n2.y, n3.y),
                kreate_math::DVec3::new(-1.0, -1.0, -1.0),
            );
            if m.determinant().abs() < 1e-18 {
                return None;
            }
            let x = m.inverse() * kreate_math::DVec3::new(*d1, *d2, *d3);
            (x.z > 0.0).then(|| (DVec2::new(x.x, x.y), x.z))
        }
        _ => None,
    }
}

/// A circle tangent to two lines at a radius, or to three — Fusion's
/// 2-Tangent and 3-Tangent Circle — held tangent to them.
pub struct DraftTangentCircle;

impl Operator for DraftTangentCircle {
    type Params = DraftTangentCircleParams;
    fn id() -> &'static str {
        "draft.tangent_circle"
    }
    fn label() -> &'static str {
        "Tangent Circle"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let ids: Vec<EntityId> = params.lines.iter().map(|i| EntityId(*i)).collect();
        let near = xy(params.near);
        draw(doc, |d| {
            let lines: Vec<(DVec2, DVec2)> = ids
                .iter()
                .map(|id| match d.curve(*id) {
                    Some(Curve2D::Line { a, b }) => Ok((*a, *b)),
                    _ => Err(OpError::Failed("a tangent circle touches lines".into())),
                })
                .collect::<Result<_, _>>()?;
            if !(2..=3).contains(&lines.len()) {
                return Err(OpError::Failed("pick two or three lines".into()));
            }
            let (c, r) = tangent_circle(&lines, params.radius, near)
                .ok_or_else(|| OpError::Failed("no circle touches them there".into()))?;
            let circle = d.add_curve(Curve2D::Circle { c, r });
            for id in ids {
                d.constraints.push(Constraint::Tangent(circle, id));
            }
            Ok(())
        })
    }
}

// --- control-point spline -----------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftControlSplineParams {
    /// Its control points, as x, y pairs.
    #[param(unit = "length")]
    pub points: Vec<f64>,
    #[param(min = 1.0, max = 5.0)]
    pub degree: u32,
}

impl Default for DraftControlSplineParams {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            degree: 3,
        }
    }
}

/// A spline shaped by its control points, from the first to the last, as
/// Fusion's Control Point Spline: smoother than one through its points,
/// and moved by pulling the points that are not on it.
pub struct DraftControlSpline;

impl Operator for DraftControlSpline {
    type Params = DraftControlSplineParams;
    fn id() -> &'static str {
        "draft.control_spline"
    }
    fn label() -> &'static str {
        "Control Point Spline"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let ctrl: Vec<DVec2> = params
            .points
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| DVec2::new(c[0], c[1]))
            .collect();
        if ctrl.len() < 2 {
            return Err(OpError::Failed("a spline needs two points".into()));
        }
        draw(doc, |d| {
            d.add_curve(Curve2D::Spline(Spline::clamped(
                params.degree.max(1) as usize,
                ctrl,
            )));
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::{Mode, OperatorRegistry};
    use kreate_scene::drawing::EntityKind;

    fn bench() -> (OperatorRegistry, History, Document) {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "draft.new()").unwrap();
        (r, h, doc)
    }

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        doc: &mut Document,
        line: &str,
    ) -> Result<OpResult, OpError> {
        let call = crate::script::parse_call(r, line).unwrap_or_else(|e| panic!("{line}: {e}"));
        h.dispatch(
            r,
            doc,
            &Context { mode: Mode::Object },
            call.op_id,
            &call.params,
        )
    }

    fn drawing(doc: &Document) -> &Drawing {
        doc.object(active_drawing(doc).unwrap())
            .unwrap()
            .drawing
            .as_deref()
            .unwrap()
    }

    fn areas(d: &Drawing) -> Vec<f64> {
        let mut a: Vec<f64> = d.regions(1e-9).iter().map(|r| r.area).collect();
        a.sort_by(f64::total_cmp);
        a
    }

    #[test]
    fn a_turned_rectangle_by_three_points_is_square_cornered() {
        let (r, mut h, mut doc) = bench();
        // A 30 mm side at 30°, reaching 10 mm.
        let (s, c) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
        run(
            &r,
            &mut h,
            &mut doc,
            &format!(
                "draft.rectangle3(a=(0, 0, 0), b=({}, {}, 0), c=({}, {}, 0))",
                0.03 * c,
                0.03 * s,
                0.03 * c - 0.01 * s,
                0.03 * s + 0.01 * c
            ),
        )
        .unwrap();
        let a = areas(drawing(&doc));
        assert_eq!(a.len(), 1);
        assert!((a[0] - 0.0003).abs() < 1e-12, "{a:?}");
        assert!(drawing(&doc)
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::Perpendicular(..))));
    }

    #[test]
    fn an_ellipse_and_an_arc_slot_close_their_regions() {
        let (r, mut h, mut doc) = bench();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.ellipse(centre=(0, 0, 0), major=(0.02, 0, 0), minor=(0, 0.01, 0))",
        )
        .unwrap();
        let a = areas(drawing(&doc));
        assert!(
            (a[0] - std::f64::consts::PI * 0.02 * 0.01).abs() < 1e-9,
            "{a:?}"
        );
        // A quarter-round slot of radius 20 mm, 6 mm wide.
        let (r2, mut h2, mut doc2) = bench();
        run(
            &r2,
            &mut h2,
            &mut doc2,
            "draft.arc_slot(centre=(0, 0, 0), start=(0.02, 0, 0), end=(0, 0.02, 0), width=0.006)",
        )
        .unwrap();
        let a = areas(drawing(&doc2));
        let pi = std::f64::consts::PI;
        let want = pi / 4.0 * (0.023f64.powi(2) - 0.017f64.powi(2)) + pi * 0.003f64.powi(2);
        assert_eq!(a.len(), 1, "{a:?}");
        assert!((a[0] - want).abs() < want * 1e-6, "{a:?} against {want}");
        let _ = r;
    }

    #[test]
    fn circles_tangent_to_two_and_three_lines() {
        // Two lines at a right angle at the origin: radius 5 mm in the
        // corner the point is in.
        let lines = [(DVec2::ZERO, DVec2::X), (DVec2::ZERO, DVec2::Y)];
        let (c, r) = tangent_circle(&lines, 0.005, DVec2::new(0.01, 0.01)).unwrap();
        assert!(c.distance(DVec2::new(0.005, 0.005)) < 1e-12 && r == 0.005);
        let (c, _) = tangent_circle(&lines, 0.005, DVec2::new(-0.01, 0.01)).unwrap();
        assert!(c.distance(DVec2::new(-0.005, 0.005)) < 1e-12);
        // A 3-4-5 triangle: its incircle has radius 1.
        let tri = [
            (DVec2::ZERO, DVec2::new(4.0, 0.0)),
            (DVec2::new(4.0, 0.0), DVec2::new(0.0, 3.0)),
            (DVec2::new(0.0, 3.0), DVec2::ZERO),
        ];
        let (c, r) = tangent_circle(&tri, 0.0, DVec2::new(1.0, 1.0)).unwrap();
        assert!(
            (r - 1.0).abs() < 1e-12 && c.distance(DVec2::ONE) < 1e-12,
            "{c} {r}"
        );
        // Through the operator: held tangent, and the drawing solved.
        let (reg, mut h, mut doc) = bench();
        run(&reg, &mut h, &mut doc, "draft.line(points=[0, 0, 0.04, 0])").unwrap();
        run(
            &reg,
            &mut h,
            &mut doc,
            "draft.line(points=[0, -0.01, 0, 0.04])",
        )
        .unwrap();
        run(
            &reg,
            &mut h,
            &mut doc,
            "draft.tangent_circle(lines=[1, 2], radius=0.005, near=(0.01, 0.01, 0))",
        )
        .unwrap();
        let d = drawing(&doc);
        assert_eq!(
            d.constraints
                .iter()
                .filter(|c| matches!(c, Constraint::Tangent(..)))
                .count(),
            2
        );
        assert!(d.entities.iter().any(|e| matches!(&e.kind,
            EntityKind::Curve(Curve2D::Circle { c, r }) if c.distance(DVec2::new(0.005, 0.005)) < 1e-9 && (*r - 0.005).abs() < 1e-12)));
    }

    #[test]
    fn a_control_point_spline_starts_and_ends_on_its_ends() {
        let (r, mut h, mut doc) = bench();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.control_spline(points=[0, 0, 0.01, 0.02, 0.03, -0.01, 0.04, 0])",
        )
        .unwrap();
        let d = drawing(&doc);
        let c = d.entities.iter().find_map(|e| match &e.kind {
            EntityKind::Curve(c @ Curve2D::Spline(_)) => Some(c.clone()),
            _ => None,
        });
        let c = c.unwrap();
        assert!(
            c.start().distance(DVec2::ZERO) < 1e-12
                && c.end().distance(DVec2::new(0.04, 0.0)) < 1e-12
        );
    }

    /// Two lines made collinear: the second lies on the first carried on.
    #[test]
    fn collinear_lines_lie_on_one_line() {
        let (r, mut h, mut doc) = bench();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.line(points=[0, 0, 0.02, 0.01])",
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.line(points=[0.03, 0.03, 0.05, 0.035])",
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.constrain(kind=\"collinear\", refs=\"2 1\")",
        )
        .unwrap();
        let d = drawing(&doc);
        let (Some(Curve2D::Line { a: a1, b: b1 }), Some(Curve2D::Line { a: a2, b: b2 })) =
            (d.curve(EntityId(1)), d.curve(EntityId(2)))
        else {
            panic!()
        };
        let dir = (*b1 - *a1).normalize();
        for p in [a2, b2] {
            assert!((*p - *a1).perp_dot(dir).abs() < 1e-9, "{p} is off the line");
        }
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "draft.constrain(kind=\"collinear\", refs=\"2 1\")"
        )
        .is_err());
    }
}
