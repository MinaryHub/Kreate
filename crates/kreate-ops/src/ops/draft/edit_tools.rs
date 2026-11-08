//! More of the drafting mode's editing tools, as Fusion's sketch has them:
//! extend a curve to what it meets, break one where others cross it, a
//! chamfer between two lines, and scaling and turning what is selected —
//! moved, or copied.

use kreate_geom2d::{Curve2D, DVec2};
use kreate_math::glam;
use kreate_scene::drawing::{Constraint, DimKind, Drawing, EntityId, EntityKind, EntityRef, Part};
use kreate_scene::{DocMut, Document};

use super::{active_drawing, edit, known, xy};
use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<DraftExtend>();
    registry.register::<DraftBreak>();
    registry.register::<DraftChamfer>();
    registry.register::<DraftTransform>();
    registry.register::<DraftTangentArc>();
}

/// The arc that leaves `start` along `dir` and ends at `end`: on the
/// circle tangent to `dir` at `start` through `end`. `None` when `end` is
/// straight ahead or behind, where no arc turns.
pub fn tangent_arc(start: DVec2, dir: DVec2, end: DVec2) -> Option<Curve2D> {
    let dir = dir.normalize_or_zero();
    let chord = end - start;
    let left = dir.perp();
    let across = chord.dot(left);
    if dir == DVec2::ZERO || across.abs() <= 1e-12 * chord.length().max(1e-12) {
        return None;
    }
    // The centre on the normal at the start, as far as makes it as far
    // from the end: signed, to the left when positive.
    let r = chord.length_squared() / (2.0 * across);
    let c = start + left * r;
    let a0 = (start - c).y.atan2((start - c).x);
    let a1 = (end - c).y.atan2((end - c).x);
    // Round the way `dir` sets off: anticlockwise when the centre is to
    // its left.
    let sweep = if r > 0.0 {
        (a1 - a0).rem_euclid(std::f64::consts::TAU)
    } else {
        -(a0 - a1).rem_euclid(std::f64::consts::TAU)
    };
    Some(Curve2D::Arc {
        c,
        r: r.abs(),
        start: a0,
        sweep,
    })
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftTangentArcParams {
    #[param(unit = "length")]
    pub start: [f64; 3],
    /// The way it sets off from its start.
    pub dir: [f64; 3],
    #[param(unit = "length")]
    pub end: [f64; 3],
}

/// An arc on from the end of what was drawn, along it, to a point: what
/// Fusion's line tool draws when dragged from a line's end. Joined to what
/// ends at its start, and held tangent to it.
pub struct DraftTangentArc;

impl Operator for DraftTangentArc {
    type Params = DraftTangentArcParams;
    fn id() -> &'static str {
        "draft.tangent_arc"
    }
    fn label() -> &'static str {
        "Tangent Arc"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let arc = tangent_arc(xy(params.start), xy(params.dir), xy(params.end))
            .ok_or_else(|| OpError::Failed("straight ahead: that is a line, not an arc".into()))?;
        super::draw(doc, |d| {
            let first = d.counters().0;
            d.add_curve(arc);
            // Tangent to a line or an arc ending where it starts, as it
            // leaves along it.
            let start = xy(params.start);
            let dir = xy(params.dir).normalize_or_zero();
            let arc_id = EntityId(first);
            let touching: Vec<EntityId> = d
                .entities
                .iter()
                .filter(|e| e.id.0 < first)
                .filter_map(|e| match &e.kind {
                    EntityKind::Curve(c) if !c.is_closed() => {
                        let (lo, hi) = c.domain();
                        [lo, hi]
                            .into_iter()
                            .find(|t| {
                                c.point(*t).distance(start) <= 1e-9 * start.length().max(1.0)
                                    && c.derivative(*t).normalize_or_zero().perp_dot(dir).abs()
                                        <= 1e-6
                            })
                            .map(|_| e.id)
                    }
                    _ => None,
                })
                .collect();
            for t in touching {
                d.constraints.push(Constraint::Tangent(arc_id, t));
            }
            Ok(())
        })
    }
}

/// Where every other curve crosses `curve`, along it.
fn crossings(d: &Drawing, target: EntityId, curve: &Curve2D) -> Vec<f64> {
    let mut cuts = Vec::new();
    for e in &d.entities {
        if e.id == target || e.construction {
            continue;
        }
        if let EntityKind::Curve(other) = &e.kind {
            for hit in kreate_geom2d::intersect(curve, other) {
                match hit {
                    kreate_geom2d::Intersection::Point { ta, .. } => cuts.push(ta),
                    kreate_geom2d::Intersection::Overlap { a, .. } => cuts.extend([a.0, a.1]),
                }
            }
        }
    }
    cuts
}

/// The constraints and dimensions on one end of an entity, which no longer
/// mean what they did once that end has moved: dropped.
fn free_end(d: &mut Drawing, id: EntityId, part: Part) {
    let on = |r: &EntityRef| r.entity == id && r.part == part;
    d.constraints.retain(|c| match c {
        Constraint::Coincident(a, b) => !on(a) && !on(b),
        Constraint::Fixed(a, _) => !on(a),
        Constraint::PointOn(a, _) | Constraint::Midpoint(a, _) => !on(a),
        Constraint::Symmetric(a, b, _) => !on(a) && !on(b),
        _ => true,
    });
    let dims: Vec<EntityId> = d
        .entities
        .iter()
        .filter(|e| matches!(&e.kind, EntityKind::Dimension(x) if x.refs.iter().any(on)))
        .map(|e| e.id)
        .collect();
    d.remove(&dims);
}

// --- extend -------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftExtendParams {
    /// The line or arc to extend.
    #[param(hidden)]
    pub id: u32,
    /// Where it was clicked: the end nearer this one goes.
    #[param(unit = "length", hidden)]
    pub at: [f64; 3],
}

/// What extending curve `target` from its end nearer `at` makes of it: a
/// line on along its own direction, an arc round its own circle, to the
/// first curve it meets.
pub fn extend_plan(d: &Drawing, target: EntityId, at: DVec2) -> Result<Curve2D, OpError> {
    let curve = d
        .curve(target)
        .cloned()
        .ok_or_else(|| OpError::Failed(format!("{} is not a curve", target.0)))?;
    // How far a line may reach: past everything drawn.
    let reach = d
        .extents()
        .map(|(lo, hi)| (hi - lo).length() * 4.0)
        .unwrap_or(1.0)
        .max(1e-3);
    match curve {
        Curve2D::Line { a, b } => {
            let from_b = at.distance(b) <= at.distance(a);
            let (fixed, end) = if from_b { (a, b) } else { (b, a) };
            let dir = (end - fixed).normalize_or_zero();
            if dir == DVec2::ZERO {
                return Err(OpError::Failed("a line of no length".into()));
            }
            let ray = Curve2D::Line {
                a: end,
                b: end + dir * reach,
            };
            let t = crossings(d, target, &ray)
                .into_iter()
                .filter(|t| *t > 1e-9)
                .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.min(t))))
                .ok_or_else(|| OpError::Failed("nothing to extend it to".into()))?;
            let to = ray.point(t);
            Ok(if from_b {
                Curve2D::Line { a, b: to }
            } else {
                Curve2D::Line { a: to, b }
            })
        }
        Curve2D::Arc { c, r, start, sweep } => {
            let s = Curve2D::Arc { c, r, start, sweep };
            let (lo, hi) = s.domain();
            let from_end = at.distance(s.point(hi)) <= at.distance(s.point(lo));
            // The rest of the circle, from the end on round the same way.
            let rest = std::f64::consts::TAU - sweep.abs();
            let sign = sweep.signum();
            let (ray_start, ray_sweep) = if from_end {
                (start + sweep, sign * rest)
            } else {
                (start, -sign * rest)
            };
            let ray = Curve2D::Arc {
                c,
                r,
                start: ray_start,
                sweep: ray_sweep,
            };
            let (rlo, rhi) = ray.domain();
            let t = crossings(d, target, &ray)
                .into_iter()
                .filter(|t| *t > rlo + (rhi - rlo) * 1e-9)
                .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.min(t))))
                .ok_or_else(|| OpError::Failed("nothing to extend it to".into()))?;
            let more = ray_sweep * (t - rlo) / (rhi - rlo);
            Ok(if from_end {
                Curve2D::Arc {
                    c,
                    r,
                    start,
                    sweep: sweep + more,
                }
            } else {
                Curve2D::Arc {
                    c,
                    r,
                    start: start + more,
                    sweep: sweep - more,
                }
            })
        }
        _ => Err(OpError::Failed("only a line or an arc is extended".into())),
    }
}

/// Lengthens a line or an arc from the end clicked to the first curve it
/// meets, as Fusion's Extend does. What held that end lets it go.
pub struct DraftExtend;

impl Operator for DraftExtend {
    type Params = DraftExtendParams;
    fn id() -> &'static str {
        "draft.extend"
    }
    fn label() -> &'static str {
        "Extend"
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
            let old = d.curve(target).cloned();
            let new = extend_plan(d, target, at)?;
            // Which end moved.
            let part = match (&old, &new) {
                (Some(Curve2D::Line { a, .. }), Curve2D::Line { a: na, .. }) if a == na => {
                    Part::End
                }
                (Some(Curve2D::Line { .. }), _) => Part::Start,
                (Some(Curve2D::Arc { start, .. }), Curve2D::Arc { start: ns, .. })
                    if start == ns =>
                {
                    Part::End
                }
                _ => Part::Start,
            };
            free_end(d, target, part);
            if let Some(e) = d.entity_mut(target) {
                e.kind = EntityKind::Curve(new);
            }
            Ok(())
        })
    }
}

// --- break --------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftBreakParams {
    #[param(hidden)]
    pub id: u32,
    /// Where it was clicked.
    #[param(unit = "length", hidden)]
    pub at: [f64; 3],
}

/// Splits a curve where the curves nearest either side of the click cross
/// it, as Fusion's Break does: the piece clicked becomes an entity of its
/// own, the pieces still joined end to end.
pub struct DraftBreak;

impl Operator for DraftBreak {
    type Params = DraftBreakParams;
    fn id() -> &'static str {
        "draft.break"
    }
    fn label() -> &'static str {
        "Break"
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
                .ok_or_else(|| OpError::Failed(format!("{} is not a curve", target.0)))?;
            if curve.is_closed() {
                return Err(OpError::Failed(
                    "a closed curve is broken by trimming it".into(),
                ));
            }
            let (lo, hi) = curve.domain();
            let span = hi - lo;
            let cuts = crossings(d, target, &curve);
            let (t, _) = curve.closest(at);
            let inner = |c: &f64| *c > lo + span * 1e-9 && *c < hi - span * 1e-9;
            let below = cuts
                .iter()
                .copied()
                .filter(|c| inner(c) && *c < t)
                .fold(None, |m: Option<f64>, c| Some(m.map_or(c, |m| m.max(c))));
            let above = cuts
                .iter()
                .copied()
                .filter(|c| inner(c) && *c > t)
                .fold(None, |m: Option<f64>, c| Some(m.map_or(c, |m| m.min(c))));
            let mut at_params: Vec<f64> = [below, above].into_iter().flatten().collect();
            if at_params.is_empty() {
                return Err(OpError::Failed("nothing crosses it to break it at".into()));
            }
            at_params.insert(0, lo);
            at_params.push(hi);
            // The first piece keeps the entity, and what held its start; the
            // last takes what held the end.
            let pieces: Vec<Curve2D> = at_params
                .windows(2)
                .map(|w| curve.sub(w[0], w[1]))
                .collect();
            let layer = d.entity(target).map(|e| e.layer);
            if let Some(e) = d.entity_mut(target) {
                e.kind = EntityKind::Curve(pieces[0].clone());
            }
            let mut ids = vec![target];
            for piece in &pieces[1..] {
                let id = d.add_curve(piece.clone());
                if let (Some(l), Some(e)) = (layer, d.entity_mut(id)) {
                    e.layer = l;
                }
                ids.push(id);
            }
            let last = *ids.last().expect("the pieces");
            // What held the old end now holds the last piece's end.
            let move_end = |r: &mut EntityRef| {
                if r.entity == target && r.part == Part::End {
                    r.entity = last;
                }
            };
            for c in &mut d.constraints {
                match c {
                    Constraint::Coincident(a, b) | Constraint::Symmetric(a, b, _) => {
                        move_end(a);
                        move_end(b);
                    }
                    Constraint::Fixed(a, _)
                    | Constraint::PointOn(a, _)
                    | Constraint::Midpoint(a, _) => move_end(a),
                    _ => {}
                }
            }
            // The pieces joined end to start.
            for w in ids.windows(2) {
                d.constraints.push(Constraint::Coincident(
                    EntityRef::at(w[0], Part::End),
                    EntityRef::at(w[1], Part::Start),
                ));
            }
            // A dimension on the whole curve no longer measures one thing.
            let dims: Vec<EntityId> = d
                .entities
                .iter()
                .filter(|e| {
                    matches!(&e.kind, EntityKind::Dimension(x)
                        if x.refs.iter().any(|r| r.entity == target && r.part == Part::Whole))
                })
                .map(|e| e.id)
                .collect();
            d.remove(&dims);
            Ok(())
        })
    }
}

// --- chamfer ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftChamferParams {
    #[param(hidden)]
    pub a: u32,
    #[param(hidden)]
    pub b: u32,
    /// How far back along each line from the corner.
    #[param(min = 0.0, unit = "length")]
    pub distance: f64,
    /// Along the second line, when it differs: zero for the same.
    #[param(min = 0.0, unit = "length")]
    pub distance_b: f64,
    /// Where each line was clicked: the side kept.
    #[param(unit = "length", hidden)]
    pub near_a: [f64; 3],
    #[param(unit = "length", hidden)]
    pub near_b: [f64; 3],
}

impl Default for DraftChamferParams {
    fn default() -> Self {
        Self {
            a: 0,
            b: 0,
            distance: 0.002,
            distance_b: 0.0,
            near_a: [0.0; 3],
            near_b: [0.0; 3],
        }
    }
}

/// Cuts a corner between two lines with a straight line, each line kept
/// from the side clicked to the cut, as Fusion's sketch Chamfer does.
pub struct DraftChamfer;

impl Operator for DraftChamfer {
    type Params = DraftChamferParams;
    fn id() -> &'static str {
        "draft.chamfer"
    }
    fn label() -> &'static str {
        "Chamfer"
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
        let da = params.distance;
        let db = if params.distance_b > 0.0 {
            params.distance_b
        } else {
            da
        };
        if !(da > 0.0 && db > 0.0 && da.is_finite() && db.is_finite()) {
            return Err(OpError::Failed("a chamfer needs a distance".into()));
        }
        edit(doc, |d| {
            let (Some(Curve2D::Line { a: a0, b: a1 }), Some(Curve2D::Line { a: b0, b: b1 })) =
                (d.curve(ia).cloned(), d.curve(ib).cloned())
            else {
                return Err(OpError::Failed("a chamfer joins two lines".into()));
            };
            // Where the two lines, carried on, meet.
            let (u, v) = (a1 - a0, b1 - b0);
            let cross = u.perp_dot(v);
            if cross.abs() < 1e-18 * u.length_squared().max(1e-30) {
                return Err(OpError::Failed("parallel lines have no corner".into()));
            }
            let corner = a0 + u * ((b0 - a0).perp_dot(v) / cross);
            // Each line's kept end: the one on the clicked side of the corner.
            let keep = |p0: DVec2, p1: DVec2, near: DVec2| -> DVec2 {
                let dir = (p1 - p0).normalize();
                let s = (near - corner).dot(dir);
                if s >= 0.0 {
                    if (p1 - corner).dot(dir) >= (p0 - corner).dot(dir) {
                        p1
                    } else {
                        p0
                    }
                } else if (p1 - corner).dot(dir) <= (p0 - corner).dot(dir) {
                    p1
                } else {
                    p0
                }
            };
            let (ka, kb) = (keep(a0, a1, pa), keep(b0, b1, pb));
            let (ca, cb) = (
                corner + (ka - corner).normalize_or_zero() * da,
                corner + (kb - corner).normalize_or_zero() * db,
            );
            if (ka - corner).length() < da || (kb - corner).length() < db {
                return Err(OpError::Failed(
                    "that distance does not fit the corner".into(),
                ));
            }
            // The lines from their kept ends to the cut, starting where they
            // are kept so their start's constraints still hold.
            let set =
                |d: &mut Drawing, id: EntityId, old: (DVec2, DVec2), kept: DVec2, cut: DVec2| {
                    let line = if old.0 == kept {
                        Curve2D::Line { a: kept, b: cut }
                    } else {
                        Curve2D::Line { a: cut, b: kept }
                    };
                    let cut_part = if old.0 == kept {
                        Part::End
                    } else {
                        Part::Start
                    };
                    free_end(d, id, cut_part);
                    if let Some(e) = d.entity_mut(id) {
                        e.kind = EntityKind::Curve(line);
                    }
                    cut_part
                };
            let pa_part = set(d, ia, (a0, a1), ka, ca);
            let pb_part = set(d, ib, (b0, b1), kb, cb);
            d.constraints.retain(|c| {
                !matches!(c, Constraint::Coincident(x, y)
                    if (x.entity == ia && y.entity == ib) || (x.entity == ib && y.entity == ia))
            });
            let cut = d.add_curve(Curve2D::Line { a: ca, b: cb });
            d.constraints.push(Constraint::Coincident(
                EntityRef::at(ia, pa_part),
                EntityRef::at(cut, Part::Start),
            ));
            d.constraints.push(Constraint::Coincident(
                EntityRef::at(ib, pb_part),
                EntityRef::at(cut, Part::End),
            ));
            Ok(())
        })
    }
}

// --- scale and turn -----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftTransformParams {
    pub ids: Vec<u32>,
    /// The point it is turned and scaled about.
    #[param(unit = "length")]
    pub centre: [f64; 3],
    /// Anticlockwise.
    #[param(unit = "angle")]
    pub angle: f64,
    #[param(min = 0.0001, soft_max = 10.0)]
    pub scale: f64,
    /// Moved further after the turn and the scale.
    #[param(unit = "length")]
    pub offset: [f64; 3],
    /// Copies, with the constraints among them; the originals stay.
    pub copy: bool,
}

impl Default for DraftTransformParams {
    fn default() -> Self {
        Self {
            ids: Vec::new(),
            centre: [0.0; 3],
            angle: 0.0,
            scale: 1.0,
            offset: [0.0; 3],
            copy: false,
        }
    }
}

/// Turns, scales and moves entities about a point — or copies of them —
/// as Fusion's sketch Move/Copy and Sketch Scale do. A scale changes the
/// sizes the moved entities' own dimensions give, so they stay what holds
/// them; their held places go with them.
pub struct DraftTransform;

impl Operator for DraftTransform {
    type Params = DraftTransformParams;
    fn id() -> &'static str {
        "draft.transform"
    }
    fn label() -> &'static str {
        "Move, Turn or Scale"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !(params.scale > 0.0 && params.scale.is_finite() && params.angle.is_finite()) {
            return Err(OpError::Failed("a scale above zero and an angle".into()));
        }
        let centre = xy(params.centre);
        let m = glam::DAffine2::from_translation(xy(params.offset) + centre)
            * glam::DAffine2::from_angle(params.angle)
            * glam::DAffine2::from_scale(DVec2::splat(params.scale))
            * glam::DAffine2::from_translation(-centre);
        let ids: Vec<EntityId> = params.ids.iter().map(|i| EntityId(*i)).collect();
        if params.copy {
            return edit(doc, |d| {
                known(
                    d,
                    &ids.iter().map(|i| EntityRef::whole(*i)).collect::<Vec<_>>(),
                )?;
                super::copy_moved(d, &ids, &[m]);
                Ok(())
            });
        }
        edit(doc, |d| {
            known(
                d,
                &ids.iter().map(|i| EntityRef::whole(*i)).collect::<Vec<_>>(),
            )?;
            let inside = |r: &EntityRef| ids.contains(&r.entity);
            // Their own sizes scale with them.
            for e in &mut d.entities {
                if let EntityKind::Dimension(x) = &mut e.kind {
                    if !x.refs.is_empty() && x.refs.iter().all(inside) && x.kind != DimKind::Angular
                    {
                        x.value *= params.scale;
                    }
                    if x.refs.iter().all(inside) {
                        x.text_at = m.transform_point2(x.text_at);
                    }
                }
            }
            // Their held places go with them.
            for c in &mut d.constraints {
                if let Constraint::Fixed(r, at) = c {
                    if inside(r) {
                        *at = m.transform_point2(*at);
                    }
                }
            }
            // Level and upright do not survive a turn.
            if params.angle.rem_euclid(std::f64::consts::FRAC_PI_2).abs() > 1e-9 {
                d.constraints.retain(|c| match c {
                    Constraint::Horizontal(a, b) | Constraint::Vertical(a, b) => {
                        !inside(a) && !b.as_ref().is_some_and(inside)
                    }
                    _ => true,
                });
            }
            for id in &ids {
                if let Some(e) = d.entity_mut(*id) {
                    match &mut e.kind {
                        EntityKind::Point(p) => *p = m.transform_point2(*p),
                        EntityKind::Curve(c) => *c = c.transformed(&m),
                        EntityKind::Text(t) => {
                            t.at = m.transform_point2(t.at);
                            t.height *= params.scale;
                            t.rotation += params.angle;
                        }
                        EntityKind::Dimension(_) => {}
                    }
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::{Mode, OperatorRegistry};

    struct Bench {
        registry: OperatorRegistry,
        history: History,
        doc: Document,
    }

    impl Bench {
        fn new() -> Bench {
            let mut registry = OperatorRegistry::new();
            crate::register_builtin(&mut registry);
            let mut b = Bench {
                registry,
                history: History::new(),
                doc: Document::new(),
            };
            b.run("draft.new()").unwrap();
            b
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
        fn curve(&self, id: u32) -> Curve2D {
            self.drawing().curve(EntityId(id)).cloned().unwrap()
        }
    }

    fn near(a: DVec2, b: DVec2) -> bool {
        a.distance(b) < 1e-9
    }

    /// A line short of a wall is carried on to it; an arc short of a line
    /// goes on round its circle to meet it; with nothing ahead, refused.
    #[test]
    fn a_line_and_an_arc_extend_to_what_they_meet() {
        let mut b = Bench::new();
        b.run("draft.line(points=[0, 0, 0.01, 0])").unwrap();
        b.run("draft.line(points=[0.03, -0.01, 0.03, 0.01])")
            .unwrap();
        b.run("draft.extend(id=1, at=(0.009, 0, 0))").unwrap();
        let Curve2D::Line { a, b: end } = b.curve(1) else {
            panic!()
        };
        assert!(
            near(a, DVec2::ZERO) && near(end, DVec2::new(0.03, 0.0)),
            "{end}"
        );
        // The other way: nothing there.
        assert!(b.run("draft.extend(id=1, at=(0.001, 0, 0))").is_err());
        // A quarter arc of radius 10 mm about the origin, from +X round to
        // +Y, extended from its end to the line x = -0.005.
        let mut b = Bench::new();
        b.run(
            "draft.arc(start=(0.01, 0, 0), through=(0.00707107, 0.00707107, 0), end=(0, 0.01, 0))",
        )
        .unwrap();
        b.run("draft.line(points=[-0.005, -0.02, -0.005, 0.02])")
            .unwrap();
        b.run("draft.extend(id=1, at=(0, 0.01, 0))").unwrap();
        let arc = b.curve(1);
        let (_, hi) = arc.domain();
        let end = arc.point(hi);
        assert!((end.x + 0.005).abs() < 1e-9 && end.y > 0.0, "{end}");
    }

    /// A line crossed twice is broken into three, joined, the middle one
    /// on its own; a chamfer cuts a square corner to the distances given.
    #[test]
    fn break_splits_where_crossed_and_chamfer_cuts_the_corner() {
        let mut b = Bench::new();
        b.run("draft.line(points=[0, 0, 0.03, 0])").unwrap();
        b.run("draft.line(points=[0.01, -0.01, 0.01, 0.01])")
            .unwrap();
        b.run("draft.line(points=[0.02, -0.01, 0.02, 0.01])")
            .unwrap();
        b.run("draft.break(id=1, at=(0.015, 0, 0))").unwrap();
        let lines: Vec<Curve2D> = b
            .drawing()
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c @ Curve2D::Line { a, b }) if a.y == 0.0 && b.y == 0.0 => {
                    Some(c.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(lines.len(), 3, "{lines:?}");
        let middle = lines
            .iter()
            .find(|c| matches!(c, Curve2D::Line { a, b } if near(*a, DVec2::new(0.01, 0.0)) && near(*b, DVec2::new(0.02, 0.0))));
        assert!(middle.is_some(), "{lines:?}");

        let mut b = Bench::new();
        b.run("draft.line(points=[0, 0, 0.02, 0, 0.02, 0.02])")
            .unwrap();
        b.run("draft.chamfer(a=1, b=2, distance=0.005, near_a=(0.005, 0, 0), near_b=(0.02, 0.015, 0))")
            .unwrap();
        let (Curve2D::Line { a: a0, b: a1 }, Curve2D::Line { a: b0, b: b1 }) =
            (b.curve(1), b.curve(2))
        else {
            panic!()
        };
        assert!(
            near(a0, DVec2::ZERO) && near(a1, DVec2::new(0.015, 0.0)),
            "{a1}"
        );
        assert!(
            near(b1, DVec2::new(0.02, 0.02)) && near(b0, DVec2::new(0.02, 0.005)),
            "{b0}"
        );
        let cut = b
            .drawing()
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c @ Curve2D::Line { .. }) => Some(c.clone()),
                _ => None,
            })
            .count();
        assert_eq!(cut, 3, "the two lines and the cut");
    }

    /// A rectangle with its width dimensioned, scaled two times about its
    /// corner: twice as big, the dimension twice its value; turned a
    /// quarter as a copy, the original stays and the copy stands up.
    #[test]
    fn scaling_takes_its_dimensions_with_it_and_a_turned_copy_leaves_the_original() {
        let mut b = Bench::new();
        b.run("draft.line(points=[0, 0, 0.02, 0])").unwrap();
        b.run("draft.dimension(kind=\"aligned\", refs=\"1\", value=0.02)")
            .unwrap();
        b.run("draft.transform(ids=[1], centre=(0, 0, 0), scale=2)")
            .unwrap();
        let Curve2D::Line { b: end, .. } = b.curve(1) else {
            panic!()
        };
        assert!(near(end, DVec2::new(0.04, 0.0)), "{end}");
        let dim = b
            .drawing()
            .entities
            .iter()
            .find_map(|e| match &e.kind {
                EntityKind::Dimension(x) => Some(x.value),
                _ => None,
            })
            .unwrap();
        assert!((dim - 0.04).abs() < 1e-12);
        b.run("draft.transform(ids=[1], centre=(0, 0, 0), angle=1.5707963267948966, copy=true)")
            .unwrap();
        let curves: Vec<Curve2D> = b
            .drawing()
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(curves.len(), 2);
        assert!(curves
            .iter()
            .any(|c| matches!(c, Curve2D::Line { b, .. } if near(*b, DVec2::new(0.0, 0.04)))));
    }

    /// Saying a held thing again is refused as a constraint and made a
    /// driven dimension as a dimension — and something new is still taken.
    #[test]
    fn what_is_already_held_is_not_held_twice() {
        let mut b = Bench::new();
        // Drawn level, so held level.
        b.run("draft.line(points=[0, 0, 0.02, 0])").unwrap();
        assert!(b
            .drawing()
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::Horizontal(..))));
        let err = b.run("draft.constrain(kind=\"horizontal\", refs=\"1\")");
        assert!(err.is_err(), "level twice: {err:?}");
        // Its length, then its length again: the second only measures.
        b.run("draft.dimension(kind=\"aligned\", refs=\"1\", value=0.02)")
            .unwrap();
        b.run("draft.dimension(kind=\"horizontal\", refs=\"1.start 1.end\", value=0.02)")
            .unwrap();
        let driving: Vec<bool> = b
            .drawing()
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Dimension(x) => Some(x.driving),
                _ => None,
            })
            .collect();
        assert_eq!(driving, [true, false]);
        // Something it does not yet say is still taken.
        b.run("draft.constrain(kind=\"fixed\", refs=\"1.start\")")
            .unwrap();
    }

    /// What the line tool turned a segment onto is held as drawn: along
    /// another line, square to it, on from an arc's end, ending on a curve
    /// or at a line's middle.
    #[test]
    fn what_a_line_was_turned_onto_is_held() {
        let has =
            |b: &Bench, f: &dyn Fn(&Constraint) -> bool| b.drawing().constraints.iter().any(f);
        let mut b = Bench::new();
        // A slanted line, then one exactly along it and one exactly square.
        b.run("draft.line(points=[0, 0, 0.02, 0.01])").unwrap();
        b.run("draft.line(points=[0, 0.02, 0.04, 0.04])").unwrap();
        assert!(has(
            &b,
            &|c| matches!(c, Constraint::Parallel(x, y) if x.0 == 2 && y.0 == 1)
        ));
        b.run("draft.line(points=[0.05, 0, 0.04, 0.02])").unwrap();
        assert!(has(
            &b,
            &|c| matches!(c, Constraint::Perpendicular(x, y) if x.0 == 3 && y.0 == 1)
        ));
        // A line ending on the first one's middle, and one on it elsewhere.
        b.run("draft.line(points=[0.01, -0.01, 0.01, 0.005])")
            .unwrap();
        assert!(has(
            &b,
            &|c| matches!(c, Constraint::Midpoint(r, l) if r.entity.0 == 4 && l.0 == 1)
        ));
        b.run("draft.line(points=[0.016, -0.01, 0.016, 0.008])")
            .unwrap();
        assert!(has(
            &b,
            &|c| matches!(c, Constraint::PointOn(r, l) if r.entity.0 == 5 && l.0 == 1)
        ));
        // A line leaving a quarter arc's end along it.
        let mut b = Bench::new();
        b.run(
            "draft.arc(start=(0.01, 0, 0), through=(0.00707107, 0.00707107, 0), end=(0, 0.01, 0))",
        )
        .unwrap();
        let arc = b.curve(1);
        let (_, hi) = arc.domain();
        let (end, dir) = (arc.point(hi), arc.derivative(hi).normalize());
        let to = end + dir * 0.02;
        b.run(&format!(
            "draft.line(points=[{}, {}, {}, {}])",
            end.x, end.y, to.x, to.y
        ))
        .unwrap();
        assert!(has(
            &b,
            &|c| matches!(c, Constraint::Tangent(x, y) if x.0 == 2 && y.0 == 1)
        ));
    }

    /// An arc on from a line's end: tangent to it, through the point, the
    /// right way round, joined and held tangent.
    #[test]
    fn a_tangent_arc_goes_on_from_a_lines_end() {
        let arc = tangent_arc(DVec2::ZERO, DVec2::X, DVec2::new(0.01, 0.01)).unwrap();
        let Curve2D::Arc { c, r, sweep, .. } = arc else {
            panic!()
        };
        assert!(near(c, DVec2::new(0.0, 0.01)) && (r - 0.01).abs() < 1e-12);
        assert!(
            (sweep - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "{sweep}"
        );
        // Turning right: clockwise.
        let Curve2D::Arc { sweep, .. } =
            tangent_arc(DVec2::ZERO, DVec2::X, DVec2::new(0.01, -0.01)).unwrap()
        else {
            panic!()
        };
        assert!(sweep < 0.0);
        assert!(tangent_arc(DVec2::ZERO, DVec2::X, DVec2::new(0.02, 0.0)).is_none());
        let mut b = Bench::new();
        b.run("draft.line(points=[-0.02, 0, 0, 0])").unwrap();
        b.run("draft.tangent_arc(start=(0, 0, 0), dir=(1, 0, 0), end=(0.01, 0.01, 0))")
            .unwrap();
        let held = &b.drawing().constraints;
        assert!(held.iter().any(|c| matches!(c, Constraint::Tangent(..))));
        assert!(held.iter().any(|c| matches!(c, Constraint::Coincident(..))));
    }
}
