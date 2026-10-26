//! Holding a drawing to its constraints and driving dimensions.
//!
//! # By connected piece
//!
//! The entities a constraint or a driving dimension joins are one piece;
//! each piece is solved on its own, and an entity nothing holds is not
//! touched at all. A drafted sheet is mostly free geometry with a few
//! held clusters, and a solve over all of it at once — what the sketch
//! solver did — grew with the whole sheet instead of with what was held.
//!
//! # The method
//!
//! Levenberg–Marquardt on the residuals. Each residual depends on the
//! variables of the one to three entities it names, so its row of the
//! Jacobian is found by moving only those — the cost grows with the
//! relations, not with their count times the variables.
//!
//! # What is reported
//!
//! For each piece: whether it converged, what is left over, and its degrees
//! of freedom — the variables less the rank of the Jacobian at the
//! solution. Zero is fully held; above zero something can still move; a
//! residual left over means the relations contradict each other.

use crate::drawing::{
    Constraint, DimKind, Dimension, Drawing, EntityId, EntityKind, EntityRef, Part,
};
use kreate_geom2d::{Curve2D, DVec2, Spline};
use std::collections::BTreeMap;

/// How a piece came out.
#[derive(Clone, Debug, PartialEq)]
pub struct PieceSolve {
    pub entities: Vec<EntityId>,
    pub converged: bool,
    /// The residuals' size when it stopped.
    pub error: f64,
    /// What can still move: zero when fully held.
    pub freedom: usize,
}

/// A point the user is dragging, and where to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    pub at: EntityRef,
    pub to: DVec2,
}

/// The variables of one entity, laid out as the solver holds them.
#[derive(Clone, Debug)]
enum Vars {
    Point,
    Line,
    Circle,
    /// Centre, radius, and the angles of its two ends.
    Arc,
    Ellipse,
    /// A polyline's vertices; its bulges stay.
    Polyline(usize),
    /// A spline's fit points, or its control points when it has none.
    Spline(usize),
    Text,
}

impl Vars {
    fn of(kind: &EntityKind) -> Option<Vars> {
        Some(match kind {
            EntityKind::Point(_) => Vars::Point,
            EntityKind::Text(_) => Vars::Text,
            EntityKind::Dimension(_) => return None,
            EntityKind::Curve(c) => match c {
                Curve2D::Line { .. } => Vars::Line,
                Curve2D::Circle { .. } => Vars::Circle,
                Curve2D::Arc { .. } => Vars::Arc,
                Curve2D::Ellipse { .. } => Vars::Ellipse,
                Curve2D::Polyline { verts, .. } => Vars::Polyline(verts.len()),
                Curve2D::Spline(s) => Vars::Spline(if s.fit.is_empty() {
                    s.ctrl.len()
                } else {
                    s.fit.len()
                }),
            },
        })
    }

    fn len(&self) -> usize {
        match self {
            Vars::Point | Vars::Text => 2,
            Vars::Line => 4,
            Vars::Circle => 3,
            Vars::Arc => 5,
            Vars::Ellipse => 5,
            Vars::Polyline(n) | Vars::Spline(n) => 2 * n,
        }
    }
}

/// An entity's variables read out of it.
fn read(kind: &EntityKind) -> Vec<f64> {
    match kind {
        EntityKind::Point(p) => vec![p.x, p.y],
        EntityKind::Text(t) => vec![t.at.x, t.at.y],
        EntityKind::Dimension(_) => Vec::new(),
        EntityKind::Curve(c) => match c {
            Curve2D::Line { a, b } => vec![a.x, a.y, b.x, b.y],
            Curve2D::Circle { c, r } => vec![c.x, c.y, *r],
            Curve2D::Arc { c, r, start, sweep } => vec![c.x, c.y, *r, *start, start + sweep],
            Curve2D::Ellipse {
                c, major, ratio, ..
            } => vec![c.x, c.y, major.x, major.y, *ratio],
            Curve2D::Polyline { verts, .. } => verts.iter().flat_map(|(p, _)| [p.x, p.y]).collect(),
            Curve2D::Spline(s) => {
                let pts = if s.fit.is_empty() { &s.ctrl } else { &s.fit };
                pts.iter().flat_map(|p| [p.x, p.y]).collect()
            }
        },
    }
}

/// An entity with its variables written back into it.
fn write(kind: &EntityKind, x: &[f64]) -> EntityKind {
    let p = |i: usize| DVec2::new(x[i], x[i + 1]);
    match kind {
        EntityKind::Point(_) => EntityKind::Point(p(0)),
        EntityKind::Text(t) => {
            let mut t = t.clone();
            t.at = p(0);
            EntityKind::Text(t)
        }
        EntityKind::Dimension(d) => EntityKind::Dimension(d.clone()),
        EntityKind::Curve(c) => EntityKind::Curve(match c {
            Curve2D::Line { .. } => Curve2D::Line { a: p(0), b: p(2) },
            Curve2D::Circle { .. } => Curve2D::Circle {
                c: p(0),
                r: x[2].abs(),
            },
            Curve2D::Arc { .. } => Curve2D::Arc {
                c: p(0),
                r: x[2].abs(),
                start: x[3],
                sweep: x[4] - x[3],
            },
            Curve2D::Ellipse { start, sweep, .. } => Curve2D::Ellipse {
                c: p(0),
                major: p(2),
                ratio: x[4],
                start: *start,
                sweep: *sweep,
            },
            Curve2D::Polyline { verts, closed } => Curve2D::Polyline {
                verts: verts
                    .iter()
                    .enumerate()
                    .map(|(i, (_, b))| (p(2 * i), *b))
                    .collect(),
                closed: *closed,
            },
            Curve2D::Spline(s) => {
                let pts: Vec<DVec2> = (0..x.len() / 2).map(|i| p(2 * i)).collect();
                if s.fit.is_empty() {
                    let mut s = s.clone();
                    s.ctrl = pts;
                    Curve2D::Spline(s)
                } else {
                    Curve2D::Spline(Spline::through(&pts))
                }
            }
        }),
    }
}

/// The variables of a piece: where each entity's start in one vector, and
/// the entity itself to rebuild from them.
struct Layout {
    order: Vec<EntityId>,
    offset: BTreeMap<EntityId, (usize, Vars)>,
    kinds: BTreeMap<EntityId, EntityKind>,
    len: usize,
}

impl Layout {
    fn new(drawing: &Drawing, ids: &[EntityId]) -> Layout {
        let mut offset = BTreeMap::new();
        let mut kinds = BTreeMap::new();
        let mut order = Vec::new();
        let mut at = 0;
        for id in ids {
            let Some(e) = drawing.entity(*id) else {
                continue;
            };
            let Some(vars) = Vars::of(&e.kind) else {
                continue;
            };
            let n = vars.len();
            offset.insert(*id, (at, vars));
            kinds.insert(*id, e.kind.clone());
            order.push(*id);
            at += n;
        }
        Layout {
            order,
            offset,
            kinds,
            len: at,
        }
    }

    fn read(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.len];
        for id in &self.order {
            let (at, vars) = &self.offset[id];
            x[*at..*at + vars.len()].copy_from_slice(&read(&self.kinds[id]));
        }
        x
    }

    /// The entity `id` as the variables `x` make it.
    fn entity(&self, id: EntityId, x: &[f64]) -> Option<EntityKind> {
        let (at, vars) = self.offset.get(&id)?;
        Some(write(&self.kinds[&id], &x[*at..*at + vars.len()]))
    }

    /// The variable indices of an entity.
    fn range(&self, id: EntityId) -> std::ops::Range<usize> {
        match self.offset.get(&id) {
            Some((at, vars)) => *at..*at + vars.len(),
            None => 0..0,
        }
    }
}

/// Where a part is, given the variables.
fn point(layout: &Layout, x: &[f64], r: EntityRef) -> Option<DVec2> {
    let kind = layout.entity(r.entity, x)?;
    part_point(&kind, r.part)
}

fn part_point(kind: &EntityKind, part: Part) -> Option<DVec2> {
    match (kind, part) {
        (EntityKind::Point(p), _) => Some(*p),
        (EntityKind::Text(t), _) => Some(t.at),
        (EntityKind::Curve(c), part) => match part {
            Part::Start => Some(c.start()),
            Part::End => Some(c.end()),
            Part::Mid => {
                let (lo, hi) = c.domain();
                Some(c.point((lo + hi) / 2.0))
            }
            Part::Center => match c {
                Curve2D::Circle { c, .. } | Curve2D::Arc { c, .. } | Curve2D::Ellipse { c, .. } => {
                    Some(*c)
                }
                _ => None,
            },
            Part::Vertex(i) => match c {
                Curve2D::Polyline { verts, .. } => verts.get(i as usize).map(|v| v.0),
                Curve2D::Spline(s) => s.fit.get(i as usize).copied(),
                _ => None,
            },
            Part::Whole => None,
        },
        _ => None,
    }
}

fn curve(layout: &Layout, x: &[f64], id: EntityId) -> Option<Curve2D> {
    match layout.entity(id, x)? {
        EntityKind::Curve(c) => Some(c),
        _ => None,
    }
}

/// A line's direction, for a line or the part of a polyline it is.
fn direction(c: &Curve2D) -> Option<DVec2> {
    match c {
        Curve2D::Line { a, b } => Some(*b - *a),
        _ => None,
    }
}

fn radius(c: &Curve2D) -> Option<(DVec2, f64)> {
    match c {
        Curve2D::Circle { c, r } | Curve2D::Arc { c, r, .. } => Some((*c, *r)),
        _ => None,
    }
}

/// One relation the solver holds: a constraint, a driving dimension, or a
/// drag. Its residuals, given the variables.
enum Relation<'a> {
    Constraint(&'a Constraint),
    Dimension(&'a Dimension, f64),
    Drag(Drag),
}

impl Relation<'_> {
    fn entities(&self) -> Vec<EntityId> {
        match self {
            Relation::Constraint(c) => c.entities(),
            Relation::Dimension(d, _) => d.refs.iter().map(|r| r.entity).collect(),
            Relation::Drag(d) => vec![d.at.entity],
        }
    }

    /// Its residuals, appended to `out`. Each is scaled to a length, so the
    /// solver weighs a millimetre the same whatever relation it is in.
    fn residuals(&self, layout: &Layout, x: &[f64], scale: f64, out: &mut Vec<f64>) {
        match self {
            Relation::Drag(d) => {
                // A drag is a wish, not a rule: weak beside the relations.
                if let Some(p) = point(layout, x, d.at) {
                    let w = 1e-3;
                    out.extend([(p.x - d.to.x) * w, (p.y - d.to.y) * w]);
                }
            }
            Relation::Constraint(c) => constraint(c, layout, x, scale, out),
            Relation::Dimension(d, sign) => dimension(d, *sign, layout, x, out),
        }
    }
}

fn constraint(c: &Constraint, layout: &Layout, x: &[f64], scale: f64, out: &mut Vec<f64>) {
    let pt = |r: &EntityRef| point(layout, x, *r);
    let cv = |id: &EntityId| curve(layout, x, *id);
    match c {
        Constraint::Coincident(a, b) => {
            if let (Some(p), Some(q)) = (pt(a), pt(b)) {
                out.extend([p.x - q.x, p.y - q.y]);
            }
        }
        Constraint::Horizontal(a, b) | Constraint::Vertical(a, b) => {
            let horizontal = matches!(c, Constraint::Horizontal(..));
            let (p, q) = match b {
                Some(b) => (pt(a), pt(b)),
                None => match cv(&a.entity) {
                    Some(Curve2D::Line { a, b }) => (Some(a), Some(b)),
                    _ => (None, None),
                },
            };
            if let (Some(p), Some(q)) = (p, q) {
                out.push(if horizontal { p.y - q.y } else { p.x - q.x });
            }
        }
        Constraint::Parallel(a, b) | Constraint::Perpendicular(a, b) => {
            if let (Some(da), Some(db)) = (
                cv(a).and_then(|c| direction(&c)),
                cv(b).and_then(|c| direction(&c)),
            ) {
                let (la, lb) = (da.length().max(1e-12), db.length().max(1e-12));
                let v = if matches!(c, Constraint::Parallel(..)) {
                    da.perp_dot(db)
                } else {
                    da.dot(db)
                } / (la * lb);
                out.push(v * scale);
            }
        }
        Constraint::Equal(a, b) => {
            if let (Some(ca), Some(cb)) = (cv(a), cv(b)) {
                if let (Some(da), Some(db)) = (direction(&ca), direction(&cb)) {
                    out.push(da.length() - db.length());
                } else if let (Some((_, ra)), Some((_, rb))) = (radius(&ca), radius(&cb)) {
                    out.push(ra - rb);
                }
            }
        }
        Constraint::Concentric(a, b) => {
            let ca = pt(&EntityRef::at(*a, Part::Center));
            let cb = pt(&EntityRef::at(*b, Part::Center));
            if let (Some(p), Some(q)) = (ca, cb) {
                out.extend([p.x - q.x, p.y - q.y]);
            }
        }
        Constraint::Tangent(a, b) | Constraint::Smooth(a, b) => {
            let (Some(ca), Some(cb)) = (cv(a), cv(b)) else {
                return;
            };
            let line_circle = |l: &Curve2D, k: &Curve2D| -> Option<f64> {
                let Curve2D::Line { a, b } = l else {
                    return None;
                };
                let (c, r) = radius(k)?;
                let d = *b - *a;
                Some((c - *a).perp_dot(d).abs() / d.length().max(1e-12) - r)
            };
            if let Some(v) = line_circle(&ca, &cb).or_else(|| line_circle(&cb, &ca)) {
                out.push(v);
            } else if let (Some((c1, r1)), Some((c2, r2))) = (radius(&ca), radius(&cb)) {
                // Inside or outside, whichever they are nearer to now.
                let d = c1.distance(c2);
                let outer = d - (r1 + r2);
                let inner = d - (r1 - r2).abs();
                out.push(if outer.abs() <= inner.abs() {
                    outer
                } else {
                    inner
                });
            }
        }
        Constraint::PointOn(p, on) => {
            let (Some(p), Some(c)) = (pt(p), cv(on)) else {
                return;
            };
            out.push(match &c {
                // On the line's extension too.
                Curve2D::Line { a, b } => {
                    let d = *b - *a;
                    (p - *a).perp_dot(d) / d.length().max(1e-12)
                }
                Curve2D::Circle { c, r } | Curve2D::Arc { c, r, .. } => p.distance(*c) - r,
                other => {
                    let (_, q) = other.closest(p);
                    q.distance(p)
                }
            });
        }
        Constraint::Symmetric(p, q, line) => {
            let (Some(p), Some(q), Some(Curve2D::Line { a, b })) = (pt(p), pt(q), cv(line)) else {
                return;
            };
            let d = (b - a).normalize_or_zero();
            let m = (p + q) / 2.0;
            out.push((m - a).perp_dot(d));
            out.push((q - p).dot(d));
        }
        Constraint::Midpoint(p, line) => {
            if let (Some(p), Some(c)) = (pt(p), cv(line)) {
                let m = c.point({
                    let (lo, hi) = c.domain();
                    (lo + hi) / 2.0
                });
                out.extend([p.x - m.x, p.y - m.y]);
            }
        }
        Constraint::Fixed(p, at) => {
            if let Some(p) = pt(p) {
                out.extend([p.x - at.x, p.y - at.y]);
            }
        }
    }
}

/// What a dimension measures, given the variables.
pub fn measure(
    d: &Dimension,
    kinds: &dyn Fn(EntityRef) -> Option<DVec2>,
    curve_of: &dyn Fn(EntityId) -> Option<Curve2D>,
) -> Option<f64> {
    let two = || -> Option<(DVec2, DVec2)> {
        match d.refs.as_slice() {
            [a, b] => Some((kinds(*a)?, kinds(*b)?)),
            [a] => match curve_of(a.entity)? {
                Curve2D::Line { a, b } => Some((a, b)),
                _ => None,
            },
            _ => None,
        }
    };
    Some(match d.kind {
        DimKind::Horizontal => {
            let (p, q) = two()?;
            q.x - p.x
        }
        DimKind::Vertical => {
            let (p, q) = two()?;
            q.y - p.y
        }
        DimKind::Aligned => {
            let (p, q) = two()?;
            p.distance(q)
        }
        DimKind::Radius | DimKind::Diameter | DimKind::ArcLength => {
            let c = curve_of(d.refs.first()?.entity)?;
            let (_, r) = radius(&c)?;
            match (d.kind, &c) {
                (DimKind::Radius, _) => r,
                (DimKind::Diameter, _) => 2.0 * r,
                (_, Curve2D::Arc { sweep, .. }) => r * sweep.abs(),
                _ => return None,
            }
        }
        DimKind::Angular => {
            let (a, b) = match d.refs.as_slice() {
                [a, b] => (curve_of(a.entity)?, curve_of(b.entity)?),
                _ => return None,
            };
            let (u, v) = (direction(&a)?, direction(&b)?);
            u.perp_dot(v).atan2(u.dot(v))
        }
        DimKind::OrdinateX | DimKind::OrdinateY => {
            let (origin, p) = match d.refs.as_slice() {
                [o, p] => (kinds(*o)?, kinds(*p)?),
                _ => return None,
            };
            if d.kind == DimKind::OrdinateX {
                p.x - origin.x
            } else {
                p.y - origin.y
            }
        }
    })
}

fn dimension(d: &Dimension, sign: f64, layout: &Layout, x: &[f64], out: &mut Vec<f64>) {
    let got = measure(d, &|r| point(layout, x, r), &|id| curve(layout, x, id));
    if let Some(got) = got {
        // Horizontal and vertical are signed by how they were drawn, so a
        // value holds the points on the side they are on.
        let want = match d.kind {
            DimKind::Horizontal | DimKind::Vertical | DimKind::OrdinateX | DimKind::OrdinateY => {
                d.value * sign
            }
            _ => d.value,
        };
        out.push(got - want);
    }
}

impl Drawing {
    /// Holds the drawing to its constraints and driving dimensions, piece
    /// by piece, and measures its driven dimensions afterwards.
    pub fn solve(&mut self) -> Vec<PieceSolve> {
        self.solve_dragging(&[])
    }

    /// As [`Drawing::solve`], with points being dragged toward places: they
    /// go as near as the relations let them.
    pub fn solve_dragging(&mut self, drags: &[Drag]) -> Vec<PieceSolve> {
        // The relations, and the sign each horizontal or vertical
        // dimension was drawn with.
        let mut relations: Vec<Relation> =
            self.constraints.iter().map(Relation::Constraint).collect();
        for e in &self.entities {
            if let EntityKind::Dimension(d) = &e.kind {
                if d.driving {
                    let now = measure(d, &|r| self.point_of(r), &|id| self.curve(id).cloned());
                    let sign = now.map_or(1.0, |v| if v < 0.0 { -1.0 } else { 1.0 });
                    relations.push(Relation::Dimension(d, sign));
                }
            }
        }
        relations.extend(drags.iter().map(|d| Relation::Drag(*d)));
        // Pieces: entities joined by a relation.
        let mut parent: BTreeMap<EntityId, EntityId> = BTreeMap::new();
        fn find(parent: &mut BTreeMap<EntityId, EntityId>, e: EntityId) -> EntityId {
            let p = *parent.entry(e).or_insert(e);
            if p == e {
                return e;
            }
            let root = find(parent, p);
            parent.insert(e, root);
            root
        }
        for r in &relations {
            let es = r.entities();
            for e in &es {
                find(&mut parent, *e);
            }
            for w in es.windows(2) {
                let (a, b) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
                if a != b {
                    parent.insert(a, b);
                }
            }
        }
        let mut pieces: BTreeMap<EntityId, Vec<EntityId>> = BTreeMap::new();
        let ids: Vec<EntityId> = parent.keys().copied().collect();
        for e in ids {
            let root = find(&mut parent, e);
            pieces.entry(root).or_default().push(e);
        }
        let scale = self.extent();
        let mut results = Vec::new();
        let mut updates: Vec<(EntityId, EntityKind)> = Vec::new();
        for (_, members) in pieces {
            let mine: Vec<&Relation> = relations
                .iter()
                .filter(|r| r.entities().first().is_some_and(|e| members.contains(e)))
                .collect();
            let layout = Layout::new(self, &members);
            if layout.len == 0 {
                continue;
            }
            let start = layout.read();
            let (x, outcome) = if drags.is_empty() {
                levenberg_marquardt(&layout, &mine, scale, start)
            } else {
                // Toward the drag first, then held exactly from there: the
                // drag is a wish the relations bend to, never one they give
                // way to.
                let (toward, _) = levenberg_marquardt(&layout, &mine, scale, start);
                let firm: Vec<&Relation> = mine
                    .iter()
                    .copied()
                    .filter(|r| !matches!(r, Relation::Drag(_)))
                    .collect();
                levenberg_marquardt(&layout, &firm, scale, toward)
            };
            for id in &layout.order {
                if let Some(kind) = layout.entity(*id, &x) {
                    updates.push((*id, kind));
                }
            }
            results.push(PieceSolve {
                entities: members,
                ..outcome
            });
        }
        for (id, kind) in updates {
            if let Some(e) = self.entity_mut(id) {
                e.kind = kind;
            }
        }
        self.measure_driven();
        results
    }

    /// Driven dimensions: what the geometry measures now.
    pub fn measure_driven(&mut self) {
        let values: Vec<(EntityId, f64)> = self
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Dimension(d) if !d.driving => {
                    let v = measure(d, &|r| self.point_of(r), &|id| self.curve(id).cloned())?;
                    Some((e.id, v.abs()))
                }
                _ => None,
            })
            .collect();
        for (id, v) in values {
            if let Some(EntityKind::Dimension(d)) = self.entity_mut(id).map(|e| &mut e.kind) {
                d.value = v;
            }
        }
    }

    /// The size of what is drawn, for tolerances relative to it.
    fn extent(&self) -> f64 {
        let (lo, hi) = self
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c) => Some(c.bounds()),
                EntityKind::Point(p) => Some((*p, *p)),
                _ => None,
            })
            .fold(
                (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
                |(lo, hi), (a, b)| (lo.min(a), hi.max(b)),
            );
        if lo.x > hi.x {
            1.0
        } else {
            (hi - lo).length().max(1e-6)
        }
    }
}

fn levenberg_marquardt(
    layout: &Layout,
    relations: &[&Relation],
    scale: f64,
    start: Vec<f64>,
) -> (Vec<f64>, PieceSolve) {
    let mut x = start;
    let n = x.len();
    // Which variables each relation's residuals depend on.
    let deps: Vec<Vec<usize>> = relations
        .iter()
        .map(|r| r.entities().iter().flat_map(|e| layout.range(*e)).collect())
        .collect();
    let residuals = |x: &[f64]| -> Vec<Vec<f64>> {
        relations
            .iter()
            .map(|r| {
                let mut out = Vec::new();
                r.residuals(layout, x, scale, &mut out);
                out
            })
            .collect()
    };
    let flat = |r: &[Vec<f64>]| -> Vec<f64> { r.iter().flatten().copied().collect() };
    let norm = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let tolerance = scale * 1e-10;
    let mut r = residuals(&x);
    let mut lambda = 1e-3;
    let h = scale * 1e-7;
    let jacobian = |x: &[f64], r0: &[Vec<f64>]| -> (Vec<f64>, usize) {
        let m: usize = r0.iter().map(|v| v.len()).sum();
        let mut jac = vec![0.0; m * n];
        let mut row = 0;
        for (k, rel) in relations.iter().enumerate() {
            let rows = r0[k].len();
            for &j in &deps[k] {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[j] += h;
                xm[j] -= h;
                let (mut rp, mut rm) = (Vec::new(), Vec::new());
                rel.residuals(layout, &xp, scale, &mut rp);
                rel.residuals(layout, &xm, scale, &mut rm);
                for i in 0..rows.min(rp.len()).min(rm.len()) {
                    jac[(row + i) * n + j] = (rp[i] - rm[i]) / (2.0 * h);
                }
            }
            row += rows;
        }
        (jac, m)
    };
    for _ in 0..100 {
        let rf = flat(&r);
        if norm(&rf) <= tolerance {
            break;
        }
        let (jac, m) = jacobian(&x, &r);
        // (JᵀJ + λ diag) δ = -Jᵀr
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for i in 0..m {
            for j in 0..n {
                let jij = jac[i * n + j];
                if jij == 0.0 {
                    continue;
                }
                g[j] -= jij * rf[i];
                for k in 0..n {
                    a[j * n + k] += jij * jac[i * n + k];
                }
            }
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut damped = a.clone();
            for j in 0..n {
                damped[j * n + j] += lambda * (1.0 + a[j * n + j]);
            }
            let Some(step) = gauss(damped, g.clone(), n) else {
                lambda *= 10.0;
                continue;
            };
            let trial: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            let rt = residuals(&trial);
            if norm(&flat(&rt)) < norm(&rf) {
                x = trial;
                r = rt;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    // What is left, and how much can still move. Drags are wishes and do
    // not hold anything, so they are left out of both.
    let held: Vec<&Relation> = relations
        .iter()
        .copied()
        .filter(|r| !matches!(r, Relation::Drag(_)))
        .collect();
    let mut rest = Vec::new();
    for rel in &held {
        rel.residuals(layout, &x, scale, &mut rest);
    }
    let error = norm(&rest);
    let held_layout_relations: Vec<&Relation> = held.clone();
    let rank = {
        let deps: Vec<Vec<usize>> = held_layout_relations
            .iter()
            .map(|r| r.entities().iter().flat_map(|e| layout.range(*e)).collect())
            .collect();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (k, rel) in held_layout_relations.iter().enumerate() {
            let mut r0 = Vec::new();
            rel.residuals(layout, &x, scale, &mut r0);
            let mut block = vec![vec![0.0; n]; r0.len()];
            for &j in &deps[k] {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[j] += h;
                xm[j] -= h;
                let (mut rp, mut rm) = (Vec::new(), Vec::new());
                rel.residuals(layout, &xp, scale, &mut rp);
                rel.residuals(layout, &xm, scale, &mut rm);
                for i in 0..block.len().min(rp.len()).min(rm.len()) {
                    block[i][j] = (rp[i] - rm[i]) / (2.0 * h);
                }
            }
            rows.extend(block);
        }
        rank(rows, n)
    };
    let converged = error <= tolerance.max(scale * 1e-9);
    (
        x,
        PieceSolve {
            entities: Vec::new(),
            converged,
            error,
            freedom: n.saturating_sub(rank),
        },
    )
}

/// The rank of a matrix by rows, by elimination with a tolerance.
fn rank(mut rows: Vec<Vec<f64>>, n: usize) -> usize {
    let scale = rows
        .iter()
        .flatten()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    let eps = scale * 1e-8;
    let mut rank = 0;
    for col in 0..n {
        let Some(pivot) =
            (rank..rows.len()).max_by(|a, b| rows[*a][col].abs().total_cmp(&rows[*b][col].abs()))
        else {
            break;
        };
        if rows[pivot][col].abs() <= eps {
            continue;
        }
        rows.swap(rank, pivot);
        let (top, below) = rows.split_at_mut(rank + 1);
        let pivot_row = &top[rank];
        for row in below {
            let f = row[col] / pivot_row[col];
            if f != 0.0 {
                for (x, p) in row[col..n].iter_mut().zip(&pivot_row[col..n]) {
                    *x -= f * p;
                }
            }
        }
        rank += 1;
    }
    rank
}

fn gauss(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|i, j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..n {
            let f = a[row * n + col] / a[col * n + col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row * n + k] * x[k];
        }
        x[row] = s / a[row * n + row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawing::{Drawing, EntityRef as R, Part};
    use std::f64::consts::FRAC_PI_2;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn line(d: &mut Drawing, a: DVec2, b: DVec2) -> EntityId {
        d.add_curve(Curve2D::Line { a, b })
    }

    fn dim(kind: DimKind, refs: Vec<EntityRef>, value: f64) -> EntityKind {
        EntityKind::Dimension(Dimension {
            kind,
            refs,
            text_at: DVec2::ZERO,
            style: String::new(),
            driving: true,
            value,
            text: None,
            tolerance: None,
        })
    }

    /// The case the sketch solver could not do: a level line, and a
    /// slanted one made parallel to it.
    #[test]
    fn a_line_made_parallel_to_a_level_one_levels() {
        let mut d = Drawing::new();
        let a = line(&mut d, v(0.0, 0.0), v(0.04, 0.0));
        let b = line(&mut d, v(0.0, 0.02), v(0.03, 0.04));
        d.constraints
            .push(Constraint::Horizontal(R::whole(a), None));
        d.constraints.push(Constraint::Parallel(a, b));
        let out = d.solve();
        assert!(out.iter().all(|p| p.converged), "{out:?}");
        let Some(Curve2D::Line { a: p, b: q }) = d.curve(b).cloned() else {
            panic!()
        };
        assert!((p.y - q.y).abs() < 1e-12, "{p} {q}");
    }

    /// A rectangle held by its sides and two dimensions, with one corner
    /// fixed: fully held, and the sizes it was given.
    #[test]
    fn a_dimensioned_rectangle_is_fully_held() {
        let mut d = Drawing::new();
        let c = [
            v(0.0, 0.0),
            v(0.035, 0.001),
            v(0.036, 0.018),
            v(0.001, 0.019),
        ];
        let s: Vec<EntityId> = (0..4).map(|i| line(&mut d, c[i], c[(i + 1) % 4])).collect();
        for i in 0..4 {
            d.constraints.push(Constraint::Coincident(
                R::at(s[i], Part::End),
                R::at(s[(i + 1) % 4], Part::Start),
            ));
        }
        d.constraints
            .push(Constraint::Horizontal(R::whole(s[0]), None));
        d.constraints
            .push(Constraint::Horizontal(R::whole(s[2]), None));
        d.constraints
            .push(Constraint::Vertical(R::whole(s[1]), None));
        d.constraints
            .push(Constraint::Vertical(R::whole(s[3]), None));
        d.constraints
            .push(Constraint::Fixed(R::at(s[0], Part::Start), v(0.0, 0.0)));
        d.add(dim(DimKind::Aligned, vec![R::whole(s[0])], 0.04));
        d.add(dim(DimKind::Aligned, vec![R::whole(s[1])], 0.02));
        let out = d.solve();
        assert_eq!(out.len(), 1, "one piece");
        assert!(out[0].converged, "{out:?}");
        assert_eq!(out[0].freedom, 0, "fully held: {out:?}");
        let Some(Curve2D::Line { a: p, b: q }) = d.curve(s[1]).cloned() else {
            panic!()
        };
        assert!(
            (p.x - 0.04).abs() < 1e-10 && (q.y - 0.02).abs() < 1e-10,
            "{p} {q}"
        );
        // One dimension off: one degree of freedom.
        let width = d
            .entities
            .iter()
            .find(|e| matches!(&e.kind, EntityKind::Dimension(x) if x.value == 0.04))
            .unwrap()
            .id;
        d.remove(&[width]);
        let out = d.solve();
        assert_eq!(out[0].freedom, 1, "{out:?}");
    }

    /// What nothing holds is not moved, and is not a piece.
    #[test]
    fn free_geometry_is_left_alone() {
        let mut d = Drawing::new();
        let free = line(&mut d, v(0.1, 0.1), v(0.2, 0.3));
        let held = line(&mut d, v(0.0, 0.0), v(0.01, 0.002));
        d.constraints
            .push(Constraint::Horizontal(R::whole(held), None));
        let before = d.curve(free).cloned();
        let out = d.solve();
        assert_eq!(out.len(), 1);
        assert_eq!(d.curve(free).cloned(), before);
    }

    #[test]
    fn a_line_tangent_to_a_circle_touches_it() {
        let mut d = Drawing::new();
        let circle = d.add_curve(Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 0.01,
        });
        let l = line(&mut d, v(-0.02, 0.013), v(0.02, 0.012));
        d.constraints.push(Constraint::Tangent(l, circle));
        d.constraints
            .push(Constraint::Horizontal(R::whole(l), None));
        d.add(dim(DimKind::Radius, vec![R::whole(circle)], 0.01));
        d.constraints
            .push(Constraint::Fixed(R::at(circle, Part::Center), v(0.0, 0.0)));
        let out = d.solve();
        assert!(out.iter().all(|p| p.converged), "{out:?}");
        let Some(Curve2D::Line { a, .. }) = d.curve(l).cloned() else {
            panic!()
        };
        assert!((a.y.abs() - 0.01).abs() < 1e-10, "{a}");
    }

    /// Relations that cannot all hold are reported, not hidden.
    #[test]
    fn a_contradiction_does_not_converge() {
        let mut d = Drawing::new();
        let l = line(&mut d, v(0.0, 0.0), v(0.03, 0.01));
        d.constraints
            .push(Constraint::Horizontal(R::whole(l), None));
        d.constraints.push(Constraint::Vertical(R::whole(l), None));
        d.add(dim(DimKind::Aligned, vec![R::whole(l)], 0.03));
        let out = d.solve();
        assert!(!out[0].converged, "{out:?}");
    }

    #[test]
    fn an_angle_dimension_turns_a_line() {
        let mut d = Drawing::new();
        let a = line(&mut d, v(0.0, 0.0), v(0.04, 0.0));
        let b = line(&mut d, v(0.0, 0.0), v(0.03, 0.03));
        d.constraints
            .push(Constraint::Horizontal(R::whole(a), None));
        d.constraints.push(Constraint::Coincident(
            R::at(a, Part::Start),
            R::at(b, Part::Start),
        ));
        d.add(dim(
            DimKind::Angular,
            vec![R::whole(a), R::whole(b)],
            FRAC_PI_2,
        ));
        let out = d.solve();
        assert!(out[0].converged, "{out:?}");
        let Some(Curve2D::Line { a: p, b: q }) = d.curve(b).cloned() else {
            panic!()
        };
        assert!((p.x - q.x).abs() < 1e-10, "upright: {p} {q}");
    }

    /// A drag goes as far as the relations let it: along a level line,
    /// not off it.
    #[test]
    fn a_dragged_end_slides_along_what_holds_it() {
        let mut d = Drawing::new();
        let l = line(&mut d, v(0.0, 0.0), v(0.03, 0.0));
        d.constraints
            .push(Constraint::Horizontal(R::whole(l), None));
        d.constraints
            .push(Constraint::Fixed(R::at(l, Part::Start), v(0.0, 0.0)));
        d.solve_dragging(&[Drag {
            at: R::at(l, Part::End),
            to: v(0.05, 0.02),
        }]);
        let Some(Curve2D::Line { b, .. }) = d.curve(l).cloned() else {
            panic!()
        };
        assert!((b.x - 0.05).abs() < 1e-6, "{b}");
        assert!(b.y.abs() < 1e-10, "{b}");
    }

    /// Driven dimensions report what the geometry measures.
    #[test]
    fn a_driven_dimension_measures() {
        let mut d = Drawing::new();
        let c = d.add_curve(Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 0.0125,
        });
        let mut k = dim(DimKind::Diameter, vec![R::whole(c)], 0.0);
        if let EntityKind::Dimension(x) = &mut k {
            x.driving = false;
        }
        let id = d.add(k);
        d.solve();
        let Some(EntityKind::Dimension(x)) = d.entity(id).map(|e| e.kind.clone()) else {
            panic!()
        };
        assert!((x.value - 0.025).abs() < 1e-12);
    }
}
