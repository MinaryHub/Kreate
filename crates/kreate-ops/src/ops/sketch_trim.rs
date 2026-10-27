//! Trim: the piece of a line, circle or arc between the crossings either side
//! of where it was clicked, taken out — Fusion's Trim.
//!
//! What it is cut at is every other line of the sketch and every line of
//! the other sketches lying in its plane, so a sketch can be trimmed against
//! the one beside it. A piece with no crossing on one side runs to its end;
//! a line or arc with none at all goes whole, and so does a circle crossed
//! fewer than twice.
//!
//! A circle trimmed becomes an arc, and an arc trimmed in the middle two arcs,
//! each still held round the same centre; a line trimmed keeps the relations
//! that still hold of what is left of it — level, upright, parallel — and
//! loses its length.

use kreate_math::DVec2;
use kreate_mesh::MeshData;
use kreate_scene::document::ObjectId;
use kreate_scene::sketch::Constraint;
use kreate_scene::{DocMut, Document};

use super::sketch_draw::{active_sketch, points, solve};
use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

use std::f64::consts::TAU;

/// How near the end a crossing has to be to count as the end: a line drawn
/// from another's end crosses it there, and that is not a place to cut.
const END: f64 = 1e-6;

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TrimParams {
    /// The line clicked, by its place in the sketch's list.
    #[param(hidden)]
    pub line: u32,
    /// Where on it, in the sketch's plane: where it was clicked, not a
    /// number to adjust afterwards.
    #[param(unit = "length", hidden)]
    pub at: [f64; 3],
}

/// Takes out the piece of a curve between the crossings either side of a
/// point on it.
pub struct Trim;

impl Operator for Trim {
    type Params = TrimParams;
    fn id() -> &'static str {
        "sketch.trim"
    }
    fn label() -> &'static str {
        "Trim"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = active_sketch(doc.document())
            .ok_or_else(|| OpError::Failed("no sketch is being drawn".into()))?;
        let at = DVec2::new(params.at[0], params.at[1]);
        let Some(plan) = plan(doc.document(), id, params.line as usize, at) else {
            return Ok(OpResult::Cancelled);
        };
        apply(doc, id, plan)
    }
}

/// What a trim at `at` on line `line` would take out, as segments in the
/// sketch's plane: for showing it before the click.
pub fn preview(doc: &Document, id: ObjectId, line: usize, at: DVec2) -> Vec<[DVec2; 2]> {
    let Some(plan) = plan(doc, id, line, at) else {
        return Vec::new();
    };
    match &plan.curve {
        Curve::Segment { a, b, .. } => {
            let (pa, pb) = (plan.points[*a as usize], plan.points[*b as usize]);
            vec![[pa.lerp(pb, plan.cut.0), pa.lerp(pb, plan.cut.1)]]
        }
        Curve::Round { .. } => {
            let round = plan.round().expect("a round");
            let (lo, hi) = plan.cut;
            let sweep = if hi > lo {
                hi - lo
            } else {
                hi + round.extent - lo
            };
            let n = ((sweep / TAU * 96.0).ceil() as usize).max(2);
            (0..n)
                .map(|i| {
                    let t = |k: usize| round.at(lo + sweep * k as f64 / n as f64);
                    [t(i), t(i + 1)]
                })
                .collect()
        }
    }
}

/// The curve a line belongs to.
#[derive(Clone, Debug)]
enum Curve {
    /// A straight line on its own, by its place in the list and its ends.
    Segment { line: usize, a: u32, b: u32 },
    /// A circle or an arc, by its constraint's place in the list.
    Round {
        constraint: usize,
        centre: u32,
        ring: Vec<u32>,
        closed: bool,
    },
}

/// A circle or arc measured by angle: anticlockwise from its first point.
struct RoundGeom {
    centre: DVec2,
    radius: f64,
    start: f64,
    /// How far round it goes: a full turn for a circle.
    extent: f64,
}

impl RoundGeom {
    fn at(&self, t: f64) -> DVec2 {
        let a = self.start + t;
        self.centre + DVec2::new(a.cos(), a.sin()) * self.radius
    }

    fn param(&self, p: DVec2) -> f64 {
        let d = p - self.centre;
        (d.y.atan2(d.x) - self.start).rem_euclid(TAU)
    }
}

struct Plan {
    points: Vec<DVec2>,
    curve: Curve,
    /// What goes: from `cut.0` to `cut.1`, by the curve's parameter — 0 to 1
    /// along a line, the angle round a circle or arc. A circle's may wrap.
    cut: (f64, f64),
    /// The whole curve goes.
    whole: bool,
}

impl Plan {
    fn round(&self) -> Option<RoundGeom> {
        let Curve::Round {
            centre,
            ring,
            closed,
            ..
        } = &self.curve
        else {
            return None;
        };
        let c = self.points[*centre as usize];
        let first = self.points[ring[0] as usize];
        let start = (first - c).y.atan2((first - c).x);
        let mut geom = RoundGeom {
            centre: c,
            radius: (first - c).length(),
            start,
            extent: TAU,
        };
        if !closed {
            let last = self.points[*ring.last()? as usize];
            geom.extent = geom.param(last);
            if geom.extent < END {
                geom.extent = TAU;
            }
        }
        Some(geom)
    }
}

/// Where segment `p0 p1` crosses segment `q0 q1`: the parameter along the
/// first. Touching counts; lying along each other does not.
fn crossing(p0: DVec2, p1: DVec2, q0: DVec2, q1: DVec2) -> Option<f64> {
    let r = p1 - p0;
    let s = q1 - q0;
    let denom = r.perp_dot(s);
    if denom.abs() < 1e-18 {
        return None;
    }
    let qp = q0 - p0;
    let t = qp.perp_dot(s) / denom;
    let u = qp.perp_dot(r) / denom;
    let slack = 1e-9;
    ((-slack..=1.0 + slack).contains(&t) && (-slack..=1.0 + slack).contains(&u))
        .then_some(t.clamp(0.0, 1.0))
}

/// Every line of the other sketches in this one's plane, in its own space.
fn neighbours(doc: &Document, id: ObjectId) -> Vec<[DVec2; 2]> {
    let back = doc.world_transform(id).inverse();
    let mut out = Vec::new();
    for (other, object) in doc.objects() {
        if other == id || object.sketch_lines.is_empty() || object.mesh().is_none() {
            continue;
        }
        let to_here = back * doc.world_transform(other);
        let pts: Vec<kreate_math::DVec3> = points(doc, other)
            .iter()
            .map(|p| to_here.transform_point3(kreate_math::DVec3::new(p.x, p.y, 0.0)))
            .collect();
        // In the plane: a small slack for the float the mesh keeps.
        let flat = |p: &kreate_math::DVec3| p.z.abs() < 1e-5;
        for l in &object.sketch_lines {
            let (Some(a), Some(b)) = (pts.get(l[0] as usize), pts.get(l[1] as usize)) else {
                continue;
            };
            if flat(a) && flat(b) {
                out.push([a.truncate(), b.truncate()]);
            }
        }
    }
    out
}

fn plan(doc: &Document, id: ObjectId, line: usize, at: DVec2) -> Option<Plan> {
    let object = doc.object(id)?;
    let pts = points(doc, id);
    let picked = *object.sketch_lines.get(line)?;
    if picked.iter().any(|v| *v as usize >= pts.len()) {
        return None;
    }
    // The circle or arc it is a piece of, if any.
    let curve = object
        .sketch
        .iter()
        .enumerate()
        .find_map(|(i, c)| {
            let (centre, ring, closed) = match c {
                Constraint::Circle { centre, ring } => (*centre, ring, true),
                Constraint::Arc { centre, ring } => (*centre, ring, false),
                _ => return None,
            };
            let n = ring.len();
            let pairs = if closed { n } else { n.saturating_sub(1) };
            let on = (0..pairs).any(|k| {
                let (u, v) = (ring[k], ring[(k + 1) % n]);
                (u, v) == (picked[0], picked[1]) || (v, u) == (picked[0], picked[1])
            });
            let sound = ring.len() >= 2
                && ring
                    .iter()
                    .chain([&centre])
                    .all(|v| (*v as usize) < pts.len());
            (on && sound).then(|| Curve::Round {
                constraint: i,
                centre,
                ring: ring.clone(),
                closed,
            })
        })
        .unwrap_or(Curve::Segment {
            line,
            a: picked[0],
            b: picked[1],
        });

    // Its own lines, which do not cut it.
    let own: Vec<[u32; 2]> = match &curve {
        Curve::Segment { a, b, .. } => vec![[*a, *b]],
        Curve::Round { ring, closed, .. } => {
            let n = ring.len();
            let pairs = if *closed { n } else { n - 1 };
            (0..pairs).map(|k| [ring[k], ring[(k + 1) % n]]).collect()
        }
    };
    let is_own = |l: &[u32; 2]| own.iter().any(|o| o == l || [o[1], o[0]] == *l);
    let mut cutters: Vec<[DVec2; 2]> = object
        .sketch_lines
        .iter()
        .filter(|l| !is_own(l))
        .filter_map(|l| Some([*pts.get(l[0] as usize)?, *pts.get(l[1] as usize)?]))
        .collect();
    cutters.extend(neighbours(doc, id));

    let mut plan = Plan {
        points: pts,
        curve,
        cut: (0.0, 0.0),
        whole: false,
    };
    match plan.curve.clone() {
        Curve::Segment { a, b, .. } => {
            let (pa, pb) = (plan.points[a as usize], plan.points[b as usize]);
            let d = pb - pa;
            let t = ((at - pa).dot(d) / d.length_squared().max(1e-24)).clamp(0.0, 1.0);
            let cuts: Vec<f64> = cutters
                .iter()
                .filter_map(|c| crossing(pa, pb, c[0], c[1]))
                .filter(|s| *s > END && *s < 1.0 - END)
                .collect();
            let lo = cuts.iter().copied().filter(|s| *s < t).fold(0.0, f64::max);
            let hi = cuts.iter().copied().filter(|s| *s > t).fold(1.0, f64::min);
            plan.cut = (lo, hi);
            plan.whole = lo <= 0.0 && hi >= 1.0;
        }
        Curve::Round { ring, closed, .. } => {
            let geom = plan.round()?;
            let n = ring.len();
            let pairs = if closed { n } else { n - 1 };
            let mut cuts: Vec<f64> = Vec::new();
            for k in 0..pairs {
                let (p0, p1) = (
                    plan.points[ring[k] as usize],
                    plan.points[ring[(k + 1) % n] as usize],
                );
                for c in &cutters {
                    if let Some(s) = crossing(p0, p1, c[0], c[1]) {
                        let t = geom.param(p0.lerp(p1, s));
                        // The arc's first point reads as a full turn when the
                        // crossing is a hair before it.
                        let t = if !closed && t > geom.extent + END {
                            0.0
                        } else {
                            t
                        };
                        if closed || (t > END && t < geom.extent - END) {
                            cuts.push(t);
                        }
                    }
                }
            }
            cuts.sort_by(f64::total_cmp);
            cuts.dedup_by(|a, b| (*a - *b).abs() < END);
            let t = geom.param(at).min(geom.extent);
            if closed {
                if cuts.len() < 2 {
                    plan.whole = true;
                    plan.cut = (0.0, TAU);
                } else {
                    let lo = cuts
                        .iter()
                        .copied()
                        .filter(|s| *s < t)
                        .fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s))))
                        .unwrap_or(*cuts.last().expect("two"));
                    let hi = cuts.iter().copied().find(|s| *s > t).unwrap_or(cuts[0]);
                    plan.cut = (lo, hi);
                }
            } else {
                let lo = cuts.iter().copied().filter(|s| *s < t).fold(0.0, f64::max);
                let hi = cuts
                    .iter()
                    .copied()
                    .filter(|s| *s > t)
                    .fold(geom.extent, f64::min);
                plan.cut = (lo, hi);
                plan.whole = lo <= 0.0 && hi >= geom.extent;
            }
        }
    }
    Some(plan)
}

/// A relation a line's two ends took part in, carried to what is left of it:
/// `None` for one that no longer holds — its length, its equality to another.
fn carried(c: &Constraint, (a, b): (u32, u32), (na, nb): (u32, u32)) -> Option<Constraint> {
    let pair = |x: u32, y: u32| (x, y) == (a, b) || (x, y) == (b, a);
    let swap = |x: u32, y: u32| if (x, y) == (a, b) { (na, nb) } else { (nb, na) };
    let mut out = c.clone();
    match &mut out {
        Constraint::Horizontal { a: x, b: y } | Constraint::Vertical { a: x, b: y } => {
            if pair(*x, *y) {
                (*x, *y) = swap(*x, *y);
            }
        }
        Constraint::Distance { a: x, b: y, .. } => {
            if pair(*x, *y) {
                return None;
            }
        }
        Constraint::OnLine { a: x, b: y, .. } | Constraint::Tangent { a: x, b: y, .. } => {
            if pair(*x, *y) {
                (*x, *y) = swap(*x, *y);
            }
        }
        Constraint::Angle {
            a: x,
            b: y,
            c: z,
            d: w,
            ..
        }
        | Constraint::Parallel {
            a: x,
            b: y,
            c: z,
            d: w,
        }
        | Constraint::Perpendicular {
            a: x,
            b: y,
            c: z,
            d: w,
        } => {
            if pair(*x, *y) {
                (*x, *y) = swap(*x, *y);
            }
            if pair(*z, *w) {
                (*z, *w) = swap(*z, *w);
            }
        }
        Constraint::Equal {
            a: x,
            b: y,
            c: z,
            d: w,
        } if pair(*x, *y) || pair(*z, *w) => return None,
        _ => {}
    }
    Some(out)
}

/// Whether a relation is about the line from `a` to `b` itself.
fn about_line(c: &Constraint, (a, b): (u32, u32)) -> bool {
    let pair = |x: u32, y: u32| (x, y) == (a, b) || (x, y) == (b, a);
    match c {
        Constraint::Horizontal { a: x, b: y }
        | Constraint::Vertical { a: x, b: y }
        | Constraint::Distance { a: x, b: y, .. }
        | Constraint::OnLine { a: x, b: y, .. }
        | Constraint::Tangent { a: x, b: y, .. } => pair(*x, *y),
        Constraint::Angle {
            a: x,
            b: y,
            c: z,
            d: w,
            ..
        }
        | Constraint::Parallel {
            a: x,
            b: y,
            c: z,
            d: w,
        }
        | Constraint::Perpendicular {
            a: x,
            b: y,
            c: z,
            d: w,
        }
        | Constraint::Equal {
            a: x,
            b: y,
            c: z,
            d: w,
        } => pair(*x, *y) || pair(*z, *w),
        _ => false,
    }
}

fn apply(doc: &mut DocMut<'_>, id: ObjectId, plan: Plan) -> Result<OpResult, OpError> {
    let object = doc.document().object(id).expect("the sketch");
    let mut pts = plan.points.clone();
    let mut lines = object.sketch_lines.clone();
    let mut constraints = object.sketch.clone();
    // Points that may be left on no line: they go if nothing else uses them.
    let mut loose: Vec<u32> = Vec::new();
    let add = |pts: &mut Vec<DVec2>, p: DVec2| {
        pts.push(p);
        pts.len() as u32 - 1
    };

    match &plan.curve {
        Curve::Segment { line, a, b } => {
            let (a, b) = (*a, *b);
            let (pa, pb) = (pts[a as usize], pts[b as usize]);
            lines.remove(*line);
            loose.extend([a, b]);
            let (lo, hi) = plan.cut;
            let mut pieces: Vec<[u32; 2]> = Vec::new();
            if lo > 0.0 {
                let n = add(&mut pts, pa.lerp(pb, lo));
                pieces.push([a, n]);
            }
            if hi < 1.0 {
                let n = add(&mut pts, pa.lerp(pb, hi));
                pieces.push([n, b]);
            }
            lines.extend(pieces.iter().copied());
            // What was said of the line is said of what is left of it — of
            // both pieces, if it was cut in two.
            let mut out = Vec::new();
            for c in &constraints {
                if !about_line(c, (a, b)) {
                    out.push(c.clone());
                    continue;
                }
                for p in &pieces {
                    if let Some(c) = carried(c, (a, b), (p[0], p[1])) {
                        out.push(c);
                    }
                }
            }
            // Two pieces of one line stay in one line.
            if let [first, second] = pieces.as_slice() {
                out.push(Constraint::OnLine {
                    p: second[0],
                    a: first[0],
                    b: first[1],
                });
            }
            constraints = out;
        }
        Curve::Round {
            constraint,
            centre,
            ring,
            closed,
        } => {
            let geom = plan.round().expect("a round");
            let n = ring.len();
            let pairs = if *closed { n } else { n - 1 };
            let own: Vec<[u32; 2]> = (0..pairs).map(|k| [ring[k], ring[(k + 1) % n]]).collect();
            lines.retain(|l| !own.iter().any(|o| o == l || [o[1], o[0]] == *l));
            loose.extend(ring.iter().copied());
            // What is left, anticlockwise, as spans of angle.
            let (lo, hi) = plan.cut;
            let spans: Vec<(f64, f64)> = if plan.whole {
                Vec::new()
            } else if *closed {
                vec![(hi, if lo > hi { lo } else { lo + TAU })]
            } else {
                let mut s = Vec::new();
                if lo > END {
                    s.push((0.0, lo));
                }
                if hi < geom.extent - END {
                    s.push((hi, geom.extent));
                }
                s
            };
            if spans.is_empty() {
                loose.push(*centre);
            }
            let segments_per_turn = pairs as f64 / geom.extent * TAU;
            // The radius's dimension, kept for each arc that is left.
            let radius = constraints.iter().find_map(|c| match c {
                Constraint::Distance { a, b, value }
                    if (*a == *centre && ring.contains(b))
                        || (*b == *centre && ring.contains(a)) =>
                {
                    Some(*value)
                }
                _ => None,
            });
            let mut arcs: Vec<Vec<u32>> = Vec::new();
            for (from, to) in &spans {
                let count = ((segments_per_turn * (to - from) / TAU).round() as usize).max(2);
                let mut new_ring = Vec::with_capacity(count + 1);
                for k in 0..=count {
                    let t = from + (to - from) * k as f64 / count as f64;
                    // An arc's own end is kept as the point it was, so what
                    // was joined to it stays joined.
                    let v = if !closed && k == 0 && *from <= END {
                        ring[0]
                    } else if !closed && k == count && *to >= geom.extent - END {
                        *ring.last().expect("an arc")
                    } else {
                        add(&mut pts, geom.at(t))
                    };
                    new_ring.push(v);
                }
                lines.extend(new_ring.windows(2).map(|w| [w[0], w[1]]));
                arcs.push(new_ring);
            }
            let first_on = arcs.first().map(|r| r[0]);
            let mut out: Vec<Constraint> = Vec::new();
            for (i, c) in constraints.iter().enumerate() {
                if i == *constraint {
                    continue;
                }
                match c {
                    // The radius, said again of each arc below.
                    Constraint::Distance { a, b, .. }
                        if radius.is_some()
                            && ((*a == *centre && ring.contains(b))
                                || (*b == *centre && ring.contains(a))) => {}
                    // A tangent to it, to what is left of it.
                    Constraint::Tangent {
                        a,
                        b,
                        centre: c0,
                        on,
                    } if *c0 == *centre && ring.contains(on) => {
                        if let Some(on) = first_on {
                            out.push(Constraint::Tangent {
                                a: *a,
                                b: *b,
                                centre: *c0,
                                on,
                            });
                        }
                    }
                    _ => out.push(c.clone()),
                }
            }
            for r in &arcs {
                out.push(Constraint::Arc {
                    centre: *centre,
                    ring: r.clone(),
                });
                if let Some(value) = radius {
                    out.push(Constraint::Distance {
                        a: *centre,
                        b: r[0],
                        value,
                    });
                }
            }
            constraints = out;
        }
    }

    // The loose points nothing is on any more go, and what is said of them.
    let mut used = vec![false; pts.len()];
    for l in &lines {
        for v in l {
            if let Some(u) = used.get_mut(*v as usize) {
                *u = true;
            }
        }
    }
    for c in &constraints {
        if let Constraint::Arc { centre, .. } | Constraint::Circle { centre, .. } = c {
            if let Some(u) = used.get_mut(*centre as usize) {
                *u = true;
            }
        }
    }
    let mut gone = vec![false; pts.len()];
    for v in loose {
        if v != 0 && !used[v as usize] {
            gone[v as usize] = true;
        }
    }
    let mut map = vec![u32::MAX; pts.len()];
    let mut kept = Vec::new();
    for (i, p) in pts.iter().enumerate() {
        if !gone[i] {
            map[i] = kept.len() as u32;
            kept.push(*p);
        }
    }
    let renumber = |v: u32| map[v as usize];
    let constraints: Vec<Constraint> = constraints
        .iter()
        .filter(|c| {
            c.points()
                .iter()
                .all(|v| (*v as usize) < gone.len() && !gone[*v as usize])
        })
        .map(|c| c.remapped(&renumber))
        .collect();
    let lines: Vec<[u32; 2]> = lines
        .iter()
        .filter(|l| {
            l.iter()
                .all(|v| (*v as usize) < gone.len() && !gone[*v as usize])
        })
        .map(|l| [renumber(l[0]), renumber(l[1])])
        .collect();

    let mesh_id = object.mesh().expect("a sketch is a mesh");
    let positions: Vec<kreate_math::Vec3> = kept
        .iter()
        .map(|p| kreate_math::Vec3::new(p.x as f32, p.y as f32, 0.0))
        .collect();
    *doc.document_mut()
        .mesh_mut(mesh_id)
        .expect("the sketch's mesh") = MeshData::from_faces(positions, &[]);
    let o = doc.document_mut().object_mut(id).expect("the sketch");
    o.sketch_lines = lines;
    o.sketch = constraints;
    solve(doc, id);
    Ok(OpResult::Finished)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};

    struct Bench {
        r: OperatorRegistry,
        h: History,
        doc: Document,
    }

    impl Bench {
        fn new() -> Self {
            let mut r = OperatorRegistry::new();
            crate::register_builtin(&mut r);
            let mut b = Self {
                r,
                h: History::new(),
                doc: Document::new(),
            };
            b.run("sketch.new", &[]);
            b
        }

        fn run(&mut self, id: &str, pairs: &[(&'static str, PropertyValue)]) -> OpResult {
            let mut p = PropertyMap::new();
            for (k, v) in pairs {
                p.set(k, v.clone());
            }
            self.h
                .dispatch(
                    &self.r,
                    &mut self.doc,
                    &Context { mode: Mode::Object },
                    id,
                    &p,
                )
                .unwrap()
        }

        fn line(&mut self, pts: &[f64]) {
            self.run(
                "sketch.line",
                &[("points", PropertyValue::Floats(pts.to_vec()))],
            );
        }

        fn sketch(&self) -> ObjectId {
            active_sketch(&self.doc).unwrap()
        }

        /// The line nearest `at`, by its place in the list.
        fn line_at(&self, at: DVec2) -> u32 {
            let id = self.sketch();
            let p = points(&self.doc, id);
            let o = self.doc.object(id).unwrap();
            let dist = |l: &[u32; 2]| {
                let (a, b) = (p[l[0] as usize], p[l[1] as usize]);
                let d = b - a;
                let t = ((at - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
                (a + d * t).distance(at)
            };
            o.sketch_lines
                .iter()
                .enumerate()
                .min_by(|x, y| dist(x.1).total_cmp(&dist(y.1)))
                .unwrap()
                .0 as u32
        }

        fn trim(&mut self, x: f64, y: f64) -> OpResult {
            let line = self.line_at(DVec2::new(x, y));
            self.run(
                "sketch.trim",
                &[
                    ("line", PropertyValue::Int(line as i64)),
                    ("at", PropertyValue::Vec3([x, y, 0.0])),
                ],
            )
        }

        /// Every line, as its two ends.
        fn segments(&self) -> Vec<[DVec2; 2]> {
            let id = self.sketch();
            let p = points(&self.doc, id);
            self.doc
                .object(id)
                .unwrap()
                .sketch_lines
                .iter()
                .map(|l| [p[l[0] as usize], p[l[1] as usize]])
                .collect()
        }
    }

    fn near(a: DVec2, b: DVec2) -> bool {
        a.distance(b) < 1e-6
    }

    #[test]
    fn a_cross_trimmed_leaves_a_t() {
        let mut b = Bench::new();
        b.line(&[-1.0, 0.0, 1.0, 0.0]);
        b.line(&[0.0, -1.0, 0.0, 1.0]);
        // The upright's lower half goes, from the crossing down.
        assert_eq!(b.trim(0.0, -0.5), OpResult::Finished);
        let segs = b.segments();
        assert_eq!(segs.len(), 2);
        let upright = segs
            .iter()
            .find(|s| (s[0].x - s[1].x).abs() < 1e-9)
            .unwrap();
        let ends = [upright[0], upright[1]];
        assert!(
            ends.iter().any(|p| near(*p, DVec2::new(0.0, 0.0))),
            "{ends:?}"
        );
        assert!(
            ends.iter().any(|p| near(*p, DVec2::new(0.0, 1.0))),
            "{ends:?}"
        );
    }

    #[test]
    fn the_middle_between_two_crossings_goes() {
        let mut b = Bench::new();
        b.line(&[0.0, 0.0, 4.0, 0.0]);
        b.line(&[1.0, -1.0, 1.0, 1.0]);
        b.line(&[3.0, -1.0, 3.0, 1.0]);
        b.trim(2.0, 0.0);
        let level: Vec<_> = b
            .segments()
            .into_iter()
            .filter(|s| (s[0].y - s[1].y).abs() < 1e-9)
            .collect();
        assert_eq!(level.len(), 2, "{level:?}");
        let mut lengths: Vec<f64> = level.iter().map(|s| s[0].distance(s[1])).collect();
        lengths.sort_by(f64::total_cmp);
        assert!((lengths[0] - 1.0).abs() < 1e-6 && (lengths[1] - 1.0).abs() < 1e-6);
        // Both pieces still level.
        let o = b.doc.object(b.sketch()).unwrap();
        let level_count = o
            .sketch
            .iter()
            .filter(|c| matches!(c, Constraint::Horizontal { .. }))
            .count();
        assert_eq!(level_count, 2, "{:?}", o.sketch);
    }

    #[test]
    fn a_line_crossing_nothing_goes_whole() {
        let mut b = Bench::new();
        b.line(&[1.0, 1.0, 2.0, 1.0]);
        b.trim(1.5, 1.0);
        assert!(b.segments().is_empty());
        assert_eq!(points(&b.doc, b.sketch()).len(), 1, "the origin stays");
    }

    #[test]
    fn a_circle_cut_by_a_line_becomes_an_arc() {
        let mut b = Bench::new();
        b.run(
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("radius", PropertyValue::Float(1.0)),
            ],
        );
        b.line(&[-2.0, 0.0, 2.0, 0.0]);
        // Take the top half off.
        b.trim(0.0, 1.0);
        let id = b.sketch();
        let o = b.doc.object(id).unwrap();
        let arc = o
            .sketch
            .iter()
            .find_map(|c| match c {
                Constraint::Arc { ring, .. } => Some(ring.clone()),
                _ => None,
            })
            .expect("an arc");
        assert!(!o
            .sketch
            .iter()
            .any(|c| matches!(c, Constraint::Circle { .. })));
        let p = points(&b.doc, id);
        assert!(
            arc.iter().all(|v| p[*v as usize].y <= 1e-6),
            "the bottom half"
        );
        // Anticlockwise, round the bottom: from the left to the right.
        assert!(near(p[arc[0] as usize], DVec2::new(-1.0, 0.0)));
        assert!(near(p[*arc.last().unwrap() as usize], DVec2::new(1.0, 0.0)));
    }

    #[test]
    fn a_sketch_is_trimmed_against_another_in_its_plane() {
        let mut b = Bench::new();
        b.line(&[0.0, 0.0, 2.0, 0.0]);
        let first = b.sketch();
        // A second sketch on the same ground, its line across the first's.
        b.run("sketch.new", &[]);
        b.line(&[1.0, -1.0, 1.0, 1.0]);
        let second = b.sketch();
        assert_ne!(first, second);
        // Back to the first, and trim its right half against the second.
        b.doc.selection.clear();
        b.doc.selection.add(first);
        b.doc.selection.active = Some(first);
        b.trim(1.5, 0.0);
        let segs = b.segments();
        assert_eq!(segs.len(), 1);
        let xs = [segs[0][0].x, segs[0][1].x];
        assert!(xs.iter().any(|x| x.abs() < 1e-6) && xs.iter().any(|x| (x - 1.0).abs() < 1e-6));
        assert_eq!(
            b.doc.object(second).unwrap().sketch_lines.len(),
            1,
            "the other sketch is untouched"
        );
    }

    #[test]
    fn the_preview_is_the_piece_that_goes() {
        let mut b = Bench::new();
        b.line(&[-1.0, 0.0, 1.0, 0.0]);
        b.line(&[0.0, -1.0, 0.0, 1.0]);
        let line = b.line_at(DVec2::new(0.5, 0.0));
        let shown = preview(&b.doc, b.sketch(), line as usize, DVec2::new(0.5, 0.0));
        assert_eq!(shown.len(), 1);
        let s = shown[0];
        assert!(
            (near(s[0], DVec2::ZERO) && near(s[1], DVec2::X))
                || (near(s[1], DVec2::ZERO) && near(s[0], DVec2::X)),
            "{s:?}"
        );
    }
}
