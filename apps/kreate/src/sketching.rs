//! The sketch tools' state and their decisions: which tool is in hand, what
//! has been clicked, what is under the pointer, what a set of picks
//! constrains or dimensions, and what a typed number means.
//!
//! Kept apart from the window so it can be tested without one: the
//! application turns the pointer into a point on the sketch's plane and
//! into the sketch's points on screen, and hands both in; what comes back is
//! an operator to run, or nothing yet.

use kreate_geom2d::Curve2D;
use kreate_math::DVec2;
use kreate_scene::drawing::{Drawing, EntityId, EntityKind, EntityRef, Part};
use kreate_scene::sketch::Constraint;

/// The tool in hand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    /// Click a point, a line or a circle to select it, Shift to add; drag
    /// what is selected to move it.
    #[default]
    Select,
    Line,
    Rectangle,
    Circle,
    Arc,
    Point,
    /// A smooth curve through the points clicked: Fusion's fit-point spline.
    Spline,
    /// A regular polygon: its centre, then a corner.
    Polygon,
    /// A slot: one end's centre, the other's, then how wide.
    Slot,
    /// Click a corner to round it.
    Fillet,
    /// What is selected, mirrored across the line clicked.
    Mirror,
    /// The selected lines, copied a distance to the side clicked.
    Offset,
    /// Click a line, circle or arc to take out the piece between the
    /// crossings either side.
    Trim,
    /// A constraint, by its tag in `kreate_scene::sketch::KINDS` — or
    /// `concentric`, which is a coincidence of two centres.
    Constrain(&'static str),
    Dimension,
    Extrude,
    Revolve,
    /// Copies of what is selected, in rows and columns or round a centre.
    Pattern,
    /// Text, put where it is clicked.
    Text,
}

impl Tool {
    /// From the name the menus and keys use.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "select" => Tool::Select,
            "line" => Tool::Line,
            "rectangle" => Tool::Rectangle,
            "circle" => Tool::Circle,
            "arc" => Tool::Arc,
            "point" => Tool::Point,
            "spline" => Tool::Spline,
            "polygon" => Tool::Polygon,
            "slot" => Tool::Slot,
            "fillet" => Tool::Fillet,
            "mirror" => Tool::Mirror,
            "offset" => Tool::Offset,
            "trim" => Tool::Trim,
            "dimension" => Tool::Dimension,
            "extrude" => Tool::Extrude,
            "revolve" => Tool::Revolve,
            "pattern" => Tool::Pattern,
            "text" => Tool::Text,
            "coincident" => Tool::Constrain("coincident"),
            "horizontal" => Tool::Constrain("horizontal"),
            "vertical" => Tool::Constrain("vertical"),
            "parallel" => Tool::Constrain("parallel"),
            "perpendicular" => Tool::Constrain("perpendicular"),
            "equal" => Tool::Constrain("equal"),
            "tangent" => Tool::Constrain("tangent"),
            "concentric" => Tool::Constrain("concentric"),
            "fixed" => Tool::Constrain("fixed"),
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Tool::Select => "select",
            Tool::Line => "line",
            Tool::Rectangle => "rectangle",
            Tool::Circle => "circle",
            Tool::Arc => "arc",
            Tool::Point => "point",
            Tool::Spline => "spline",
            Tool::Polygon => "polygon",
            Tool::Slot => "slot",
            Tool::Fillet => "fillet",
            Tool::Mirror => "mirror",
            Tool::Offset => "offset",
            Tool::Trim => "trim",
            Tool::Dimension => "dimension",
            Tool::Extrude => "extrude",
            Tool::Revolve => "revolve",
            Tool::Pattern => "pattern",
            Tool::Text => "text",
            Tool::Constrain(kind) => kind,
        }
    }

    /// Whether it draws: what the clicks are points of, and what a typed
    /// number measures.
    pub fn draws(self) -> bool {
        matches!(
            self,
            Tool::Line
                | Tool::Rectangle
                | Tool::Circle
                | Tool::Arc
                | Tool::Point
                | Tool::Spline
                | Tool::Polygon
                | Tool::Slot
        )
    }
}

/// A drag of the select tool: the points that move, and the place on the
/// plane that follows the pointer — the point taken, or where a line or a
/// circle was taken.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SketchDrag {
    pub points: Vec<u32>,
    pub anchor: DVec2,
    /// The one point taken, when a single point is dragged: what is joined to
    /// a point it is dropped on.
    pub single: Option<u32>,
    /// In a drawing: the part the one point is, dragged through the solver;
    /// otherwise the entities that move whole.
    pub at: Option<EntityRef>,
    pub entities: Vec<u32>,
}

/// Something of the sketch under the pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Point(u32),
    /// A line, by its place in the sketch's list.
    Line(u32),
    /// A circle or an arc: its centre and its first point.
    Round {
        centre: u32,
        on: u32,
    },
}

/// The sketch as the tools see it: its points in its plane and on screen,
/// its lines, and its circles and arcs.
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub points: Vec<DVec2>,
    /// Each point on screen, in interface points; `None` behind the camera.
    pub screen: Vec<Option<DVec2>>,
    pub lines: Vec<[u32; 2]>,
    /// Centre, ring, and whether it goes all the way round (a circle) or
    /// not (an arc).
    pub rounds: Vec<(u32, Vec<u32>, bool)>,
    /// Points that lie along a spline between the points it passes through:
    /// not picked as points, as a circle's ring is not.
    pub along: Vec<u32>,
    /// For a drawing, what each point, line and round is in it.
    pub map: Option<DrawingMap>,
}

/// What a drawing's points, lines and rounds stand for, when the tools see
/// a drawing as a sketch: each point is the parts of entities at that
/// place, each line a straight piece of one curve, each round a circle or
/// an arc. See [`Geometry::from_drawing`].
#[derive(Clone, Debug, Default)]
pub struct DrawingMap {
    pub refs: Vec<Vec<EntityRef>>,
    pub line_entity: Vec<EntityId>,
    pub round_entity: Vec<EntityId>,
    /// Where drawing snaps that is no entity's point: a line's or an arc's
    /// middle, and where two curves cross — exactly, not where two chords do.
    pub snaps: Vec<DVec2>,
}

impl Geometry {
    /// From a sketch's points, lines and constraints; `project` puts a point
    /// of the plane on screen.
    pub fn new(
        points: Vec<DVec2>,
        lines: Vec<[u32; 2]>,
        constraints: &[Constraint],
        project: impl Fn(DVec2) -> Option<DVec2>,
    ) -> Self {
        let screen = points.iter().map(|p| project(*p)).collect();
        let rounds = constraints
            .iter()
            .filter_map(|c| match c {
                Constraint::Circle { centre, ring } => Some((*centre, ring.clone(), true)),
                Constraint::Arc { centre, ring } => Some((*centre, ring.clone(), false)),
                _ => None,
            })
            .collect();
        let along = constraints
            .iter()
            .filter_map(|c| match c {
                Constraint::Spline { ring, .. } => Some(ring.iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        Self {
            points,
            screen,
            lines,
            rounds,
            along,
            map: None,
        }
    }

    /// A drawing as the tools see a sketch: point 0 its origin, then every
    /// entity's points — one point for parts held coincident — and after
    /// them the points along curves, each curve drawn as straight pieces
    /// within `tol`, each circle and arc a round whose ring is those
    /// pieces' ends. The drawing's points keep their numbers while a drag
    /// moves them, since neither where they are nor how finely a curve is
    /// drawn counts them.
    pub fn from_drawing(d: &Drawing, tol: f64, project: impl Fn(DVec2) -> Option<DVec2>) -> Self {
        use std::collections::HashMap;
        let mut g = Geometry::default();
        let mut map = DrawingMap::default();
        g.points.push(DVec2::ZERO);
        map.refs.push(Vec::new());
        // The parts, one point for each set held together.
        let parts: Vec<(EntityRef, DVec2)> =
            d.entities.iter().flat_map(|e| d.parts(e.id)).collect();
        let slot: HashMap<EntityRef, usize> = parts
            .iter()
            .enumerate()
            .map(|(i, (r, _))| (*r, i))
            .collect();
        let mut parent: Vec<usize> = (0..parts.len()).collect();
        fn root(parent: &mut [usize], i: usize) -> usize {
            let mut r = i;
            while parent[r] != r {
                r = parent[r];
            }
            parent[i] = r;
            r
        }
        for c in &d.constraints {
            if let kreate_scene::drawing::Constraint::Coincident(a, b) = c {
                if let (Some(&i), Some(&j)) = (slot.get(a), slot.get(b)) {
                    let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                    if ri != rj {
                        parent[ri.max(rj)] = ri.min(rj);
                    }
                }
            }
        }
        let mut index: HashMap<EntityRef, u32> = HashMap::new();
        let mut of_root: HashMap<usize, u32> = HashMap::new();
        for (i, (r, p)) in parts.iter().enumerate() {
            let k = root(&mut parent, i);
            let v = *of_root.entry(k).or_insert_with(|| {
                g.points.push(*p);
                map.refs.push(Vec::new());
                (g.points.len() - 1) as u32
            });
            map.refs[v as usize].push(*r);
            index.insert(*r, v);
        }
        let idx = |r: EntityRef| index.get(&r).copied();
        // A curve's pieces from point `from` to point `to`, through points
        // of its own along the way.
        let pieces = |g: &mut Geometry,
                      map: &mut DrawingMap,
                      c: &Curve2D,
                      from: u32,
                      to: u32,
                      id: EntityId,
                      mark: bool|
         -> Vec<u32> {
            let pts = c.tessellate(tol);
            let mut ring = vec![from];
            if pts.len() > 2 {
                for p in &pts[1..pts.len() - 1] {
                    g.points.push(*p);
                    map.refs.push(Vec::new());
                    let v = (g.points.len() - 1) as u32;
                    if mark {
                        g.along.push(v);
                    }
                    ring.push(v);
                }
            }
            ring.push(to);
            for w in ring.windows(2) {
                if w[0] != w[1] {
                    g.lines.push([w[0], w[1]]);
                    map.line_entity.push(id);
                }
            }
            ring
        };
        for e in &d.entities {
            let id = e.id;
            let at = |part| idx(EntityRef::at(id, part));
            let EntityKind::Curve(c) = &e.kind else {
                continue;
            };
            match c {
                Curve2D::Line { .. } => {
                    if let (Some(a), Some(b)) = (at(Part::Start), at(Part::End)) {
                        g.lines.push([a, b]);
                        map.line_entity.push(id);
                    }
                }
                Curve2D::Circle { .. } | Curve2D::Ellipse { .. } if c.is_closed() => {
                    let Some(centre) = at(Part::Center) else {
                        continue;
                    };
                    // The ring begins at a point of its own, not the drawing's.
                    g.points.push(c.start());
                    map.refs.push(Vec::new());
                    let first = (g.points.len() - 1) as u32;
                    let mut ring = pieces(&mut g, &mut map, c, first, first, id, false);
                    ring.pop();
                    g.rounds.push((centre, ring, true));
                    map.round_entity.push(id);
                }
                Curve2D::Circle { .. } => {}
                Curve2D::Arc { .. } | Curve2D::Ellipse { .. } => {
                    let (Some(centre), Some(s), Some(t)) =
                        (at(Part::Center), at(Part::Start), at(Part::End))
                    else {
                        continue;
                    };
                    let ring = pieces(&mut g, &mut map, c, s, t, id, false);
                    g.rounds.push((centre, ring, false));
                    map.round_entity.push(id);
                }
                Curve2D::Polyline { verts, closed } => {
                    let vs: Vec<Option<u32>> = (0..verts.len())
                        .map(|i| at(Part::Vertex(i as u32)))
                        .collect();
                    let n = vs.len();
                    let segments = if *closed { n } else { n.saturating_sub(1) };
                    for k in 0..segments {
                        let (Some(a), Some(b)) = (vs[k], vs[(k + 1) % n]) else {
                            continue;
                        };
                        if verts[k].1 == 0.0 {
                            g.lines.push([a, b]);
                            map.line_entity.push(id);
                        } else {
                            let seg = c.sub(k as f64, (k + 1) as f64);
                            pieces(&mut g, &mut map, &seg, a, b, id, true);
                        }
                    }
                }
                Curve2D::Spline(_) => {
                    let ends: Vec<u32> = d
                        .parts(id)
                        .into_iter()
                        .filter_map(|(r, _)| idx(r))
                        .collect();
                    let (Some(first), Some(last)) = (ends.first(), ends.last()) else {
                        continue;
                    };
                    let last = if c.is_closed() { *first } else { *last };
                    pieces(&mut g, &mut map, c, *first, last, id, true);
                }
            }
        }
        let curves: Vec<&Curve2D> = d
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c) => Some(c),
                _ => None,
            })
            .collect();
        for c in &curves {
            if matches!(c, Curve2D::Line { .. } | Curve2D::Arc { .. }) {
                let (lo, hi) = c.domain();
                map.snaps.push(c.point((lo + hi) / 2.0));
            }
        }
        // Crossings, pair by pair, as a sketch's are: a drawing past a few
        // hundred curves is spared the cost.
        if curves.len() <= 400 {
            for (i, a) in curves.iter().enumerate() {
                for b in &curves[i + 1..] {
                    for hit in kreate_geom2d::intersect(a, b) {
                        if let kreate_geom2d::Intersection::Point { p, .. } = hit {
                            map.snaps.push(p);
                        }
                    }
                }
            }
        }
        g.screen = g.points.iter().map(|p| project(*p)).collect();
        g.map = Some(map);
        g
    }

    /// The drawing's entity a line is a piece of.
    pub fn entity_of_line(&self, l: u32) -> Option<EntityId> {
        self.map.as_ref()?.line_entity.get(l as usize).copied()
    }

    /// The drawing's entity a round is: the circle or arc round `centre`
    /// whose ring holds `on`.
    pub fn entity_of_round(&self, centre: u32, on: u32) -> Option<EntityId> {
        let map = self.map.as_ref()?;
        let k = self
            .rounds
            .iter()
            .position(|(c, ring, _)| *c == centre && ring.contains(&on))?;
        map.round_entity.get(k).copied()
    }

    /// A drawing's point as a part of one of its entities — of a curve
    /// before a lone point, which is what a constraint means by it.
    pub fn point_ref(&self, v: u32) -> Option<EntityRef> {
        let refs = self.map.as_ref()?.refs.get(v as usize)?;
        refs.iter()
            .find(|r| r.part != Part::Whole)
            .or(refs.first())
            .copied()
    }

    /// What a pick is in a drawing: its entity, for a line or a round.
    pub fn entity_of(&self, pick: Pick) -> Option<EntityId> {
        match pick {
            Pick::Line(l) => self.entity_of_line(l),
            Pick::Round { centre, on } => self.entity_of_round(centre, on),
            Pick::Point(_) => None,
        }
    }

    /// The lines a pick selects: a line alone in a sketch, every piece of
    /// its curve in a drawing; a round's ring.
    pub fn lines_of(&self, pick: Pick) -> Vec<u32> {
        match (pick, &self.map) {
            (Pick::Point(_), _) => Vec::new(),
            (Pick::Round { centre, on }, None) => self.lines_of_round(centre, on),
            (Pick::Line(l), None) => vec![l],
            (_, Some(map)) => match self.entity_of(pick) {
                Some(e) => (0..map.line_entity.len() as u32)
                    .filter(|l| map.line_entity[*l as usize] == e)
                    .collect(),
                None => Vec::new(),
            },
        }
    }

    /// A drawing's entities among selected points and lines: each line's
    /// curve, and the lone points.
    pub fn entities_in(&self, points: &[u32], lines: &[u32]) -> Vec<u32> {
        let Some(map) = &self.map else {
            return Vec::new();
        };
        let mut out: Vec<u32> = lines
            .iter()
            .filter_map(|l| map.line_entity.get(*l as usize))
            .map(|e| e.0)
            .collect();
        for v in points {
            for r in map.refs.get(*v as usize).into_iter().flatten() {
                if r.part == Part::Whole {
                    out.push(r.entity.0);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether a point is a ring point of a circle or arc but not an end of
    /// one: what a click on a curve should not pick as a point.
    fn inner_ring_point(&self, v: u32) -> bool {
        if self.along.contains(&v) {
            return true;
        }
        self.rounds.iter().any(|(_, ring, closed)| {
            if *closed {
                ring.contains(&v)
            } else {
                ring.len() > 2 && ring[1..ring.len() - 1].contains(&v)
            }
        })
    }

    /// The nearest point to `at` on screen within `radius`, skipping the
    /// points a circle is made of.
    pub fn point_near(&self, at: DVec2, radius: f64) -> Option<u32> {
        self.screen
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Some((i as u32, (*s)?.distance(at))))
            .filter(|(i, d)| *d <= radius && !self.inner_ring_point(*i))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// What is under the pointer: a point first, then a curve, then a line.
    pub fn pick(&self, at: DVec2) -> Option<Pick> {
        if let Some(p) = self.point_near(at, 8.0) {
            return Some(Pick::Point(p));
        }
        let near_segment = |a: u32, b: u32| -> Option<f64> {
            let (pa, pb) = (
                (*self.screen.get(a as usize)?)?,
                (*self.screen.get(b as usize)?)?,
            );
            let d = pb - pa;
            let t = ((at - pa).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
            Some((pa + d * t).distance(at))
        };
        // A curve is its ring's segments.
        let mut best: Option<(Pick, f64)> = None;
        for (centre, ring, closed) in &self.rounds {
            let n = ring.len();
            let pairs = if *closed { n } else { n.saturating_sub(1) };
            for k in 0..pairs {
                if let Some(d) = near_segment(ring[k], ring[(k + 1) % n]) {
                    if d <= 6.0 && best.is_none_or(|b| d < b.1) {
                        best = Some((
                            Pick::Round {
                                centre: *centre,
                                on: ring[0],
                            },
                            d,
                        ));
                    }
                }
            }
        }
        if best.is_some() {
            return best.map(|b| b.0);
        }
        for (i, l) in self.lines.iter().enumerate() {
            if let Some(d) = near_segment(l[0], l[1]) {
                if d <= 6.0 && best.is_none_or(|b| d < b.1) {
                    best = Some((Pick::Line(i as u32), d));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// The sketch without what is being dragged: what a dragged point can
    /// snap to is everything but itself and the lines that move with it.
    pub fn without(&self, moving: &[u32]) -> Geometry {
        let mut g = self.clone();
        for v in moving {
            if let Some(s) = g.screen.get_mut(*v as usize) {
                *s = None;
            }
        }
        g.lines
            .retain(|l| !moving.contains(&l[0]) && !moving.contains(&l[1]));
        g.rounds
            .retain(|(c, ring, _)| !moving.contains(c) && !ring.iter().any(|v| moving.contains(v)));
        g
    }

    /// The points a pick stands for when it is dragged: a point, a line's
    /// two ends, a circle's centre and ring.
    pub fn points_of(&self, pick: Pick) -> Vec<u32> {
        match pick {
            Pick::Point(v) => vec![v],
            Pick::Line(l) => self
                .lines
                .get(l as usize)
                .map(|l| l.to_vec())
                .unwrap_or_default(),
            Pick::Round { centre, on } => self
                .rounds
                .iter()
                .find(|(c, ring, _)| *c == centre && ring.contains(&on))
                .map(|(c, ring, _)| std::iter::once(*c).chain(ring.iter().copied()).collect())
                .unwrap_or_default(),
        }
    }

    /// The lines of a circle or arc, by their place in the list.
    pub fn lines_of_round(&self, centre: u32, on: u32) -> Vec<u32> {
        let Some((_, ring, _)) = self
            .rounds
            .iter()
            .find(|(c, ring, _)| *c == centre && ring.contains(&on))
        else {
            return Vec::new();
        };
        self.lines
            .iter()
            .enumerate()
            .filter(|(_, l)| ring.contains(&l[0]) && ring.contains(&l[1]))
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Where drawing can snap that is not a point of the sketch: each line's
    /// middle and every place two lines cross, in the plane.
    pub fn snap_points(&self) -> Vec<DVec2> {
        if let Some(map) = &self.map {
            return map.snaps.clone();
        }
        let seg = |l: &[u32; 2]| -> Option<(DVec2, DVec2)> {
            Some((
                *self.points.get(l[0] as usize)?,
                *self.points.get(l[1] as usize)?,
            ))
        };
        let segments: Vec<(DVec2, DVec2)> = self.lines.iter().filter_map(seg).collect();
        let mut out: Vec<DVec2> = segments.iter().map(|(a, b)| (*a + *b) * 0.5).collect();
        // Crossings, pair by pair: a sketch is drawn by hand, a few hundred
        // lines at the most, and a sketch past that is spared the cost.
        if segments.len() <= 600 {
            for (i, (p0, p1)) in segments.iter().enumerate() {
                for (q0, q1) in &segments[i + 1..] {
                    if let Some(x) = crossing(*p0, *p1, *q0, *q1) {
                        out.push(x);
                    }
                }
            }
        }
        out
    }

    /// The point of a line nearest `on_plane`, for a line within `radius`
    /// of `at` on screen.
    pub fn on_line(&self, at: DVec2, on_plane: DVec2, radius: f64) -> Option<DVec2> {
        let line = self.line_near(at, radius)?;
        let [a, b] = *self.lines.get(line as usize)?;
        let (pa, pb) = (*self.points.get(a as usize)?, *self.points.get(b as usize)?);
        let d = pb - pa;
        let t = ((on_plane - pa).dot(d) / d.length_squared().max(1e-24)).clamp(0.0, 1.0);
        Some(pa + d * t)
    }

    /// The line nearest `at` on screen within `radius`, points or no points:
    /// what the trim tool acts on.
    pub fn line_near(&self, at: DVec2, radius: f64) -> Option<u32> {
        self.lines
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                let pa = (*self.screen.get(l[0] as usize)?)?;
                let pb = (*self.screen.get(l[1] as usize)?)?;
                let d = pb - pa;
                let t = ((at - pa).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
                Some((i as u32, (pa + d * t).distance(at)))
            })
            .filter(|(_, d)| *d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn ends(&self, line: u32) -> Option<[u32; 2]> {
        self.lines.get(line as usize).copied()
    }
}

/// Picks that can never make what the tool makes: a point fixed as a line,
/// two circles made perpendicular.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CannotConstrain;

/// What a set of picks makes under a constraint tool: the operator's kind,
/// by its tag, and its points — or `None` while more picks are wanted.
/// `Err` when the picks can never make it.
pub fn constraint_for(
    kind: &'static str,
    picks: &[Pick],
    g: &Geometry,
) -> Result<Option<(&'static str, Vec<u32>)>, CannotConstrain> {
    use Pick::*;
    let line = |l: u32| g.ends(l).ok_or(CannotConstrain);
    Ok(Some(match (kind, picks) {
        ("horizontal" | "vertical", [Line(l)]) => (kind, line(*l)?.to_vec()),
        ("horizontal" | "vertical", [Point(a), Point(b)]) => (kind, vec![*a, *b]),
        ("fixed", [Point(a)]) => ("fixed", vec![*a]),
        ("coincident", [Point(a), Point(b)]) => ("coincident", vec![*a, *b]),
        ("coincident", [Point(p), Line(l)] | [Line(l), Point(p)]) => {
            let [a, b] = line(*l)?;
            ("on_line", vec![*p, a, b])
        }
        ("coincident", [Point(p), Round { centre, .. }] | [Round { centre, .. }, Point(p)]) => {
            ("coincident", vec![*p, *centre])
        }
        ("concentric", [Round { centre: a, .. }, Round { centre: b, .. }]) => {
            ("coincident", vec![*a, *b])
        }
        ("parallel" | "perpendicular" | "equal", [Line(x), Line(y)]) => {
            let ([a, b], [c, d]) = (line(*x)?, line(*y)?);
            (kind, vec![a, b, c, d])
        }
        ("equal", [Round { centre: a, on: b }, Round { centre: c, on: d }]) => {
            ("equal", vec![*a, *b, *c, *d])
        }
        ("tangent", [Line(l), Round { centre, on }] | [Round { centre, on }, Line(l)]) => {
            let [a, b] = line(*l)?;
            ("tangent", vec![a, b, *centre, *on])
        }
        (_, []) => return Ok(None),
        // One pick of a pair: wait for the second, if it could still make it.
        ("coincident" | "concentric" | "parallel" | "perpendicular" | "equal" | "tangent", [_]) => {
            return Ok(None)
        }
        ("horizontal" | "vertical", [Point(_)]) => return Ok(None),
        _ => return Err(CannotConstrain),
    }))
}

/// What the dimension tool makes of its picks: a distance between two
/// points (a line's ends, a circle's centre and rim), or an angle between
/// two lines — or `None` while it waits for a second pick.
pub fn dimension_for(
    picks: &[Pick],
    g: &Geometry,
) -> Result<Option<(&'static str, Vec<u32>)>, CannotConstrain> {
    use Pick::*;
    Ok(Some(match picks {
        [Line(l)] => ("distance", g.ends(*l).ok_or(CannotConstrain)?.to_vec()),
        [Round { centre, on }] => ("distance", vec![*centre, *on]),
        [Point(a), Point(b)] => ("distance", vec![*a, *b]),
        [Line(x), Line(y)] => {
            let ([a, b], [c, d]) = (
                g.ends(*x).ok_or(CannotConstrain)?,
                g.ends(*y).ok_or(CannotConstrain)?,
            );
            ("angle", vec![a, b, c, d])
        }
        [Point(_)] | [] => return Ok(None),
        _ => return Err(CannotConstrain),
    }))
}

/// [`constraint_for`] in a drawing: the `draft.constrain` kind, by its tag,
/// and what it holds. A point on a line or a circle is held on it, as in a
/// CAD program; a point that is no entity's (the origin) holds nothing.
pub fn drawing_constraint_for(
    kind: &'static str,
    picks: &[Pick],
    g: &Geometry,
) -> Result<Option<(&'static str, Vec<EntityRef>)>, CannotConstrain> {
    use Pick::*;
    let point = |v: u32| g.point_ref(v).ok_or(CannotConstrain);
    let whole = |p: Pick| g.entity_of(p).map(EntityRef::whole).ok_or(CannotConstrain);
    let curve = |p: &Pick| matches!(p, Line(_) | Round { .. });
    Ok(Some(match (kind, picks) {
        ("horizontal" | "vertical", [l @ Line(_)]) => (kind, vec![whole(*l)?]),
        ("horizontal" | "vertical", [Point(a), Point(b)]) => (kind, vec![point(*a)?, point(*b)?]),
        ("fixed", [Point(a)]) => ("fixed", vec![point(*a)?]),
        ("coincident", [Point(a), Point(b)]) => ("coincident", vec![point(*a)?, point(*b)?]),
        ("coincident", [Point(p), c] | [c, Point(p)]) if curve(c) => {
            ("point_on", vec![point(*p)?, whole(*c)?])
        }
        ("concentric", [a @ Round { .. }, b @ Round { .. }]) => {
            ("concentric", vec![whole(*a)?, whole(*b)?])
        }
        ("parallel" | "perpendicular", [a @ Line(_), b @ Line(_)]) => {
            (kind, vec![whole(*a)?, whole(*b)?])
        }
        ("equal", [a, b]) if curve(a) && curve(b) => ("equal", vec![whole(*a)?, whole(*b)?]),
        ("tangent", [a, b]) if curve(a) && curve(b) => ("tangent", vec![whole(*a)?, whole(*b)?]),
        (_, []) => return Ok(None),
        ("coincident" | "concentric" | "parallel" | "perpendicular" | "equal" | "tangent", [_]) => {
            return Ok(None)
        }
        ("horizontal" | "vertical", [Point(_)]) => return Ok(None),
        _ => return Err(CannotConstrain),
    }))
}

/// [`dimension_for`] in a drawing: the `draft.dimension` kind, by its tag,
/// and what it measures — a line's length, a circle's diameter, an arc's
/// radius, two points apart, two lines' angle.
pub fn drawing_dimension_for(
    picks: &[Pick],
    g: &Geometry,
    d: &Drawing,
) -> Result<Option<(&'static str, Vec<EntityRef>)>, CannotConstrain> {
    use Pick::*;
    let point = |v: u32| g.point_ref(v).ok_or(CannotConstrain);
    let entity = |p: Pick| g.entity_of(p).ok_or(CannotConstrain);
    let is_line = |p: Pick| {
        g.entity_of(p)
            .is_some_and(|e| matches!(d.curve(e), Some(Curve2D::Line { .. })))
    };
    Ok(Some(match picks {
        [l @ Line(_)] if is_line(*l) => ("aligned", vec![EntityRef::whole(entity(*l)?)]),
        // A piece of a polyline: between its ends.
        [Line(l)] => {
            let [a, b] = g.ends(*l).ok_or(CannotConstrain)?;
            ("aligned", vec![point(a)?, point(b)?])
        }
        [r @ Round { .. }] => {
            let e = entity(*r)?;
            let kind = match d.curve(e) {
                Some(Curve2D::Circle { .. }) => "diameter",
                Some(Curve2D::Arc { .. }) => "radius",
                _ => return Err(CannotConstrain),
            };
            (kind, vec![EntityRef::whole(e)])
        }
        [Point(a), Point(b)] => ("aligned", vec![point(*a)?, point(*b)?]),
        [x @ Line(_), y @ Line(_)] if is_line(*x) && is_line(*y) => (
            "angular",
            vec![EntityRef::whole(entity(*x)?), EntityRef::whole(entity(*y)?)],
        ),
        [Point(_)] | [] => return Ok(None),
        _ => return Err(CannotConstrain),
    }))
}

/// `3.start`: a reference as the drafting operators read it.
pub fn ref_text(r: EntityRef) -> String {
    let part = match r.part {
        Part::Whole => return r.entity.0.to_string(),
        Part::Start => "start".to_string(),
        Part::End => "end".to_string(),
        Part::Center => "center".to_string(),
        Part::Mid => "mid".to_string(),
        Part::Vertex(i) => format!("v{i}"),
    };
    format!("{}.{part}", r.entity.0)
}

/// A typed length in the scene's unit, as metres: "40", "12.5", or "40,20"
/// for two. Commas and spaces separate.
pub fn typed_lengths(text: &str, unit_metres: f64) -> Vec<f64> {
    text.split([',', ' ', ';'])
        .filter(|t| !t.trim().is_empty())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v * unit_metres)
        .collect()
}

/// Whether a key's text belongs in a typed number.
pub fn typing(ch: char) -> bool {
    ch.is_ascii_digit() || matches!(ch, '.' | '-' | ',' | ' ')
}

/// Where the next point of a line goes when a length is typed: from the
/// last point toward the pointer, that far.
pub fn along(from: DVec2, toward: DVec2, length: f64) -> DVec2 {
    let d = (toward - from).normalize_or(DVec2::X);
    from + d * length
}

/// A drawn rectangle's far corner when a width and a height are typed: the
/// pointer says which way each goes.
pub fn corner(from: DVec2, toward: DVec2, w: f64, h: f64) -> DVec2 {
    let sx = if toward.x < from.x { -1.0 } else { 1.0 };
    let sy = if toward.y < from.y { -1.0 } else { 1.0 };
    from + DVec2::new(sx * w.abs(), sy * h.abs())
}

/// The tools' state.
#[derive(Clone, Debug, Default)]
pub struct SketchUi {
    pub tool: Tool,
    /// Points clicked so far by a drawing tool, each with the existing point
    /// it landed on (plus one; zero for none).
    pub clicks: Vec<(DVec2, u32)>,
    /// The pointer on the plane, snapped, and what it snapped to.
    pub hover: Option<(DVec2, u32)>,
    /// What a constraint or dimension tool has picked so far.
    pub picks: Vec<Pick>,
    /// What has been typed since the last click.
    pub typed: String,
    /// The regions picked for an extrude or a revolve: a point in each.
    pub seeds: Vec<DVec2>,
    /// The revolve's axis: a line's two points.
    pub axis: Option<[u32; 2]>,
    /// Selected points and lines.
    pub points: Vec<u32>,
    pub lines: Vec<u32>,
    /// What is being dragged, if anything.
    pub drag: Option<SketchDrag>,
    /// The line the trim tool is over, and where on the plane.
    pub trim: Option<(u32, DVec2)>,
    /// The hover was caught by snapping — a midpoint, a crossing, the grid,
    /// the scene — rather than by a point of the sketch.
    pub snapped: bool,
    /// The line tool's pointer is on the first point of the outline being
    /// drawn: a click there closes it.
    pub closing: bool,
    /// A drag-box: where it began and where the pointer is, on screen.
    pub band: Option<[DVec2; 2]>,
}

impl SketchUi {
    /// A tool taken up: whatever the last one was doing is dropped.
    pub fn take(&mut self, tool: Tool) {
        *self = SketchUi {
            tool,
            hover: self.hover,
            ..Default::default()
        };
    }

    /// Clicks, picks and typing dropped; the tool stays in hand.
    pub fn reset(&mut self) {
        let (tool, hover) = (self.tool, self.hover);
        *self = SketchUi {
            tool,
            hover,
            ..Default::default()
        };
    }
}

// --- the overlay's picture of the sketches, and the feature list ------------

/// How finely a drawing's curves are drawn: a thousandth of its size.
pub fn drawing_tolerance(d: &Drawing) -> f64 {
    let (lo, hi) = d
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
        return 1e-5;
    }
    ((hi - lo).length() * 1e-3).max(1e-7)
}

/// A drawing's curves as straight pieces in its own plane, a thousandth of
/// its size off at the most: what is drawn of it when it is not being drawn
/// in, and what is snapped to and clicked.
pub fn drawing_pieces(d: &Drawing) -> Vec<[DVec2; 2]> {
    let tol = drawing_tolerance(d);
    let curves = d.entities.iter().filter_map(|e| match &e.kind {
        EntityKind::Curve(c) => Some(c),
        _ => None,
    });
    let mut out = Vec::new();
    for c in curves {
        let pts = c.tessellate(tol);
        for w in pts.windows(2) {
            out.push([w[0], w[1]]);
        }
    }
    out
}

/// Every sketch's and drawing's lines in the world, with the object they
/// are of: sketches by their line list, drawings by [`drawing_pieces`].
fn world_lines(
    doc: &kreate_scene::Document,
    id: kreate_scene::document::ObjectId,
    o: &kreate_scene::document::Object,
) -> Vec<[kreate_math::DVec3; 2]> {
    use kreate_math::DVec3;
    let world = doc.world_transform(id);
    let w = |p: DVec2| world.transform_point3(DVec3::new(p.x, p.y, 0.0));
    if let Some(d) = o.drawing.as_deref() {
        return drawing_pieces(d)
            .into_iter()
            .map(|[a, b]| [w(a), w(b)])
            .collect();
    }
    let pts = kreate_ops::ops::sketch_draw::points(doc, id);
    o.sketch_lines
        .iter()
        .filter_map(|[a, b]| Some([w(*pts.get(*a as usize)?), w(*pts.get(*b as usize)?)]))
        .collect()
}

/// Every sketch's lines but `active`'s, in the world: what the overlay draws
/// thin, since a sketch has no faces for the renderer to draw.
pub fn other_lines(
    doc: &kreate_scene::Document,
    active: Option<kreate_scene::document::ObjectId>,
) -> Vec<[kreate_math::DVec3; 2]> {
    let mut others = Vec::new();
    for (id, o) in doc.objects() {
        if (o.sketch_lines.is_empty() && o.drawing.is_none())
            || Some(id) == active
            || !doc.is_shown(id)
        {
            continue;
        }
        others.extend(world_lines(doc, id, o));
    }
    others
}

/// The shown sketch whose line passes nearest `at` on screen, within
/// `reach` points: what a click in object mode selects. A sketch has no
/// surface, so the id buffer never finds it.
pub fn sketch_under(
    doc: &kreate_scene::Document,
    projector: &kreate_ui::object_glyph::Projector,
    at: egui::Pos2,
    reach: f32,
) -> Option<kreate_scene::document::ObjectId> {
    let mut best: Option<(f32, kreate_scene::document::ObjectId)> = None;
    for (id, o) in doc.objects() {
        if (o.sketch_lines.is_empty() && o.drawing.is_none()) || !doc.is_shown(id) {
            continue;
        }
        for [p, q] in world_lines(doc, id, o) {
            let (Some(p), Some(q)) = (projector.project(p), projector.project(q)) else {
                continue;
            };
            let d = segment_distance(at, p, q);
            if d <= reach && best.is_none_or(|(b, _)| d < b) {
                best = Some((d, id));
            }
        }
    }
    best.map(|(_, id)| id)
}

/// How far `at` is from the segment `p`–`q`, on screen.
fn segment_distance(at: egui::Pos2, p: egui::Pos2, q: egui::Pos2) -> f32 {
    let (d, w) = (q - p, at - p);
    let t = if d.length_sq() > 1e-12 {
        (w.dot(d) / d.length_sq()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p + d * t).distance(at)
}

/// [`other_lines`], each with how it is selected: 0 not, 1 its sketch is
/// selected, 2 its sketch is the active object — for the overlay to draw a
/// selected sketch as a selected object is drawn.
pub fn other_lines_marked(
    doc: &kreate_scene::Document,
    active: Option<kreate_scene::document::ObjectId>,
) -> Vec<([kreate_math::DVec3; 2], u8)> {
    let mut out = Vec::new();
    for (id, o) in doc.objects() {
        if (o.sketch_lines.is_empty() && o.drawing.is_none())
            || Some(id) == active
            || !doc.is_shown(id)
        {
            continue;
        }
        let mark = if doc.selection.active == Some(id) && doc.selection.contains(id) {
            2
        } else if doc.selection.contains(id) {
            1
        } else {
            0
        };
        out.extend(world_lines(doc, id, o).into_iter().map(|l| (l, mark)));
    }
    out
}

/// The other sketches' points a drawing can start from, in the world: their
/// ends and corners, lone points and the centres of their circles and arcs
/// — not the points round a circle, which are only how it is drawn, nor any
/// sketch's origin.
pub fn other_points(
    doc: &kreate_scene::Document,
    active: Option<kreate_scene::document::ObjectId>,
) -> Vec<kreate_math::DVec3> {
    use kreate_math::DVec3;
    let mut out = Vec::new();
    for (id, o) in doc.objects() {
        if (o.sketch_lines.is_empty() && o.sketch.is_empty() && o.drawing.is_none())
            || Some(id) == active
            || !doc.is_shown(id)
        {
            continue;
        }
        let world = doc.world_transform(id);
        if let Some(d) = o.drawing.as_deref() {
            for e in &d.entities {
                for (_, p) in d.parts(e.id) {
                    out.push(world.transform_point3(DVec3::new(p.x, p.y, 0.0)));
                }
            }
            continue;
        }
        let pts = kreate_ops::ops::sketch_draw::points(doc, id);
        let mut along = vec![false; pts.len()];
        for c in &o.sketch {
            match c {
                Constraint::Circle { ring, .. } => {
                    for v in ring {
                        if let Some(a) = along.get_mut(*v as usize) {
                            *a = true;
                        }
                    }
                }
                Constraint::Arc { ring, .. } if ring.len() > 2 => {
                    for v in &ring[1..ring.len() - 1] {
                        if let Some(a) = along.get_mut(*v as usize) {
                            *a = true;
                        }
                    }
                }
                Constraint::Spline { ring, .. } => {
                    for v in ring {
                        if let Some(a) = along.get_mut(*v as usize) {
                            *a = true;
                        }
                    }
                }
                _ => {}
            }
        }
        for (i, p) in pts.iter().enumerate().skip(1) {
            if !along[i] {
                out.push(world.transform_point3(DVec3::new(p.x, p.y, 0.0)));
            }
        }
    }
    out
}

/// The feature list: every feature, and each sketch just before the first
/// feature made from it, in the order they were made.
pub fn feature_rows(
    doc: &kreate_scene::Document,
    catalog: &kreate_i18n::Catalog,
) -> Vec<crate::shell::FeatureRow> {
    let mut rows = Vec::new();
    for (_, o) in doc.objects() {
        for (i, f) in o.features.iter().enumerate() {
            let (revolve, amount, symmetric) = match f.shape {
                kreate_scene::feature::Shape::Extrude {
                    distance,
                    symmetric,
                } => (false, distance, symmetric),
                kreate_scene::feature::Shape::Revolve { angle, .. } => (true, angle, false),
                kreate_scene::feature::Shape::Round { radius, .. } => (false, radius, false),
                kreate_scene::feature::Shape::Shell { thickness, .. } => (false, thickness, false),
                kreate_scene::feature::Shape::Pattern { count, .. }
                | kreate_scene::feature::Shape::Circular { count, .. } => {
                    (false, f64::from(count), false)
                }
                kreate_scene::feature::Shape::Mirror { .. } => (false, 0.0, false),
            };
            let kind = catalog.get(match f.kind_tag() {
                "revolve" => "sketch-tool-revolve",
                "fillet" => "feature-fillet",
                "chamfer" => "feature-chamfer",
                "shell" => "feature-shell",
                "pattern" => "feature-pattern",
                "circular" => "feature-circular",
                "mirror" => "feature-mirror",
                _ => "sketch-tool-extrude",
            });
            rows.push(crate::shell::FeatureRow {
                order: f.order * 2 + 1,
                // i18n-exempt: a translated name and its place on the timeline
                label: format!("{kind} {}", f.order),
                body: o.name.clone(),
                index: i as u32,
                is_sketch: false,
                kind: f.kind_tag().to_string(),
                revolve,
                amount,
                symmetric,
                cut: f.operation == kreate_scene::feature::Operation::Cut,
                expression: doc
                    .design
                    .binding(&kreate_scene::design::Bound::FeatureAmount {
                        body: o.uid,
                        order: f.order,
                    })
                    .map(|b| b.expression.clone()),
            });
        }
    }
    // Each sketch just before the first feature made from it.
    for (_, o) in doc.objects() {
        if o.sketch_lines.is_empty() && o.sketch.is_empty() && o.drawing.is_none() {
            continue;
        }
        let first = doc
            .objects()
            .flat_map(|(_, b)| b.features.iter())
            .filter(|f| Some(f.sketch) == Some(o.uid))
            .map(|f| f.order * 2)
            .min()
            .unwrap_or(u64::MAX);
        rows.push(crate::shell::FeatureRow {
            order: first,
            label: o.name.clone(),
            body: String::new(),
            index: 0,
            is_sketch: true,
            kind: "sketch".to_string(),
            revolve: false,
            amount: 0.0,
            symmetric: false,
            cut: false,
            expression: None,
        });
    }
    rows.sort_by_key(|r| r.order);
    rows
}

/// The sketch being drawn as the overlay draws it: its lines and points by
/// how pinned they are, its marks and dimensions, and what the tool in hand
/// would draw. `pinned` is `kreate_scene::sketch::pinned` of its points.
/// Which points the constraints hold, for colouring: as
/// [`kreate_scene::sketch::pinned`], but a circle's points count as held once
/// its centre and size are. Turning a circle round its centre changes
/// nothing anyone drew, so the freedom is real to the solver and invisible to
/// the user, and a circle that showed blue however fully it was dimensioned
/// would make the colour mean nothing.
pub fn pinned(points: &[DVec2], constraints: &[Constraint]) -> Vec<bool> {
    let mut held = constraints.to_vec();
    for c in constraints {
        if let Constraint::Circle { centre, ring } = c {
            if let Some(first) = ring.first() {
                held.push(Constraint::Horizontal {
                    a: *centre,
                    b: *first,
                });
            }
        }
    }
    kreate_scene::sketch::pinned(points, &held)
}

pub fn view(
    doc: &kreate_scene::Document,
    id: kreate_scene::document::ObjectId,
    ui: &SketchUi,
    catalog: &kreate_i18n::Catalog,
    pinned: &[bool],
) -> crate::shell::SketchView {
    use crate::shell::{DimKind, SketchDim, SketchView};
    use kreate_math::DVec3;
    let world = doc.world_transform(id);
    let Some(object) = doc.object(id) else {
        return SketchView::default();
    };
    let points = kreate_ops::ops::sketch_draw::points(doc, id);
    let constraints = object.sketch.clone();
    let lines = object.sketch_lines.clone();
    let to_world = |p: DVec2| world.transform_point3(DVec3::new(p.x, p.y, 0.0));
    let pin = |v: u32| pinned.get(v as usize).copied().unwrap_or(false);
    // A dimension held to the parameters shows what holds it.
    let bound = |index: usize| {
        doc.design
            .binding(&kreate_scene::design::Bound::SketchDimension {
                sketch: object.uid,
                index: index as u32,
            })
            .map(|b| b.expression.clone())
    };
    let ring_point = |v: u32| {
        constraints.iter().any(|c| match c {
            Constraint::Circle { ring, .. } => ring.contains(&v),
            Constraint::Arc { ring, .. } => ring.len() > 2 && ring[1..ring.len() - 1].contains(&v),
            Constraint::Spline { ring, .. } => ring.contains(&v),
            _ => false,
        })
    };
    let mut view = SketchView {
        unit: (
            doc.design.length_unit.metres(),
            doc.design.length_unit.symbol(),
        ),
        tool: ui.tool.name(),
        typed: ui.typed.clone(),
        ..Default::default()
    };
    for (i, [a, b]) in lines.iter().enumerate() {
        let (Some(p), Some(q)) = (points.get(*a as usize), points.get(*b as usize)) else {
            continue;
        };
        view.lines.push((
            [to_world(*p), to_world(*q)],
            pin(*a) && pin(*b),
            ui.lines.contains(&(i as u32)),
            Constraint::is_construction(&constraints, [*a, *b]),
        ));
    }
    for (i, p) in points.iter().enumerate() {
        let v = i as u32;
        if !ring_point(v) {
            view.points
                .push((to_world(*p), pin(v), ui.points.contains(&v)));
        }
    }
    // The marks and the dimensions.
    let at = |v: u32| points.get(v as usize).copied().unwrap_or_default();
    let mid = |a: u32, b: u32| (at(a) + at(b)) * 0.5;
    let (m, sym) = view.unit;
    // The drawing's middle and size: a length's label goes on the side of
    // its line away from the middle, outside the shape, the way a drawing is
    // dimensioned, and stands off by a share of the whole.
    let centroid = if points.is_empty() {
        DVec2::ZERO
    } else {
        points.iter().copied().sum::<DVec2>() / points.len() as f64
    };
    let extent = points
        .iter()
        .map(|p| (*p - centroid).length())
        .fold(0.0f64, f64::max);
    for (index, c) in constraints.iter().enumerate() {
        match c {
            Constraint::Horizontal { a, b } => {
                view.glyphs.push((to_world(mid(*a, *b)), "H".into()))
            }
            Constraint::Vertical { a, b } => view.glyphs.push((to_world(mid(*a, *b)), "V".into())),
            Constraint::Parallel { a, b, c, d } => {
                view.glyphs.push((to_world(mid(*a, *b)), "//".into()));
                view.glyphs.push((to_world(mid(*c, *d)), "//".into()));
            }
            Constraint::Perpendicular { a, b, c, d } => {
                view.glyphs.push((to_world(mid(*a, *b)), "⊥".into()));
                view.glyphs.push((to_world(mid(*c, *d)), "⊥".into()));
            }
            Constraint::Equal { a, b, c, d } => {
                view.glyphs.push((to_world(mid(*a, *b)), "=".into()));
                view.glyphs.push((to_world(mid(*c, *d)), "=".into()));
            }
            Constraint::Tangent { a, b, .. } => {
                view.glyphs.push((to_world(mid(*a, *b)), "T".into()))
            }
            Constraint::OnLine { p, .. } => view.glyphs.push((to_world(at(*p)), "∘".into())),
            Constraint::Coincident { a, .. } => view.glyphs.push((to_world(at(*a)), "∘".into())),
            Constraint::Fixed { a, .. } if *a != 0 => {
                view.glyphs.push((to_world(at(*a)), "F".into()))
            }
            Constraint::Distance { a, b, value } => {
                let diameter = constraints.iter().any(|k| {
                        matches!(k, Constraint::Circle { centre, ring } if centre == a && ring.contains(b))
                    });
                let (shown, text, kind) = if diameter {
                    let d = value * 2.0 / m;
                    (d, format!("Ø{d:.2}"), DimKind::Diameter)
                } else {
                    let d = value / m;
                    // i18n-exempt: a number and its unit
                    (d, format!("{d:.2} {sym}"), DimKind::Length)
                };
                let (pa, pb) = (at(*a), at(*b));
                let mut side = (pb - pa).perp().normalize_or_zero();
                let middle = (pa + pb) * 0.5;
                if side.dot(middle - centroid) < 0.0 {
                    side = -side;
                }
                let span = (pb - pa).length();
                let label = if diameter {
                    pb + (pb - pa).normalize_or_zero() * span * 0.4
                } else {
                    middle + side * (span * 0.12).max(extent * 0.3).max(1e-6)
                };
                let expression = bound(index);
                let text = match &expression {
                    // i18n-exempt: an expression and its value
                    Some(e) => format!("{e} = {text}"),
                    None => text,
                };
                view.dims.push(SketchDim {
                    index,
                    at: to_world(label),
                    ends: [to_world(pa), to_world(pb)],
                    text,
                    shown,
                    kind,
                    expression,
                });
            }
            Constraint::Angle { a, b, c, d, value } => {
                let deg = value.to_degrees();
                let expression = bound(index);
                let text = match &expression {
                    // i18n-exempt: an expression and its value
                    Some(e) => format!("{e} = {deg:.1}°"),
                    None => format!("{deg:.1}°"),
                };
                view.dims.push(SketchDim {
                    index,
                    at: to_world((mid(*a, *b) + mid(*c, *d)) * 0.5),
                    ends: [to_world(at(*b)), to_world(at(*c))],
                    text,
                    shown: deg,
                    kind: DimKind::Angle,
                    expression,
                });
            }
            _ => {}
        }
    }
    overlay(&mut view, doc, id, ui, catalog, &points, &to_world, None);
    view
}

/// What every sketch and drawing shows over its geometry: what the tool in
/// hand would draw, the trim's piece, the regions picked, the prompt.
/// `points` are the plane's points, for the size of things; `drawing` is
/// the geometry when the sketch is a drawing.
#[allow(clippy::too_many_arguments)]
fn overlay(
    view: &mut crate::shell::SketchView,
    doc: &kreate_scene::Document,
    id: kreate_scene::document::ObjectId,
    ui: &SketchUi,
    catalog: &kreate_i18n::Catalog,
    points: &[DVec2],
    to_world: &dyn Fn(DVec2) -> kreate_math::DVec3,
    drawing: Option<&Geometry>,
) {
    // What the tool would draw.
    let hover = ui.hover;
    if let Some((h, join)) = hover {
        if join > 0 || ui.snapped {
            view.snap = Some(to_world(h));
        }
        let clicks: Vec<DVec2> = ui.clicks.iter().map(|c| c.0).collect();
        let unit = view.unit.0;
        let typed = typed_lengths(&ui.typed, unit);
        let polyline = |pts: Vec<DVec2>| pts.into_iter().map(to_world).collect::<Vec<_>>();
        let circle = |c: DVec2, r: f64| {
            (0..=48)
                .map(|i| {
                    let a = i as f64 * std::f64::consts::TAU / 48.0;
                    to_world(c + DVec2::new(a.cos(), a.sin()) * r)
                })
                .collect::<Vec<_>>()
        };
        match (ui.tool, clicks.as_slice()) {
            (Tool::Line, [.., last]) => {
                let next = typed.first().map(|l| along(*last, h, *l)).unwrap_or(h);
                let mut pts = clicks.clone();
                pts.push(next);
                view.preview.push(polyline(pts));
            }
            // The curve so far and on to the pointer, smooth through them.
            (Tool::Spline, [_, ..]) => {
                let mut pts = clicks.clone();
                pts.push(h);
                let ring = kreate_scene::sketch::spline_ring(&pts, (pts.len() - 1) * 8, false);
                let mut line = Vec::with_capacity(pts.len() + ring.len());
                for (i, p) in pts.iter().enumerate() {
                    line.push(*p);
                    if i + 1 < pts.len() {
                        line.extend_from_slice(&ring[i * 8..(i + 1) * 8]);
                    }
                }
                view.preview.push(polyline(line));
            }
            (Tool::Rectangle, [a]) => {
                let b = match typed.as_slice() {
                    [w, hh, ..] => corner(*a, h, *w, *hh),
                    _ => h,
                };
                view.preview.push(polyline(vec![
                    *a,
                    DVec2::new(b.x, a.y),
                    b,
                    DVec2::new(a.x, b.y),
                    *a,
                ]));
            }
            (Tool::Circle, [c]) => {
                let r = typed
                    .first()
                    .map(|d| d.abs() * 0.5)
                    .unwrap_or(c.distance(h));
                view.preview.push(circle(*c, r));
            }
            (Tool::Polygon, [c]) => {
                let r = typed.first().map(|d| d.abs()).unwrap_or(c.distance(h));
                let a0 = (h - *c).y.atan2((h - *c).x);
                let pts: Vec<DVec2> = (0..=6)
                    .map(|i| {
                        let a = a0 + i as f64 * std::f64::consts::TAU / 6.0;
                        *c + DVec2::new(a.cos(), a.sin()) * r
                    })
                    .collect();
                view.preview.push(polyline(pts));
            }
            (Tool::Slot, [a]) => view.preview.push(polyline(vec![*a, h])),
            (Tool::Slot, [a, b]) => {
                let d = *b - *a;
                let dir = d.normalize_or_zero();
                let side = DVec2::new(-dir.y, dir.x);
                let r = typed
                    .first()
                    .map(|w| w.abs() * 0.5)
                    .unwrap_or_else(|| (h - *a).dot(side).abs().max(1e-6));
                let mut pts = Vec::new();
                for k in 0..=12 {
                    let t = std::f64::consts::FRAC_PI_2 - std::f64::consts::PI * k as f64 / 12.0;
                    let (s, c) = t.sin_cos();
                    pts.push(*b + dir * (c * r) + side * (s * r));
                }
                for k in 0..=12 {
                    let t = -std::f64::consts::FRAC_PI_2 - std::f64::consts::PI * k as f64 / 12.0;
                    let (s, c) = t.sin_cos();
                    pts.push(*a + dir * (c * r) + side * (s * r));
                }
                pts.push(pts[0]);
                view.preview.push(polyline(pts));
            }
            (Tool::Arc, [s]) => view.preview.push(polyline(vec![*s, h])),
            (Tool::Arc, [s, e]) => {
                if let Some(c) = kreate_ops::ops::sketch_draw::circumcentre(*s, *e, h) {
                    view.preview.push(circle(c, c.distance(*s)));
                }
            }
            _ => {}
        }
        if let (Tool::Trim, Some((line, on))) = (ui.tool, ui.trim) {
            match drawing {
                Some(g) => {
                    let gone = g.entity_of_line(line).and_then(|e| {
                        let d = doc.object(id)?.drawing.as_deref()?;
                        kreate_ops::ops::draft::trim_plan(d, e, on).ok()
                    });
                    if let Some((_, gone)) = gone {
                        let pts = gone.tessellate(trim_tolerance(points));
                        for w in pts.windows(2) {
                            view.cut.push([to_world(w[0]), to_world(w[1])]);
                        }
                    }
                }
                None => {
                    for [a, b] in kreate_ops::ops::sketch_trim::preview(doc, id, line as usize, on)
                    {
                        view.cut.push([to_world(a), to_world(b)]);
                    }
                }
            }
        }
    }
    // The regions picked, and the one under the pointer.
    if matches!(ui.tool, Tool::Extrude | Tool::Revolve) {
        let regions = match (ui.tool, ui.axis) {
            (Tool::Revolve, Some(axis)) => {
                // The axis, drawn across the sketch and past it.
                if let Some((origin, dir)) = kreate_scene::feature::revolve_axis(doc, id, axis) {
                    let (origin, dir) = (origin.as_dvec2(), dir.normalize_or_zero().as_dvec2());
                    let reach = points
                        .iter()
                        .fold(0.01f64, |m, p| m.max((*p - origin).length()))
                        * 1.5;
                    view.axis_line = Some([
                        to_world(origin - dir * reach),
                        to_world(origin + dir * reach),
                    ]);
                }
                kreate_scene::feature::shape_regions(doc, id, &revolve_shape(axis))
            }
            _ => kreate_scene::feature::sketch_regions(doc, id),
        };
        // Each region as its rings and its triangles, holes and all: a fan
        // from one corner filled outside a concave outline and over a hole.
        let outline = |r: &kreate_mesh::regions::Region| {
            let rings: Vec<Vec<DVec2>> = std::iter::once(&r.outline)
                .chain(r.holes.iter())
                .map(|ring| ring.iter().map(|p| p.as_dvec2()).collect())
                .collect();
            let flat: Vec<DVec2> = rings.iter().flatten().copied().collect();
            crate::shell::RegionShape {
                rings: rings
                    .iter()
                    .map(|ring| ring.iter().map(|p| to_world(*p)).collect())
                    .collect(),
                tris: kreate_mesh::earcut::triangulate(&rings)
                    .into_iter()
                    .map(|t| t.map(|i| to_world(flat[i])))
                    .collect(),
            }
        };
        for s in &ui.seeds {
            if let Some(i) = kreate_mesh::regions::region_at(&regions, s.as_vec2()) {
                view.regions.push(outline(&regions[i]));
            }
        }
        if let Some(i) =
            hover.and_then(|(h, _)| kreate_mesh::regions::region_at(&regions, h.as_vec2()))
        {
            view.hover_region = Some(outline(&regions[i]));
        }
        view.seeds = ui.seeds.iter().map(|s| [s.x, s.y]).collect();
        view.axis = ui.axis;
    }
    let prompt_key = match (ui.tool, ui.clicks.len()) {
        (Tool::Spline, 0) => "sketch-prompt-spline-start",
        (Tool::Spline, _) => "sketch-prompt-spline-next",
        (Tool::Line, 0) => "sketch-prompt-line-start",
        (Tool::Line, _) => "sketch-prompt-line-next",
        (Tool::Rectangle, 0) => "sketch-prompt-corner",
        (Tool::Rectangle, _) => "sketch-prompt-rectangle-far",
        (Tool::Circle, 0) => "sketch-prompt-centre",
        (Tool::Circle, _) => "sketch-prompt-diameter",
        (Tool::Arc, 0) => "sketch-prompt-arc-start",
        (Tool::Arc, 1) => "sketch-prompt-arc-end",
        (Tool::Arc, _) => "sketch-prompt-arc-through",
        (Tool::Point, _) => "sketch-prompt-point",
        (Tool::Text, _) => "sketch-prompt-text",
        (Tool::Polygon, 0) => "sketch-prompt-centre",
        (Tool::Polygon, _) => "sketch-prompt-polygon-corner",
        (Tool::Slot, 0) => "sketch-prompt-slot-start",
        (Tool::Slot, 1) => "sketch-prompt-slot-end",
        (Tool::Slot, _) => "sketch-prompt-slot-width",
        (Tool::Fillet, _) => "sketch-prompt-fillet",
        (Tool::Mirror, _) => "sketch-prompt-mirror",
        (Tool::Offset, _) => "sketch-prompt-offset",
        (Tool::Trim, _) => "sketch-prompt-trim",
        (Tool::Dimension, _) => "sketch-prompt-dimension",
        (Tool::Constrain(_), _) => "sketch-prompt-constrain",
        (Tool::Extrude, _) => "sketch-prompt-region",
        (Tool::Revolve, _) => {
            if ui.axis.is_none() {
                "sketch-prompt-axis"
            } else {
                "sketch-prompt-region"
            }
        }
        (Tool::Select, _) => "sketch-prompt-select",
        (Tool::Pattern, _) => "sketch-prompt-pattern",
    };
    view.prompt = format!(
        "{} — {}",
        catalog.get(&format!("sketch-tool-{}", ui.tool.name())),
        catalog.get(prompt_key)
    );
    view.picked_points = ui.points.clone();
    view.picked_lines = ui.lines.clone();
    view.band = ui.band;
    view.drawing = ui.tool.draws();
}

/// How finely a preview draws a curve: a thousandth of the drawing's size.
fn trim_tolerance(points: &[DVec2]) -> f64 {
    let (lo, hi) = points.iter().fold(
        (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    ((hi - lo).length() * 1e-3).max(1e-7)
}

/// The entities of a drawing nothing can move: those of every piece the
/// solver finds held all round. What draws in the text colour, the rest
/// blue, as Fusion colours a sketch.
pub fn drawing_pinned(d: &Drawing) -> Vec<EntityId> {
    let mut d = d.clone();
    d.solve()
        .into_iter()
        .filter(|p| p.converged && p.freedom == 0)
        .flat_map(|p| p.entities)
        .collect()
}

/// [`view`] for a drawing, seen through `g` ([`Geometry::from_drawing`]).
pub fn drawing_view(
    doc: &kreate_scene::Document,
    id: kreate_scene::document::ObjectId,
    ui: &SketchUi,
    catalog: &kreate_i18n::Catalog,
    g: &Geometry,
    pinned: &[EntityId],
) -> crate::shell::SketchView {
    use crate::shell::{DimKind, SketchDim, SketchView};
    use kreate_math::DVec3;
    use kreate_scene::drawing::{Constraint as C, DimKind as K};
    let world = doc.world_transform(id);
    let (Some(object), Some(map)) = (doc.object(id), g.map.as_ref()) else {
        return SketchView::default();
    };
    let Some(d) = object.drawing.as_deref() else {
        return SketchView::default();
    };
    let to_world = |p: DVec2| world.transform_point3(DVec3::new(p.x, p.y, 0.0));
    let held = |e: EntityId| pinned.contains(&e);
    let construction = |e: EntityId| d.entity(e).is_some_and(|x| x.construction);
    let mut view = SketchView {
        unit: (
            doc.design.length_unit.metres(),
            doc.design.length_unit.symbol(),
        ),
        tool: ui.tool.name(),
        typed: ui.typed.clone(),
        ..Default::default()
    };
    for (i, [a, b]) in g.lines.iter().enumerate() {
        let e = map.line_entity[i];
        view.lines.push((
            [
                to_world(g.points[*a as usize]),
                to_world(g.points[*b as usize]),
            ],
            held(e),
            ui.lines.contains(&(i as u32)),
            construction(e),
        ));
    }
    for (v, p) in g.points.iter().enumerate().skip(1) {
        let refs = &map.refs[v];
        if refs.is_empty() {
            continue;
        }
        view.points.push((
            to_world(*p),
            refs.iter().all(|r| held(r.entity)),
            ui.points.contains(&(v as u32)),
        ));
    }
    for e in &d.entities {
        let kreate_scene::drawing::EntityKind::Text(t) = &e.kind else {
            continue;
        };
        let along = DVec2::new(t.rotation.cos(), t.rotation.sin());
        let up = DVec2::new(-along.y, along.x);
        view.texts.push(crate::shell::SketchText {
            at: to_world(t.at),
            up: to_world(t.at + up * t.height),
            along: to_world(t.at + along * t.height),
            content: t.content.clone(),
            anchor: match t.align {
                kreate_scene::drawing::TextAlign::Left => 0.0,
                kreate_scene::drawing::TextAlign::Center => 0.5,
                kreate_scene::drawing::TextAlign::Right => 1.0,
            },
        });
    }
    // The marks: at a point for what holds a point, at a curve's middle for
    // what holds a curve.
    let at = |r: EntityRef| d.point_of(r);
    let mid = |e: EntityId| {
        d.point_of(EntityRef::at(e, Part::Mid))
            .or_else(|| d.point_of(EntityRef::at(e, Part::Center)))
    };
    let mut mark = |p: Option<DVec2>, text: &str| {
        if let Some(p) = p {
            view.glyphs.push((to_world(p), text.to_string()));
        }
    };
    for c in &d.constraints {
        match c {
            C::Coincident(a, _) => mark(at(*a), "∘"),
            C::Horizontal(a, None) => mark(mid(a.entity), "H"),
            C::Vertical(a, None) => mark(mid(a.entity), "V"),
            C::Horizontal(a, Some(_)) => mark(at(*a), "H"),
            C::Vertical(a, Some(_)) => mark(at(*a), "V"),
            C::Parallel(a, b) => {
                mark(mid(*a), "//");
                mark(mid(*b), "//");
            }
            C::Perpendicular(a, b) => {
                mark(mid(*a), "⊥");
                mark(mid(*b), "⊥");
            }
            C::Equal(a, b) => {
                mark(mid(*a), "=");
                mark(mid(*b), "=");
            }
            C::Tangent(a, _) => mark(mid(*a), "T"),
            C::Smooth(a, _) => mark(mid(*a), "G2"),
            C::Concentric(a, _) => mark(mid(*a), "C"),
            C::PointOn(p, _) => mark(at(*p), "∘"),
            C::Midpoint(p, _) => mark(at(*p), "M"),
            C::Symmetric(a, b, _) => {
                mark(at(*a), "S");
                mark(at(*b), "S");
            }
            C::Fixed(p, _) => mark(at(*p), "F"),
        }
    }
    let (m, sym) = view.unit;
    for e in &d.entities {
        let EntityKind::Dimension(dim) = &e.kind else {
            continue;
        };
        // What it measures, end to end, as the overlay draws its line.
        let ends: Option<[DVec2; 2]> = match (dim.kind, dim.refs.as_slice()) {
            (K::Radius | K::Diameter | K::ArcLength, [r]) => match d.curve(r.entity) {
                Some(Curve2D::Circle { c, r: radius })
                | Some(Curve2D::Arc { c, r: radius, .. }) => {
                    let toward = (dim.text_at - *c).normalize_or(DVec2::X);
                    let far = *c + toward * *radius;
                    Some(if dim.kind == K::Diameter {
                        [*c - toward * *radius, far]
                    } else {
                        [*c, far]
                    })
                }
                _ => None,
            },
            (K::Angular, [a, b]) => mid(a.entity).zip(mid(b.entity)).map(|(p, q)| [p, q]),
            (_, [a, b]) => at(*a).zip(at(*b)).map(|(p, q)| [p, q]),
            (_, [r]) => d.curve(r.entity).map(|c| [c.start(), c.end()]),
            _ => None,
        };
        let Some([pa, pb]) = ends else {
            continue;
        };
        let expression = doc
            .design
            .binding(&kreate_scene::design::Bound::DrawingDimension {
                drawing: object.uid,
                dim: e.id.0,
            })
            .map(|b| b.expression.clone());
        let (shown, number, kind) = match dim.kind {
            K::Angular => {
                let deg = dim.value.to_degrees();
                (deg, format!("{deg:.1}°"), DimKind::Angle)
            }
            K::Diameter => {
                let v = dim.value / m;
                (v, format!("Ø{v:.2}"), DimKind::Length)
            }
            K::Radius => {
                let v = dim.value / m;
                (v, format!("R{v:.2}"), DimKind::Length)
            }
            _ => {
                let v = dim.value / m;
                // i18n-exempt: a number and its unit
                (v, format!("{v:.2} {sym}"), DimKind::Length)
            }
        };
        // A driven dimension only measures: shown in brackets, as a
        // reference dimension is on a drawing.
        let number = if dim.driving {
            number
        } else {
            format!("({number})")
        };
        let text = match &expression {
            // i18n-exempt: an expression and its value
            Some(x) => format!("{x} = {number}"),
            None => number,
        };
        let label = if dim.text_at == DVec2::ZERO {
            (pa + pb) * 0.5
        } else {
            dim.text_at
        };
        view.dims.push(SketchDim {
            index: e.id.0 as usize,
            at: to_world(label),
            ends: [to_world(pa), to_world(pb)],
            text,
            shown,
            kind,
            expression,
        });
    }
    view.drafting = true;
    view.picked_entities = g.entities_in(&ui.points, &ui.lines);
    overlay(
        &mut view,
        doc,
        id,
        ui,
        catalog,
        &g.points,
        &to_world,
        Some(g),
    );
    view
}

/// A whole turn about `axis`: the shape a revolve's regions are found for,
/// which does not depend on the angle.
pub fn revolve_shape(axis: [u32; 2]) -> kreate_scene::feature::Shape {
    kreate_scene::feature::Shape::Revolve {
        axis,
        angle: std::f64::consts::TAU,
    }
}

/// What a drag-box from `a` to `b` on screen takes: the points inside it,
/// and every line with both ends inside — a circle or an arc whole.
pub fn in_band(g: &Geometry, a: DVec2, b: DVec2) -> (Vec<u32>, Vec<u32>) {
    let (lo, hi) = (a.min(b), a.max(b));
    let inside = |v: u32| {
        g.screen
            .get(v as usize)
            .copied()
            .flatten()
            .is_some_and(|q| q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y)
    };
    let ring: Vec<u32> = g
        .rounds
        .iter()
        .flat_map(|(_, ring, _)| ring.iter().copied())
        .collect();
    // The origin is not something to select; a circle's ring points are
    // its, not points of their own.
    let points: Vec<u32> = (1..g.points.len() as u32)
        .filter(|v| inside(*v) && !ring.contains(v))
        .filter(|v| {
            g.map
                .as_ref()
                .is_none_or(|m| !m.refs[*v as usize].is_empty())
        })
        .collect();
    let mut lines: Vec<u32> = g
        .lines
        .iter()
        .enumerate()
        .filter(|(_, [p, q])| inside(*p) && inside(*q))
        .map(|(i, _)| i as u32)
        .collect();
    // In a drawing a curve is taken whole or not at all.
    if g.map.is_some() {
        let all = lines.clone();
        lines.retain(|l| g.lines_of(Pick::Line(*l)).iter().all(|k| all.contains(k)));
    }
    (points, lines)
}

/// Where two segments cross, strictly inside both: two lines meeting at an
/// end already have a point there to snap to.
fn crossing(p0: DVec2, p1: DVec2, q0: DVec2, q1: DVec2) -> Option<DVec2> {
    let (r, s) = (p1 - p0, q1 - q0);
    let denom = r.perp_dot(s);
    if denom.abs() < 1e-18 {
        return None;
    }
    let qp = q0 - p0;
    let (t, u) = (qp.perp_dot(s) / denom, qp.perp_dot(r) / denom);
    let inside = |x: f64| x > 1e-9 && x < 1.0 - 1e-9;
    (inside(t) && inside(u)).then(|| p0 + r * t)
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_selected_sketchs_lines_are_marked_so_and_the_active_ones_more() {
        let mut doc = kreate_scene::Document::new();
        let mut add = |name: &str| {
            let m = doc.add_mesh(kreate_mesh::MeshData::from_faces(
                vec![kreate_math::Vec3::ZERO, kreate_math::Vec3::X],
                &[],
            ));
            let mut o = kreate_scene::document::Object::new(
                name,
                kreate_scene::document::ObjectData::Mesh(m),
            );
            o.sketch_lines = vec![[0, 1]];
            doc.add_object(o)
        };
        let (a, b, c) = (add("A"), add("B"), add("C"));
        doc.selection.add(a);
        doc.selection.add(b);
        doc.selection.active = Some(b);
        let marks: Vec<u8> = other_lines_marked(&doc, None)
            .iter()
            .map(|(_, m)| *m)
            .collect();
        assert_eq!(marks, vec![1, 2, 0]);
        let _ = c;
    }

    #[test]
    fn a_click_near_a_sketchs_line_finds_the_sketch() {
        let mut doc = kreate_scene::Document::new();
        let mut mesh = kreate_mesh::MeshData::from_faces(
            vec![
                kreate_math::Vec3::new(-1.0, 0.0, 0.0),
                kreate_math::Vec3::new(1.0, 0.0, 0.0),
            ],
            &[],
        );
        let _ = &mut mesh;
        let m = doc.add_mesh(mesh);
        let mut o = kreate_scene::document::Object::new(
            "Sketch",
            kreate_scene::document::ObjectData::Mesh(m),
        );
        o.sketch_lines = vec![[0, 1]];
        let id = doc.add_object(o);
        // Looking straight down at the line from above.
        let camera = kreate_render::Camera {
            pivot: kreate_math::DVec3::ZERO,
            rotation: kreate_math::DQuat::IDENTITY,
            distance: 5.0,
            ..kreate_render::Camera::default()
        };
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0));
        let projector = kreate_ui::object_glyph::Projector {
            eye: camera.eye(),
            rotation: camera.rotation,
            projection: camera.projection_matrix(1.0).as_dmat4(),
            rect,
        };
        let on = projector
            .project(kreate_math::DVec3::new(0.3, 0.0, 0.0))
            .unwrap();
        assert_eq!(
            sketch_under(&doc, &projector, on + egui::vec2(0.0, 3.0), 6.0),
            Some(id)
        );
        assert_eq!(
            sketch_under(&doc, &projector, on + egui::vec2(0.0, 30.0), 6.0),
            None
        );
    }

    use super::*;

    /// A unit square on screen at 100 px a unit, and a circle.
    /// A rectangle's corners joined, a circle and an arc: one point per
    /// corner, each line its entity, the round its circle.
    fn drawn() -> Drawing {
        use kreate_scene::drawing::Constraint as C;
        let mut d = Drawing::new();
        let corners = [
            DVec2::new(0.0, 0.0),
            DVec2::new(2.0, 0.0),
            DVec2::new(2.0, 1.0),
            DVec2::new(0.0, 1.0),
        ];
        let ids: Vec<EntityId> = (0..4)
            .map(|k| {
                d.add_curve(Curve2D::Line {
                    a: corners[k],
                    b: corners[(k + 1) % 4],
                })
            })
            .collect();
        for k in 0..4 {
            d.constraints.push(C::Coincident(
                EntityRef::at(ids[k], Part::End),
                EntityRef::at(ids[(k + 1) % 4], Part::Start),
            ));
        }
        d.add_curve(Curve2D::Circle {
            c: DVec2::new(5.0, 0.0),
            r: 1.0,
        });
        d.add_curve(Curve2D::Arc {
            c: DVec2::new(8.0, 0.0),
            r: 1.0,
            start: 0.0,
            sweep: std::f64::consts::PI,
        });
        d
    }

    #[test]
    fn a_drawing_is_seen_as_points_lines_and_rounds() {
        let d = drawn();
        let g = Geometry::from_drawing(&d, 1e-3, |p| Some(p * 100.0));
        let map = g.map.as_ref().unwrap();
        // The origin, four corners, the circle's centre, the arc's centre
        // and two ends; then the points along the curves.
        let real = map.refs.iter().filter(|r| !r.is_empty()).count();
        assert_eq!(real, 8);
        assert!(map.refs[1..9].iter().all(|r| !r.is_empty()));
        assert_eq!(map.refs[1].len(), 2, "a corner is two lines' ends");
        // Each line its entity; a click near the rectangle's bottom is line 1.
        match g.pick(DVec2::new(100.0, 1.0)) {
            Some(p @ Pick::Line(_)) => assert_eq!(g.entity_of(p), Some(EntityId(1))),
            other => panic!("{other:?}"),
        }
        // The circle as a round, and all its pieces selected with it.
        let at = DVec2::new(500.0, 100.0);
        let pick = g.pick(at).expect("the circle");
        assert!(matches!(pick, Pick::Round { .. }));
        assert_eq!(g.entity_of(pick), Some(EntityId(5)));
        let lines = g.lines_of(pick);
        assert!(lines.len() > 8);
        assert!(lines
            .iter()
            .all(|l| g.entity_of_line(*l) == Some(EntityId(5))));
        // A corner is a curve's part, never a lone point.
        assert_eq!(g.point_ref(1).map(|r| r.part), Some(Part::Start));
        assert_eq!(g.entities_in(&[1], &[0]), vec![1]);
    }

    /// The drawing's points keep their numbers when it moves: a drag finds
    /// the point it took hold of on every step.
    #[test]
    fn a_drawings_points_keep_their_numbers_as_it_moves() {
        let d = drawn();
        let before = Geometry::from_drawing(&d, 1e-3, Some);
        let mut moved = d.clone();
        if let Some(e) = moved.entity_mut(EntityId(6)) {
            e.kind = EntityKind::Curve(Curve2D::Arc {
                c: DVec2::new(8.0, 0.0),
                r: 3.0,
                start: 0.0,
                sweep: std::f64::consts::PI,
            });
        }
        let after = Geometry::from_drawing(&moved, 1e-3, Some);
        let (a, b) = (before.map.unwrap(), after.map.unwrap());
        assert_eq!(a.refs[..9], b.refs[..9]);
        assert!(after.points.len() > before.points.len(), "drawn finer");
    }

    #[test]
    fn picks_in_a_drawing_become_the_drafting_constraint_they_mean() {
        let d = drawn();
        let g = Geometry::from_drawing(&d, 1e-3, |p| Some(p * 100.0));
        let (bottom, right) = (Pick::Line(0), Pick::Line(1));
        let circle = g.pick(DVec2::new(500.0, 100.0)).unwrap();
        let whole = |i| EntityRef::whole(EntityId(i));
        assert_eq!(
            drawing_constraint_for("perpendicular", &[bottom, right], &g),
            Ok(Some(("perpendicular", vec![whole(1), whole(2)])))
        );
        assert_eq!(
            drawing_constraint_for("tangent", &[bottom, circle], &g),
            Ok(Some(("tangent", vec![whole(1), whole(5)])))
        );
        // A point on a curve is held on it.
        assert_eq!(
            drawing_constraint_for("coincident", &[Pick::Point(1), circle], &g),
            Ok(Some((
                "point_on",
                vec![EntityRef::at(EntityId(1), Part::Start), whole(5)]
            )))
        );
        assert_eq!(
            drawing_constraint_for("fixed", &[bottom], &g),
            Err(CannotConstrain)
        );
        // A line's length, a circle's diameter, two lines' angle.
        assert_eq!(
            drawing_dimension_for(&[bottom], &g, &d),
            Ok(Some(("aligned", vec![whole(1)])))
        );
        assert_eq!(
            drawing_dimension_for(&[circle], &g, &d),
            Ok(Some(("diameter", vec![whole(5)])))
        );
        assert_eq!(
            drawing_dimension_for(&[bottom, right], &g, &d),
            Ok(Some(("angular", vec![whole(1), whole(2)])))
        );
        assert_eq!(
            ref_text(EntityRef::at(EntityId(3), Part::Vertex(2))),
            "3.v2"
        );
    }

    fn geometry() -> Geometry {
        let mut points = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 0.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(0.0, 1.0),
            DVec2::new(3.0, 0.5), // 4: a centre
        ];
        let ring: Vec<u32> = (0..8)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 8.0;
                points.push(DVec2::new(3.0, 0.5) + DVec2::new(a.cos(), a.sin()) * 0.5);
                points.len() as u32 - 1
            })
            .collect();
        let mut lines = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
        for k in 0..8 {
            lines.push([ring[k], ring[(k + 1) % 8]]);
        }
        Geometry::new(
            points,
            lines,
            &[Constraint::Circle { centre: 4, ring }],
            |p| Some(p * 100.0),
        )
    }

    #[test]
    fn a_drag_box_takes_the_points_inside_and_the_lines_wholly_in_it() {
        let g = geometry();
        // Round the square's right half: its two right corners and the side
        // between them, not the origin's sides.
        let (points, lines) = in_band(&g, DVec2::new(50.0, -10.0), DVec2::new(150.0, 110.0));
        assert_eq!(points, vec![1, 2]);
        assert_eq!(lines, vec![1]);
        // Round the circle, drawn either way: its centre and every piece of
        // it, but not the points it is drawn through.
        let (points, lines) = in_band(&g, DVec2::new(360.0, 110.0), DVec2::new(240.0, -10.0));
        assert_eq!(points, vec![4]);
        assert_eq!(lines, (4..12).collect::<Vec<u32>>());
        // The origin is never taken.
        let (points, _) = in_band(&g, DVec2::new(-10.0, -10.0), DVec2::new(10.0, 10.0));
        assert!(points.is_empty());
    }

    #[test]
    fn drawing_snaps_to_a_lines_middle_and_where_two_lines_cross() {
        let mut g = geometry();
        // A diagonal across the square, crossing nothing but at its corners;
        // and a line across the middle, crossing the diagonal at the centre.
        g.lines.push([0, 2]);
        g.points.push(DVec2::new(0.0, 0.5));
        g.points.push(DVec2::new(1.0, 0.5));
        let n = g.points.len() as u32;
        g.lines.push([n - 2, n - 1]);
        let snaps = g.snap_points();
        let has = |p: DVec2| snaps.iter().any(|q| q.distance(p) < 1e-9);
        assert!(has(DVec2::new(0.5, 0.0)), "the bottom's middle");
        assert!(has(DVec2::new(0.5, 0.5)), "the crossing at the centre");
        // Where lines meet at an end is a point already, not a crossing.
        assert!(
            !snaps
                .iter()
                .any(|q| q.distance(DVec2::new(1.0, 0.0)) < 1e-9),
            "{snaps:?}"
        );
    }

    #[test]
    fn a_line_drags_by_its_ends_and_a_circle_with_its_centre() {
        let g = geometry();
        assert_eq!(g.points_of(Pick::Line(1)), vec![1, 2]);
        let circle = g.points_of(Pick::Round { centre: 4, on: 5 });
        assert_eq!(circle.len(), 9, "the centre and its eight: {circle:?}");
        assert_eq!(circle[0], 4);
        assert_eq!(g.lines_of_round(4, 5), (4..12).collect::<Vec<u32>>());
    }

    #[test]
    fn what_is_dragged_is_not_snapped_to() {
        let g = geometry().without(&[1, 2]);
        // Point 1 is under the pointer but moving: not caught.
        assert_eq!(g.point_near(DVec2::new(101.0, 1.0), 10.0), None);
        // The lines on the moving points went; the others stay.
        assert!(g.lines.iter().all(|l| !l.contains(&1) && !l.contains(&2)));
        assert!(g.lines.contains(&[3, 0]));
        assert_eq!(g.point_near(DVec2::new(1.0, 1.0), 10.0), Some(0));
    }

    #[test]
    fn drawing_snaps_onto_a_line_under_the_pointer() {
        let g = geometry();
        // 3 px under the bottom edge on screen, a little off it on the plane.
        let on = g
            .on_line(DVec2::new(40.0, 3.0), DVec2::new(0.4, 0.03), 6.0)
            .unwrap();
        assert!(on.distance(DVec2::new(0.4, 0.0)) < 1e-9, "{on:?}");
        assert_eq!(
            g.on_line(DVec2::new(40.0, 30.0), DVec2::new(0.4, 0.3), 6.0),
            None
        );
    }

    #[test]
    fn a_point_beats_a_line_and_a_circle_is_picked_as_a_circle() {
        let g = geometry();
        assert_eq!(g.pick(DVec2::new(102.0, 3.0)), Some(Pick::Point(1)));
        assert_eq!(g.pick(DVec2::new(50.0, 2.0)), Some(Pick::Line(0)));
        // On the circle's rim, near one of its ring points: the circle.
        assert!(matches!(
            g.pick(DVec2::new(350.0, 50.0)),
            Some(Pick::Round { centre: 4, .. })
        ));
        assert_eq!(g.pick(DVec2::new(200.0, 300.0)), None);
    }

    #[test]
    fn picks_become_the_constraint_they_mean() {
        let g = geometry();
        let round = Pick::Round { centre: 4, on: 5 };
        assert_eq!(
            constraint_for("horizontal", &[Pick::Line(0)], &g),
            Ok(Some(("horizontal", vec![0, 1])))
        );
        assert_eq!(
            constraint_for("perpendicular", &[Pick::Line(0)], &g),
            Ok(None),
            "waits for the second line"
        );
        assert_eq!(
            constraint_for("perpendicular", &[Pick::Line(0), Pick::Line(1)], &g),
            Ok(Some(("perpendicular", vec![0, 1, 1, 2])))
        );
        assert_eq!(
            constraint_for("tangent", &[Pick::Line(2), round], &g),
            Ok(Some(("tangent", vec![2, 3, 4, 5])))
        );
        assert_eq!(
            constraint_for("coincident", &[Pick::Point(4), Pick::Line(0)], &g),
            Ok(Some(("on_line", vec![4, 0, 1])))
        );
        assert_eq!(
            constraint_for("concentric", &[round, round], &g),
            Ok(Some(("coincident", vec![4, 4])))
        );
        assert!(constraint_for("fixed", &[Pick::Line(0)], &g).is_err());
    }

    #[test]
    fn a_dimension_is_a_length_a_radius_or_an_angle() {
        let g = geometry();
        assert_eq!(
            dimension_for(&[Pick::Line(1)], &g),
            Ok(Some(("distance", vec![1, 2])))
        );
        assert_eq!(
            dimension_for(&[Pick::Round { centre: 4, on: 5 }], &g),
            Ok(Some(("distance", vec![4, 5])))
        );
        assert_eq!(
            dimension_for(&[Pick::Line(0), Pick::Line(1)], &g),
            Ok(Some(("angle", vec![0, 1, 1, 2])))
        );
        assert_eq!(dimension_for(&[Pick::Point(0)], &g), Ok(None));
    }

    #[test]
    fn typed_numbers_are_in_the_scenes_unit() {
        let mm = 0.001;
        assert_eq!(typed_lengths("40", mm), vec![0.04]);
        assert_eq!(typed_lengths("40, 20", mm), vec![0.04, 0.02]);
        assert_eq!(typed_lengths("abc", mm), Vec::<f64>::new());
        let p = along(DVec2::ZERO, DVec2::new(0.0, 5.0), 0.02);
        assert!((p - DVec2::new(0.0, 0.02)).length() < 1e-12);
        let c = corner(DVec2::ZERO, DVec2::new(-1.0, 1.0), 0.04, 0.02);
        assert_eq!(c, DVec2::new(-0.04, 0.02));
    }
}
