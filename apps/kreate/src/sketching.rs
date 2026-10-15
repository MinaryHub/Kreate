//! The sketch tools' state and their decisions: which tool is in hand, what
//! has been clicked, what is under the pointer, what a set of picks
//! constrains or dimensions, and what a typed number means.
//!
//! Kept apart from the window so it can be tested without one: the
//! application turns the pointer into a point on the sketch's plane and
//! into the sketch's points on screen, and hands both in; what comes back is
//! an operator to run, or nothing yet.

use kreate_math::DVec2;
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
            "trim" => Tool::Trim,
            "dimension" => Tool::Dimension,
            "extrude" => Tool::Extrude,
            "revolve" => Tool::Revolve,
            "pattern" => Tool::Pattern,
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
            Tool::Trim => "trim",
            Tool::Dimension => "dimension",
            Tool::Extrude => "extrude",
            Tool::Revolve => "revolve",
            Tool::Pattern => "pattern",
            Tool::Constrain(kind) => kind,
        }
    }

    /// Whether it draws: what the clicks are points of, and what a typed
    /// number measures.
    pub fn draws(self) -> bool {
        matches!(
            self,
            Tool::Line | Tool::Rectangle | Tool::Circle | Tool::Arc | Tool::Point
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
        Self {
            points,
            screen,
            lines,
            rounds,
        }
    }

    /// Whether a point is a ring point of a circle or arc but not an end of
    /// one: what a click on a curve should not pick as a point.
    fn inner_ring_point(&self, v: u32) -> bool {
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

/// Every sketch's lines but `active`'s, in the world: what the overlay draws
/// thin, since a sketch has no faces for the renderer to draw.
pub fn other_lines(
    doc: &kreate_scene::Document,
    active: Option<kreate_scene::document::ObjectId>,
) -> Vec<[kreate_math::DVec3; 2]> {
    use kreate_math::DVec3;
    let mut others = Vec::new();
    for (id, o) in doc.objects() {
        if o.sketch_lines.is_empty() || Some(id) == active || !doc.is_shown(id) {
            continue;
        }
        let world = doc.world_transform(id);
        let pts = kreate_ops::ops::sketch_draw::points(doc, id);
        for [a, b] in &o.sketch_lines {
            if let (Some(p), Some(q)) = (pts.get(*a as usize), pts.get(*b as usize)) {
                others.push([
                    world.transform_point3(DVec3::new(p.x, p.y, 0.0)),
                    world.transform_point3(DVec3::new(q.x, q.y, 0.0)),
                ]);
            }
        }
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
    use kreate_math::DVec3;
    let mut best: Option<(f32, kreate_scene::document::ObjectId)> = None;
    for (id, o) in doc.objects() {
        if o.sketch_lines.is_empty() || !doc.is_shown(id) {
            continue;
        }
        let world = doc.world_transform(id);
        let pts = kreate_ops::ops::sketch_draw::points(doc, id);
        for [a, b] in &o.sketch_lines {
            let (Some(p), Some(q)) = (pts.get(*a as usize), pts.get(*b as usize)) else {
                continue;
            };
            let (Some(p), Some(q)) = (
                projector.project(world.transform_point3(DVec3::new(p.x, p.y, 0.0))),
                projector.project(world.transform_point3(DVec3::new(q.x, q.y, 0.0))),
            ) else {
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
    use kreate_math::DVec3;
    let mut out = Vec::new();
    for (id, o) in doc.objects() {
        if o.sketch_lines.is_empty() || Some(id) == active || !doc.is_shown(id) {
            continue;
        }
        let mark = if doc.selection.active == Some(id) && doc.selection.contains(id) {
            2
        } else if doc.selection.contains(id) {
            1
        } else {
            0
        };
        let world = doc.world_transform(id);
        let pts = kreate_ops::ops::sketch_draw::points(doc, id);
        for [a, b] in &o.sketch_lines {
            if let (Some(p), Some(q)) = (pts.get(*a as usize), pts.get(*b as usize)) {
                out.push((
                    [
                        world.transform_point3(DVec3::new(p.x, p.y, 0.0)),
                        world.transform_point3(DVec3::new(q.x, q.y, 0.0)),
                    ],
                    mark,
                ));
            }
        }
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
        if (o.sketch_lines.is_empty() && o.sketch.is_empty())
            || Some(id) == active
            || !doc.is_shown(id)
        {
            continue;
        }
        let world = doc.world_transform(id);
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
            };
            let kind = catalog.get(match f.kind_tag() {
                "revolve" => "sketch-tool-revolve",
                "fillet" => "feature-fillet",
                "chamfer" => "feature-chamfer",
                _ => "sketch-tool-extrude",
            });
            rows.push(crate::shell::FeatureRow {
                order: f.order * 2 + 1,
                // i18n-exempt: a translated name and its place on the timeline
                label: format!("{kind} {}", f.order),
                body: o.name.clone(),
                index: i as u32,
                is_sketch: false,
                revolve,
                amount,
                symmetric,
                cut: f.operation == kreate_scene::feature::Operation::Cut,
            });
        }
    }
    // Each sketch just before the first feature made from it.
    for (_, o) in doc.objects() {
        if o.sketch_lines.is_empty() && o.sketch.is_empty() {
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
            revolve: false,
            amount: 0.0,
            symmetric: false,
            cut: false,
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
    let ring_point = |v: u32| {
        constraints.iter().any(|c| match c {
            Constraint::Circle { ring, .. } => ring.contains(&v),
            Constraint::Arc { ring, .. } => ring.len() > 2 && ring[1..ring.len() - 1].contains(&v),
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
                view.dims.push(SketchDim {
                    index,
                    at: to_world(label),
                    ends: [to_world(pa), to_world(pb)],
                    text,
                    shown,
                    kind,
                });
            }
            Constraint::Angle { a, b, c, d, value } => {
                let deg = value.to_degrees();
                view.dims.push(SketchDim {
                    index,
                    at: to_world((mid(*a, *b) + mid(*c, *d)) * 0.5),
                    ends: [to_world(at(*b)), to_world(at(*c))],
                    text: format!("{deg:.1}°"),
                    shown: deg,
                    kind: DimKind::Angle,
                });
            }
            _ => {}
        }
    }
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
            (Tool::Arc, [s]) => view.preview.push(polyline(vec![*s, h])),
            (Tool::Arc, [s, e]) => {
                if let Some(c) = kreate_ops::ops::sketch_draw::circumcentre(*s, *e, h) {
                    view.preview.push(circle(c, c.distance(*s)));
                }
            }
            _ => {}
        }
        if let (Tool::Trim, Some((line, on))) = (ui.tool, ui.trim) {
            for [a, b] in kreate_ops::ops::sketch_trim::preview(doc, id, line as usize, on) {
                view.cut.push([to_world(a), to_world(b)]);
            }
        }
    }
    // The regions picked, and the one under the pointer.
    if matches!(ui.tool, Tool::Extrude | Tool::Revolve) {
        let regions = kreate_scene::feature::sketch_regions(doc, id);
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
    view
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
        .collect();
    let lines: Vec<u32> = g
        .lines
        .iter()
        .enumerate()
        .filter(|(_, [p, q])| inside(*p) && inside(*q))
        .map(|(i, _)| i as u32)
        .collect();
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
