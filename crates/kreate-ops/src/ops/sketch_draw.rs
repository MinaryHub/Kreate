//! Drawing in a sketch: a new sketch on a plane, and the lines, rectangles,
//! circles, arcs and points the sketch tools put in it, each one operator
//! and one undo step — with the constraints a careful drawing implies added
//! as it is drawn, as Fusion adds them.
//!
//! A sketch's points are its mesh's vertices, its lines the object's own
//! (`Object::sketch_lines`), its constraints `Object::sketch`. Point 0 is
//! the sketch's origin, fixed where the sketch's axes cross: what a line
//! drawn from it is tied down by.
//!
//! Every position here is in the sketch's own plane, in metres.

use kreate_math::{DMat3, DQuat, DVec2, DVec3, Vec3};
use kreate_mesh::MeshData;
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::sketch::{Constraint, KINDS};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<NewSketch>();
    registry.register::<Reattach>();
    registry.register::<Line>();
    registry.register::<Rectangle>();
    registry.register::<Circle>();
    registry.register::<Arc>();
    registry.register::<Point>();
    registry.register::<Spline>();
    registry.register::<Polygon>();
    registry.register::<Slot>();
    registry.register::<Fillet>();
    registry.register::<Constrain>();
    registry.register::<MovePoints>();
    registry.register::<Delete>();
    registry.register::<Pattern>();
}

/// How near level or upright a line has to be drawn to be made so: about a
/// degree and a half.
const SQUARE: f64 = 0.026;

/// The active object, when it is a sketch: one that has its origin.
pub fn active_sketch(doc: &Document) -> Option<ObjectId> {
    let id = doc.selection.active?;
    let o = doc.object(id)?;
    o.mesh()?;
    (!o.sketch.is_empty() || !o.sketch_lines.is_empty()).then_some(id)
}

/// The sketch's points, in its plane.
pub fn points(doc: &Document, id: ObjectId) -> Vec<DVec2> {
    doc.object(id)
        .and_then(|o| o.mesh())
        .and_then(|m| doc.mesh(m))
        .map(|m| {
            m.positions()
                .iter()
                .map(|p| DVec2::new(p.x as f64, p.y as f64))
                .collect()
        })
        .unwrap_or_default()
}

/// Writes a sketch's points: its mesh made again with them, its faces kept.
fn set_points(doc: &mut DocMut<'_>, id: ObjectId, points: &[DVec2]) -> Option<()> {
    let mesh_id = doc.document().object(id)?.mesh()?;
    let old = doc.document().mesh(mesh_id)?;
    let faces: Vec<Vec<u32>> = old.faces().map(|f| f.to_vec()).collect();
    let positions: Vec<Vec3> = points
        .iter()
        .map(|p| Vec3::new(p.x as f32, p.y as f32, 0.0))
        .collect();
    *doc.document_mut().mesh_mut(mesh_id)? = MeshData::from_faces(positions, &faces);
    Some(())
}

/// Solves the sketch and writes the points where the constraints put them.
pub fn solve(doc: &mut DocMut<'_>, id: ObjectId) -> Option<kreate_scene::sketch::Solve> {
    let mut p = points(doc.document(), id);
    let c = doc.document().object(id)?.sketch.clone();
    let result = kreate_scene::sketch::solve(&mut p, &c);
    set_points(doc, id, &p)?;
    Some(result)
}

/// What a tool hands over for one of its points: where, and an existing
/// point it lands on (its index plus one; zero for none) — a drawing that
/// ends where another begins is joined there, not two points on one spot.
fn place(points: &mut Vec<DVec2>, at: DVec2, join: Option<&u32>) -> u32 {
    match join
        .copied()
        .filter(|j| *j > 0 && (*j as usize) <= points.len())
    {
        Some(j) => j - 1,
        None => {
            points.push(at);
            points.len() as u32 - 1
        }
    }
}

/// A point parameter's place in the sketch's plane: its Z is ignored.
fn xy(v: [f64; 3]) -> DVec2 {
    DVec2::new(v[0], v[1])
}

fn pair(v: &[f64], i: usize) -> Option<DVec2> {
    Some(DVec2::new(*v.get(2 * i)?, *v.get(2 * i + 1)?)).filter(|p| p.is_finite())
}

/// The level or upright constraint a line drawn nearly so earns.
fn squared(points: &[DVec2], a: u32, b: u32) -> Option<Constraint> {
    let d = points[b as usize] - points[a as usize];
    if d.length() < 1e-12 {
        return None;
    }
    let angle = d.y.atan2(d.x).rem_euclid(std::f64::consts::PI);
    if angle < SQUARE || angle > std::f64::consts::PI - SQUARE {
        Some(Constraint::Horizontal { a, b })
    } else if (angle - std::f64::consts::FRAC_PI_2).abs() < SQUARE {
        Some(Constraint::Vertical { a, b })
    } else {
        None
    }
}

/// Lines, points and constraints added to the active sketch, and the sketch
/// solved: what every drawing operator ends with.
fn commit(
    doc: &mut DocMut<'_>,
    id: ObjectId,
    points: Vec<DVec2>,
    lines: Vec<[u32; 2]>,
    constraints: Vec<Constraint>,
) -> Result<OpResult, OpError> {
    set_points(doc, id, &points).ok_or_else(|| OpError::Failed("no sketch".into()))?;
    let o = doc.document_mut().object_mut(id).expect("the sketch");
    o.sketch_lines.extend(lines);
    o.sketch.extend(constraints);
    solve(doc, id);
    Ok(OpResult::Finished)
}

fn need_sketch(doc: &Document) -> Result<ObjectId, OpError> {
    active_sketch(doc).ok_or_else(|| OpError::Failed("no sketch is being drawn".into()))
}

// --- a new sketch ------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct NewSketchParams {
    /// Which plane: the ground (XY), the front (XZ), the side (YZ), the
    /// work plane at the 3D cursor, or a flat face of a part.
    #[param(variants("xy", "xz", "yz", "cursor", "face"))]
    pub plane: u32,
    /// For a face: the part, by name; empty for the active object.
    pub object: String,
    /// For a face: its index in the part's mesh, as get_mesh(faces=true)
    /// lists them; −1 to find it by `at`.
    #[param(min = -1.0)]
    pub face: i64,
    /// For a face: a point on it, in the world — where the sketch's origin
    /// goes, and, without `face`, how the face is found.
    #[param(unit = "length")]
    pub at: [f64; 3],
}

impl Default for NewSketchParams {
    fn default() -> Self {
        Self {
            plane: 0,
            object: String::new(),
            face: -1,
            at: [0.0; 3],
        }
    }
}

/// A sketch's plane on a part's face, from the parameters: its origin, its
/// turn, and the face's outlines in the sketch's plane.
fn face_plane(
    doc: &Document,
    params: &NewSketchParams,
) -> Result<crate::ops::sketch_face::FaceFrame, OpError> {
    use crate::ops::sketch_face::{face_at, face_frame};
    let id = if params.object.trim().is_empty() {
        doc.selection.active
    } else {
        doc.objects()
            .find(|(_, o)| o.name == params.object)
            .map(|(id, _)| id)
    }
    .filter(|id| {
        doc.object(*id)
            .is_some_and(|o| o.mesh().is_some() && o.sketch.is_empty())
    })
    .ok_or_else(|| OpError::Failed("name the part to sketch on, or make it active".into()))?;
    let world = doc.world_transform(id);
    let at = DVec3::from_array(params.at);
    let found = if params.face >= 0 {
        // The part's own faces, as get_mesh numbers them.
        let mesh = doc
            .object(id)
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh(m))
            .expect("a mesh")
            .clone();
        let given = (params.at != [0.0; 3]).then_some(at);
        face_frame(&mesh, world, params.face as u32, given)
    } else {
        // What is drawn — a body's features, a boolean's cut — under `at`.
        let mut cache = kreate_scene::modifier::ModifierCache::new();
        let mesh = cache
            .evaluated(doc, id, &kreate_scene::modifier::EvalCtx::still())
            .ok_or_else(|| OpError::Failed("the part has no surface".into()))?;
        let face = face_at(&mesh, world, at).ok_or_else(|| {
            OpError::Failed(format!(
                "no face of the part at ({:.4}, {:.4}, {:.4}): give a point on its surface, or `face`",
                at.x, at.y, at.z
            ))
        })?;
        match face_frame(&mesh, world, face, Some(at)) {
            // A curved surface: the plane that touches it at the point, as
            // Fusion offers a tangent plane. No outline to project.
            Err(why) if why.contains("curved") => {
                let n = crate::ops::sketch_face::facet_normal(&mesh, world, face);
                crate::report::say("a curved face: sketched on its tangent plane at the point");
                Ok(crate::ops::sketch_face::tangent_frame(at, n))
            }
            other => other,
        }
    };
    found.map_err(OpError::Failed)
}

/// The part a sketch on a face was made on: the object `face_plane` reads.
fn face_host(doc: &Document, params: &NewSketchParams) -> Option<ObjectId> {
    if params.object.trim().is_empty() {
        doc.selection.active
    } else {
        doc.objects()
            .find(|(_, o)| o.name == params.object)
            .map(|(id, _)| id)
    }
}

/// Starts a sketch on a plane: an object whose point 0 is its origin, held.
pub struct NewSketch;

impl Operator for NewSketch {
    type Params = NewSketchParams;
    fn id() -> &'static str {
        "sketch.new"
    }
    fn label() -> &'static str {
        "New Sketch"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // A face's outlines, to draw against: put in fixed, as Fusion
        // projects a face's edges into a sketch made on it.
        let mut outlines: Vec<Vec<DVec2>> = Vec::new();
        let mut host: Option<ObjectId> = None;
        let (location, rotation) = match params.plane {
            // Front: the sketch's Y up the world's Z, its X along the world's.
            1 => (
                DVec3::ZERO,
                DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
            ),
            // Side: its X along the world's Y, its Y up the world's Z.
            2 => (
                DVec3::ZERO,
                DQuat::from_mat3(&DMat3::from_cols(DVec3::Y, DVec3::Z, DVec3::X)),
            ),
            3 => {
                let d = doc.document();
                (d.cursor, d.cursor_rotation)
            }
            4 => {
                let frame = face_plane(doc.document(), params)?;
                host = face_host(doc.document(), params);
                outlines = frame.loops;
                (frame.origin, frame.rotation)
            }
            _ => (DVec3::ZERO, DQuat::IDENTITY),
        };
        let d = doc.document_mut();
        let mut points = vec![Vec3::ZERO];
        let mut lines: Vec<[u32; 2]> = Vec::new();
        let mut fixed = vec![Constraint::Fixed {
            a: 0,
            at: [0.0, 0.0],
        }];
        for ring in &outlines {
            let first = points.len() as u32;
            for (k, p) in ring.iter().enumerate() {
                let i = points.len() as u32;
                points.push(Vec3::new(p.x as f32, p.y as f32, 0.0));
                fixed.push(Constraint::Fixed {
                    a: i,
                    at: [p.x, p.y],
                });
                let next = if k + 1 == ring.len() { first } else { i + 1 };
                lines.push([i, next]);
            }
        }
        let mesh = d.add_mesh(MeshData::from_faces(points, &[]));
        let name = d.unique_object_name("Sketch");
        let mut object = Object::new(name, ObjectData::Mesh(mesh));
        object.transform.translation = location;
        object.transform.rotation = rotation;
        object.sketch.extend(fixed);
        object.sketch_lines = lines;
        let id = d.add_object(object);
        // On a part's face: a child of the part, so it goes where the part
        // goes, staying where it is now.
        if let Some(h) = host {
            let _ = d.set_parent(id, Some(h));
        }
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        Ok(OpResult::Finished)
    }
}

// --- back onto its face --------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ReattachParams {
    /// Put in the face's outline as it is now, in place of the one projected
    /// before.
    pub outline: bool,
}

impl Default for ReattachParams {
    fn default() -> Self {
        Self { outline: true }
    }
}

/// A sketch on a face put back on it after the part changed — its top
/// pushed up, a pocket made deeper: the face of the part along the sketch's
/// normal, the nearest that faces the same way. What was drawn keeps its
/// place across the plane and its turn; the projected outline is the face's
/// again.
pub struct Reattach;

impl Operator for Reattach {
    type Params = ReattachParams;
    fn id() -> &'static str {
        "sketch.reattach"
    }
    fn label() -> &'static str {
        "Reattach Sketch to Face"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some_and(|id| doc.object(id).is_some_and(|o| o.parent.is_some()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use crate::ops::sketch_face::{face_at, face_frame};
        let id = need_sketch(doc.document())?;
        let d = doc.document();
        let host = d
            .object(id)
            .and_then(|o| o.parent)
            .filter(|h| d.object(*h).is_some_and(|o| o.mesh().is_some()))
            .ok_or_else(|| OpError::Failed("the sketch was not made on a part's face".into()))?;
        let world = d.world_transform(id);
        let origin = world.translation;
        let (_, turn, _) = world.to_scale_rotation_translation();
        let normal = (turn * DVec3::Z).normalize();
        let old_x = (turn * DVec3::X).normalize();
        let mut cache = kreate_scene::modifier::ModifierCache::new();
        let mesh = cache
            .evaluated(d, host, &kreate_scene::modifier::EvalCtx::still())
            .ok_or_else(|| OpError::Failed("the part has no surface".into()))?;
        let host_world = d.world_transform(host);
        // Along the normal, both ways, the nearest face facing the same way.
        let positions: Vec<DVec3> = mesh
            .positions()
            .iter()
            .map(|p| host_world.transform_point3(p.as_dvec3()))
            .collect();
        let mut best: Option<(f64, u32, DVec3)> = None;
        for f in 0..mesh.num_faces() {
            let ring = mesh.face_verts(f);
            let mut n = DVec3::ZERO;
            for i in 0..ring.len() {
                n += positions[ring[i] as usize]
                    .cross(positions[ring[(i + 1) % ring.len()] as usize]);
            }
            let n = n.normalize_or_zero();
            if n.dot(normal) < 0.999 {
                continue;
            }
            let t = (positions[ring[0] as usize] - origin).dot(n) / normal.dot(n);
            if best.is_some_and(|(b, _, _)| b <= t.abs()) {
                continue;
            }
            let hit = origin + normal * t;
            if face_at(&mesh, host_world, hit) == Some(f) {
                best = Some((t.abs(), f, hit));
            }
        }
        let Some((_, face, hit)) = best else {
            return Err(OpError::Failed(
                "no face of the part faces the sketch along its normal any more".into(),
            ));
        };
        let frame = face_frame(&mesh, host_world, face, Some(hit)).map_err(OpError::Failed)?;
        // The sketch's own turn kept: its X laid onto the new plane.
        let n = frame.rotation * DVec3::Z;
        let x = (old_x - n * old_x.dot(n)).normalize_or(frame.rotation * DVec3::X);
        let rotation = DQuat::from_mat3(&DMat3::from_cols(x, n.cross(x), n)).normalize();
        let new_world = kreate_math::DAffine3::from_rotation_translation(rotation, frame.origin);
        let local = d.above(id).inverse() * new_world;
        let (_, local_turn, local_at) = local.to_scale_rotation_translation();
        // The old projected outline out, the face's as it is now in.
        let mut pts = points(d, id);
        let object = d.object(id).expect("the sketch");
        let mut lines = object.sketch_lines.clone();
        let mut constraints = object.sketch.clone();
        if params.outline {
            let fixed: Vec<bool> = (0..pts.len())
                .map(|v| {
                    v != 0
                        && constraints
                            .iter()
                            .any(|c| matches!(c, Constraint::Fixed { a, .. } if *a as usize == v))
                })
                .collect();
            let mut map = vec![u32::MAX; pts.len()];
            let mut kept = Vec::new();
            for (i, p) in pts.iter().enumerate() {
                if !fixed[i] {
                    map[i] = kept.len() as u32;
                    kept.push(*p);
                }
            }
            lines = lines
                .iter()
                .filter(|l| !fixed[l[0] as usize] && !fixed[l[1] as usize])
                .map(|l| [map[l[0] as usize], map[l[1] as usize]])
                .collect();
            constraints = constraints
                .iter()
                .filter(|c| {
                    c.points()
                        .iter()
                        .all(|v| !fixed.get(*v as usize).copied().unwrap_or(true))
                })
                .map(|c| c.remapped(&|v| map[v as usize]))
                .collect();
            pts = kept;
            // The new outline, in the sketch's own axes.
            let to_local = |p: DVec2| {
                let w = frame.origin + frame.rotation * DVec3::new(p.x, p.y, 0.0);
                let d = w - frame.origin;
                DVec2::new(d.dot(x), d.dot(n.cross(x)))
            };
            for ring in &frame.loops {
                let first = pts.len() as u32;
                for (k, p) in ring.iter().enumerate() {
                    let q = to_local(*p);
                    let i = pts.len() as u32;
                    pts.push(q);
                    constraints.push(Constraint::Fixed {
                        a: i,
                        at: [q.x, q.y],
                    });
                    lines.push([i, if k + 1 == ring.len() { first } else { i + 1 }]);
                }
            }
        }
        set_points(doc, id, &pts).ok_or_else(|| OpError::Failed("no sketch".into()))?;
        let o = doc.document_mut().object_mut(id).expect("the sketch");
        o.transform.translation = local_at;
        o.transform.rotation = local_turn;
        o.sketch_lines = lines;
        o.sketch = constraints;
        solve(doc, id);
        Ok(OpResult::Finished)
    }
}

// --- lines -------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LineParams {
    /// The points in order, as x, y pairs.
    #[param(hidden)]
    pub points: Vec<f64>,
    /// For each point, the existing point it lands on, plus one; zero, or
    /// nothing, for a new one.
    #[param(hidden)]
    pub joins: Vec<u32>,
    /// Back to the first point at the end.
    pub closed: bool,
}

/// A run of lines, one after another: Fusion's Line tool.
pub struct Line;

impl Operator for Line {
    type Params = LineParams;
    fn id() -> &'static str {
        "sketch.line"
    }
    fn label() -> &'static str {
        "Line"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let mut pts = points(doc.document(), id);
        let count = params.points.len() / 2;
        let mut ids = Vec::new();
        for i in 0..count {
            let at = pair(&params.points, i)
                .ok_or_else(|| OpError::Failed("a point is not a number".into()))?;
            let v = place(&mut pts, at, params.joins.get(i));
            if ids.last() != Some(&v) {
                ids.push(v);
            }
        }
        if params.closed && ids.len() > 2 && ids.first() != ids.last() {
            ids.push(ids[0]);
        }
        if ids.len() < 2 {
            crate::report::say("a line needs two different points");
            return Ok(OpResult::Cancelled);
        }
        let lines: Vec<[u32; 2]> = ids.windows(2).map(|w| [w[0], w[1]]).collect();
        let constraints = lines
            .iter()
            .filter_map(|l| squared(&pts, l[0], l[1]))
            .collect();
        commit(doc, id, pts, lines, constraints)
    }
}

// --- rectangles --------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RectangleParams {
    /// One corner — or the centre, for a centred one.
    #[param(unit = "length")]
    pub a: [f64; 3],
    /// The opposite corner.
    #[param(unit = "length")]
    pub b: [f64; 3],
    /// `a` is the middle rather than a corner.
    pub centred: bool,
    /// The existing point the first corner lands on, plus one.
    #[param(hidden)]
    pub join: u32,
}

/// Four lines square to the sketch's axes: its sides level and upright.
pub struct Rectangle;

impl Operator for Rectangle {
    type Params = RectangleParams;
    fn id() -> &'static str {
        "sketch.rectangle"
    }
    fn label() -> &'static str {
        "Rectangle"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let (a, b) = (xy(params.a), xy(params.b));
        let (lo, hi) = if params.centred {
            let half = (b - a).abs();
            (a - half, a + half)
        } else {
            (a.min(b), a.max(b))
        };
        if (hi - lo).min_element() < 1e-9 || !lo.is_finite() || !hi.is_finite() {
            return Err(OpError::Failed(
                "a rectangle needs a width and a height".into(),
            ));
        }
        let mut pts = points(doc.document(), id);
        let corners = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
        // The corner drawn from keeps the join, whichever corner it is.
        let first = if params.centred {
            None
        } else {
            corners.iter().position(|c| (*c - a).length() < 1e-12)
        };
        let ids: Vec<u32> = corners
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let join = (Some(i) == first).then_some(&params.join);
                place(&mut pts, *c, join)
            })
            .collect();
        let lines: Vec<[u32; 2]> = (0..4).map(|i| [ids[i], ids[(i + 1) % 4]]).collect();
        let constraints = vec![
            Constraint::Horizontal {
                a: ids[0],
                b: ids[1],
            },
            Constraint::Vertical {
                a: ids[1],
                b: ids[2],
            },
            Constraint::Horizontal {
                a: ids[2],
                b: ids[3],
            },
            Constraint::Vertical {
                a: ids[3],
                b: ids[0],
            },
        ];
        commit(doc, id, pts, lines, constraints)
    }
}

// --- circles and arcs --------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct CircleParams {
    #[param(unit = "length")]
    pub centre: [f64; 3],
    #[param(min = 0.0, unit = "length")]
    pub radius: f64,
    /// Points round it: the circle a mesh can be.
    #[param(min = 8, max = 256, unit = "count")]
    pub segments: u32,
    /// The existing point the centre lands on, plus one.
    #[param(hidden)]
    pub join: u32,
}

impl Default for CircleParams {
    fn default() -> Self {
        Self {
            centre: [0.0; 3],
            radius: 0.01,
            segments: 48,
            join: 0,
        }
    }
}

/// A circle about a centre: a ring of points held round it.
pub struct Circle;

impl Operator for Circle {
    type Params = CircleParams;
    fn id() -> &'static str {
        "sketch.circle"
    }
    fn label() -> &'static str {
        "Circle"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        if !(params.radius > 0.0 && params.radius.is_finite()) {
            return Err(OpError::Failed("a circle needs a radius".into()));
        }
        let mut pts = points(doc.document(), id);
        let c = xy(params.centre);
        let centre = place(&mut pts, c, Some(&params.join));
        let n = params.segments.clamp(8, 256);
        let ring: Vec<u32> = (0..n)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / n as f64;
                pts.push(c + DVec2::new(a.cos(), a.sin()) * params.radius);
                pts.len() as u32 - 1
            })
            .collect();
        let lines = (0..n as usize)
            .map(|i| [ring[i], ring[(i + 1) % n as usize]])
            .collect();
        commit(
            doc,
            id,
            pts,
            lines,
            vec![Constraint::Circle { centre, ring }],
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ArcParams {
    #[param(unit = "length")]
    pub start: [f64; 3],
    /// A point it passes through, between the ends.
    #[param(unit = "length")]
    pub through: [f64; 3],
    #[param(unit = "length")]
    pub end: [f64; 3],
    #[param(min = 2, max = 256, unit = "count")]
    pub segments: u32,
    /// The existing points the start and the end land on, plus one.
    #[param(hidden)]
    pub joins: Vec<u32>,
}

impl Default for ArcParams {
    fn default() -> Self {
        Self {
            start: [0.0; 3],
            through: [0.0; 3],
            end: [0.0; 3],
            segments: 24,
            joins: Vec::new(),
        }
    }
}

/// The centre of the circle through three points, if they are not in a row.
pub fn circumcentre(a: DVec2, b: DVec2, c: DVec2) -> Option<DVec2> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-18 {
        return None;
    }
    let (a2, b2, c2) = (a.length_squared(), b.length_squared(), c.length_squared());
    Some(DVec2::new(
        (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
        (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d,
    ))
}

/// An arc through three points: Fusion's three-point arc.
pub struct Arc;

impl Operator for Arc {
    type Params = ArcParams;
    fn id() -> &'static str {
        "sketch.arc"
    }
    fn label() -> &'static str {
        "Arc"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let (s, m, e) = (xy(params.start), xy(params.through), xy(params.end));
        let c = circumcentre(s, m, e)
            .ok_or_else(|| OpError::Failed("three points in a row make no arc".into()))?;
        let angle = |p: DVec2| (p - c).y.atan2((p - c).x);
        let ccw = |from: f64, to: f64| (to - from).rem_euclid(std::f64::consts::TAU);
        // Anticlockwise from start to end, if that passes the middle point;
        // otherwise the other way, which is anticlockwise from end to start.
        let forward = ccw(angle(s), angle(m)) < ccw(angle(s), angle(e));
        let (from, to, from_join, to_join) = if forward {
            (s, e, params.joins.first(), params.joins.get(1))
        } else {
            (e, s, params.joins.get(1), params.joins.first())
        };
        let r = (from - c).length();
        let sweep = ccw(angle(from), angle(to));
        let n = params.segments.clamp(2, 256);
        let mut pts = points(doc.document(), id);
        let centre = {
            pts.push(c);
            pts.len() as u32 - 1
        };
        let mut ring = vec![place(&mut pts, from, from_join)];
        for i in 1..n {
            let a = angle(from) + sweep * i as f64 / n as f64;
            pts.push(c + DVec2::new(a.cos(), a.sin()) * r);
            ring.push(pts.len() as u32 - 1);
        }
        ring.push(place(&mut pts, to, to_join));
        let lines = ring.windows(2).map(|w| [w[0], w[1]]).collect();
        commit(doc, id, pts, lines, vec![Constraint::Arc { centre, ring }])
    }
}

// --- splines -----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SplineParams {
    /// The points it passes through, in order, as x, y pairs.
    #[param(hidden)]
    pub points: Vec<f64>,
    /// For each point, the existing point it lands on, plus one.
    #[param(hidden)]
    pub joins: Vec<u32>,
    /// Back round to the first point at the end.
    pub closed: bool,
    /// Points along the curve between each pair it passes through.
    #[param(min = 1, max = 64, unit = "count")]
    pub segments: u32,
}

impl Default for SplineParams {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            joins: Vec::new(),
            closed: false,
            segments: 8,
        }
    }
}

/// A smooth curve through the points given — Fusion's fit-point spline —
/// held to them: move one and the curve follows.
pub struct Spline;

impl Operator for Spline {
    type Params = SplineParams;
    fn id() -> &'static str {
        "sketch.spline"
    }
    fn label() -> &'static str {
        "Spline"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let count = params.points.len() / 2;
        if count < 2 {
            return Err(OpError::Failed("a spline needs two points or more".into()));
        }
        let mut pts = points(doc.document(), id);
        let through: Vec<u32> = (0..count)
            .filter_map(|i| {
                let at = pair(&params.points, i)?;
                Some(place(&mut pts, at, params.joins.get(i)))
            })
            .collect();
        if through.len() < 2 {
            return Err(OpError::Failed("a spline needs two points or more".into()));
        }
        let positions: Vec<DVec2> = through.iter().map(|v| pts[*v as usize]).collect();
        let per = params.segments.clamp(1, 64) as usize;
        let spans = if params.closed {
            through.len()
        } else {
            through.len() - 1
        };
        let ring_at = kreate_scene::sketch::spline_ring(&positions, spans * per, params.closed);
        let ring: Vec<u32> = ring_at
            .iter()
            .map(|p| {
                pts.push(*p);
                pts.len() as u32 - 1
            })
            .collect();
        // Lines: through, its ring, the next through — round the loop when
        // closed.
        let mut lines = Vec::new();
        for s in 0..spans {
            let mut run = vec![through[s]];
            run.extend_from_slice(&ring[s * per..(s + 1) * per]);
            run.push(through[(s + 1) % through.len()]);
            lines.extend(run.windows(2).map(|w| [w[0], w[1]]));
        }
        commit(
            doc,
            id,
            pts,
            lines,
            vec![Constraint::Spline {
                through,
                ring,
                closed: params.closed,
            }],
        )
    }
}

// --- polygons, slots and fillets --------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PolygonParams {
    #[param(unit = "length")]
    pub centre: [f64; 3],
    /// From the centre to a corner.
    #[param(min = 0.0, unit = "length")]
    pub radius: f64,
    #[param(min = 3, max = 64, unit = "count")]
    pub sides: u32,
    /// Turned, in radians: 0 puts the first corner along +X.
    #[param(unit = "angle")]
    pub rotation: f64,
    /// The existing point the centre lands on, plus one.
    #[param(hidden)]
    pub join: u32,
}

impl Default for PolygonParams {
    fn default() -> Self {
        Self {
            centre: [0.0; 3],
            radius: 0.01,
            sides: 6,
            rotation: 0.0,
            join: 0,
        }
    }
}

/// A regular polygon about a centre, held regular: Fusion's Circumscribed
/// Polygon by its corners.
pub struct Polygon;

impl Operator for Polygon {
    type Params = PolygonParams;
    fn id() -> &'static str {
        "sketch.polygon"
    }
    fn label() -> &'static str {
        "Polygon"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        if !(params.radius > 0.0 && params.radius.is_finite()) {
            return Err(OpError::Failed("a polygon needs a radius".into()));
        }
        let mut pts = points(doc.document(), id);
        let c = xy(params.centre);
        let centre = place(&mut pts, c, Some(&params.join));
        let n = params.sides.clamp(3, 64);
        let ring: Vec<u32> = (0..n)
            .map(|i| {
                let a = params.rotation + i as f64 * std::f64::consts::TAU / n as f64;
                pts.push(c + DVec2::new(a.cos(), a.sin()) * params.radius);
                pts.len() as u32 - 1
            })
            .collect();
        let lines = (0..n as usize)
            .map(|i| [ring[i], ring[(i + 1) % n as usize]])
            .collect();
        commit(
            doc,
            id,
            pts,
            lines,
            vec![Constraint::Polygon { centre, ring }],
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SlotParams {
    /// The centres of its two round ends.
    #[param(unit = "length")]
    pub a: [f64; 3],
    #[param(unit = "length")]
    pub b: [f64; 3],
    /// Across the slot.
    #[param(min = 0.0, unit = "length")]
    pub width: f64,
    /// Points round each end.
    #[param(min = 2, max = 128, unit = "count")]
    pub segments: u32,
    /// The existing points the centres land on, plus one.
    #[param(hidden)]
    pub joins: Vec<u32>,
}

impl Default for SlotParams {
    fn default() -> Self {
        Self {
            a: [0.0; 3],
            b: [0.02, 0.0, 0.0],
            width: 0.005,
            segments: 12,
            joins: Vec::new(),
        }
    }
}

/// A slot between two centres: two lines and two half circles, each line
/// held tangent to both ends. Fusion's Center to Center Slot.
pub struct Slot;

impl Operator for Slot {
    type Params = SlotParams;
    fn id() -> &'static str {
        "sketch.slot"
    }
    fn label() -> &'static str {
        "Slot"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let (a, b) = (xy(params.a), xy(params.b));
        let r = params.width * 0.5;
        let d = b - a;
        if !(r > 0.0 && r.is_finite()) || d.length() < 1e-9 {
            return Err(OpError::Failed(
                "a slot needs two centres apart and a width".into(),
            ));
        }
        let dir = d.normalize();
        let side = DVec2::new(-dir.y, dir.x);
        let mut pts = points(doc.document(), id);
        let ca = place(&mut pts, a, params.joins.first());
        let cb = place(&mut pts, b, params.joins.get(1));
        let n = params.segments.clamp(2, 128) as usize;
        // Each end anticlockwise from its first ring point to its last, as
        // an arc is held: the far end from below round through b + dir to
        // above, the near end from above round through a - dir to below.
        let mut arc = |c: DVec2, from: DVec2| -> Vec<u32> {
            let start = from.y.atan2(from.x);
            (0..=n)
                .map(|k| {
                    let t = start + std::f64::consts::PI * k as f64 / n as f64;
                    pts.push(c + DVec2::new(t.cos(), t.sin()) * r);
                    pts.len() as u32 - 1
                })
                .collect()
        };
        let far = arc(b, -side);
        let near = arc(a, side);
        let mut lines: Vec<[u32; 2]> = Vec::new();
        lines.extend(far.windows(2).map(|w| [w[0], w[1]]));
        lines.extend(near.windows(2).map(|w| [w[0], w[1]]));
        // The two straight sides join the arcs' ends.
        let (top_a, top_b) = (near[0], far[n]);
        let (bottom_b, bottom_a) = (far[0], near[n]);
        lines.push([top_a, top_b]);
        lines.push([bottom_b, bottom_a]);
        let constraints = vec![
            Constraint::Arc {
                centre: cb,
                ring: far.clone(),
            },
            Constraint::Arc {
                centre: ca,
                ring: near.clone(),
            },
            Constraint::Tangent {
                a: top_a,
                b: top_b,
                centre: cb,
                on: far[0],
            },
            Constraint::Tangent {
                a: top_a,
                b: top_b,
                centre: ca,
                on: near[0],
            },
            Constraint::Tangent {
                a: bottom_b,
                b: bottom_a,
                centre: cb,
                on: far[0],
            },
            Constraint::Tangent {
                a: bottom_b,
                b: bottom_a,
                centre: ca,
                on: near[0],
            },
            // The two ends the same size.
            Constraint::Equal {
                a: ca,
                b: near[0],
                c: cb,
                d: far[0],
            },
        ];
        commit(doc, id, pts, lines, constraints)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct FilletParams {
    /// The corner: a point two lines meet at.
    #[param(hidden)]
    pub point: u32,
    #[param(min = 0.0, unit = "length")]
    pub radius: f64,
    /// Points along the round.
    #[param(min = 2, max = 64, unit = "count")]
    pub segments: u32,
}

impl Default for FilletParams {
    fn default() -> Self {
        Self {
            point: 0,
            radius: 0.002,
            segments: 8,
        }
    }
}

/// Rounds a corner: the two lines meeting at a point are cut back and an
/// arc tangent to both put between them, its radius dimensioned.
pub struct Fillet;

impl Operator for Fillet {
    type Params = FilletParams;
    fn id() -> &'static str {
        "sketch.fillet"
    }
    fn label() -> &'static str {
        "Sketch Fillet"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let mut pts = points(doc.document(), id);
        let object = doc.document().object(id).expect("the sketch");
        let p = params.point;
        if p as usize >= pts.len() {
            return Err(OpError::Failed("no such point".into()));
        }
        let at: Vec<(usize, u32)> = object
            .sketch_lines
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if l[0] == p {
                    Some((i, l[1]))
                } else if l[1] == p {
                    Some((i, l[0]))
                } else {
                    None
                }
            })
            .collect();
        let [(la, a), (lb, b)] = at[..] else {
            return Err(OpError::Failed(
                "a fillet wants a corner where exactly two lines meet".into(),
            ));
        };
        let (pa, pb, pp) = (pts[a as usize], pts[b as usize], pts[p as usize]);
        let (ua, ub) = ((pa - pp).normalize_or_zero(), (pb - pp).normalize_or_zero());
        let cos = ua.dot(ub).clamp(-1.0, 1.0);
        let theta = cos.acos();
        if theta < 1e-6 || theta > std::f64::consts::PI - 1e-6 {
            return Err(OpError::Failed(
                "the lines are in a row: nothing to round".into(),
            ));
        }
        let r = params.radius;
        if !(r > 0.0 && r.is_finite()) {
            return Err(OpError::Failed("a fillet needs a radius".into()));
        }
        let back = r / (theta * 0.5).tan();
        if back >= (pa - pp).length() - 1e-9 || back >= (pb - pp).length() - 1e-9 {
            return Err(OpError::Failed(
                "the radius is too big for these lines".into(),
            ));
        }
        let (ta, tb) = (pp + ua * back, pp + ub * back);
        let bisector = (ua + ub).normalize_or_zero();
        let centre_at = pp + bisector * (r / (theta * 0.5).sin());
        // The arc from ta to tb, the short way round the centre.
        let n = params.segments.clamp(2, 64) as usize;
        let (sa, sb) = (
            (ta - centre_at).y.atan2((ta - centre_at).x),
            (tb - centre_at).y.atan2((tb - centre_at).x),
        );
        let mut sweep = (sb - sa).rem_euclid(std::f64::consts::TAU);
        let (from, to, sweep) = if sweep > std::f64::consts::PI {
            sweep = std::f64::consts::TAU - sweep;
            (tb, ta, sweep)
        } else {
            (ta, tb, sweep)
        };
        let start = (from - centre_at).y.atan2((from - centre_at).x);
        let _ = to;
        let centre = {
            pts.push(centre_at);
            pts.len() as u32 - 1
        };
        let ring: Vec<u32> = (0..=n)
            .map(|k| {
                let t = start + sweep * k as f64 / n as f64;
                pts.push(centre_at + DVec2::new(t.cos(), t.sin()) * r);
                pts.len() as u32 - 1
            })
            .collect();
        // Which ring end sits on which line.
        let (end_a, end_b) = if from == ta {
            (ring[0], ring[n])
        } else {
            (ring[n], ring[0])
        };
        let mut lines = object.sketch_lines.clone();
        let mut constraints = object.sketch.clone();
        lines[la] = [a, end_a];
        lines[lb] = [b, end_b];
        lines.extend(ring.windows(2).map(|w| [w[0], w[1]]));
        // What was said of the lines through the corner is said of them cut
        // back: a level line stays level.
        for c in constraints.iter_mut() {
            *c = c.remapped(&|v| v);
            let swap = |x: &mut u32, y: &mut u32, from_other: u32, to: u32| {
                if (*x == p && *y == from_other) || (*y == p && *x == from_other) {
                    if *x == p {
                        *x = to;
                    } else {
                        *y = to;
                    }
                }
            };
            match c {
                Constraint::Horizontal { a: x, b: y }
                | Constraint::Vertical { a: x, b: y }
                | Constraint::OnLine { a: x, b: y, .. }
                | Constraint::Tangent { a: x, b: y, .. } => {
                    swap(x, y, a, end_a);
                    swap(x, y, b, end_b);
                }
                Constraint::Parallel {
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
                | Constraint::Angle {
                    a: x,
                    b: y,
                    c: z,
                    d: w,
                    ..
                } => {
                    swap(x, y, a, end_a);
                    swap(x, y, b, end_b);
                    swap(z, w, a, end_a);
                    swap(z, w, b, end_b);
                }
                _ => {}
            }
        }
        // The corner's own holds — a distance to it, its fix — go with it.
        constraints.retain(|c| !c.points().contains(&p));
        constraints.push(Constraint::Arc {
            centre,
            ring: ring.clone(),
        });
        constraints.push(Constraint::Tangent {
            a,
            b: end_a,
            centre,
            on: ring[0],
        });
        constraints.push(Constraint::Tangent {
            a: b,
            b: end_b,
            centre,
            on: ring[0],
        });
        constraints.push(Constraint::Distance {
            a: centre,
            b: ring[0],
            value: r,
        });
        // The corner point itself goes: nothing is on it any more.
        let mut gone = vec![false; pts.len()];
        gone[p as usize] = true;
        let mut map = vec![u32::MAX; pts.len()];
        let mut kept = Vec::new();
        for (i, q) in pts.iter().enumerate() {
            if !gone[i] {
                map[i] = kept.len() as u32;
                kept.push(*q);
            }
        }
        let renumber = |v: u32| map[v as usize];
        let constraints: Vec<Constraint> =
            constraints.iter().map(|c| c.remapped(&renumber)).collect();
        let lines: Vec<[u32; 2]> = lines
            .iter()
            .filter(|l| !gone[l[0] as usize] && !gone[l[1] as usize])
            .map(|l| [renumber(l[0]), renumber(l[1])])
            .collect();
        let mesh_id = object.mesh().expect("a sketch is a mesh");
        let positions: Vec<Vec3> = kept
            .iter()
            .map(|q| Vec3::new(q.x as f32, q.y as f32, 0.0))
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
}

// --- points ------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PointParams {
    #[param(unit = "length")]
    pub at: [f64; 3],
}

/// A point on its own, to dimension to or revolve about.
pub struct Point;

impl Operator for Point {
    type Params = PointParams;
    fn id() -> &'static str {
        "sketch.point"
    }
    fn label() -> &'static str {
        "Point"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let mut pts = points(doc.document(), id);
        pts.push(xy(params.at));
        commit(doc, id, pts, Vec::new(), Vec::new())
    }
}

// --- constraints and dimensions, by point -------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ConstrainParams {
    /// What kind: see `kreate_scene::sketch::KINDS`.
    #[param(variants(
        "distance",
        "horizontal",
        "vertical",
        "fixed",
        "angle",
        "parallel",
        "perpendicular",
        "equal",
        "coincident",
        "on_line",
        "circle",
        "arc",
        "tangent"
    ))]
    pub kind: u32,
    /// The points it is on, in the order the kind names them.
    #[param(hidden)]
    pub points: Vec<u32>,
    /// Its dimension, for a distance or an angle (radians); zero takes it as
    /// it is drawn.
    pub value: f64,
}

/// Adds a constraint or a dimension on the points the tool picked, and
/// solves: the sketch tools' way in, where `sketch.constraint_add` reads the
/// edit-mode selection.
pub struct Constrain;

impl Operator for Constrain {
    type Params = ConstrainParams;
    fn id() -> &'static str {
        "sketch.constrain"
    }
    fn label() -> &'static str {
        "Constrain"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let pts = points(doc.document(), id);
        if params.points.iter().any(|p| *p as usize >= pts.len()) {
            return Err(OpError::Failed("a point that is not in the sketch".into()));
        }
        let tag = KINDS
            .get(params.kind as usize)
            .copied()
            .ok_or_else(|| OpError::Failed("no such constraint".into()))?;
        let given = (params.value != 0.0 && params.value.is_finite()).then_some(params.value);
        let p = &params.points;
        let at = |i: usize| pts[p[i] as usize];
        let value = match tag {
            "distance" if p.len() >= 2 => given.unwrap_or_else(|| at(0).distance(at(1))),
            "angle" if p.len() >= 4 => given.unwrap_or_else(|| {
                let (u, v) = (at(1) - at(0), at(3) - at(2));
                u.perp_dot(v).atan2(u.dot(v))
            }),
            _ => 0.0,
        };
        let held = p.first().map_or([0.0; 2], |a| pts[*a as usize].to_array());
        let c = Constraint::from_parts(tag, p, value, held)
            .ok_or_else(|| OpError::Failed(format!("{tag} needs other points")))?;
        doc.document_mut()
            .object_mut(id)
            .expect("the sketch")
            .sketch
            .push(c);
        match solve(doc, id) {
            Some(s) if !s.converged => Err(OpError::Failed(
                "that contradicts what is already there".into(),
            )),
            _ => Ok(OpResult::Finished),
        }
    }
}

// --- moving and deleting -----------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MovePointsParams {
    #[param(hidden)]
    pub points: Vec<u32>,
    #[param(unit = "length")]
    pub offset: [f64; 3],
}

/// Drags points: they go where they are taken, as nearly as the
/// constraints let them, and the rest of the sketch follows.
pub struct MovePoints;

impl Operator for MovePoints {
    type Params = MovePointsParams;
    fn id() -> &'static str {
        "sketch.move"
    }
    fn label() -> &'static str {
        "Move Sketch Points"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let mut pts = points(doc.document(), id);
        let offset = xy(params.offset);
        let mut held = doc
            .document()
            .object(id)
            .expect("the sketch")
            .sketch
            .clone();
        for p in &params.points {
            if let Some(q) = pts.get_mut(*p as usize) {
                *q += offset;
                // Held where it was taken while the rest settles round it.
                held.push(Constraint::Fixed {
                    a: *p,
                    at: q.to_array(),
                });
            }
        }
        kreate_scene::sketch::solve(&mut pts, &held);
        set_points(doc, id, &pts).ok_or_else(|| OpError::Failed("no sketch".into()))?;
        // Then the sketch's own constraints alone, which win where they and
        // the drag disagree.
        solve(doc, id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteParams {
    /// Lines to take out, by their place in the sketch's list.
    #[param(hidden)]
    pub lines: Vec<u32>,
    /// Points to take out; with them every line and constraint on them.
    #[param(hidden)]
    pub points: Vec<u32>,
}

/// Takes lines and points out of the sketch, and what hangs on them: a point
/// no line uses any more goes too, and a constraint on a point that went.
/// The origin stays.
pub struct Delete;

impl Operator for Delete {
    type Params = DeleteParams;
    fn id() -> &'static str {
        "sketch.delete"
    }
    fn label() -> &'static str {
        "Delete from Sketch"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let pts = points(doc.document(), id);
        let object = doc.document().object(id).expect("the sketch");
        let mut gone = vec![false; pts.len()];
        for p in &params.points {
            if (*p as usize) < gone.len() && *p != 0 {
                gone[*p as usize] = true;
            }
        }
        // A line or constraint naming a point the mesh no longer has — its
        // vertices deleted in edit mode — goes too, rather than being read
        // past the end of the points.
        let missing = |v: u32| v as usize >= pts.len();
        let gone_at = |gone: &[bool], v: u32| gone.get(v as usize).copied().unwrap_or(true);
        let lines: Vec<[u32; 2]> = object
            .sketch_lines
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                !params.lines.contains(&(*i as u32))
                    && !gone_at(&gone, l[0])
                    && !gone_at(&gone, l[1])
            })
            .map(|(_, l)| *l)
            .collect();
        // A point that only the lines taken out used goes with them; a point
        // on its own, drawn as one, stays unless named.
        let mut used = vec![false; pts.len()];
        for l in &lines {
            used[l[0] as usize] = true;
            used[l[1] as usize] = true;
        }
        for (i, l) in object.sketch_lines.iter().enumerate() {
            if params.lines.contains(&(i as u32)) {
                for v in l {
                    if !missing(*v) && !used[*v as usize] && *v != 0 {
                        gone[*v as usize] = true;
                    }
                }
            }
        }
        if !gone.iter().any(|g| *g) && lines.len() == object.sketch_lines.len() {
            return Ok(OpResult::Cancelled);
        }
        // Renumbered without the points that went.
        let mut map = vec![u32::MAX; pts.len()];
        let mut kept = Vec::new();
        for (i, p) in pts.iter().enumerate() {
            if !gone[i] {
                map[i] = kept.len() as u32;
                kept.push(*p);
            }
        }
        let renumber = |v: u32| map[v as usize];
        let constraints: Vec<Constraint> = object
            .sketch
            .iter()
            .filter(|c| c.points().iter().all(|v| !gone_at(&gone, *v)))
            .map(|c| c.remapped(&renumber))
            .collect();
        let lines: Vec<[u32; 2]> = lines
            .iter()
            .map(|l| [renumber(l[0]), renumber(l[1])])
            .collect();
        // Faces from a sketch drawn before lines were its own go with it.
        let mesh_id = object.mesh().expect("a sketch is a mesh");
        let positions: Vec<Vec3> = kept
            .iter()
            .map(|p| Vec3::new(p.x as f32, p.y as f32, 0.0))
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
}

// --- patterns ----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PatternParams {
    /// The points to repeat, with every line between two of them.
    #[param(hidden)]
    pub points: Vec<u32>,
    /// Lines to repeat, by their place in the sketch's list; their ends
    /// count as picked points.
    #[param(hidden)]
    pub lines: Vec<u32>,
    /// In rows and columns, or round a centre.
    #[param(variants("rectangular", "circular"))]
    pub kind: u32,
    /// How many along the first direction — or round the circle — the
    /// original counted.
    #[param(min = 1.0, max = 500.0)]
    pub count: u32,
    /// The step along the sketch's X.
    #[param(unit = "length")]
    pub spacing: f64,
    /// How many along the sketch's Y, the original counted.
    #[param(min = 1.0, max = 500.0)]
    pub count2: u32,
    /// The step along the sketch's Y.
    #[param(unit = "length")]
    pub spacing2: f64,
    /// What a circular pattern turns round, in the sketch's plane.
    #[param(unit = "length")]
    pub centre: [f64; 3],
    /// How far round a circular pattern goes, in radians. A full turn puts
    /// the copies evenly round it; less, the last copy lands on the angle.
    #[param(unit = "angle")]
    pub angle: f64,
}

impl Default for PatternParams {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            lines: Vec::new(),
            kind: 0,
            count: 3,
            spacing: 0.01,
            count2: 1,
            spacing2: 0.01,
            centre: [0.0; 3],
            angle: std::f64::consts::TAU,
        }
    }
}

/// Where each copy of a pattern goes: the moves that take the original to
/// each copy, the original's own left out.
pub fn pattern_moves(params: &PatternParams) -> Vec<kreate_math::glam::DAffine2> {
    use kreate_math::glam::DAffine2;
    let mut out = Vec::new();
    if params.kind == 1 {
        let n = params.count.clamp(1, 500);
        let full = (params.angle.abs() - std::f64::consts::TAU).abs() < 1e-9;
        let step = if full || n < 2 {
            params.angle / n as f64
        } else {
            params.angle / (n - 1) as f64
        };
        let c = xy(params.centre);
        for k in 1..n {
            out.push(
                DAffine2::from_translation(c)
                    * DAffine2::from_angle(step * k as f64)
                    * DAffine2::from_translation(-c),
            );
        }
    } else {
        for j in 0..params.count2.clamp(1, 500) {
            for i in 0..params.count.clamp(1, 500) {
                if i == 0 && j == 0 {
                    continue;
                }
                out.push(DAffine2::from_translation(DVec2::new(
                    params.spacing * i as f64,
                    params.spacing2 * j as f64,
                )));
            }
        }
    }
    out
}

/// What a pattern repeats: the points picked, the ends of the lines picked,
/// and the centre of any circle or arc wholly among them; then every line
/// with both ends in that set.
pub fn pattern_source(
    lines: &[[u32; 2]],
    constraints: &[Constraint],
    points: &[u32],
    picked_lines: &[u32],
    count: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut set: Vec<u32> = points
        .iter()
        .copied()
        .filter(|v| (*v as usize) < count)
        .collect();
    for l in picked_lines {
        if let Some(ends) = lines.get(*l as usize) {
            set.extend_from_slice(ends);
        }
    }
    for c in constraints {
        if let Constraint::Circle { centre, ring } | Constraint::Arc { centre, ring } = c {
            if !ring.is_empty() && ring.iter().all(|v| set.contains(v)) {
                set.push(*centre);
            }
        }
    }
    set.retain(|v| (*v as usize) < count);
    set.sort_unstable();
    set.dedup();
    let with: Vec<u32> = lines
        .iter()
        .enumerate()
        .filter(|(_, [a, b])| set.contains(a) && set.contains(b))
        .map(|(i, _)| i as u32)
        .collect();
    (set, with)
}

/// Copies of picked lines and points in rows and columns, or round a
/// centre — Fusion's Rectangular and Circular Pattern for a sketch. The
/// copies keep the shape the picked ones have: a circle stays a circle,
/// lines that were parallel stay so; the dimensions and fixings stay with
/// the original. One step.
pub struct Pattern;

impl Operator for Pattern {
    type Params = PatternParams;
    fn id() -> &'static str {
        "sketch.pattern"
    }
    fn label() -> &'static str {
        "Sketch Pattern"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_sketch(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let id = need_sketch(doc.document())?;
        let mut pts = points(doc.document(), id);
        let object = doc.document().object(id).expect("the sketch");
        let (set, with) = pattern_source(
            &object.sketch_lines,
            &object.sketch,
            &params.points,
            &params.lines,
            pts.len(),
        );
        let moves = pattern_moves(params);
        if set.is_empty() || moves.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        if (set.len() + with.len()) * moves.len() > 200_000 {
            return Err(OpError::Failed("that pattern is too large".into()));
        }
        let turning = params.kind == 1;
        // The shape, not where it is: kept on every copy. Level and upright
        // only where the copy is not turned.
        let shape: Vec<Constraint> = object
            .sketch
            .iter()
            .filter(|c| c.points().iter().all(|v| set.contains(v)))
            .filter(|c| match c {
                Constraint::Horizontal { .. } | Constraint::Vertical { .. } => !turning,
                Constraint::Distance { .. }
                | Constraint::Angle { .. }
                | Constraint::Fixed { .. } => false,
                _ => true,
            })
            .cloned()
            .collect();
        let source_lines: Vec<[u32; 2]> = with
            .iter()
            .map(|l| object.sketch_lines[*l as usize])
            .collect();
        let mut lines = Vec::new();
        let mut constraints = Vec::new();
        for m in &moves {
            let mut map = std::collections::HashMap::new();
            for v in &set {
                map.insert(*v, pts.len() as u32);
                pts.push(m.transform_point2(pts[*v as usize]));
            }
            let to = |v: u32| map.get(&v).copied().unwrap_or(v);
            lines.extend(source_lines.iter().map(|[a, b]| [to(*a), to(*b)]));
            constraints.extend(shape.iter().map(|c| c.remapped(&to)));
        }
        commit(doc, id, pts, lines, constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};

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

    fn setup() -> (OperatorRegistry, History, Document) {
        let mut registry = OperatorRegistry::new();
        crate::register_builtin(&mut registry);
        (registry, History::new(), Document::new())
    }

    /// A sketch on a box's top: in the face's plane at the point given, the
    /// top's square in it, held; by the face's number, the same plane; and
    /// a point off the part, refused.
    #[test]
    fn a_sketch_on_a_face_lies_on_it_with_its_outline() {
        let (r, mut h, mut doc) = setup();
        let part = doc.add_mesh_object("Part", kreate_mesh::primitives::cube(0.04));
        doc.object_mut(part).unwrap().transform.translation = DVec3::new(0.1, 0.0, 0.0);
        doc.selection.add(part);
        doc.selection.active = Some(part);
        let at = [0.105, 0.003, 0.02];
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[
                ("plane", PropertyValue::Enum(4)),
                ("at", PropertyValue::Vec3(at)),
            ],
        )
        .unwrap();
        let s = doc.selection.active.unwrap();
        assert_ne!(s, part);
        let o = doc.object(s).unwrap();
        let w = doc.world_transform(s);
        assert!(w.translation.distance(DVec3::from_array(at)) < 1e-9);
        assert!((w.transform_vector3(DVec3::Z) - DVec3::Z).length() < 1e-9);
        assert_eq!(o.parent, Some(part), "a child of the part");
        assert_eq!(o.sketch_lines.len(), 4, "the top's square");
        assert_eq!(o.sketch.len(), 5, "the origin and four corners, held");
        // The square's corners are the top's, in the world.
        let world = doc.world_transform(s);
        let corners: Vec<DVec3> = points(&doc, s)[1..]
            .iter()
            .map(|p| world.transform_point3(DVec3::new(p.x, p.y, 0.0)))
            .collect();
        for c in &corners {
            assert!(
                (c.z - 0.02).abs() < 1e-9 && ((c.x - 0.1).abs() - 0.02).abs() < 1e-6,
                "{c}"
            );
        }
        // Off the part: said, nothing made.
        doc.selection.active = Some(part);
        let err = run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[
                ("plane", PropertyValue::Enum(4)),
                ("object", PropertyValue::Text("Part".into())),
                ("at", PropertyValue::Vec3([1.0, 1.0, 1.0])),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("no face"), "{err}");
    }

    /// A sketch on a face goes with the part; after the part's top is
    /// pushed up, Reattach puts it back on the top, what was drawn kept and
    /// the outline the top's again.
    #[test]
    fn a_sketch_on_a_face_follows_the_part_and_is_put_back_on_it() {
        let (r, mut h, mut doc) = setup();
        let part = doc.add_mesh_object("Part", kreate_mesh::primitives::cube(0.04));
        doc.selection.add(part);
        doc.selection.active = Some(part);
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[
                ("plane", PropertyValue::Enum(4)),
                ("at", PropertyValue::Vec3([0.0, 0.0, 0.02])),
            ],
        )
        .unwrap();
        let s = doc.selection.active.unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.005, 0.0, 0.0])),
                ("radius", PropertyValue::Float(0.003)),
            ],
        )
        .ok();
        let drawn = points(&doc, s).len();
        // The part moved: the sketch with it.
        doc.object_mut(part).unwrap().transform.translation = DVec3::new(0.1, 0.0, 0.0);
        assert!((doc.world_transform(s).translation - DVec3::new(0.1, 0.0, 0.02)).length() < 1e-9);
        // Its top pushed up 10 mm: back onto it.
        let m = doc.object(part).unwrap().mesh().unwrap();
        for p in doc.mesh_mut(m).unwrap().positions_mut() {
            if p.z > 0.0 {
                p.z += 0.01;
            }
        }
        run(&r, &mut h, &mut doc, "sketch.reattach", &[]).unwrap();
        let w = doc.world_transform(s);
        assert!(
            (w.translation - DVec3::new(0.1, 0.0, 0.03)).length() < 1e-9,
            "{}",
            w.translation
        );
        assert_eq!(
            points(&doc, s).len(),
            drawn,
            "the same points: the outline swapped, the rest kept"
        );
        for p in points(&doc, s).iter().skip(1) {
            let at = w.transform_point3(DVec3::new(p.x, p.y, 0.0));
            assert!((at.z - 0.03).abs() < 1e-9, "{at}");
        }
    }

    /// A round wall is sketched on where it was clicked, on the plane that
    /// touches it there.
    #[test]
    fn a_click_on_a_curved_face_sketches_on_its_tangent_plane() {
        let (r, mut h, mut doc) = setup();
        let post = doc.add_mesh_object("Post", kreate_mesh::primitives::cylinder(0.02, 0.05, 32));
        doc.selection.add(post);
        doc.selection.active = Some(post);
        crate::report::take();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[
                ("plane", PropertyValue::Enum(4)),
                ("at", PropertyValue::Vec3([0.02, 0.0, 0.0])),
            ],
        )
        .unwrap();
        let s = doc.selection.active.unwrap();
        let n = doc.world_transform(s).transform_vector3(DVec3::Z);
        assert!(n.dot(DVec3::X) > 0.99, "{n}");
        assert!(doc.object(s).unwrap().sketch_lines.is_empty());
        assert!(crate::report::take().unwrap().contains("tangent"));
    }

    #[test]
    fn a_new_sketch_has_its_origin_held_and_lies_in_its_plane() {
        let (r, mut h, mut doc) = setup();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.new",
            &[("plane", PropertyValue::Enum(1))],
        )
        .unwrap();
        let id = active_sketch(&doc).expect("a sketch");
        assert_eq!(points(&doc, id), vec![DVec2::ZERO]);
        // The front plane: the sketch's Y is the world's Z.
        let up = doc.world_transform(id).transform_vector3(DVec3::Y);
        assert!((up - DVec3::Z).length() < 1e-12, "{up:?}");
    }

    #[test]
    fn a_line_drawn_nearly_level_is_made_level_and_joins_where_it_lands() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        // From the origin, nearly level, then nearly upright.
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.line",
            &[
                (
                    "points",
                    PropertyValue::Floats(vec![0.0, 0.0, 0.03, 0.0004, 0.0302, 0.02]),
                ),
                ("joins", PropertyValue::Ints(vec![1, 0, 0])),
            ],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        assert_eq!(o.sketch_lines, vec![[0, 1], [1, 2]], "from the origin");
        let p = points(&doc, id);
        assert!(p[1].y.abs() < 1e-7, "level: {p:?}");
        assert!((p[2].x - p[1].x).abs() < 1e-7, "upright: {p:?}");
        assert_eq!(h.entries().len(), 2, "one step a stroke");
    }

    #[test]
    fn a_circle_patterned_in_rows_and_columns_is_six_circles_one_step() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.01, 0.01, 0.0])),
                ("radius", PropertyValue::Float(0.002)),
            ],
        )
        .unwrap();
        let circle_lines: Vec<i64> =
            (0..doc.object(id).unwrap().sketch_lines.len() as i64).collect();
        let before = points(&doc, id).len();
        let steps = h.entries().len();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.pattern",
            &[
                (
                    "lines",
                    PropertyValue::Ints(circle_lines.iter().map(|l| *l as u32).collect()),
                ),
                ("count", PropertyValue::Int(3)),
                ("spacing", PropertyValue::Float(0.01)),
                ("count2", PropertyValue::Int(2)),
                ("spacing2", PropertyValue::Float(0.02)),
            ],
        )
        .unwrap();
        assert_eq!(h.entries().len(), steps + 1, "one step");
        let o = doc.object(id).unwrap();
        let circles: Vec<_> = o
            .sketch
            .iter()
            .filter_map(|c| match c {
                Constraint::Circle { centre, .. } => Some(*centre),
                _ => None,
            })
            .collect();
        assert_eq!(circles.len(), 6, "every copy is still a circle");
        let p = points(&doc, id);
        // The origin aside, the circle's points six times over.
        assert_eq!(p.len() - 1, (before - 1) * 6);
        let mut centres: Vec<(i64, i64)> = circles
            .iter()
            .map(|c| {
                let q = p[*c as usize];
                ((q.x * 1e4).round() as i64, (q.y * 1e4).round() as i64)
            })
            .collect();
        centres.sort();
        assert_eq!(
            centres,
            vec![
                (100, 100),
                (100, 300),
                (200, 100),
                (200, 300),
                (300, 100),
                (300, 300)
            ]
        );
    }

    #[test]
    fn a_line_patterned_round_a_centre_turns_with_each_copy() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.line",
            &[("points", PropertyValue::Floats(vec![0.01, 0.0, 0.02, 0.0]))],
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.pattern",
            &[
                ("lines", PropertyValue::Ints(vec![0])),
                ("kind", PropertyValue::Enum(1)),
                ("count", PropertyValue::Int(4)),
            ],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        assert_eq!(o.sketch_lines.len(), 4);
        // Level once, not on the turned copies.
        let level = o
            .sketch
            .iter()
            .filter(|c| matches!(c, Constraint::Horizontal { .. }))
            .count();
        assert_eq!(level, 1);
        let p = points(&doc, id);
        let [a, b] = o.sketch_lines[1];
        // A quarter turn: from (0, 0.01) to (0, 0.02).
        assert!(
            (p[a as usize] - DVec2::new(0.0, 0.01)).length() < 1e-9,
            "{:?}",
            p[a as usize]
        );
        assert!((p[b as usize] - DVec2::new(0.0, 0.02)).length() < 1e-9);
    }

    #[test]
    fn deleting_from_a_sketch_whose_lines_name_missing_points_does_not_crash() {
        // Its vertices deleted in edit mode, a sketch's lines and constraints
        // can name points it no longer has; deleting must clear them, not
        // read past the end.
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.line",
            &[("points", PropertyValue::Floats(vec![0.01, 0.0, 0.02, 0.01]))],
        )
        .unwrap();
        let o = doc.object_mut(id).unwrap();
        o.sketch_lines.push([1, 999]);
        o.sketch.push(Constraint::Horizontal { a: 1, b: 999 });
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.delete",
            &[("points", PropertyValue::Ints(vec![2]))],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        let n = points(&doc, id).len() as u32;
        assert!(
            o.sketch_lines.iter().flatten().all(|v| *v < n),
            "{:?}",
            o.sketch_lines
        );
        assert!(o.sketch.iter().all(|c| c.points().iter().all(|v| *v < n)));
    }

    /// A spline through three points is held to them: move the middle one
    /// and the ring follows; closed, it is a region.
    #[test]
    fn a_spline_follows_the_points_it_passes_through() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.spline",
            &[
                (
                    "points",
                    PropertyValue::Floats(vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0]),
                ),
                ("joins", PropertyValue::Ints(vec![1, 0, 0])),
                ("segments", PropertyValue::Int(4)),
            ],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        // Through the origin and two points, 4 ring points a span: 2 spans.
        assert_eq!(points(&doc, id).len(), 1 + 2 + 8);
        assert_eq!(o.sketch_lines.len(), 10);
        let spline = o
            .sketch
            .iter()
            .find(|c| matches!(c, Constraint::Spline { .. }))
            .expect("a spline");
        let Constraint::Spline { through, ring, .. } = spline.clone() else {
            unreachable!()
        };
        assert_eq!((through.len(), ring.len()), (3, 8));
        // The middle point moved up: the ring near it goes up with it.
        let before = points(&doc, id)[ring[3] as usize];
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.move",
            &[
                ("points", PropertyValue::Ints(vec![through[1]])),
                ("offset", PropertyValue::Vec3([0.0, 1.0, 0.0])),
            ],
        )
        .unwrap();
        let after = points(&doc, id)[ring[3] as usize];
        assert!(after.y > before.y + 0.5, "{before:?} -> {after:?}");
        // Closed: one region to extrude.
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.spline",
            &[
                (
                    "points",
                    PropertyValue::Floats(vec![5.0, 0.0, 6.0, 0.0, 6.0, 1.0, 5.0, 1.0]),
                ),
                ("closed", PropertyValue::Bool(true)),
            ],
        )
        .unwrap();
        let regions = kreate_scene::feature::sketch_regions(&doc, id);
        assert!(!regions.is_empty(), "a closed spline encloses a region");
    }

    #[test]
    fn a_polygon_is_regular_and_stays_so_when_a_corner_is_pulled() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.polygon",
            &[
                ("centre", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("radius", PropertyValue::Float(1.0)),
                ("sides", PropertyValue::Int(6)),
                ("join", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        assert_eq!(o.sketch_lines.len(), 6);
        assert_eq!(
            points(&doc, id).len(),
            7,
            "the origin as centre, six corners"
        );
        // Pulled out: still regular, larger.
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.move",
            &[
                ("points", PropertyValue::Ints(vec![1])),
                ("offset", PropertyValue::Vec3([0.5, 0.0, 0.0])),
            ],
        )
        .unwrap();
        let p = points(&doc, id);
        let radii: Vec<f64> = (1..7).map(|i| p[i].length()).collect();
        assert!(radii.iter().all(|r| (r - 1.5).abs() < 1e-4), "{radii:?}");
        assert_eq!(kreate_scene::feature::sketch_regions(&doc, id).len(), 1);
    }

    #[test]
    fn a_slot_is_closed_and_its_ends_are_round() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.slot",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([4.0, 0.0, 0.0])),
                ("width", PropertyValue::Float(2.0)),
                ("segments", PropertyValue::Int(8)),
            ],
        )
        .unwrap();
        let regions = kreate_scene::feature::sketch_regions(&doc, id);
        assert_eq!(regions.len(), 1, "closed");
        // Two round ends of radius 1 and a 4 x 2 middle: 8 + π, near.
        let area = regions[0].area().abs() as f64;
        let want = 8.0 + std::f64::consts::PI;
        assert!((area - want).abs() < want * 0.03, "{area} vs {want}");
        let p = points(&doc, id);
        assert!(p
            .iter()
            .all(|q| q.y.abs() <= 1.0 + 1e-6 && q.x >= -1.0 - 1e-6 && q.x <= 5.0 + 1e-6));
    }

    #[test]
    fn a_fillet_rounds_a_corner_tangent_to_both_lines() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        // An L: along X then up Y, the corner at (2, 0).
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.line",
            &[(
                "points",
                PropertyValue::Floats(vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0]),
            )],
        )
        .unwrap();
        let corner = points(&doc, id)
            .iter()
            .position(|p| (p.x - 2.0).abs() < 1e-9 && p.y.abs() < 1e-9)
            .unwrap() as i64;
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.fillet",
            &[
                ("point", PropertyValue::Int(corner)),
                ("radius", PropertyValue::Float(0.5)),
            ],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        let p = points(&doc, id);
        assert!(
            !p.iter()
                .any(|q| (q.x - 2.0).abs() < 1e-9 && q.y.abs() < 1e-9),
            "the corner went"
        );
        let arc = o
            .sketch
            .iter()
            .find_map(|c| match c {
                Constraint::Arc { centre, ring } => Some((*centre, ring.clone())),
                _ => None,
            })
            .expect("an arc");
        assert!(
            (p[arc.0 as usize] - DVec2::new(1.5, 0.5)).length() < 1e-6,
            "{:?}",
            p[arc.0 as usize]
        );
        for v in &arc.1 {
            assert!(((p[*v as usize] - DVec2::new(1.5, 0.5)).length() - 0.5).abs() < 1e-6);
        }
        assert_eq!(
            o.sketch
                .iter()
                .filter(|c| matches!(c, Constraint::Tangent { .. }))
                .count(),
            2
        );
        // Too big for the lines: said so.
        let other = points(&doc, id).len() as i64; // no such point
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "sketch.fillet",
            &[("point", PropertyValue::Int(other))]
        )
        .is_err());
    }

    #[test]
    fn a_rectangle_dimensioned_and_a_circle_make_a_plate_with_a_hole() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.rectangle",
            &[
                ("a", PropertyValue::Vec3([0.0, 0.0, 0.0])),
                ("b", PropertyValue::Vec3([0.037, 0.022, 0.0])),
                ("join", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        // Corners 0 (the origin), 1, 2, 3: dimension two sides.
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
        let p = points(&doc, id);
        assert!((p[2] - DVec2::new(0.04, 0.02)).length() < 1e-9, "{p:?}");
        let pinned = kreate_scene::sketch::pinned(&p, &doc.object(id).unwrap().sketch);
        assert!(pinned[..4].iter().all(|f| *f), "{pinned:?}");

        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.circle",
            &[
                ("centre", PropertyValue::Vec3([0.02, 0.01, 0.0])),
                ("radius", PropertyValue::Float(0.004)),
            ],
        )
        .unwrap();
        let regions = kreate_scene::feature::sketch_regions(&doc, id);
        assert_eq!(regions.len(), 2, "the plate and the disc");
    }

    #[test]
    fn an_arc_passes_through_its_middle_point() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        // A half circle over the top, drawn right to left.
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.arc",
            &[
                ("start", PropertyValue::Vec3([1.0, 0.0, 0.0])),
                ("through", PropertyValue::Vec3([0.0, 1.0, 0.0])),
                ("end", PropertyValue::Vec3([-1.0, 0.0, 0.0])),
            ],
        )
        .unwrap();
        let p = points(&doc, id);
        assert!(
            p.iter().skip(2).all(|q| (q.length() - 1.0).abs() < 1e-6),
            "{p:?}"
        );
        assert!(
            p.iter().skip(2).all(|q| q.y >= -1e-9),
            "over the top: {p:?}"
        );
    }

    #[test]
    fn deleting_a_line_takes_its_loose_points_and_their_constraints() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "sketch.new", &[]).unwrap();
        let id = active_sketch(&doc).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.line",
            &[("points", PropertyValue::Floats(vec![0.1, 0.1, 0.2, 0.1]))],
        )
        .unwrap();
        assert_eq!(points(&doc, id).len(), 3);
        run(
            &r,
            &mut h,
            &mut doc,
            "sketch.delete",
            &[("lines", PropertyValue::Ints(vec![0]))],
        )
        .unwrap();
        let o = doc.object(id).unwrap();
        assert!(o.sketch_lines.is_empty());
        assert_eq!(points(&doc, id).len(), 1, "the origin stays");
        assert_eq!(
            o.sketch.len(),
            1,
            "the origin's hold, and not the line's level"
        );
    }
}
