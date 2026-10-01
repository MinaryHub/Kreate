//! Snapping to geometry: vertices, edges, midpoints and faces.
//!
//! # The rule that shapes everything here
//!
//! **The index excludes whatever is moving.** §2.4 states it and it is not a
//! detail: a selection that can snap to itself locks solid the instant the drag
//! begins, because the nearest vertex to a moving vertex is itself, at distance
//! zero, always. Every candidate in a [`SnapWorld`] is geometry that will still
//! be where it is when the drag ends.
//!
//! # Why the index is built once, at the start
//!
//! Nothing outside the selection moves during a transform, so the candidate set
//! cannot change. Building it per frame would be the same answer at a hundred
//! times the cost, and the cost is proportional to the scene.
//!
//! # Why resolution is in screen space
//!
//! "Within fourteen pixels of the cursor" is a tolerance a user can see and that
//! behaves the same whether they are zoomed in on a bolt or looking at a
//! building. A world-space radius is neither: it catches everything when zoomed
//! out and nothing when zoomed in, and the user has to keep adjusting it. §4.9
//! makes the same argument for picking, and for the same reason — the pick
//! tolerance being in screen space is most of why picking feels right.
//!
//! Faces are the exception and are resolved by a ray, because a face is a
//! surface rather than a point: what the user means by snapping to one is the
//! place on it that they are pointing at.

use glam::{DVec3, Vec2, Vec4};
use kreate_render::Camera;

/// What a snap looks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Round the *change* to an increment. Needs no geometry.
    #[default]
    Increment,
    /// Whatever is under the cursor: a vertex if one is near, else an edge,
    /// else the face — Blender's Vertex + Edge + Face, what `B` turns on.
    Element,
    Vertex,
    Edge,
    /// The middle of an edge, which is what centring something on a face wants.
    EdgeMidpoint,
    /// The point on a face under the cursor.
    Face,
}

impl Mode {
    /// Whether this mode needs a [`SnapWorld`] to resolve.
    pub fn needs_geometry(self) -> bool {
        !matches!(self, Mode::Increment)
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Increment => "increment",
            Mode::Element => "vertex/edge/face",
            Mode::Vertex => "vertex",
            Mode::Edge => "edge",
            Mode::EdgeMidpoint => "midpoint",
            Mode::Face => "face",
        }
    }

    /// The next mode, for a key that cycles through them.
    pub fn next(self) -> Self {
        match self {
            Mode::Increment => Mode::Element,
            Mode::Element => Mode::Vertex,
            Mode::Vertex => Mode::Edge,
            Mode::Edge => Mode::EdgeMidpoint,
            Mode::EdgeMidpoint => Mode::Face,
            Mode::Face => Mode::Increment,
        }
    }
}

/// How near the cursor a candidate has to be, in viewport pixels.
///
/// Larger than the pick tolerance, because a snap is a deliberate request: the
/// user has chosen a snap mode and is looking for something to catch on.
pub const RADIUS_PX: f32 = 14.0;

/// The geometry a drag may snap to: everything that is not moving.
#[derive(Debug, Default, Clone)]
pub struct SnapWorld {
    pub verts: Vec<DVec3>,
    pub edges: Vec<[DVec3; 2]>,
    pub tris: Vec<[DVec3; 3]>,
}

impl SnapWorld {
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty() && self.edges.is_empty() && self.tris.is_empty()
    }

    /// Adds a mesh's geometry, in world space, skipping the vertices given.
    ///
    /// `moving` is what the transform is about to displace, sorted. An edge or a
    /// face with *any* moving vertex is skipped whole: half of it would move and
    /// the point snapped to would not be where it appeared to be.
    pub fn add_mesh(
        &mut self,
        mesh: &kreate_mesh::MeshData,
        to_world: &glam::DAffine3,
        moving: &[u32],
    ) {
        let is_moving = |v: u32| moving.binary_search(&v).is_ok();
        let world = |v: u32| -> DVec3 {
            let p = mesh.positions()[v as usize];
            to_world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64))
        };

        for v in 0..mesh.num_verts() {
            if !is_moving(v) {
                self.verts.push(world(v));
            }
        }
        for edge in mesh.edges() {
            if is_moving(edge[0]) || is_moving(edge[1]) {
                continue;
            }
            self.edges.push([world(edge[0]), world(edge[1])]);
        }
        for face in 0..mesh.num_faces() {
            let ring = mesh.face_verts(face);
            if ring.len() < 3 || ring.iter().any(|v| is_moving(*v)) {
                continue;
            }
            // A fan, which is enough for a ray test against a planar face and
            // close enough for one that is not. A face being snapped to is being
            // pointed at, so an exact triangulation would buy nothing a user
            // could see.
            for i in 1..ring.len() - 1 {
                self.tris
                    .push([world(ring[0]), world(ring[i]), world(ring[i + 1])]);
            }
        }
    }
}

/// What kind of thing a snap caught, for drawing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Vertex,
    Edge,
    Face,
}

impl Element {
    pub fn label(self) -> &'static str {
        match self {
            Element::Vertex => "vertex",
            Element::Edge => "edge",
            Element::Face => "face",
        }
    }
}

/// A point a drag can snap to.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub point: DVec3,
    pub element: Element,
    /// Distance from the cursor in pixels, for choosing between candidates.
    pub distance_px: f32,
}

/// Finds what the cursor is near, or `None` if nothing is within the radius.
pub fn resolve(
    world: &SnapWorld,
    mode: Mode,
    cursor: (f32, f32),
    camera: &Camera,
    viewport: (f32, f32),
) -> Option<Hit> {
    match mode {
        Mode::Increment => None,
        // A vertex beats the edges it ends and an edge beats the faces it
        // borders: the smaller thing is the more deliberate aim, and a vertex
        // could never be caught if the face around it always won.
        Mode::Element => nearest_point(world.verts.iter().copied(), cursor, camera, viewport)
            .or_else(|| nearest_on_edges(world, cursor, camera, viewport))
            .or_else(|| under_cursor(world, cursor, camera, viewport)),
        Mode::Vertex => nearest_point(world.verts.iter().copied(), cursor, camera, viewport),
        Mode::EdgeMidpoint => nearest_point(
            world.edges.iter().map(|e| (e[0] + e[1]) * 0.5),
            cursor,
            camera,
            viewport,
        )
        .map(|hit| Hit { element: Element::Edge, ..hit }),
        Mode::Edge => nearest_on_edges(world, cursor, camera, viewport),
        Mode::Face => under_cursor(world, cursor, camera, viewport),
    }
}

/// The nearest of a set of world points to the cursor, in screen space.
fn nearest_point(
    points: impl Iterator<Item = DVec3>,
    cursor: (f32, f32),
    camera: &Camera,
    viewport: (f32, f32),
) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for point in points {
        let Some(px) = project(point, camera, viewport) else {
            continue;
        };
        let distance = (px.0 - cursor.0).hypot(px.1 - cursor.1);
        if distance > RADIUS_PX {
            continue;
        }
        if best.is_none_or(|b| distance < b.distance_px) {
            best = Some(Hit {
                point,
                element: Element::Vertex,
                distance_px: distance,
            });
        }
    }
    best
}

/// The nearest point *along* an edge, rather than at its ends.
///
/// Resolved on the projected segment and then mapped back along the edge, so
/// that sliding towards a long edge catches the part of it beside the cursor
/// rather than whichever end happens to be closer.
fn nearest_on_edges(
    world: &SnapWorld,
    cursor: (f32, f32),
    camera: &Camera,
    viewport: (f32, f32),
) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for edge in &world.edges {
        let (Some(a), Some(b)) = (
            project(edge[0], camera, viewport),
            project(edge[1], camera, viewport),
        ) else {
            continue;
        };
        let ab = Vec2::new(b.0 - a.0, b.1 - a.1);
        let ap = Vec2::new(cursor.0 - a.0, cursor.1 - a.1);
        let length2 = ab.length_squared();
        // A degenerate projection — an edge seen end-on — has no direction to
        // slide along. Its endpoints are still candidates through the vertex
        // mode, so nothing is lost by skipping it here.
        if length2 < 1e-6 {
            continue;
        }
        let t = (ap.dot(ab) / length2).clamp(0.0, 1.0);
        let on_screen = (a.0 + ab.x * t, a.1 + ab.y * t);
        let distance = (on_screen.0 - cursor.0).hypot(on_screen.1 - cursor.1);
        if distance > RADIUS_PX {
            continue;
        }
        // The same fraction along the edge in world space. Not exact under
        // perspective — the screen midpoint of a receding edge is not its world
        // midpoint — and near enough for a snap the user is aiming by eye.
        let point = edge[0] + (edge[1] - edge[0]) * t as f64;
        if best.is_none_or(|best| distance < best.distance_px) {
            best = Some(Hit {
                point,
                element: Element::Edge,
                distance_px: distance,
            });
        }
    }
    best
}

/// Where a ray through the cursor first meets a face.
fn under_cursor(
    world: &SnapWorld,
    cursor: (f32, f32),
    camera: &Camera,
    viewport: (f32, f32),
) -> Option<Hit> {
    let aspect = (viewport.0 / viewport.1.max(1.0)) as f64;
    let ndc = glam::Vec2::new(
        cursor.0 / viewport.0.max(1.0) * 2.0 - 1.0,
        1.0 - cursor.1 / viewport.1.max(1.0) * 2.0,
    );
    let direction = camera.ray_direction(ndc, aspect);
    let origin = camera.eye();
    let direction = DVec3::new(direction.x as f64, direction.y as f64, direction.z as f64);

    let mut nearest: Option<(f64, DVec3)> = None;
    for tri in &world.tris {
        if let Some(t) = ray_triangle(origin, direction, tri) {
            if nearest.is_none_or(|(best, _)| t < best) {
                nearest = Some((t, origin + direction * t));
            }
        }
    }
    nearest.map(|(_, point)| Hit {
        point,
        element: Element::Face,
        // A face hit is under the cursor by construction, so there is no
        // distance to rank it by. Zero, which also means it beats nothing.
        distance_px: 0.0,
    })
}

/// Möller–Trumbore, double sided.
///
/// Double sided on purpose: a snap is aiming at a surface, and refusing the back
/// of one would make snapping stop working when the user orbits past it.
fn ray_triangle(origin: DVec3, direction: DVec3, tri: &[DVec3; 3]) -> Option<f64> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let h = direction.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-12 {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - tri[0];
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    (t > 1e-9).then_some(t)
}

/// A world point in viewport pixels, or `None` when it is behind the camera.
///
/// Camera-relative, because the view projection is built that way: §4.10 puts
/// the camera at the origin so that a scene far from the world origin does not
/// jitter, and a projection that forgot to subtract the eye would be wrong by
/// exactly the camera's position.
pub fn project(point: DVec3, camera: &Camera, viewport: (f32, f32)) -> Option<(f32, f32)> {
    let aspect = (viewport.0 / viewport.1.max(1.0)) as f64;
    let relative = point - camera.eye();
    let clip = camera.view_projection(aspect)
        * Vec4::new(relative.x as f32, relative.y as f32, relative.z as f32, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some((
        (ndc.x * 0.5 + 0.5) * viewport.0,
        (0.5 - ndc.y * 0.5) * viewport.1,
    ))
}
