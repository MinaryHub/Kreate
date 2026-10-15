//! Geometry nodes: a mesh made by a graph.
//!
//! Blender's Geometry Nodes, its first slice. An object can carry a graph that
//! runs after its modifier stack: Group Input is the mesh as the stack left
//! it, Group Output is what is drawn, and between them nodes transform, repeat,
//! subdivide, mirror and join geometry, or make some from nothing. One kind of
//! wire — geometry — and each node one output.
//!
//! The graph is data on the object, saved and copied with it, and evaluated
//! by the modifier cache like the stack, keyed on its input and its own value.
//!
//! # Fields
//!
//! Wires carry geometry, or a *field*: a number or a direction that is not
//! one value but a rule for one at every point — the point's position, its
//! normal, its index, a noise over space, and sums and products of those.
//! A field is worked out where it arrives: Set Position reads its offset at
//! each vertex of the geometry it moves, Instance on Points reads a scale and
//! a turn at each point it puts a copy on. Blender's fields, the same idea
//! with fewer nodes.
//!
//! # Curves
//!
//! A curve is a row of points in order, with no faces: the point layer
//! [`CURVE`] says which curve each point is on and [`CYCLIC`] whether that
//! curve closes. Made by the curve primitives, evened out by Resample, turned
//! into points (pointing along the curve) or into a tube.

use kreate_math::{DAffine3, DQuat, DVec3, Vec3};
use kreate_mesh::edit::join::{join, Part};
use kreate_mesh::modifiers::{
    array, mirror, subdivide, ArrayParams, MirrorParams, SubdivideParams,
};
use kreate_mesh::MeshData;

/// What a node does.
#[derive(Clone, Debug, PartialEq)]
pub enum GeoNodeKind {
    /// The object's mesh, as its modifier stack left it.
    Input,
    /// What the object shows.
    Output,
    /// Moved, turned (degrees, XYZ) and scaled.
    Transform {
        translate: [f32; 3],
        rotate: [f32; 3],
        scale: [f32; 3],
    },
    /// Copies along an offset, in metres.
    Array { count: u32, offset: [f32; 3] },
    /// Each level splits every edge; smoothed toward the limit surface.
    Subdivide { levels: u32, smooth: bool },
    /// Reflected across one plane: 0 X, 1 Y, 2 Z.
    Mirror { axis: u32 },
    /// Two geometries as one.
    Join,
    /// A shape from nothing: 0 cube, 1 UV sphere, 2 cylinder, 3 cone,
    /// 4 torus, 5 icosphere.
    Primitive { shape: u32, size: f32 },
    /// A sketch's outline pushed out along its Z: a solid, holes and all. See
    /// `kreate_mesh::profile`.
    ExtrudeProfile { height: f32 },
    /// A sketch's outline turned a full turn about its Y axis.
    RevolveProfile { segments: u32 },
    /// Points scattered over the faces, so many to a square metre: what
    /// Instance on Points puts copies on. Each remembers the face's normal.
    DistributePoints { density: f32, seed: u32 },
    /// A copy of the second input at every point of the first: sized,
    /// turned at random and stood on the surface's normal if asked.
    InstanceOnPoints {
        scale: f32,
        random_scale: f32,
        random_rotation: bool,
        align: bool,
        seed: u32,
    },
    /// Every vertex moved: by an offset, and along its normal by a noise
    /// that varies over the surface — a field, sampled where each vertex is.
    /// A vector field wired into its second input is added on top.
    SetPosition {
        offset: [f32; 3],
        noise: f32,
        noise_scale: f32,
        seed: u32,
    },
    /// Where each point is: a vector field.
    Position,
    /// Which way each point faces: a vector field.
    Normal,
    /// Each point's number, from zero: a number field.
    Index,
    /// One number everywhere.
    Value { value: f32 },
    /// One direction everywhere.
    Vector { value: [f32; 3] },
    /// Smooth noise read at a point — its position when nothing is wired —
    /// from -1 to 1.
    NoiseField { scale: f32, seed: u32 },
    /// Two numbers met by one of [`MATH_OPS`]; `a` and `b` are what an
    /// unwired input takes.
    Math { op: u32, a: f32, b: f32 },
    /// A direction times a number.
    VectorScale { scale: f32 },
    /// Three numbers as a direction.
    CombineXyz { x: f32, y: f32, z: f32 },
    /// A straight curve up Z.
    CurveLine { length: f32, resolution: u32 },
    /// A closed circle in XY.
    CurveCircle { radius: f32, resolution: u32 },
    /// A helix up Z.
    CurveSpiral {
        turns: f32,
        height: f32,
        radius: f32,
        resolution: u32,
    },
    /// Each curve given `count` points, evenly along it.
    ResampleCurve { count: u32 },
    /// The curves' points, each facing along its curve: for instancing.
    CurveToPoints,
    /// Each curve swept into a tube.
    CurveToMesh {
        radius: f32,
        segments: u32,
        caps: bool,
    },
}

/// What a wire carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoSocket {
    Geometry,
    /// A number at every point.
    Float,
    /// A direction at every point.
    Vector,
}

/// [`GeoNodeKind::Math`]'s operations, in its `op`'s order.
pub const MATH_OPS: [&str; 8] = [
    "add", "subtract", "multiply", "divide", "power", "minimum", "maximum", "sine",
];

/// The point layer a curve's points carry: which curve each is on.
pub const CURVE: &str = ".geo_curve";
/// The point layer saying the point's curve closes.
pub const CYCLIC: &str = ".geo_cyclic";

/// A field: a rule for a value at every point of a geometry.
#[derive(Clone, Debug, PartialEq)]
enum Field {
    Float(f32),
    Vector(Vec3),
    Position,
    Normal,
    Index,
    Noise {
        at: Box<Field>,
        scale: f32,
        seed: u32,
    },
    Math {
        op: u32,
        a: Box<Field>,
        b: Box<Field>,
    },
    Scale {
        v: Box<Field>,
        s: Box<Field>,
    },
    Combine(Box<[Field; 3]>),
}

/// The geometry a field is read on, and its normals once asked for.
struct FieldCtx<'a> {
    m: &'a MeshData,
    normals: std::cell::OnceCell<Vec<Vec3>>,
}

impl<'a> FieldCtx<'a> {
    fn new(m: &'a MeshData) -> Self {
        Self {
            m,
            normals: std::cell::OnceCell::new(),
        }
    }

    fn normals(&self) -> &[Vec3] {
        self.normals.get_or_init(|| point_normals(self.m))
    }
}

fn math(op: u32, a: f32, b: f32) -> f32 {
    match op {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => {
            if b == 0.0 {
                0.0
            } else {
                a / b
            }
        }
        4 => a.powf(b),
        5 => a.min(b),
        6 => a.max(b),
        _ => a.sin(),
    }
}

impl Field {
    fn floats(&self, c: &FieldCtx<'_>) -> Vec<f32> {
        let n = c.m.num_verts() as usize;
        match self {
            Field::Float(v) => vec![*v; n],
            Field::Index => (0..n).map(|i| i as f32).collect(),
            Field::Noise { at, scale, seed } => at
                .vectors(c)
                .iter()
                .map(|p| noise3(*p * *scale, *seed))
                .collect(),
            Field::Math { op, a, b } => a
                .floats(c)
                .into_iter()
                .zip(b.floats(c))
                .map(|(a, b)| math(*op, a, b))
                .collect(),
            // A direction where a number is wanted: its length.
            Field::Vector(_)
            | Field::Position
            | Field::Normal
            | Field::Scale { .. }
            | Field::Combine(_) => self.vectors(c).iter().map(|v| v.length()).collect(),
        }
    }

    fn vectors(&self, c: &FieldCtx<'_>) -> Vec<Vec3> {
        let n = c.m.num_verts() as usize;
        match self {
            Field::Vector(v) => vec![*v; n],
            Field::Position => c.m.positions().to_vec(),
            Field::Normal => c.normals().to_vec(),
            Field::Scale { v, s } => v
                .vectors(c)
                .into_iter()
                .zip(s.floats(c))
                .map(|(v, s)| v * s)
                .collect(),
            Field::Combine(f) => {
                let (x, y, z) = (f[0].floats(c), f[1].floats(c), f[2].floats(c));
                (0..n).map(|i| Vec3::new(x[i], y[i], z[i])).collect()
            }
            // A number where a direction is wanted: the same on every axis.
            Field::Float(_) | Field::Index | Field::Noise { .. } | Field::Math { .. } => {
                self.floats(c).into_iter().map(Vec3::splat).collect()
            }
        }
    }
}

/// What a node gives.
#[derive(Clone, Debug)]
enum Value {
    Geometry(MeshData),
    Field(Field),
}

/// Each point's normal: the one it was scattered with, or its faces'.
fn point_normals(m: &MeshData) -> Vec<Vec3> {
    match m.attributes().get_f32x3(POINT_NORMAL) {
        Some(n) if n.len() == m.num_verts() as usize => n.to_vec(),
        _ => vertex_normals(m),
    }
}

/// Curves as one geometry: their points in a row, each tagged with its
/// curve and whether that closes.
pub fn curves_mesh(curves: &[(Vec<Vec3>, bool)]) -> MeshData {
    let mut positions = Vec::new();
    let mut index = Vec::new();
    let mut cyclic = Vec::new();
    for (i, (points, closed)) in curves.iter().enumerate() {
        for p in points {
            positions.push(*p);
            index.push(i as i32);
            cyclic.push(*closed);
        }
    }
    let mut m = MeshData::from_faces(positions, &[]);
    m.attributes_mut().insert(
        CURVE,
        kreate_mesh::Domain::Point,
        kreate_mesh::AttrData::I32(std::sync::Arc::new(index)),
    );
    m.attributes_mut().insert(
        CYCLIC,
        kreate_mesh::Domain::Point,
        kreate_mesh::AttrData::Bool(std::sync::Arc::new(cyclic)),
    );
    m
}

/// The curves a geometry holds. Points with no faces and no curve layer are
/// one open curve, in their order; a mesh with faces holds none.
pub fn curves_of(m: &MeshData) -> Vec<(Vec<Vec3>, bool)> {
    let p = m.positions();
    let index = m.attributes().get_i32(CURVE).filter(|v| v.len() == p.len());
    let Some(index) = index else {
        return if m.num_faces() == 0 && p.len() >= 2 {
            vec![(p.to_vec(), false)]
        } else {
            Vec::new()
        };
    };
    let cyclic = m
        .attributes()
        .get_bool(CYCLIC)
        .filter(|v| v.len() == p.len());
    let mut out: Vec<(Vec<Vec3>, bool)> = Vec::new();
    let mut last = None;
    for (i, q) in p.iter().enumerate() {
        if last != Some(index[i]) {
            out.push((Vec::new(), cyclic.is_some_and(|c| c[i])));
            last = Some(index[i]);
        }
        out.last_mut().expect("one").0.push(*q);
    }
    out
}

/// `count` points evenly along a curve, by length.
fn resample(points: &[Vec3], closed: bool, count: u32) -> Vec<Vec3> {
    let count = count.max(if closed { 3 } else { 2 }) as usize;
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut ring: Vec<Vec3> = points.to_vec();
    if closed {
        ring.push(points[0]);
    }
    let mut along = vec![0.0f32];
    for w in ring.windows(2) {
        along.push(along.last().expect("one") + w[0].distance(w[1]));
    }
    let total = *along.last().expect("one");
    if total <= 0.0 {
        return vec![points[0]; count];
    }
    // A closed curve's last point is not its first again.
    let steps = if closed { count } else { count - 1 };
    (0..count)
        .map(|k| {
            let d = total * k as f32 / steps as f32;
            let i = along.partition_point(|a| *a <= d).clamp(1, ring.len() - 1);
            let (a, b) = (along[i - 1], along[i]);
            let t = if b > a { (d - a) / (b - a) } else { 0.0 };
            ring[i - 1].lerp(ring[i], t.clamp(0.0, 1.0))
        })
        .collect()
}

/// Each point's direction along its curve.
fn tangents(points: &[Vec3], closed: bool) -> Vec<Vec3> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let prev = if i > 0 {
                points[i - 1]
            } else if closed {
                points[n - 1]
            } else {
                points[i]
            };
            let next = if i + 1 < n {
                points[i + 1]
            } else if closed {
                points[0]
            } else {
                points[i]
            };
            (next - prev).normalize_or(Vec3::Z)
        })
        .collect()
}

/// The point layer [`GeoNodeKind::DistributePoints`] leaves each point's
/// normal in.
pub const POINT_NORMAL: &str = ".geo_normal";

/// Area-weighted vertex normals; a vertex on no face points up.
fn vertex_normals(m: &MeshData) -> Vec<Vec3> {
    let p = m.positions();
    let mut out = vec![Vec3::ZERO; p.len()];
    for f in 0..m.num_faces() {
        let ring = m.face_verts(f);
        let mut n = Vec3::ZERO;
        for i in 0..ring.len() {
            n += p[ring[i] as usize].cross(p[ring[(i + 1) % ring.len()] as usize]);
        }
        for v in ring {
            out[*v as usize] += n;
        }
    }
    out.into_iter().map(|n| n.normalize_or(Vec3::Z)).collect()
}

/// Points over `m`'s faces, `density` to a unit of area, by a seeded
/// scatter: vertices with no faces, each with its face's normal.
fn distribute_points(m: &MeshData, density: f32, seed: u32) -> MeshData {
    let surface = crate::physics::particles::Surface::of(m);
    let tri = m.triangulate();
    let area: f64 = tri
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| m.positions()[i as usize]);
            0.5 * (b - a).cross(c - a).length() as f64
        })
        .sum();
    let count = ((area * density.max(0.0) as f64).round() as usize).min(1_000_000);
    if surface.is_empty() || count == 0 {
        return MeshData::empty();
    }
    let mut rng = crate::physics::Rng::new(seed as u64 + 7);
    let mut positions = Vec::with_capacity(count);
    let mut normals = Vec::with_capacity(count);
    for _ in 0..count {
        let (p, n) = surface.sample(&mut rng);
        positions.push(p.as_vec3());
        normals.push(n.as_vec3());
    }
    let mut out = MeshData::from_faces(positions, &[]);
    out.attributes_mut().insert(
        POINT_NORMAL,
        kreate_mesh::Domain::Point,
        kreate_mesh::AttrData::F32x3(std::sync::Arc::new(normals)),
    );
    out
}

/// A copy of `shape` at every vertex of `points`.
#[allow(clippy::too_many_arguments)]
fn instance_on_points(
    points: &MeshData,
    shape: &MeshData,
    scale: f32,
    random_scale: f32,
    random_rotation: bool,
    align: bool,
    seed: u32,
    scales: Option<&[f32]>,
    turns: Option<&[Vec3]>,
) -> MeshData {
    if points.num_verts() == 0 || shape.num_verts() == 0 {
        return MeshData::empty();
    }
    let normals: Vec<Vec3> = match points.attributes().get_f32x3(POINT_NORMAL) {
        Some(n) if n.len() == points.num_verts() as usize => n.to_vec(),
        _ => vertex_normals(points),
    };
    let mut rng = crate::physics::Rng::new(seed as u64 + 11);
    let transforms: Vec<DAffine3> = points
        .positions()
        .iter()
        .zip(&normals)
        .enumerate()
        .map(|(i, (p, n))| {
            let up = if align {
                n.as_dvec3().normalize_or(DVec3::Z)
            } else {
                DVec3::Z
            };
            let stand = DQuat::from_rotation_arc(DVec3::Z, up);
            let spin = if random_rotation {
                DQuat::from_axis_angle(DVec3::Z, rng.unit() * std::f64::consts::TAU)
            } else {
                DQuat::IDENTITY
            };
            let s = scale as f64 * (1.0 - random_scale.clamp(0.0, 1.0) as f64 * rng.unit());
            let s = s * scales.and_then(|v| v.get(i)).map_or(1.0, |v| *v as f64);
            let turn = turns.and_then(|v| v.get(i)).map_or(DQuat::IDENTITY, |r| {
                DQuat::from_euler(
                    kreate_math::glam::EulerRot::ZYX,
                    r.z as f64,
                    r.y as f64,
                    r.x as f64,
                )
            });
            DAffine3::from_scale_rotation_translation(
                DVec3::splat(s),
                stand * spin * turn,
                p.as_dvec3(),
            )
        })
        .collect();
    let parts: Vec<Part<'_>> = transforms
        .into_iter()
        .map(|transform| Part {
            mesh: shape,
            transform,
        })
        .collect();
    join(&parts).0
}

/// Smooth noise in space, -1 to 1: a hashed value at each lattice corner,
/// blended. The field Set Position samples.
fn noise3(p: Vec3, seed: u32) -> f32 {
    let hash = |x: i32, y: i32, z: i32| {
        let mut h = (x as u32).wrapping_mul(0x8da6_b343)
            ^ (y as u32).wrapping_mul(0xd816_3841)
            ^ (z as u32).wrapping_mul(0xcb1a_b31f)
            ^ seed.wrapping_mul(0x2545_f491);
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1_e995);
        h ^= h >> 15;
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let f = p.floor();
    let t = p - f;
    let s = t * t * (Vec3::splat(3.0) - 2.0 * t);
    let (x, y, z) = (f.x as i32, f.y as i32, f.z as i32);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c = |dx: i32, dy: i32, dz: i32| hash(x + dx, y + dy, z + dz);
    lerp(
        lerp(
            lerp(c(0, 0, 0), c(1, 0, 0), s.x),
            lerp(c(0, 1, 0), c(1, 1, 0), s.x),
            s.y,
        ),
        lerp(
            lerp(c(0, 0, 1), c(1, 0, 1), s.x),
            lerp(c(0, 1, 1), c(1, 1, 1), s.x),
            s.y,
        ),
        s.z,
    )
}

/// Every vertex by `offset`, and along its normal by the noise there.
fn set_position(m: &mut MeshData, offset: Vec3, noise: f32, scale: f32, seed: u32) {
    let normals = vertex_normals(m);
    for (p, n) in m.positions_mut().iter_mut().zip(&normals) {
        let along = if noise != 0.0 {
            noise * noise3(*p * scale, seed)
        } else {
            0.0
        };
        *p += offset + *n * along;
    }
}

/// The primitive shapes, in [`GeoNodeKind::Primitive`]'s order.
pub const PRIMITIVES: [&str; 6] = [
    "cube",
    "uv_sphere",
    "cylinder",
    "cone",
    "torus",
    "ico_sphere",
];

impl GeoNodeKind {
    /// How many inputs it has.
    pub fn inputs(&self) -> usize {
        self.input_types().len()
    }

    /// What each input takes, in order.
    pub fn input_types(&self) -> &'static [GeoSocket] {
        use GeoSocket::{Float as F, Geometry as G, Vector as V};
        match self {
            GeoNodeKind::Input
            | GeoNodeKind::Primitive { .. }
            | GeoNodeKind::Position
            | GeoNodeKind::Normal
            | GeoNodeKind::Index
            | GeoNodeKind::Value { .. }
            | GeoNodeKind::Vector { .. }
            | GeoNodeKind::CurveLine { .. }
            | GeoNodeKind::CurveCircle { .. }
            | GeoNodeKind::CurveSpiral { .. } => &[],
            GeoNodeKind::Join => &[G, G],
            GeoNodeKind::InstanceOnPoints { .. } => &[G, G, F, V],
            GeoNodeKind::SetPosition { .. } => &[G, V],
            GeoNodeKind::NoiseField { .. } => &[V],
            GeoNodeKind::Math { .. } => &[F, F],
            GeoNodeKind::VectorScale { .. } => &[V, F],
            GeoNodeKind::CombineXyz { .. } => &[F, F, F],
            _ => &[G],
        }
    }

    /// The key tails of its inputs' labels: `geo-socket-<name>`.
    pub fn input_labels(&self) -> &'static [&'static str] {
        match self {
            GeoNodeKind::Join => &["geometry", "geometry"],
            GeoNodeKind::InstanceOnPoints { .. } => &["points", "instance", "scale", "rotation"],
            GeoNodeKind::SetPosition { .. } => &["geometry", "offset"],
            GeoNodeKind::NoiseField { .. } => &["vector"],
            GeoNodeKind::Math { .. } => &["value", "value"],
            GeoNodeKind::VectorScale { .. } => &["vector", "scale"],
            GeoNodeKind::CombineXyz { .. } => &["x", "y", "z"],
            GeoNodeKind::ResampleCurve { .. }
            | GeoNodeKind::CurveToPoints
            | GeoNodeKind::CurveToMesh { .. } => &["curve"],
            _ if self.inputs() == 1 => &["geometry"],
            _ => &[],
        }
    }

    /// Whether it has an output: every kind but the group's output.
    pub fn has_output(&self) -> bool {
        !matches!(self, GeoNodeKind::Output)
    }

    /// What its output gives, if it has one.
    pub fn output_type(&self) -> Option<GeoSocket> {
        Some(match self {
            GeoNodeKind::Output => return None,
            GeoNodeKind::Position
            | GeoNodeKind::Normal
            | GeoNodeKind::Vector { .. }
            | GeoNodeKind::VectorScale { .. }
            | GeoNodeKind::CombineXyz { .. } => GeoSocket::Vector,
            GeoNodeKind::Index
            | GeoNodeKind::Value { .. }
            | GeoNodeKind::NoiseField { .. }
            | GeoNodeKind::Math { .. } => GeoSocket::Float,
            _ => GeoSocket::Geometry,
        })
    }

    /// The key tail of its output's label.
    pub fn output_label(&self) -> &'static str {
        match self.output_type() {
            Some(GeoSocket::Float) => "value",
            Some(GeoSocket::Vector) => "vector",
            _ if matches!(
                self,
                GeoNodeKind::CurveLine { .. }
                    | GeoNodeKind::CurveCircle { .. }
                    | GeoNodeKind::CurveSpiral { .. }
                    | GeoNodeKind::ResampleCurve { .. }
            ) =>
            {
                "curve"
            }
            _ => "geometry",
        }
    }

    /// A stable name for files and scripts.
    pub fn tag(&self) -> &'static str {
        match self {
            GeoNodeKind::Input => "input",
            GeoNodeKind::Output => "output",
            GeoNodeKind::Transform { .. } => "transform",
            GeoNodeKind::Array { .. } => "array",
            GeoNodeKind::Subdivide { .. } => "subdivide",
            GeoNodeKind::Mirror { .. } => "mirror",
            GeoNodeKind::Join => "join",
            GeoNodeKind::Primitive { .. } => "primitive",
            GeoNodeKind::ExtrudeProfile { .. } => "extrude_profile",
            GeoNodeKind::RevolveProfile { .. } => "revolve_profile",
            GeoNodeKind::DistributePoints { .. } => "distribute_points",
            GeoNodeKind::InstanceOnPoints { .. } => "instance_on_points",
            GeoNodeKind::SetPosition { .. } => "set_position",
            GeoNodeKind::Position => "position",
            GeoNodeKind::Normal => "normal",
            GeoNodeKind::Index => "index",
            GeoNodeKind::Value { .. } => "value",
            GeoNodeKind::Vector { .. } => "vector",
            GeoNodeKind::NoiseField { .. } => "noise_field",
            GeoNodeKind::Math { .. } => "math",
            GeoNodeKind::VectorScale { .. } => "vector_scale",
            GeoNodeKind::CombineXyz { .. } => "combine_xyz",
            GeoNodeKind::CurveLine { .. } => "curve_line",
            GeoNodeKind::CurveCircle { .. } => "curve_circle",
            GeoNodeKind::CurveSpiral { .. } => "curve_spiral",
            GeoNodeKind::ResampleCurve { .. } => "resample_curve",
            GeoNodeKind::CurveToPoints => "curve_to_points",
            GeoNodeKind::CurveToMesh { .. } => "curve_to_mesh",
        }
    }

    /// The kinds the Add menu offers, in its order. Not the group's input or
    /// output: a graph has one of each.
    pub const ADDABLE: [&'static str; 26] = [
        "transform",
        "array",
        "subdivide",
        "mirror",
        "join",
        "primitive",
        "extrude_profile",
        "revolve_profile",
        "distribute_points",
        "instance_on_points",
        "set_position",
        "position",
        "normal",
        "index",
        "value",
        "vector",
        "noise_field",
        "math",
        "vector_scale",
        "combine_xyz",
        "curve_line",
        "curve_circle",
        "curve_spiral",
        "resample_curve",
        "curve_to_points",
        "curve_to_mesh",
    ];

    /// The node's numbers, flat — for a file, and for its fields by index.
    pub fn values(&self) -> Vec<f32> {
        match self {
            GeoNodeKind::Input
            | GeoNodeKind::Output
            | GeoNodeKind::Join
            | GeoNodeKind::Position
            | GeoNodeKind::Normal
            | GeoNodeKind::Index
            | GeoNodeKind::CurveToPoints => Vec::new(),
            GeoNodeKind::Value { value } => vec![*value],
            GeoNodeKind::Vector { value } => value.to_vec(),
            GeoNodeKind::NoiseField { scale, seed } => vec![*scale, *seed as f32],
            GeoNodeKind::Math { op, a, b } => vec![*op as f32, *a, *b],
            GeoNodeKind::VectorScale { scale } => vec![*scale],
            GeoNodeKind::CombineXyz { x, y, z } => vec![*x, *y, *z],
            GeoNodeKind::CurveLine { length, resolution } => vec![*length, *resolution as f32],
            GeoNodeKind::CurveCircle { radius, resolution } => vec![*radius, *resolution as f32],
            GeoNodeKind::CurveSpiral {
                turns,
                height,
                radius,
                resolution,
            } => vec![*turns, *height, *radius, *resolution as f32],
            GeoNodeKind::ResampleCurve { count } => vec![*count as f32],
            GeoNodeKind::CurveToMesh {
                radius,
                segments,
                caps,
            } => vec![*radius, *segments as f32, if *caps { 1.0 } else { 0.0 }],
            GeoNodeKind::Transform {
                translate,
                rotate,
                scale,
            } => translate
                .iter()
                .chain(rotate)
                .chain(scale)
                .copied()
                .collect(),
            GeoNodeKind::Array { count, offset } => {
                vec![*count as f32, offset[0], offset[1], offset[2]]
            }
            GeoNodeKind::Subdivide { levels, smooth } => {
                vec![*levels as f32, if *smooth { 1.0 } else { 0.0 }]
            }
            GeoNodeKind::Mirror { axis } => vec![*axis as f32],
            GeoNodeKind::Primitive { shape, size } => vec![*shape as f32, *size],
            GeoNodeKind::ExtrudeProfile { height } => vec![*height],
            GeoNodeKind::RevolveProfile { segments } => vec![*segments as f32],
            GeoNodeKind::DistributePoints { density, seed } => vec![*density, *seed as f32],
            GeoNodeKind::InstanceOnPoints {
                scale,
                random_scale,
                random_rotation,
                align,
                seed,
            } => vec![
                *scale,
                *random_scale,
                if *random_rotation { 1.0 } else { 0.0 },
                if *align { 1.0 } else { 0.0 },
                *seed as f32,
            ],
            GeoNodeKind::SetPosition {
                offset,
                noise,
                noise_scale,
                seed,
            } => vec![
                offset[0],
                offset[1],
                offset[2],
                *noise,
                *noise_scale,
                *seed as f32,
            ],
        }
    }

    /// A kind from its tag and numbers; missing numbers take the defaults.
    /// `None` for a tag this build does not know.
    pub fn from_parts(tag: &str, values: &[f32]) -> Option<Self> {
        let at = |i: usize, d: f32| {
            values
                .get(i)
                .copied()
                .filter(|v| v.is_finite())
                .unwrap_or(d)
        };
        Some(match tag {
            "input" => GeoNodeKind::Input,
            "output" => GeoNodeKind::Output,
            "transform" => GeoNodeKind::Transform {
                translate: [at(0, 0.0), at(1, 0.0), at(2, 0.0)],
                rotate: [at(3, 0.0), at(4, 0.0), at(5, 0.0)],
                scale: [at(6, 1.0), at(7, 1.0), at(8, 1.0)],
            },
            "array" => GeoNodeKind::Array {
                count: at(0, 3.0).clamp(1.0, kreate_mesh::modifiers::array::MAX_COUNT as f32)
                    as u32,
                offset: [at(1, 2.5), at(2, 0.0), at(3, 0.0)],
            },
            "subdivide" => GeoNodeKind::Subdivide {
                levels: at(0, 1.0).clamp(0.0, 4.0) as u32,
                smooth: at(1, 1.0) >= 0.5,
            },
            "mirror" => GeoNodeKind::Mirror {
                axis: at(0, 0.0).clamp(0.0, 2.0) as u32,
            },
            "join" => GeoNodeKind::Join,
            "primitive" => GeoNodeKind::Primitive {
                shape: at(0, 0.0).clamp(0.0, (PRIMITIVES.len() - 1) as f32) as u32,
                size: at(1, 1.0).max(0.001),
            },
            "extrude_profile" => GeoNodeKind::ExtrudeProfile { height: at(0, 1.0) },
            "revolve_profile" => GeoNodeKind::RevolveProfile {
                segments: at(0, 32.0).clamp(3.0, 512.0) as u32,
            },
            "distribute_points" => GeoNodeKind::DistributePoints {
                density: at(0, 10.0).clamp(0.0, 100_000.0),
                seed: at(1, 0.0).clamp(0.0, 1e6) as u32,
            },
            "instance_on_points" => GeoNodeKind::InstanceOnPoints {
                scale: at(0, 1.0),
                random_scale: at(1, 0.0).clamp(0.0, 1.0),
                random_rotation: at(2, 0.0) >= 0.5,
                align: at(3, 1.0) >= 0.5,
                seed: at(4, 0.0).clamp(0.0, 1e6) as u32,
            },
            "set_position" => GeoNodeKind::SetPosition {
                offset: [at(0, 0.0), at(1, 0.0), at(2, 0.0)],
                noise: at(3, 0.1),
                noise_scale: at(4, 2.0).max(0.0),
                seed: at(5, 0.0).clamp(0.0, 1e6) as u32,
            },
            "position" => GeoNodeKind::Position,
            "normal" => GeoNodeKind::Normal,
            "index" => GeoNodeKind::Index,
            "value" => GeoNodeKind::Value { value: at(0, 1.0) },
            "vector" => GeoNodeKind::Vector {
                value: [at(0, 0.0), at(1, 0.0), at(2, 1.0)],
            },
            "noise_field" => GeoNodeKind::NoiseField {
                scale: at(0, 2.0).max(0.0),
                seed: at(1, 0.0).clamp(0.0, 1e6) as u32,
            },
            "math" => GeoNodeKind::Math {
                op: at(0, 2.0).clamp(0.0, (MATH_OPS.len() - 1) as f32) as u32,
                a: at(1, 1.0),
                b: at(2, 1.0),
            },
            "vector_scale" => GeoNodeKind::VectorScale { scale: at(0, 1.0) },
            "combine_xyz" => GeoNodeKind::CombineXyz {
                x: at(0, 0.0),
                y: at(1, 0.0),
                z: at(2, 0.0),
            },
            "curve_line" => GeoNodeKind::CurveLine {
                length: at(0, 1.0),
                resolution: at(1, 16.0).clamp(2.0, 10_000.0) as u32,
            },
            "curve_circle" => GeoNodeKind::CurveCircle {
                radius: at(0, 1.0),
                resolution: at(1, 32.0).clamp(3.0, 10_000.0) as u32,
            },
            "curve_spiral" => GeoNodeKind::CurveSpiral {
                turns: at(0, 3.0),
                height: at(1, 2.0),
                radius: at(2, 1.0),
                resolution: at(3, 96.0).clamp(2.0, 10_000.0) as u32,
            },
            "resample_curve" => GeoNodeKind::ResampleCurve {
                count: at(0, 10.0).clamp(2.0, 10_000.0) as u32,
            },
            "curve_to_points" => GeoNodeKind::CurveToPoints,
            "curve_to_mesh" => GeoNodeKind::CurveToMesh {
                radius: at(0, 0.05).max(0.0),
                segments: at(1, 8.0).clamp(3.0, 256.0) as u32,
                caps: at(2, 1.0) >= 0.5,
            },
            _ => return None,
        })
    }

    /// The editable numbers: a label key's tail, the value, its range, and
    /// whether it picks from a list.
    pub fn fields(&self) -> Vec<GeoField> {
        let v = self.values();
        let number = |name: &'static str, i: usize, min: f32, max: f32| GeoField {
            name,
            index: i as u32,
            value: v[i],
            min,
            max,
            choice: None,
        };
        match self {
            GeoNodeKind::Input
            | GeoNodeKind::Output
            | GeoNodeKind::Join
            | GeoNodeKind::Position
            | GeoNodeKind::Normal
            | GeoNodeKind::Index
            | GeoNodeKind::CurveToPoints => Vec::new(),
            GeoNodeKind::Value { .. } => vec![number("value", 0, -1e6, 1e6)],
            GeoNodeKind::Vector { .. } => vec![
                number("x", 0, -1e6, 1e6),
                number("y", 1, -1e6, 1e6),
                number("z", 2, -1e6, 1e6),
            ],
            GeoNodeKind::NoiseField { .. } => {
                vec![number("scale", 0, 0.0, 1000.0), number("seed", 1, 0.0, 1e6)]
            }
            GeoNodeKind::Math { .. } => vec![
                GeoField {
                    choice: Some(&MATH_OPS),
                    ..number("operation", 0, 0.0, (MATH_OPS.len() - 1) as f32)
                },
                number("a", 1, -1e6, 1e6),
                number("b", 2, -1e6, 1e6),
            ],
            GeoNodeKind::VectorScale { .. } => vec![number("scale", 0, -1e6, 1e6)],
            GeoNodeKind::CombineXyz { .. } => vec![
                number("x", 0, -1e6, 1e6),
                number("y", 1, -1e6, 1e6),
                number("z", 2, -1e6, 1e6),
            ],
            GeoNodeKind::CurveLine { .. } => vec![
                number("length", 0, -1000.0, 1000.0),
                number("resolution", 1, 2.0, 10_000.0),
            ],
            GeoNodeKind::CurveCircle { .. } => vec![
                number("radius", 0, 0.0, 1000.0),
                number("resolution", 1, 3.0, 10_000.0),
            ],
            GeoNodeKind::CurveSpiral { .. } => vec![
                number("turns", 0, -1000.0, 1000.0),
                number("height", 1, -1000.0, 1000.0),
                number("radius", 2, 0.0, 1000.0),
                number("resolution", 3, 2.0, 10_000.0),
            ],
            GeoNodeKind::ResampleCurve { .. } => vec![number("count", 0, 2.0, 10_000.0)],
            GeoNodeKind::CurveToMesh { .. } => vec![
                number("radius", 0, 0.0, 1000.0),
                number("segments", 1, 3.0, 256.0),
                GeoField {
                    choice: Some(&["no", "yes"]),
                    ..number("caps", 2, 0.0, 1.0)
                },
            ],
            GeoNodeKind::Transform { .. } => vec![
                number("translate-x", 0, -1000.0, 1000.0),
                number("translate-y", 1, -1000.0, 1000.0),
                number("translate-z", 2, -1000.0, 1000.0),
                number("rotate-x", 3, -360.0, 360.0),
                number("rotate-y", 4, -360.0, 360.0),
                number("rotate-z", 5, -360.0, 360.0),
                number("scale-x", 6, -100.0, 100.0),
                number("scale-y", 7, -100.0, 100.0),
                number("scale-z", 8, -100.0, 100.0),
            ],
            GeoNodeKind::Array { .. } => vec![
                number("count", 0, 1.0, 100.0),
                number("offset-x", 1, -100.0, 100.0),
                number("offset-y", 2, -100.0, 100.0),
                number("offset-z", 3, -100.0, 100.0),
            ],
            GeoNodeKind::Subdivide { .. } => vec![
                number("levels", 0, 0.0, 4.0),
                GeoField {
                    choice: Some(&["off", "on"]),
                    ..number("smooth", 1, 0.0, 1.0)
                },
            ],
            GeoNodeKind::Mirror { .. } => vec![GeoField {
                choice: Some(&["x", "y", "z"]),
                ..number("axis", 0, 0.0, 2.0)
            }],
            GeoNodeKind::Primitive { .. } => vec![
                GeoField {
                    choice: Some(&PRIMITIVES),
                    ..number("shape", 0, 0.0, 5.0)
                },
                number("size", 1, 0.001, 1000.0),
            ],
            GeoNodeKind::ExtrudeProfile { .. } => vec![number("height", 0, -1000.0, 1000.0)],
            GeoNodeKind::RevolveProfile { .. } => vec![number("segments", 0, 3.0, 512.0)],
            GeoNodeKind::DistributePoints { .. } => vec![
                number("density", 0, 0.0, 100_000.0),
                number("seed", 1, 0.0, 1e6),
            ],
            GeoNodeKind::InstanceOnPoints { .. } => vec![
                number("scale", 0, -100.0, 100.0),
                number("random-scale", 1, 0.0, 1.0),
                GeoField {
                    choice: Some(&["no", "yes"]),
                    ..number("random-rotation", 2, 0.0, 1.0)
                },
                GeoField {
                    choice: Some(&["no", "yes"]),
                    ..number("align", 3, 0.0, 1.0)
                },
                number("seed", 4, 0.0, 1e6),
            ],
            GeoNodeKind::SetPosition { .. } => vec![
                number("offset-x", 0, -1000.0, 1000.0),
                number("offset-y", 1, -1000.0, 1000.0),
                number("offset-z", 2, -1000.0, 1000.0),
                number("noise", 3, -100.0, 100.0),
                number("noise-scale", 4, 0.0, 1000.0),
                number("seed", 5, 0.0, 1e6),
            ],
        }
    }

    /// The same kind with number `index` set to `value`, kept in range.
    pub fn with_value(&self, index: u32, value: f32) -> Option<Self> {
        let mut values = self.values();
        *values.get_mut(index as usize)? = value;
        GeoNodeKind::from_parts(self.tag(), &values)
    }
}

/// One editable number on a node.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoField {
    /// The tail of its label's catalogue key: `geo-field-<name>`.
    pub name: &'static str,
    /// Its place in [`GeoNodeKind::values`].
    pub index: u32,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    /// For a pick from a list: the options' key tails, `geo-option-<name>`.
    pub choice: Option<&'static [&'static str]>,
}

/// One node.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoNode {
    pub name: String,
    pub kind: GeoNodeKind,
    /// Where it sits in the editor.
    pub position: [f32; 2],
}

/// A wire: `from_node`'s geometry into `to_node`'s input `to_socket`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeoLink {
    pub from_node: usize,
    pub to_node: usize,
    pub to_socket: usize,
}

/// Why a wire was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoLinkError {
    NoSuchSocket,
    /// A field into a geometry input, or the other way.
    TypeMismatch,
    /// It would make the graph loop.
    Cycle,
}

/// A geometry graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeoGraph {
    pub nodes: Vec<GeoNode>,
    pub links: Vec<GeoLink>,
}

impl GeoGraph {
    /// Group Input straight into Group Output: the mesh as it was, which is
    /// what turning geometry nodes on must look like before anything is added.
    pub fn passthrough() -> Self {
        Self {
            nodes: vec![
                GeoNode {
                    name: "Group Input".into(),
                    kind: GeoNodeKind::Input,
                    position: [-300.0, 0.0],
                },
                GeoNode {
                    name: "Group Output".into(),
                    kind: GeoNodeKind::Output,
                    position: [300.0, 0.0],
                },
            ],
            links: vec![GeoLink {
                from_node: 0,
                to_node: 1,
                to_socket: 0,
            }],
        }
    }

    /// The wire into `node`'s input `socket`, if one is there.
    pub fn incoming(&self, node: usize, socket: usize) -> Option<&GeoLink> {
        self.links
            .iter()
            .find(|l| l.to_node == node && l.to_socket == socket)
    }

    /// Wires `from` into `to`'s input `socket`, replacing what was there.
    pub fn link(&mut self, from: usize, to: usize, socket: usize) -> Result<(), GeoLinkError> {
        let (Some(f), Some(t)) = (self.nodes.get(from), self.nodes.get(to)) else {
            return Err(GeoLinkError::NoSuchSocket);
        };
        if !f.kind.has_output() || socket >= t.kind.inputs() {
            return Err(GeoLinkError::NoSuchSocket);
        }
        // A number and a direction stand in for each other; geometry is
        // only geometry.
        let (a, b) = (f.kind.output_type(), Some(t.kind.input_types()[socket]));
        if (a == Some(GeoSocket::Geometry)) != (b == Some(GeoSocket::Geometry)) {
            return Err(GeoLinkError::TypeMismatch);
        }
        if from == to || self.reaches(to, from) {
            return Err(GeoLinkError::Cycle);
        }
        self.links
            .retain(|l| !(l.to_node == to && l.to_socket == socket));
        self.links.push(GeoLink {
            from_node: from,
            to_node: to,
            to_socket: socket,
        });
        Ok(())
    }

    /// Whether `from`'s output can reach `to` along the wires.
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut stack = vec![from];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if std::mem::replace(&mut seen[n], true) {
                continue;
            }
            stack.extend(
                self.links
                    .iter()
                    .filter(|l| l.from_node == n)
                    .map(|l| l.to_node),
            );
        }
        false
    }

    /// Removes nodes — never the group's input or output — and their wires,
    /// renumbering what is left.
    pub fn remove(&mut self, doomed: &[usize]) -> bool {
        let doomed: Vec<usize> = doomed
            .iter()
            .copied()
            .filter(|n| {
                self.nodes.get(*n).is_some_and(|node| {
                    !matches!(node.kind, GeoNodeKind::Input | GeoNodeKind::Output)
                })
            })
            .collect();
        if doomed.is_empty() {
            return false;
        }
        let mut map = vec![None; self.nodes.len()];
        let mut kept = Vec::new();
        for (i, n) in self.nodes.drain(..).enumerate() {
            if !doomed.contains(&i) {
                map[i] = Some(kept.len());
                kept.push(n);
            }
        }
        self.nodes = kept;
        self.links = self
            .links
            .iter()
            .filter_map(|l| {
                Some(GeoLink {
                    from_node: map[l.from_node]?,
                    to_node: map[l.to_node]?,
                    to_socket: l.to_socket,
                })
            })
            .collect();
        true
    }

    /// What the graph makes of `input`, or `None` when it has no output.
    pub fn evaluate(&self, input: &MeshData) -> Option<MeshData> {
        let output = self
            .nodes
            .iter()
            .position(|n| matches!(n.kind, GeoNodeKind::Output))?;
        let mut memo: Vec<Option<Value>> = vec![None; self.nodes.len()];
        Some(self.socket(output, 0, input, &mut memo, 0))
    }

    /// What arrives at `node`'s input `socket`: the wired node's result, or
    /// nothing.
    fn arriving(
        &self,
        node: usize,
        socket: usize,
        input: &MeshData,
        memo: &mut Vec<Option<Value>>,
        depth: usize,
    ) -> Option<Value> {
        match self.incoming(node, socket) {
            Some(l) if depth <= self.nodes.len() => {
                Some(self.node(l.from_node, input, memo, depth + 1))
            }
            _ => None,
        }
    }

    /// The geometry at an input: empty when nothing, or a field, is there.
    fn socket(
        &self,
        node: usize,
        socket: usize,
        input: &MeshData,
        memo: &mut Vec<Option<Value>>,
        depth: usize,
    ) -> MeshData {
        match self.arriving(node, socket, input, memo, depth) {
            Some(Value::Geometry(m)) => m,
            _ => MeshData::empty(),
        }
    }

    /// The field at an input, if one is wired.
    fn field(
        &self,
        node: usize,
        socket: usize,
        input: &MeshData,
        memo: &mut Vec<Option<Value>>,
        depth: usize,
    ) -> Option<Field> {
        match self.arriving(node, socket, input, memo, depth)? {
            Value::Field(f) => Some(f),
            Value::Geometry(_) => None,
        }
    }

    fn node(
        &self,
        n: usize,
        input: &MeshData,
        memo: &mut Vec<Option<Value>>,
        depth: usize,
    ) -> Value {
        if let Some(done) = &memo[n] {
            return done.clone();
        }
        let geometry = match &self.nodes[n].kind {
            GeoNodeKind::Position => Err(Field::Position),
            GeoNodeKind::Normal => Err(Field::Normal),
            GeoNodeKind::Index => Err(Field::Index),
            GeoNodeKind::Value { value } => Err(Field::Float(*value)),
            GeoNodeKind::Vector { value } => Err(Field::Vector(Vec3::from_array(*value))),
            GeoNodeKind::NoiseField { scale, seed } => Err(Field::Noise {
                at: Box::new(
                    self.field(n, 0, input, memo, depth)
                        .unwrap_or(Field::Position),
                ),
                scale: *scale,
                seed: *seed,
            }),
            GeoNodeKind::Math { op, a, b } => Err(Field::Math {
                op: *op,
                a: Box::new(
                    self.field(n, 0, input, memo, depth)
                        .unwrap_or(Field::Float(*a)),
                ),
                b: Box::new(
                    self.field(n, 1, input, memo, depth)
                        .unwrap_or(Field::Float(*b)),
                ),
            }),
            GeoNodeKind::VectorScale { scale } => Err(Field::Scale {
                v: Box::new(
                    self.field(n, 0, input, memo, depth)
                        .unwrap_or(Field::Vector(Vec3::ZERO)),
                ),
                s: Box::new(
                    self.field(n, 1, input, memo, depth)
                        .unwrap_or(Field::Float(*scale)),
                ),
            }),
            GeoNodeKind::CombineXyz { x, y, z } => {
                let mut f = |i: usize, d: f32| {
                    self.field(n, i, input, memo, depth)
                        .unwrap_or(Field::Float(d))
                };
                Err(Field::Combine(Box::new([f(0, *x), f(1, *y), f(2, *z)])))
            }
            _ => Ok(()),
        };
        if let Err(field) = geometry {
            let out = Value::Field(field);
            memo[n] = Some(out.clone());
            return out;
        }
        let out = match &self.nodes[n].kind {
            GeoNodeKind::Position
            | GeoNodeKind::Normal
            | GeoNodeKind::Index
            | GeoNodeKind::Value { .. }
            | GeoNodeKind::Vector { .. }
            | GeoNodeKind::NoiseField { .. }
            | GeoNodeKind::Math { .. }
            | GeoNodeKind::VectorScale { .. }
            | GeoNodeKind::CombineXyz { .. } => unreachable!("a field is made above"),
            GeoNodeKind::CurveLine { length, resolution } => {
                let k = (*resolution).max(2);
                curves_mesh(&[(
                    (0..k)
                        .map(|i| Vec3::new(0.0, 0.0, *length * i as f32 / (k - 1) as f32))
                        .collect(),
                    false,
                )])
            }
            GeoNodeKind::CurveCircle { radius, resolution } => {
                let k = (*resolution).max(3);
                curves_mesh(&[(
                    (0..k)
                        .map(|i| {
                            let a = std::f32::consts::TAU * i as f32 / k as f32;
                            Vec3::new(a.cos() * radius, a.sin() * radius, 0.0)
                        })
                        .collect(),
                    true,
                )])
            }
            GeoNodeKind::CurveSpiral {
                turns,
                height,
                radius,
                resolution,
            } => {
                let k = (*resolution).max(2);
                curves_mesh(&[(
                    (0..k)
                        .map(|i| {
                            let t = i as f32 / (k - 1) as f32;
                            let a = std::f32::consts::TAU * turns * t;
                            Vec3::new(a.cos() * radius, a.sin() * radius, height * t)
                        })
                        .collect(),
                    false,
                )])
            }
            GeoNodeKind::ResampleCurve { count } => {
                let m = self.socket(n, 0, input, memo, depth);
                let curves: Vec<(Vec<Vec3>, bool)> = curves_of(&m)
                    .into_iter()
                    .map(|(p, closed)| (resample(&p, closed, *count), closed))
                    .collect();
                curves_mesh(&curves)
            }
            GeoNodeKind::CurveToPoints => {
                let m = self.socket(n, 0, input, memo, depth);
                let mut positions = Vec::new();
                let mut normals = Vec::new();
                for (p, closed) in curves_of(&m) {
                    normals.extend(tangents(&p, closed));
                    positions.extend(p);
                }
                let mut out = MeshData::from_faces(positions, &[]);
                out.attributes_mut().insert(
                    POINT_NORMAL,
                    kreate_mesh::Domain::Point,
                    kreate_mesh::AttrData::F32x3(std::sync::Arc::new(normals)),
                );
                out
            }
            GeoNodeKind::CurveToMesh {
                radius,
                segments,
                caps,
            } => {
                let m = self.socket(n, 0, input, memo, depth);
                let profile = kreate_mesh::primitives::circle_profile(*radius, *segments);
                let tubes: Vec<MeshData> = curves_of(&m)
                    .into_iter()
                    .filter(|(p, _)| p.len() >= 2)
                    .map(|(path, closed)| {
                        kreate_mesh::primitives::sweep(&kreate_mesh::primitives::Sweep {
                            path: &path,
                            profile: &profile,
                            resolution: 1,
                            closed,
                            scale: (1.0, 1.0),
                            caps: *caps,
                        })
                    })
                    .collect();
                let parts: Vec<Part<'_>> = tubes
                    .iter()
                    .filter(|m| m.num_verts() > 0)
                    .map(|mesh| Part {
                        mesh,
                        transform: DAffine3::IDENTITY,
                    })
                    .collect();
                join(&parts).0
            }
            GeoNodeKind::DistributePoints { density, seed } => {
                let m = self.socket(n, 0, input, memo, depth);
                distribute_points(&m, *density, *seed)
            }
            GeoNodeKind::InstanceOnPoints {
                scale,
                random_scale,
                random_rotation,
                align,
                seed,
            } => {
                let points = self.socket(n, 0, input, memo, depth);
                let shape = self.socket(n, 1, input, memo, depth);
                // Read at the points, before anything stands on them.
                let ctx = FieldCtx::new(&points);
                let scales = self.field(n, 2, input, memo, depth).map(|f| f.floats(&ctx));
                let turns = self
                    .field(n, 3, input, memo, depth)
                    .map(|f| f.vectors(&ctx));
                instance_on_points(
                    &points,
                    &shape,
                    *scale,
                    *random_scale,
                    *random_rotation,
                    *align,
                    *seed,
                    scales.as_deref(),
                    turns.as_deref(),
                )
            }
            GeoNodeKind::SetPosition {
                offset,
                noise,
                noise_scale,
                seed,
            } => {
                let mut m = self.socket(n, 0, input, memo, depth);
                // The offset field is read where the points were.
                let extra = self
                    .field(n, 1, input, memo, depth)
                    .map(|f| f.vectors(&FieldCtx::new(&m)));
                set_position(
                    &mut m,
                    Vec3::from_array(*offset),
                    *noise,
                    *noise_scale,
                    *seed,
                );
                if let Some(extra) = extra {
                    for (p, d) in m.positions_mut().iter_mut().zip(extra) {
                        *p += d;
                    }
                }
                m
            }
            GeoNodeKind::Input => input.clone(),
            GeoNodeKind::Output => self.socket(n, 0, input, memo, depth),
            GeoNodeKind::Transform {
                translate,
                rotate,
                scale,
            } => {
                let mut m = self.socket(n, 0, input, memo, depth);
                let to = DAffine3::from_scale_rotation_translation(
                    Vec3::from_array(*scale).as_dvec3(),
                    DQuat::from_euler(
                        kreate_math::glam::EulerRot::ZYX,
                        (rotate[2] as f64).to_radians(),
                        (rotate[1] as f64).to_radians(),
                        (rotate[0] as f64).to_radians(),
                    ),
                    DVec3::from_array(translate.map(|v| v as f64)),
                );
                for p in m.positions_mut() {
                    *p = to.transform_point3(p.as_dvec3()).as_vec3();
                }
                m
            }
            GeoNodeKind::ExtrudeProfile { height } => {
                let m = self.socket(n, 0, input, memo, depth);
                kreate_mesh::profile::extrude_profile(&m, *height)
            }
            GeoNodeKind::RevolveProfile { segments } => {
                let m = self.socket(n, 0, input, memo, depth);
                kreate_mesh::profile::revolve_profile(&m, *segments)
            }
            GeoNodeKind::Array { count, offset } => {
                let m = self.socket(n, 0, input, memo, depth);
                array(
                    &m,
                    &ArrayParams {
                        count: *count,
                        relative: Vec3::ZERO,
                        constant: Vec3::from_array(*offset),
                    },
                )
            }
            GeoNodeKind::Subdivide { levels, smooth } => {
                let m = self.socket(n, 0, input, memo, depth);
                subdivide(
                    &m,
                    &SubdivideParams {
                        levels: *levels,
                        smooth: *smooth,
                    },
                )
            }
            GeoNodeKind::Mirror { axis } => {
                let m = self.socket(n, 0, input, memo, depth);
                mirror(
                    &m,
                    &MirrorParams {
                        x: *axis == 0,
                        y: *axis == 1,
                        z: *axis == 2,
                        ..Default::default()
                    },
                )
            }
            GeoNodeKind::Join => {
                let a = self.socket(n, 0, input, memo, depth);
                let b = self.socket(n, 1, input, memo, depth);
                let parts: Vec<Part<'_>> = [&a, &b]
                    .into_iter()
                    .filter(|m| m.num_verts() > 0)
                    .map(|mesh| Part {
                        mesh,
                        transform: DAffine3::IDENTITY,
                    })
                    .collect();
                join(&parts).0
            }
            GeoNodeKind::Primitive { shape, size } => {
                use kreate_mesh::primitives as p;
                match shape {
                    1 => p::uv_sphere(*size * 0.5, 24, 12),
                    2 => p::cylinder(*size * 0.5, *size, 24),
                    3 => p::cone(*size * 0.5, *size, 24),
                    4 => p::torus(*size * 0.5, *size * 0.15, 32, 12),
                    5 => p::ico_sphere(*size * 0.5, 2),
                    _ => p::cube(*size),
                }
            }
        };
        let out = Value::Geometry(out);
        memo[n] = Some(out.clone());
        out
    }
}

#[cfg(test)]
mod tests {

    /// A ball moved by a field: its normal, scaled by a number field made
    /// from the point's height. The top swells, the middle stays.
    #[test]
    fn set_position_reads_a_field_at_every_vertex() {
        let ball = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
        let mut g = GeoGraph::passthrough();
        let s = add(
            &mut g,
            GeoNodeKind::SetPosition {
                offset: [0.0; 3],
                noise: 0.0,
                noise_scale: 1.0,
                seed: 0,
            },
        );
        let normal = add(&mut g, GeoNodeKind::Normal);
        let position = add(&mut g, GeoNodeKind::Position);
        // max(z, 0) as a number: the length of a vector is not it, so the
        // height comes out by Math on the noise of... simpler: z alone via
        // CombineXYZ is not a separate. Use the position scaled by 0.5.
        let scale = add(&mut g, GeoNodeKind::VectorScale { scale: 0.5 });
        let _ = position;
        g.link(0, s, 0).unwrap();
        g.link(normal, scale, 0).unwrap();
        g.link(scale, s, 1).unwrap();
        g.link(s, 1, 0).unwrap();
        let out = g.evaluate(&ball).unwrap();
        // Every vertex out along its normal by a half.
        for q in out.positions() {
            assert!((q.length() - 1.5).abs() < 0.02, "{}", q.length());
        }
        // A geometry wire into a field input is refused, and the other way.
        assert_eq!(g.link(0, s, 1), Err(GeoLinkError::TypeMismatch));
        assert_eq!(g.link(normal, s, 0), Err(GeoLinkError::TypeMismatch));
        // A number field stands for a direction: Math into the offset.
        let m = add(
            &mut g,
            GeoNodeKind::Math {
                op: 2,
                a: 0.25,
                b: 2.0,
            },
        );
        g.link(m, s, 1).unwrap();
        let out = g.evaluate(&ball).unwrap();
        let moved = out.positions()[0] - ball.positions()[0];
        assert!((moved - Vec3::splat(0.5)).length() < 1e-5, "{moved:?}");
    }

    #[test]
    fn index_and_math_fields_make_a_staircase() {
        let points = MeshData::from_faces(vec![Vec3::ZERO; 4], &[]);
        let mut g = GeoGraph::passthrough();
        let s = add(
            &mut g,
            GeoNodeKind::SetPosition {
                offset: [0.0; 3],
                noise: 0.0,
                noise_scale: 1.0,
                seed: 0,
            },
        );
        let index = add(&mut g, GeoNodeKind::Index);
        let times = add(
            &mut g,
            GeoNodeKind::Math {
                op: 2,
                a: 0.0,
                b: 2.0,
            },
        );
        let up = add(
            &mut g,
            GeoNodeKind::CombineXyz {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        );
        g.link(0, s, 0).unwrap();
        g.link(index, times, 0).unwrap();
        g.link(times, up, 2).unwrap();
        g.link(up, s, 1).unwrap();
        g.link(s, 1, 0).unwrap();
        let out = g.evaluate(&points).unwrap();
        let z: Vec<f32> = out.positions().iter().map(|p| p.z).collect();
        assert_eq!(z, vec![0.0, 2.0, 4.0, 6.0]);
        assert!(out.positions().iter().all(|p| p.x == 0.0 && p.y == 0.0));
    }

    #[test]
    fn a_circle_curve_becomes_a_closed_tube() {
        let mut g = GeoGraph::passthrough();
        let c = add(
            &mut g,
            GeoNodeKind::CurveCircle {
                radius: 2.0,
                resolution: 64,
            },
        );
        let r = add(&mut g, GeoNodeKind::ResampleCurve { count: 24 });
        let t = add(
            &mut g,
            GeoNodeKind::CurveToMesh {
                radius: 0.1,
                segments: 6,
                caps: true,
            },
        );
        g.link(c, r, 0).unwrap();
        g.link(r, t, 0).unwrap();
        g.link(t, 1, 0).unwrap();
        let out = g.evaluate(&MeshData::empty()).unwrap();
        assert_eq!(
            out.num_faces(),
            24 * 6,
            "a ring of quads, no caps on a loop"
        );
        // Every point a tube's radius from the circle.
        for q in out.positions() {
            let off = (Vec3::new(q.x, q.y, 0.0).length() - 2.0).hypot(q.z);
            assert!((off - 0.1).abs() < 0.03, "{off}");
        }
        // Closed and two-sided: every edge between two faces.
        let mut count = std::collections::HashMap::new();
        for f in 0..out.num_faces() {
            let ring = out.face_verts(f);
            for k in 0..ring.len() {
                let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(count.values().all(|c| *c == 2));
    }

    #[test]
    fn resampling_spaces_points_evenly_and_curve_points_face_along_it() {
        let line = curves_mesh(&[(
            vec![
                Vec3::ZERO,
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 4.0),
            ],
            false,
        )]);
        let curves = curves_of(&line);
        let even = resample(&curves[0].0, false, 5);
        let z: Vec<f32> = even.iter().map(|p| p.z).collect();
        assert_eq!(z, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let mut g = GeoGraph::passthrough();
        let spiral = add(
            &mut g,
            GeoNodeKind::CurveSpiral {
                turns: 2.0,
                height: 1.0,
                radius: 1.0,
                resolution: 50,
            },
        );
        let pts = add(&mut g, GeoNodeKind::CurveToPoints);
        let cube = add(
            &mut g,
            GeoNodeKind::Primitive {
                shape: 0,
                size: 0.05,
            },
        );
        let inst = add(
            &mut g,
            GeoNodeKind::InstanceOnPoints {
                scale: 1.0,
                random_scale: 0.0,
                random_rotation: false,
                align: true,
                seed: 0,
            },
        );
        g.link(spiral, pts, 0).unwrap();
        g.link(pts, inst, 0).unwrap();
        g.link(cube, inst, 1).unwrap();
        g.link(inst, 1, 0).unwrap();
        let out = g.evaluate(&MeshData::empty()).unwrap();
        assert_eq!(out.num_faces(), 50 * 6);
    }

    #[test]
    fn a_scale_field_sizes_each_instance() {
        let points = MeshData::from_faces(vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)], &[]);
        let mut g = GeoGraph::passthrough();
        let cube = add(
            &mut g,
            GeoNodeKind::Primitive {
                shape: 0,
                size: 1.0,
            },
        );
        let inst = add(
            &mut g,
            GeoNodeKind::InstanceOnPoints {
                scale: 1.0,
                random_scale: 0.0,
                random_rotation: false,
                align: false,
                seed: 0,
            },
        );
        let index = add(&mut g, GeoNodeKind::Index);
        let plus = add(
            &mut g,
            GeoNodeKind::Math {
                op: 0,
                a: 0.0,
                b: 1.0,
            },
        );
        g.link(index, plus, 0).unwrap();
        g.link(0, inst, 0).unwrap();
        g.link(cube, inst, 1).unwrap();
        g.link(plus, inst, 2).unwrap();
        g.link(inst, 1, 0).unwrap();
        let out = g.evaluate(&points).unwrap();
        let width = |lo: f32, hi: f32| {
            let xs: Vec<f32> = out
                .positions()
                .iter()
                .map(|p| p.x)
                .filter(|x| *x > lo && *x < hi)
                .collect();
            xs.iter().fold(f32::MIN, |a, b| a.max(*b)) - xs.iter().fold(f32::MAX, |a, b| a.min(*b))
        };
        assert!((width(-5.0, 5.0) - 1.0).abs() < 1e-4, "the first at one");
        assert!((width(5.0, 15.0) - 2.0).abs() < 1e-4, "the second at two");
    }

    use super::*;

    fn add(g: &mut GeoGraph, kind: GeoNodeKind) -> usize {
        g.nodes.push(GeoNode {
            name: kind.tag().into(),
            kind,
            position: [0.0, 0.0],
        });
        g.nodes.len() - 1
    }

    #[test]
    fn a_passthrough_gives_the_mesh_back() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let out = GeoGraph::passthrough().evaluate(&cube).unwrap();
        assert_eq!(out.positions(), cube.positions());
        assert_eq!(out.num_faces(), cube.num_faces());
    }

    #[test]
    fn transform_array_and_join_compose() {
        let cube = kreate_mesh::primitives::cube(1.0);
        let mut g = GeoGraph::passthrough();
        // Input → Transform (up 5) → Array ×3 → Join with a fresh cube → Output.
        let t = add(
            &mut g,
            GeoNodeKind::Transform {
                translate: [0.0, 0.0, 5.0],
                rotate: [0.0, 0.0, 90.0],
                scale: [1.0, 1.0, 2.0],
            },
        );
        let a = add(
            &mut g,
            GeoNodeKind::Array {
                count: 3,
                offset: [2.0, 0.0, 0.0],
            },
        );
        let p = add(
            &mut g,
            GeoNodeKind::Primitive {
                shape: 0,
                size: 1.0,
            },
        );
        let j = add(&mut g, GeoNodeKind::Join);
        g.link(0, t, 0).unwrap();
        g.link(t, a, 0).unwrap();
        g.link(a, j, 0).unwrap();
        g.link(p, j, 1).unwrap();
        g.link(j, 1, 0).unwrap();
        let out = g.evaluate(&cube).unwrap();
        assert_eq!(out.num_faces(), 6 * 3 + 6);
        let top = out.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        assert!((top - 6.0).abs() < 1e-5, "{top}");
        let right = out.positions().iter().map(|p| p.x).fold(f32::MIN, f32::max);
        assert!((right - 4.5).abs() < 1e-5, "{right}");
    }

    #[test]
    fn a_wire_that_would_loop_is_refused_and_removing_keeps_the_group_ends() {
        let mut g = GeoGraph::passthrough();
        let t = add(&mut g, GeoNodeKind::Mirror { axis: 0 });
        let u = add(
            &mut g,
            GeoNodeKind::Subdivide {
                levels: 1,
                smooth: false,
            },
        );
        g.link(t, u, 0).unwrap();
        assert_eq!(g.link(u, t, 0), Err(GeoLinkError::Cycle));
        assert_eq!(g.link(1, t, 0), Err(GeoLinkError::NoSuchSocket));
        assert!(!g.remove(&[0, 1]));
        assert!(g.remove(&[t]));
        assert_eq!(g.nodes.len(), 3);
        assert!(g.links.iter().all(|l| l.from_node < 3 && l.to_node < 3));
    }

    #[test]
    fn points_scattered_over_a_plane_each_carry_an_instance() {
        let plane = kreate_mesh::primitives::plane(2.0);
        let mut g = GeoGraph::passthrough();
        let d = add(
            &mut g,
            GeoNodeKind::DistributePoints {
                density: 10.0,
                seed: 3,
            },
        );
        let p = add(
            &mut g,
            GeoNodeKind::Primitive {
                shape: 0,
                size: 0.1,
            },
        );
        let i = add(
            &mut g,
            GeoNodeKind::InstanceOnPoints {
                scale: 1.0,
                random_scale: 0.5,
                random_rotation: true,
                align: true,
                seed: 1,
            },
        );
        g.link(0, d, 0).unwrap();
        g.link(d, i, 0).unwrap();
        g.link(p, i, 1).unwrap();
        g.link(i, 1, 0).unwrap();
        let out = g.evaluate(&plane).unwrap();
        // Four square units at ten a unit: forty cubes, six faces each.
        assert_eq!(out.num_faces(), 40 * 6);
        // On the plane, within it.
        assert!(out
            .positions()
            .iter()
            .all(|q| q.x.abs() < 1.1 && q.y.abs() < 1.1 && q.z.abs() < 0.15));
        // The same seed scatters the same.
        assert_eq!(g.evaluate(&plane).unwrap().positions(), out.positions());
    }

    #[test]
    fn instances_stand_on_the_surface_they_were_scattered_over() {
        // A wall facing +X: aligned, the cubes' up is the wall's normal.
        let mut wall = kreate_mesh::primitives::plane(2.0);
        for q in wall.positions_mut() {
            *q = Vec3::new(0.0, q.x, q.y);
        }
        let points = distribute_points(&wall, 5.0, 0);
        let pin = kreate_mesh::MeshData::from_faces(
            vec![
                Vec3::ZERO,
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.01, 0.0, 0.0),
            ],
            &[vec![0, 1, 2]],
        );
        let out = instance_on_points(&points, &pin, 1.0, 0.0, false, true, 0, None, None);
        let n = points.attributes().get_f32x3(POINT_NORMAL).unwrap()[0];
        let tip = out.positions()[1] - out.positions()[0];
        assert!(
            tip.normalize().dot(n) > 0.999,
            "standing on the wall: {tip:?} vs {n:?}"
        );
    }

    #[test]
    fn set_position_moves_along_the_normals_by_its_noise() {
        let ball = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
        let mut lumpy = ball.clone();
        set_position(&mut lumpy, Vec3::ZERO, 0.2, 3.0, 4);
        let radii: Vec<f32> = lumpy.positions().iter().map(|q| q.length()).collect();
        assert!(
            radii.iter().any(|r| *r > 1.02) && radii.iter().any(|r| *r < 0.98),
            "in and out"
        );
        assert!(
            radii.iter().all(|r| (r - 1.0).abs() <= 0.2001),
            "no further than the noise"
        );
        let mut moved = ball.clone();
        set_position(&mut moved, Vec3::new(0.0, 0.0, 2.0), 0.0, 1.0, 0);
        assert!(moved
            .positions()
            .iter()
            .zip(ball.positions())
            .all(|(a, b)| (a.z - b.z - 2.0).abs() < 1e-6));
    }

    #[test]
    fn every_kind_round_trips_through_its_numbers() {
        for tag in GeoNodeKind::ADDABLE.iter().chain(&["input", "output"]) {
            let kind = GeoNodeKind::from_parts(tag, &[]).unwrap();
            assert_eq!(
                GeoNodeKind::from_parts(kind.tag(), &kind.values()),
                Some(kind.clone())
            );
            for f in kind.fields() {
                assert!(kind.with_value(f.index, f.value).is_some());
            }
        }
    }
}
