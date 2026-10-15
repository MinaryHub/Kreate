//! The boolean modifier: one mesh cut by, joined with, or kept inside another.
//!
//! Two ways, chosen by size (see [`Method`]). For small meshes, binary space
//! partitioning, the method of Thibault and Naylor that
//! `csg.js` made familiar: each solid's polygons build a tree of the planes
//! they lie in, and each solid is clipped against the other's tree — what is
//! inside the other kept or thrown away depending on the operation. It is
//! exact up to floating point, needs no special cases for coplanar faces, and
//! is short.
//!
//! # What it asks of its input
//!
//! Both meshes should be closed. A hole lets "inside" leak, and the result is
//! wrong in the way every boolean is wrong on an open mesh — Blender's too.
//!
//! # What it gives back
//!
//! A closed mesh, with each input face kept whole wherever the cut did not
//! reach it. The BSP alone gives neither: it splits a face along every plane of
//! the other solid's tree, cut or not — a cube's side came back as a row of
//! strips, one per facet of a cylinder that never touched it — and the pieces
//! of two neighbouring faces split at different places meet in T-junctions,
//! so the mesh had open edges everywhere and a bevel or a subdivision put on
//! top of it tore. [`assemble`] puts that right: pieces are welded, every
//! vertex lying on another piece's edge is put into that edge, and the pieces
//! of one input face are merged back into as few polygons as stay simple.
//! Attributes (UVs, materials) are not carried; the result is geometry.
//!
//! Everything is in `f64` — a plane test in `f32` misclassifies points near a
//! face, and a boolean is nothing but plane tests.

use kreate_math::{DVec3, Vec3};

use crate::MeshData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BooleanOp {
    /// What is in the first and not in the second — cutting a hole.
    #[default]
    Difference,
    /// What is in either.
    Union,
    /// What is in both.
    Intersect,
}

const EPSILON: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
struct Plane {
    normal: DVec3,
    w: f64,
}

impl Plane {
    fn from_points(a: DVec3, b: DVec3, c: DVec3) -> Option<Plane> {
        let n = (b - a).cross(c - a);
        let len = n.length();
        if len < 1e-12 {
            return None;
        }
        let normal = n / len;
        Some(Plane {
            normal,
            w: normal.dot(a),
        })
    }

    fn flip(&mut self) {
        self.normal = -self.normal;
        self.w = -self.w;
    }
}

#[derive(Clone, Debug)]
struct Polygon {
    verts: Vec<DVec3>,
    plane: Plane,
    /// Which input face this is, or a piece of — A's faces first, then B's.
    /// What lets the pieces be merged back afterwards.
    source: u32,
}

impl Polygon {
    fn new(verts: Vec<DVec3>, source: u32) -> Option<Polygon> {
        // The plane of the whole polygon, from its Newell normal, so a thin
        // first triangle does not decide it.
        let mut n = DVec3::ZERO;
        for i in 0..verts.len() {
            let (p, q) = (verts[i], verts[(i + 1) % verts.len()]);
            n += DVec3::new(
                (p.y - q.y) * (p.z + q.z),
                (p.z - q.z) * (p.x + q.x),
                (p.x - q.x) * (p.y + q.y),
            );
        }
        let len = n.length();
        if verts.len() < 3 {
            return None;
        }
        if len < 1e-12 {
            return Plane::from_points(verts[0], verts[1], verts[2]).map(|plane| Polygon {
                verts,
                plane,
                source,
            });
        }
        let normal = n / len;
        let centre = verts.iter().fold(DVec3::ZERO, |s, p| s + *p) / verts.len() as f64;
        Some(Polygon {
            plane: Plane {
                normal,
                w: normal.dot(centre),
            },
            verts,
            source,
        })
    }

    fn flip(&mut self) {
        self.verts.reverse();
        self.plane.flip();
    }
}

const COPLANAR: u8 = 0;
const FRONT: u8 = 1;
const BACK: u8 = 2;
const SPANNING: u8 = 3;

/// How long a boolean may run before it gives up. A BSP's work grows with
/// the square of the faces on a rounded mesh — every face of a sphere lies
/// behind every other's plane, so its tree is a chain as long as it has
/// faces — and a boolean on a mesh of a hundred thousand faces ran for
/// hours, on the thread that draws the window. A modelled M3 thread through
/// a plate takes some seconds; past this, the answer is no answer.
const TIME_LIMIT: std::time::Duration = std::time::Duration::from_secs(8);

/// When to give up, looked at every few thousand plane tests.
struct Budget {
    deadline: std::time::Instant,
    count: u32,
    over: bool,
}

impl Budget {
    fn new() -> Budget {
        Budget {
            deadline: std::time::Instant::now() + TIME_LIMIT,
            count: 0,
            over: false,
        }
    }

    fn spent(&self) -> bool {
        self.over
    }

    fn spend(&mut self, n: usize) {
        self.count = self.count.wrapping_add(n as u32);
        if self.count >= 1 << 16 {
            self.count = 0;
            self.over |= std::time::Instant::now() > self.deadline;
        }
    }
}

/// A boolean that would take too long: the meshes have too many faces for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooComplex;

impl std::fmt::Display for TooComplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("too many faces for a boolean: reduce them, or cut a simpler shape")
    }
}

/// Sorts `polygon` against `plane` into the four lists. Moved, not copied:
/// most polygons land whole on one side, and copying each at every level of
/// the tree was most of a boolean's time.
fn split(
    plane: &Plane,
    polygon: Polygon,
    coplanar_front: &mut Vec<Polygon>,
    coplanar_back: &mut Vec<Polygon>,
    front: &mut Vec<Polygon>,
    back: &mut Vec<Polygon>,
    budget: &mut Budget,
) {
    budget.spend(polygon.verts.len());
    let mut kind = 0u8;
    let mut small = [0u8; 16];
    let mut large = Vec::new();
    let types: &mut [u8] = if polygon.verts.len() <= small.len() {
        &mut small[..polygon.verts.len()]
    } else {
        large.resize(polygon.verts.len(), 0);
        &mut large
    };
    for (k, v) in types.iter_mut().zip(&polygon.verts) {
        let t = plane.normal.dot(*v) - plane.w;
        *k = if t < -EPSILON {
            BACK
        } else if t > EPSILON {
            FRONT
        } else {
            COPLANAR
        };
        kind |= *k;
    }
    match kind {
        COPLANAR => {
            if plane.normal.dot(polygon.plane.normal) > 0.0 {
                coplanar_front.push(polygon);
            } else {
                coplanar_back.push(polygon);
            }
        }
        FRONT => front.push(polygon),
        BACK => back.push(polygon),
        _ => {
            let (mut f, mut b) = (Vec::new(), Vec::new());
            let n = polygon.verts.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let (ti, tj) = (types[i], types[j]);
                let (vi, vj) = (polygon.verts[i], polygon.verts[j]);
                if ti != BACK {
                    f.push(vi);
                }
                if ti != FRONT {
                    b.push(vi);
                }
                if (ti | tj) == SPANNING {
                    let t = (plane.w - plane.normal.dot(vi)) / plane.normal.dot(vj - vi);
                    let v = vi.lerp(vj, t);
                    f.push(v);
                    b.push(v);
                }
            }
            if f.len() >= 3 {
                front.push(Polygon {
                    verts: f,
                    plane: polygon.plane,
                    source: polygon.source,
                });
            }
            if b.len() >= 3 {
                back.push(Polygon {
                    verts: b,
                    plane: polygon.plane,
                    source: polygon.source,
                });
            }
        }
    }
}

/// A BSP tree, its nodes in an arena so that nothing about it recurses — a
/// badly balanced tree over a few thousand polygons is thousands deep.
#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
}

/// The box round some points.
fn bounds_of<'a>(points: impl Iterator<Item = &'a DVec3>) -> (DVec3, DVec3) {
    points.fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    )
}

#[derive(Default)]
struct Node {
    plane: Option<Plane>,
    front: Option<usize>,
    back: Option<usize>,
    polygons: Vec<Polygon>,
    /// The box round every polygon this node and those under it were built
    /// with: the whole of the solid's surface inside this node's cell. Kept
    /// as built — clipping takes polygons away later, but the planes, which
    /// are what decide inside and outside, stay.
    bounds: (DVec3, DVec3),
}

impl Tree {
    fn new(polygons: Vec<Polygon>, budget: &mut Budget) -> Result<Tree, TooComplex> {
        let mut tree = Tree {
            nodes: vec![Node::default()],
        };
        tree.build(0, polygons, budget)?;
        // Children come after their parents in the arena, so from the end
        // back each node's box takes in its children's.
        for i in (0..tree.nodes.len()).rev() {
            let node = &tree.nodes[i];
            let mut b = bounds_of(node.polygons.iter().flat_map(|p| p.verts.iter()));
            for c in [node.front, node.back].into_iter().flatten() {
                let cb = tree.nodes[c].bounds;
                b = (b.0.min(cb.0), b.1.max(cb.1));
            }
            tree.nodes[i].bounds = b;
        }
        Ok(tree)
    }

    fn build(
        &mut self,
        root: usize,
        polygons: Vec<Polygon>,
        budget: &mut Budget,
    ) -> Result<(), TooComplex> {
        let mut work = vec![(root, polygons)];
        while let Some((at, polygons)) = work.pop() {
            if budget.spent() {
                return Err(TooComplex);
            }
            if polygons.is_empty() {
                continue;
            }
            let plane = *self.nodes[at].plane.get_or_insert(polygons[0].plane);
            let (mut cf, mut cb, mut f, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            // The polygon the plane was taken from stays here, whatever its
            // corners say. A face not quite flat has corners off its own
            // plane; sorted like the rest it went to a child as "in front",
            // became that child's plane, went in front of itself again — a
            // new node a turn, until the memory ran out.
            let mut rest = polygons.into_iter();
            if self.nodes[at].polygons.is_empty() && cf.is_empty() {
                if let Some(first) = rest.next() {
                    if first.plane.normal == plane.normal && first.plane.w == plane.w {
                        cf.push(first);
                    } else {
                        split(&plane, first, &mut cf, &mut cb, &mut f, &mut b, budget);
                    }
                }
            }
            for p in rest {
                split(&plane, p, &mut cf, &mut cb, &mut f, &mut b, budget);
            }
            self.nodes[at].polygons.extend(cf);
            self.nodes[at].polygons.extend(cb);
            if !f.is_empty() {
                let child = self.child(at, true);
                work.push((child, f));
            }
            if !b.is_empty() {
                let child = self.child(at, false);
                work.push((child, b));
            }
        }
        Ok(())
    }

    fn child(&mut self, at: usize, front: bool) -> usize {
        let existing = if front {
            self.nodes[at].front
        } else {
            self.nodes[at].back
        };
        if let Some(c) = existing {
            return c;
        }
        self.nodes.push(Node::default());
        let c = self.nodes.len() - 1;
        if front {
            self.nodes[at].front = Some(c);
        } else {
            self.nodes[at].back = Some(c);
        }
        c
    }

    /// Turns the solid inside out: every plane and polygon flipped, and front
    /// and back swapped.
    fn invert(&mut self) {
        for node in &mut self.nodes {
            for p in &mut node.polygons {
                p.flip();
            }
            if let Some(plane) = &mut node.plane {
                plane.flip();
            }
            std::mem::swap(&mut node.front, &mut node.back);
        }
    }

    /// What of `polygons` is outside this solid.
    fn clip_polygons(
        &self,
        polygons: Vec<Polygon>,
        budget: &mut Budget,
    ) -> Result<Vec<Polygon>, TooComplex> {
        let mut out = Vec::new();
        let margin = DVec3::splat(EPSILON * 10.0);
        let mut work = vec![(0usize, polygons)];
        while let Some((at, polygons)) = work.pop() {
            if budget.spent() {
                return Err(TooComplex);
            }
            // Nothing to sort: not walked down the rest of the tree for it.
            // Every node of one tree clipped against the other did, for
            // each of its empty lists — a walk of the whole other tree per
            // node, tens of thousands squared, and none of it counted.
            if polygons.is_empty() {
                continue;
            }
            budget.spend(1);
            let node = &self.nodes[at];
            let Some(plane) = node.plane else {
                out.extend(polygons);
                continue;
            };
            // A polygon clear of every surface in this cell lies wholly
            // inside the solid or wholly outside — its part in the cell is
            // convex and meets no boundary — so one point of it says which,
            // and it is not cut by every plane below. Cutting it was how a
            // plate's face came back in thousands of pieces from a thread
            // that never touched it.
            let (lo, hi) = (node.bounds.0 - margin, node.bounds.1 + margin);
            let (polygons, clear): (Vec<Polygon>, Vec<Polygon>) =
                polygons.into_iter().partition(|p| {
                    let (a, b) = bounds_of(p.verts.iter());
                    a.cmple(hi).all() && b.cmpge(lo).all()
                });
            for p in clear {
                let centre = p.verts.iter().fold(DVec3::ZERO, |s, v| s + *v) / p.verts.len() as f64;
                if self.outside(at, centre) {
                    out.push(p);
                }
            }
            if polygons.is_empty() {
                continue;
            }
            let (mut cf, mut cb, mut f, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for p in polygons {
                split(&plane, p, &mut cf, &mut cb, &mut f, &mut b, budget);
            }
            f.extend(cf);
            b.extend(cb);
            match node.front {
                Some(c) => work.push((c, f)),
                None => out.extend(f),
            }
            // Behind a leaf is inside: dropped.
            if let Some(c) = node.back {
                work.push((c, b));
            }
        }
        Ok(out)
    }

    /// Whether `point`, in the cell of node `at`, is outside the solid: down
    /// the tree by the side of each plane it is on, to a leaf — off the
    /// front is outside, off the back inside.
    fn outside(&self, mut at: usize, point: DVec3) -> bool {
        loop {
            let node = &self.nodes[at];
            let Some(plane) = node.plane else {
                return true;
            };
            let front = plane.normal.dot(point) - plane.w >= 0.0;
            match (front, node.front, node.back) {
                (true, Some(c), _) | (false, _, Some(c)) => at = c,
                (true, None, _) => return true,
                (false, _, None) => return false,
            }
        }
    }

    /// Removes from this tree's polygons everything inside `other`.
    fn clip_to(&mut self, other: &Tree, budget: &mut Budget) -> Result<(), TooComplex> {
        for i in 0..self.nodes.len() {
            if self.nodes[i].polygons.is_empty() {
                continue;
            }
            let polygons = std::mem::take(&mut self.nodes[i].polygons);
            self.nodes[i].polygons = other.clip_polygons(polygons, budget)?;
        }
        Ok(())
    }

    fn all_polygons(&self) -> Vec<Polygon> {
        self.nodes
            .iter()
            .flat_map(|n| n.polygons.iter().cloned())
            .collect()
    }
}

/// A mesh's faces as polygons, their sources numbered from `first`.
///
/// A face that is not flat, or not convex, goes in as triangles of the same
/// source: the tree's splitting is only right for flat convex polygons — a
/// plane cuts a concave one into pieces that are not each one piece — and
/// edit mode makes faces of both kinds as soon as a vertex is moved.
fn polygons_of(mesh: &MeshData, first: u32) -> Vec<Polygon> {
    let positions = mesh.positions();
    let mut out = Vec::new();
    let mut tris = Vec::new();
    for f in 0..mesh.num_faces() {
        let corners = mesh.face_verts(f);
        let verts: Vec<DVec3> = corners
            .iter()
            .map(|v| positions[*v as usize].as_dvec3())
            .collect();
        let source = first + f;
        let Some(polygon) = Polygon::new(verts.clone(), source) else {
            continue;
        };
        if verts.len() == 3 || flat_and_convex(&polygon) {
            out.push(polygon);
            continue;
        }
        let points: Vec<kreate_math::Vec3> =
            corners.iter().map(|v| positions[*v as usize]).collect();
        tris.clear();
        crate::tessellate::triangulate_polygon(&points, &mut tris);
        for t in &tris {
            let tri = t.iter().map(|i| verts[*i as usize]).collect();
            out.extend(Polygon::new(tri, source));
        }
    }
    out
}

/// Whether every corner is on the polygon's plane and turns the same way.
fn flat_and_convex(p: &Polygon) -> bool {
    let n = p.verts.len();
    let on_plane = p
        .verts
        .iter()
        .all(|v| (p.plane.normal.dot(*v) - p.plane.w).abs() <= EPSILON * 0.5);
    on_plane
        && (0..n).all(|i| {
            let (a, b, c) = (p.verts[i], p.verts[(i + 1) % n], p.verts[(i + 2) % n]);
            (b - a).cross(c - b).dot(p.plane.normal) >= -1e-12
        })
}

/// `a` combined with `b`, both in the same space; `a` unchanged when the
/// two are too complex to combine in reasonable time — see [`try_boolean`]
/// to be told.
pub fn boolean(a: &MeshData, b: &MeshData, op: BooleanOp) -> MeshData {
    try_boolean(a, b, op).unwrap_or_else(|_| a.clone())
}

/// `a` combined with `b`, or [`TooComplex`] when that would take more than
/// some seconds.
///
/// Worked at one size whatever the meshes' own: both scaled so their box is
/// about a hundred across, and the answer scaled back. [`EPSILON`] and
/// [`WELD`] are lengths, and lengths right for a part a metre across are
/// coarse for one a few millimetres across — a modelled M3 thread through a
/// 3 mm plate came back with its fine slivers welded shut and the plate
/// open along every one, where the same shapes a thousand times larger came
/// back closed.
pub fn try_boolean(a: &MeshData, b: &MeshData, op: BooleanOp) -> Result<MeshData, TooComplex> {
    try_boolean_by(a, b, op, Method::Auto)
}

/// [`try_boolean`], worked out the way asked.
pub fn try_boolean_by(
    a: &MeshData,
    b: &MeshData,
    op: BooleanOp,
    method: Method,
) -> Result<MeshData, TooComplex> {
    let (lo, hi) = a.positions().iter().chain(b.positions()).fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let size = (hi - lo).length();
    if !(size.is_finite() && size > 0.0) {
        return try_boolean_at_size(a, b, op, method);
    }
    let centre = (lo + hi) * 0.5;
    let k = 100.0 / size;
    let scaled = |m: &MeshData| {
        let mut m = m.clone();
        for p in m.positions_mut() {
            *p = (*p - centre) * k;
        }
        m
    };
    let (sa, sb) = (scaled(a), scaled(b));
    // A cutter whose corners fall exactly on the other mesh's planes — a
    // thread's crest on a plate's face — can leave edges that are not one
    // face each way. Tried again with the cutter moved a hair, a few
    // hundred-millionths of the size, when that happens; the cleanest kept.
    let mut out = try_boolean_at_size(&sa, &sb, op, method)?;
    let mut flaws = unmatched_edges(&out);
    for nudge in [Vec3::new(0.31, 0.57, 0.73), Vec3::new(-0.67, 0.29, -0.41)] {
        if flaws == 0 {
            break;
        }
        let mut moved = sb.clone();
        for p in moved.positions_mut() {
            *p += nudge * 3e-5;
        }
        let Ok(again) = try_boolean_at_size(&sa, &moved, op, method) else {
            break;
        };
        let f = unmatched_edges(&again);
        if f < flaws {
            out = again;
            flaws = f;
        }
    }
    for p in out.positions_mut() {
        *p = *p / k + centre;
    }
    Ok(out)
}

/// Which way to work a boolean out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// By size: see [`ARRANGE_ABOVE`].
    Auto,
    /// Binary space partitioning.
    Bsp,
    /// Cutting only where the surfaces meet, and sorting the rest.
    Arrangement,
}

/// Above this many faces between the two, the arrangement is used: the
/// tree's work grows with the square of the faces on a rounded mesh.
pub const ARRANGE_ABOVE: u32 = 3000;

fn try_boolean_at_size(
    a: &MeshData,
    b: &MeshData,
    op: BooleanOp,
    method: Method,
) -> Result<MeshData, TooComplex> {
    if a.is_empty() {
        return Ok(match op {
            BooleanOp::Union => b.clone(),
            _ => a.clone(),
        });
    }
    if b.is_empty() {
        return Ok(match op {
            BooleanOp::Intersect => MeshData::from_faces(Vec::new(), &[]),
            _ => a.clone(),
        });
    }
    // Each input face clear of the other mesh's box, as it is: see `assemble`.
    let untouched: Vec<Option<Vec<DVec3>>> = {
        let margin = Vec3::splat((WELD * 10.0) as f32);
        let box_of = |m: &MeshData| {
            m.positions().iter().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(lo, hi), p| (lo.min(*p), hi.max(*p)),
            )
        };
        let faces = |m: &MeshData, other: (Vec3, Vec3)| {
            let (lo, hi) = (other.0 - margin, other.1 + margin);
            (0..m.num_faces())
                .map(|f| {
                    let ring: Vec<Vec3> = m
                        .face_verts(f)
                        .iter()
                        .map(|v| m.positions()[*v as usize])
                        .collect();
                    let (a, b) = ring.iter().fold(
                        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
                    );
                    let near = a.cmple(hi).all() && b.cmpge(lo).all();
                    (!near).then(|| ring.iter().map(|p| p.as_dvec3()).collect())
                })
                .collect::<Vec<_>>()
        };
        let mut u = faces(a, box_of(b));
        u.extend(faces(b, box_of(a)));
        u
    };
    let polygons = match method {
        Method::Bsp => bsp(a, b, op, &mut Budget::new())?,
        Method::Arrangement => arrangement(a, b, op, &mut Budget::new())?,
        // The tree for small meshes, where it is quick and long proven; the
        // arrangement for large ones, and wherever the tree runs out of time.
        Method::Auto if a.num_faces() + b.num_faces() <= ARRANGE_ABOVE => {
            match bsp(a, b, op, &mut Budget::new()) {
                Ok(p) => p,
                Err(TooComplex) => arrangement(a, b, op, &mut Budget::new())?,
            }
        }
        Method::Auto => arrangement(a, b, op, &mut Budget::new())?,
    };
    Ok(assemble(polygons, &untouched))
}

/// The polygons of `a` combined with `b` by binary space partitioning:
/// each solid's polygons clipped by the other's tree (see the module docs).
fn bsp(
    a: &MeshData,
    b: &MeshData,
    op: BooleanOp,
    budget: &mut Budget,
) -> Result<Vec<Polygon>, TooComplex> {
    let mut ta = Tree::new(polygons_of(a, 0), budget)?;
    let mut tb = Tree::new(polygons_of(b, a.num_faces()), budget)?;
    let polygons = match op {
        BooleanOp::Union => {
            ta.clip_to(&tb, budget)?;
            tb.clip_to(&ta, budget)?;
            tb.invert();
            tb.clip_to(&ta, budget)?;
            tb.invert();
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            all
        }
        BooleanOp::Difference => {
            ta.invert();
            ta.clip_to(&tb, budget)?;
            tb.clip_to(&ta, budget)?;
            tb.invert();
            tb.clip_to(&ta, budget)?;
            tb.invert();
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            // Turned back the right way out. `csg.js` builds a whole tree
            // of them to do it, which only splits them for nothing.
            all.iter_mut().for_each(Polygon::flip);
            all
        }
        BooleanOp::Intersect => {
            ta.invert();
            tb.clip_to(&ta, budget)?;
            tb.invert();
            ta.clip_to(&tb, budget)?;
            tb.clip_to(&ta, budget)?;
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            all.iter_mut().for_each(Polygon::flip);
            all
        }
    };
    Ok(polygons)
}

/// A closed surface as triangles, with grids along each axis for counting
/// where a ray along that axis crosses it: what says whether a point is
/// inside, in time that does not grow with the surface.
struct Solid {
    tris: Vec<[DVec3; 3]>,
    grids: [AxisGrid; 3],
    size: f64,
}

/// Triangles sorted into cells by where they lie across one axis.
struct AxisGrid {
    axis: usize,
    lo: [f64; 2],
    cell: f64,
    n: usize,
    cells: Vec<Vec<u32>>,
}

impl AxisGrid {
    fn across(axis: usize) -> (usize, usize) {
        ((axis + 1) % 3, (axis + 2) % 3)
    }

    fn new(axis: usize, tris: &[[DVec3; 3]], lo: DVec3, hi: DVec3) -> AxisGrid {
        let (u, v) = AxisGrid::across(axis);
        let n = ((tris.len() as f64).sqrt() as usize).clamp(1, 256);
        let span = (hi[u] - lo[u]).max(hi[v] - lo[v]).max(1e-12);
        let cell = span / n as f64 * 1.0001;
        let mut cells = vec![Vec::new(); n * n];
        for (i, t) in tris.iter().enumerate() {
            let (mut a0, mut a1, mut b0, mut b1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for p in t {
                a0 = a0.min(p[u]);
                a1 = a1.max(p[u]);
                b0 = b0.min(p[v]);
                b1 = b1.max(p[v]);
            }
            let at = |x: f64, l: f64| (((x - l) / cell).floor().max(0.0) as usize).min(n - 1);
            for cu in at(a0, lo[u])..=at(a1, lo[u]) {
                for cv in at(b0, lo[v])..=at(b1, lo[v]) {
                    cells[cu * n + cv].push(i as u32);
                }
            }
        }
        AxisGrid {
            axis,
            lo: [lo[u], lo[v]],
            cell,
            n,
            cells,
        }
    }

    /// How many times the ray from `p` along the axis, forwards, crosses the
    /// surface.
    fn crossings(&self, tris: &[[DVec3; 3]], p: DVec3) -> usize {
        let (u, v) = AxisGrid::across(self.axis);
        let cu = (p[u] - self.lo[0]) / self.cell;
        let cv = (p[v] - self.lo[1]) / self.cell;
        if cu < 0.0 || cv < 0.0 || cu >= self.n as f64 || cv >= self.n as f64 {
            return 0;
        }
        let mut count = 0;
        for i in &self.cells[cu as usize * self.n + cv as usize] {
            let [a, b, c] = tris[*i as usize];
            let (pu, pv) = (p[u], p[v]);
            let edge =
                |s: DVec3, e: DVec3| (e[u] - s[u]) * (pv - s[v]) - (e[v] - s[v]) * (pu - s[u]);
            let (e0, e1, e2) = (edge(a, b), edge(b, c), edge(c, a));
            let inside = (e0 > 0.0 && e1 > 0.0 && e2 > 0.0) || (e0 < 0.0 && e1 < 0.0 && e2 < 0.0);
            if !inside {
                continue;
            }
            // Where the ray meets the triangle's plane, along the axis.
            let n = (b - a).cross(c - a);
            if n[self.axis].abs() < 1e-300 {
                continue;
            }
            let along = a[self.axis] - ((p[u] - a[u]) * n[u] + (p[v] - a[v]) * n[v]) / n[self.axis];
            if along > p[self.axis] {
                count += 1;
            }
        }
        count
    }
}

impl Solid {
    fn new(polygons: &[Polygon]) -> Solid {
        let mut tris = Vec::new();
        for p in polygons {
            for i in 1..p.verts.len().saturating_sub(1) {
                tris.push([p.verts[0], p.verts[i], p.verts[i + 1]]);
            }
        }
        let (lo, hi) = bounds_of(tris.iter().flatten());
        let grids = [0, 1, 2].map(|axis| AxisGrid::new(axis, &tris, lo, hi));
        Solid {
            size: (hi - lo).length().max(1e-12),
            tris,
            grids,
        }
    }

    /// Whether `p` is inside: three rays, one along each axis, and the
    /// majority of what their crossings say. Nudged off the axes' lines by a
    /// little, so a ray along a box's face does not run down its edge.
    fn inside(&self, p: DVec3) -> bool {
        let nudge = DVec3::new(0.731_9, 0.411_3, 0.193_7) * self.size * 1.3e-8;
        let votes = self
            .grids
            .iter()
            .filter(|g| g.crossings(&self.tris, p + nudge) % 2 == 1)
            .count();
        votes >= 2
    }
}

/// Where a piece of one surface is, against the other solid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Inside,
    Outside,
    /// On the other's surface, both facing the same way.
    Same,
    /// On the other's surface, facing each other: solids that touch.
    Opposite,
}

/// The polygons of `a` combined with `b` without a tree: each polygon is cut
/// only by the planes of the other's polygons it actually meets, and every
/// piece, cut or whole, is sorted by whether it lies inside the other —
/// work that grows with the faces, not their square, so a sculpted or
/// finely divided mesh takes seconds rather than hours.
fn arrangement(
    a: &MeshData,
    b: &MeshData,
    op: BooleanOp,
    budget: &mut Budget,
) -> Result<Vec<Polygon>, TooComplex> {
    let pa = polygons_of(a, 0);
    let pb = polygons_of(b, a.num_faces());
    let (sa, sb) = (Solid::new(&pa), Solid::new(&pb));
    let mut out = Vec::new();
    for (mine, theirs, other, from_a) in [(&pa, &pb, &sb, true), (&pb, &pa, &sa, false)] {
        // The other's polygons, by box, for finding which meet each of mine.
        let boxes: Vec<(Vec3, Vec3)> = theirs
            .iter()
            .map(|p| {
                let (lo, hi) = bounds_of(p.verts.iter());
                (lo.as_vec3(), hi.as_vec3())
            })
            .collect();
        let tris: Vec<[u32; 3]> = (0..boxes.len() as u32)
            .map(|i| [2 * i, 2 * i + 1, 2 * i])
            .collect();
        let corners: Vec<Vec3> = boxes.iter().flat_map(|(lo, hi)| [*lo, *hi]).collect();
        let tree = kreate_math::Bvh::build(&corners, &tris);
        let mut found = Vec::new();
        for p in mine.iter() {
            if budget.spent() {
                return Err(TooComplex);
            }
            budget.spend(p.verts.len());
            let (lo, hi) = bounds_of(p.verts.iter());
            let pad = Vec3::splat((EPSILON * 4.0) as f32);
            tree.overlaps(
                &kreate_math::Aabb::new(lo.as_vec3() - pad, hi.as_vec3() + pad),
                &mut found,
            );
            // (in front, behind, on) the plane, for some corner.
            let side_of = |plane: &Plane, verts: &[DVec3]| {
                let (mut f, mut b, mut o) = (false, false, false);
                for v in verts {
                    let d = plane.normal.dot(*v) - plane.w;
                    f |= d > EPSILON;
                    b |= d < -EPSILON;
                    o |= d.abs() <= EPSILON;
                }
                (f, b, o)
            };
            let mut pieces = vec![p.clone()];
            for q in found.iter().map(|i| &theirs[*i as usize]) {
                // Meets: its plane cuts mine across, and it reaches my plane —
                // through it, or up to it with an edge, as a box's side does
                // the top of another box it stands on.
                let (f, b, _) = side_of(&q.plane, &p.verts);
                let (qf, qb, qo) = side_of(&p.plane, &q.verts);
                if !(f && b) || !(qo || (qf && qb)) {
                    continue;
                }
                let mut next = Vec::with_capacity(pieces.len() + 1);
                for piece in pieces {
                    let (mut cf, mut cb, mut f, mut b) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    split(&q.plane, piece, &mut cf, &mut cb, &mut f, &mut b, budget);
                    next.extend(cf);
                    next.extend(cb);
                    next.extend(f);
                    next.extend(b);
                }
                pieces = next;
            }
            for piece in pieces {
                let centre =
                    piece.verts.iter().fold(DVec3::ZERO, |s, v| s + *v) / piece.verts.len() as f64;
                // As far off as a plane test reaches and no further: what is
                // nearer than that is touching, what is farther apart.
                let e = piece.plane.normal * EPSILON * 2.0;
                let (front, back) = (other.inside(centre + e), other.inside(centre - e));
                let side = match (front, back) {
                    (true, true) => Side::Inside,
                    (false, false) => Side::Outside,
                    (false, true) => Side::Same,
                    (true, false) => Side::Opposite,
                };
                let keep = match (op, from_a, side) {
                    (BooleanOp::Union, _, Side::Outside) => Some(false),
                    (BooleanOp::Union, true, Side::Same) => Some(false),
                    (BooleanOp::Difference, true, Side::Outside | Side::Opposite) => Some(false),
                    (BooleanOp::Difference, false, Side::Inside) => Some(true),
                    (BooleanOp::Intersect, _, Side::Inside) => Some(false),
                    (BooleanOp::Intersect, true, Side::Same) => Some(false),
                    _ => None,
                };
                if let Some(flip) = keep {
                    let mut piece = piece;
                    if flip {
                        piece.flip();
                    }
                    out.push(piece);
                }
            }
        }
    }
    Ok(out)
}

/// How close two points are to be the same point, and a point to lie on an
/// edge. Absolute, in the mesh's units: a boolean's numbers are plane tests
/// accurate to [`EPSILON`], and this is ten times looser than that.
const WELD: f64 = 1e-5;

/// The BSP's polygons to a closed mesh: see the module docs.
fn assemble(polygons: Vec<Polygon>, untouched: &[Option<Vec<DVec3>>]) -> MeshData {
    let mut welder = Welder::default();
    let mut pieces: Vec<(u32, Vec<u32>)> = Vec::with_capacity(polygons.len());
    // A face the other mesh comes nowhere near goes back as it came, once,
    // whatever its own tree cut it into: rebuilt from its pieces it came back
    // a little different each time — a face a boolean had already left
    // curved, cut into triangles and merged again, crossed itself — and a
    // part with several holes, cut one after another, came back open round
    // the first of them.
    let mut whole: Vec<bool> = vec![false; untouched.len()];
    for p in &polygons {
        if let Some(Some(ring)) = untouched.get(p.source as usize) {
            if std::mem::replace(&mut whole[p.source as usize], true) {
                continue;
            }
            // Facing as it did: a face clear of the other mesh is kept only
            // where it is outside it — either input's in a union, the
            // first's in a difference — and the outside keeps its way.
            // (Asked of its pieces instead, a twisted quad of a thread got
            // it wrong: its two triangles face apart.)
            let ids: Vec<u32> = ring.iter().map(|v| welder.index(*v)).collect();
            pieces.push((p.source, ids));
            continue;
        }
        let mut ring: Vec<u32> = Vec::with_capacity(p.verts.len());
        for v in &p.verts {
            let i = welder.index(*v);
            if ring.last() != Some(&i) {
                ring.push(i);
            }
        }
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() >= 3 {
            pieces.push((p.source, ring));
        }
    }
    let positions = welder.positions;
    split_t_junctions(&positions, &mut pieces);

    // The pieces of each input face, merged back together.
    pieces.sort_by_key(|(source, _)| *source);
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut at = 0;
    while at < pieces.len() {
        let source = pieces[at].0;
        let end = at
            + pieces[at..]
                .iter()
                .take_while(|(s, _)| *s == source)
                .count();
        let group: Vec<Vec<u32>> = pieces[at..end].iter().map(|(_, r)| r.clone()).collect();
        faces.extend(merge_pieces(group));
        at = end;
    }

    dissolve_straight_vertices(&positions, &mut faces);
    let mut faces = simple_loops(faces);

    // Only the vertices still in use, renumbered in the order faces meet them.
    let mut renumber = vec![u32::MAX; positions.len()];
    let mut kept: Vec<Vec3> = Vec::new();
    for face in &mut faces {
        for v in face.iter_mut() {
            if renumber[*v as usize] == u32::MAX {
                renumber[*v as usize] = kept.len() as u32;
                kept.push(positions[*v as usize].as_vec3());
            }
            *v = renumber[*v as usize];
        }
    }
    let mesh = MeshData::from_faces(kept, &faces);
    // Pieces of different faces that came to lie in one plane — the shared
    // top of two boxes joined — merged, so the result is as tidy as it is
    // correct. Left as it was when the tidying cannot be done cleanly.
    let mut bm = mesh.to_bmesh();
    match crate::edit::limited_dissolve(&mut bm, 1e-4) {
        Ok(0) | Err(_) => mesh,
        Ok(_) => {
            // Kept only if it opened nothing: merging faces across a fine
            // cut has been seen to leave an edge with one face.
            let tidied = bm.to_mesh_data();
            if tidied.validate().is_empty() && unmatched_edges(&tidied) <= unmatched_edges(&mesh) {
                tidied
            } else {
                mesh
            }
        }
    }
}

/// Edges that are not one face each way: holes, T-junctions, and edges
/// three or more faces meet at.
fn unmatched_edges(m: &MeshData) -> usize {
    // Each edge once, by its two ends; on a closed manifold every edge has
    // two faces, one each way.
    let mut e: std::collections::HashMap<(u32, u32), (u32, i32)> = std::collections::HashMap::new();
    for f in 0..m.num_faces() {
        let r = m.face_verts(f);
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            let entry = e.entry((a.min(b), a.max(b))).or_default();
            entry.0 += 1;
            entry.1 += if a < b { 1 } else { -1 };
        }
    }
    e.values().filter(|(n, d)| *n != 2 || *d != 0).count()
}

/// Faces that pass each corner once. Welding can bring two corners of one
/// piece together — a sliver thinner than the weld distance — and merging
/// pieces can walk a ring through a vertex twice where two pieces touch at a
/// point. A face may not name a vertex twice, so a repeat right after itself
/// is dropped, and a ring that comes back to a vertex further on is cut there
/// into the two rings it is.
fn simple_loops(faces: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let mut out = Vec::with_capacity(faces.len());
    let mut work = faces;
    while let Some(mut ring) = work.pop() {
        ring.dedup();
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            continue;
        }
        let repeat = (0..ring.len()).find_map(|i| {
            ring[i + 1..]
                .iter()
                .position(|v| *v == ring[i])
                .map(|d| (i, i + 1 + d))
        });
        match repeat {
            None => out.push(ring),
            Some((i, j)) => {
                let inner = ring[i..j].to_vec();
                let mut outer = ring[j..].to_vec();
                outer.extend_from_slice(&ring[..i]);
                work.push(inner);
                work.push(outer);
            }
        }
    }
    out
}

/// Points to indices, a point within [`WELD`] of one already seen taking its
/// index. Searches the neighbouring cells too: rounding alone puts two points
/// a hair apart on either side of a cell boundary into different vertices.
#[derive(Default)]
struct Welder {
    positions: Vec<DVec3>,
    cells: std::collections::HashMap<[i64; 3], Vec<u32>>,
}

impl Welder {
    fn cell(p: DVec3) -> [i64; 3] {
        [
            (p.x / WELD).floor() as i64,
            (p.y / WELD).floor() as i64,
            (p.z / WELD).floor() as i64,
        ]
    }

    fn index(&mut self, p: DVec3) -> u32 {
        let c = Self::cell(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = self.cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        if let Some(&i) = list
                            .iter()
                            .find(|i| self.positions[**i as usize].distance(p) <= WELD)
                        {
                            return i;
                        }
                    }
                }
            }
        }
        let i = self.positions.len() as u32;
        self.positions.push(p);
        self.cells.entry(c).or_default().push(i);
        i
    }
}

/// Puts every vertex that lies on a piece's edge into that edge.
///
/// A face split at a point its neighbour across the edge was not split at is
/// a T-junction: the mesh looks closed and is not, and the pieces of one face
/// cannot be merged along an edge they do not share exactly. Candidates come
/// from whichever axis the edge spans least, so a short or axis-aligned edge
/// looks at a handful of vertices rather than all of them.
fn split_t_junctions(positions: &[DVec3], pieces: &mut [(u32, Vec<u32>)]) {
    let sorted: [Vec<u32>; 3] = std::array::from_fn(|axis| {
        let mut order: Vec<u32> = (0..positions.len() as u32).collect();
        order.sort_by(|a, b| positions[*a as usize][axis].total_cmp(&positions[*b as usize][axis]));
        order
    });
    let mut on_edge: Vec<(f64, u32)> = Vec::new();
    for (_, ring) in pieces.iter_mut() {
        let n = ring.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let (a, b) = (ring[k], ring[(k + 1) % n]);
            out.push(a);
            let (pa, pb) = (positions[a as usize], positions[b as usize]);
            let d = pb - pa;
            let len2 = d.length_squared();
            if len2 <= WELD * WELD {
                continue;
            }
            // Searched along whichever axis has the fewest points in the
            // edge's span: the edge's shortest extent was the guess, and on
            // a flat face every point has the same height, so an edge
            // across a plate's top was searched against all of them.
            let range = |axis: usize| {
                let (lo, hi) = (pa[axis].min(pb[axis]) - WELD, pa[axis].max(pb[axis]) + WELD);
                let list = &sorted[axis];
                let from = list.partition_point(|i| positions[*i as usize][axis] < lo);
                let to = list.partition_point(|i| positions[*i as usize][axis] <= hi);
                (from, to.max(from))
            };
            let (axis, (from, to)) = (0..3)
                .map(|axis| (axis, range(axis)))
                .min_by_key(|(_, (from, to))| to - from)
                .expect("three axes");
            let list = &sorted[axis];
            on_edge.clear();
            for &v in &list[from..to] {
                if v == a || v == b {
                    continue;
                }
                let p = positions[v as usize];
                let t = (p - pa).dot(d) / len2;
                if t <= 0.0 || t >= 1.0 {
                    continue;
                }
                if (pa + d * t).distance(p) <= WELD {
                    on_edge.push((t, v));
                }
            }
            on_edge.sort_by(|x, y| x.0.total_cmp(&y.0));
            out.extend(on_edge.iter().map(|(_, v)| *v));
        }
        *ring = out;
    }
}

/// The pieces of one face, merged into as few polygons as stay simple.
///
/// Two pieces merge when they share one unbroken run of edges, which is
/// exactly when their union has no hole in it: a face with a hole through it
/// cannot be one polygon, and stays as however many it takes. Untouched by
/// the cut, the pieces come back together as the face they were.
/// The most corners [`merge_pieces`] grows a face to.
const MAX_MERGED: usize = 256;

fn merge_pieces(rings: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    // Only pieces that share an edge can merge, so each is tried against
    // those across its edges, found in a table of who owns which edge —
    // trying every pair, again after every merge, was most of a boolean on
    // a face a thread had cut into thousands of pieces.
    let mut rings: Vec<Option<Vec<u32>>> = rings.into_iter().map(Some).collect();
    let mut owner: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    let edges = |r: &[u32]| {
        (0..r.len())
            .map(|k| (r[k], r[(k + 1) % r.len()]))
            .collect::<Vec<_>>()
    };
    for (i, r) in rings.iter().enumerate() {
        for e in edges(r.as_ref().expect("all present")) {
            owner.insert(e, i);
        }
    }
    let mut work: Vec<usize> = (0..rings.len()).rev().collect();
    while let Some(i) = work.pop() {
        let Some(ring) = rings[i].clone() else {
            continue;
        };
        let mut neighbours: Vec<usize> = edges(&ring)
            .iter()
            .filter_map(|(a, b)| owner.get(&(*b, *a)).copied())
            .filter(|j| *j != i && rings[*j].is_some())
            .collect();
        neighbours.sort_unstable();
        neighbours.dedup();
        for j in neighbours {
            let other = rings[j].clone().expect("checked");
            // Not grown past a few hundred corners: merging one more piece
            // into a ring looks at all of it, and a plate's face round many
            // holes made that the square of thousands.
            if ring.len() + other.len() > MAX_MERGED {
                continue;
            }
            if let Some(union) = merge_two(&ring, &other) {
                for e in edges(&ring).into_iter().chain(edges(&other)) {
                    owner.remove(&e);
                }
                for e in edges(&union) {
                    owner.insert(e, i);
                }
                rings[i] = Some(union);
                rings[j] = None;
                // Grown, it may now meet pieces it did not before.
                work.push(i);
                break;
            }
        }
    }
    rings.into_iter().flatten().collect()
}

/// `p` and `q` as one polygon, if they share one unbroken run of edges and
/// the result is simple.
fn merge_two(p: &[u32], q: &[u32]) -> Option<Vec<u32>> {
    let (n, m) = (p.len(), q.len());
    // Shared: p's edge a→b is q's edge b→a.
    let q_edges: std::collections::HashSet<(u32, u32)> =
        (0..m).map(|k| (q[k], q[(k + 1) % m])).collect();
    let shared: Vec<bool> = (0..n)
        .map(|k| q_edges.contains(&(p[(k + 1) % n], p[k])))
        .collect();
    let count = shared.iter().filter(|s| **s).count();
    if count == 0 || count == n {
        return None;
    }
    // One run: exactly one place where an unshared edge is followed by a shared one.
    let starts: Vec<usize> = (0..n)
        .filter(|k| shared[*k] && !shared[(k + n - 1) % n])
        .collect();
    if starts.len() != 1 {
        return None;
    }
    let s = starts[0];
    // p[s] .. p[s + count] is the shared chain; q runs it backwards, ending at p[s].
    let first = p[s];
    let last = p[(s + count) % n];
    let at = q.iter().position(|v| *v == first)?;
    let mut union = Vec::with_capacity(n + m - 2 * count);
    // p from the chain's end round to its start...
    for k in 0..=(n - count) {
        union.push(p[(s + count + k) % n]);
    }
    // ...then q from just after the chain's start round to just before its end.
    for k in 1..m {
        let v = q[(at + k) % m];
        if v == last {
            break;
        }
        union.push(v);
    }
    if union.len() != n + m - 2 * count {
        return None;
    }
    // A vertex twice is a pinch: the two touched at a point as well.
    let mut seen = union.clone();
    seen.sort_unstable();
    seen.dedup();
    (seen.len() == union.len()).then_some(union)
}

/// Takes out every vertex that only passes along a straight edge.
///
/// What the T-junction repair put in where a neighbour was split and the
/// merge then made unnecessary. A vertex with exactly two neighbours, in line
/// with them, is on one edge between two faces, so taking it out of both
/// keeps the mesh closed.
fn dissolve_straight_vertices(positions: &[DVec3], faces: &mut [Vec<u32>]) {
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    for face in faces.iter() {
        let n = face.len();
        for k in 0..n {
            let (a, b) = (face[k], face[(k + 1) % n]);
            for (x, y) in [(a, b), (b, a)] {
                if !neighbours[x as usize].contains(&y) {
                    neighbours[x as usize].push(y);
                }
            }
        }
    }
    let mut straight: Vec<bool> = (0..positions.len())
        .map(|v| {
            let [a, b] = neighbours[v][..] else {
                return false;
            };
            let (pa, pb, p) = (positions[a as usize], positions[b as usize], positions[v]);
            let d = pb - pa;
            let t = (p - pa).dot(d) / d.length_squared().max(f64::MIN_POSITIVE);
            t > 0.0 && t < 1.0 && (pa + d * t).distance(p) <= WELD
        })
        .collect();
    // Out of every face or none: a face left too few corners keeps its own,
    // and the face across the edge has to keep them too.
    for face in faces.iter() {
        if face.iter().filter(|v| !straight[**v as usize]).count() < 3 {
            for v in face {
                straight[*v as usize] = false;
            }
        }
    }
    for face in faces.iter_mut() {
        face.retain(|v| !straight[*v as usize]);
    }
}

#[cfg(test)]
mod tests {
    /// A finely divided sphere with a box cut out of its middle: past what
    /// the tree could do in time, the arrangement does it closed, and the
    /// volume is the sphere's less the box's.
    #[test]
    fn a_fine_sphere_is_cut_by_the_arrangement() {
        let sphere = crate::primitives::uv_sphere(1.0, 128, 64);
        assert!(sphere.num_faces() > ARRANGE_ABOVE);
        let cube = crate::primitives::cube(1.0);
        let whole = crate::measure::mass_properties(&sphere, |p| p).volume;
        let t = std::time::Instant::now();
        let r = try_boolean(&sphere, &cube, BooleanOp::Difference).expect("in time");
        assert!(t.elapsed().as_secs_f64() < 8.0);
        assert_eq!(unmatched_edges(&r), 0);
        let v = crate::measure::mass_properties(&r, |p| p).volume;
        assert!(
            (v - (whole - 1.0)).abs() < 1e-3,
            "{v} against {}",
            whole - 1.0
        );
        // And a union with a box poking out: closed too.
        let mut out = crate::primitives::cube(1.0);
        for p in out.positions_mut() {
            p.x += 1.0;
        }
        let u = try_boolean(&sphere, &out, BooleanOp::Union).expect("in time");
        assert_eq!(unmatched_edges(&u), 0);
    }

    /// More solids that touch without overlapping, at a part's size: turned,
    /// offset so only part of a face meets, several bosses on a plate one
    /// after another, and triangles against quads.
    #[test]
    fn more_touching_solids_unite_closed() {
        let turn = |m: &mut MeshData, a: f32, off: Vec3, s: Vec3| {
            let q = glam::Quat::from_rotation_z(a);
            for p in m.positions_mut() {
                *p = q * (*p * s) + off;
            }
        };
        let mm = 0.001f32;
        // Rotated boxes face to face.
        let mut a = crate::primitives::cube(1.0);
        turn(&mut a, 0.5, Vec3::ZERO, Vec3::new(20.0, 10.0, 3.0) * mm);
        let mut b = crate::primitives::cube(1.0);
        let side = glam::Quat::from_rotation_z(0.5) * Vec3::new(15.0 * mm, 0.0, 0.0);
        turn(&mut b, 0.5, side, Vec3::new(10.0, 6.0, 3.0) * mm);
        let r = try_boolean(&a, &b, BooleanOp::Union).unwrap();
        assert_eq!(unmatched_edges(&r), 0, "rotated");
        // Offset in two axes, partly touching.
        let mut a = crate::primitives::cube(1.0);
        turn(&mut a, 0.0, Vec3::ZERO, Vec3::new(20.0, 10.0, 3.0) * mm);
        let mut b = crate::primitives::cube(1.0);
        turn(
            &mut b,
            0.0,
            Vec3::new(13.0, 6.0, 1.0) * mm,
            Vec3::new(6.0, 6.0, 3.0) * mm,
        );
        let r = try_boolean(&a, &b, BooleanOp::Union).unwrap();
        assert_eq!(unmatched_edges(&r), 0, "offset");
        // Several bosses on a plate, one after another.
        let mut plate = crate::primitives::cube(1.0);
        turn(&mut plate, 0.0, Vec3::ZERO, Vec3::new(60.0, 40.0, 3.0) * mm);
        for x in [-20.0f32, 0.0, 20.0] {
            let mut boss = crate::primitives::cylinder(4.0 * mm, 6.0 * mm, 24);
            let lo = boss
                .positions()
                .iter()
                .map(|p| p.z)
                .fold(f32::MAX, f32::min);
            for p in boss.positions_mut() {
                p.z += 1.5 * mm - lo;
                p.x += x * mm;
            }
            plate = try_boolean(&plate, &boss, BooleanOp::Union).unwrap();
            assert_eq!(unmatched_edges(&plate), 0, "boss");
        }
        // A triangulated block against a quad one.
        let mut a = crate::primitives::cube(1.0);
        turn(&mut a, 0.0, Vec3::ZERO, Vec3::splat(10.0 * mm));
        let c = crate::primitives::cube(1.0);
        let tri = c.triangulate();
        let (tris, _) = tri.indices.as_chunks::<3>();
        let faces: Vec<Vec<u32>> = tris.iter().map(|t| t.to_vec()).collect();
        let mut b = MeshData::from_faces(c.positions().to_vec(), &faces);
        turn(
            &mut b,
            0.0,
            Vec3::new(10.0 * mm, 2.0 * mm, 0.0),
            Vec3::splat(10.0 * mm),
        );
        let r = try_boolean(&a, &b, BooleanOp::Union).unwrap();
        assert_eq!(unmatched_edges(&r), 0, "triangles");
    }

    /// Two solids that only touch, face to face, joined: one closed solid,
    /// no seam left open where the smaller face meets the larger.
    #[test]
    fn solids_touching_face_to_face_unite_closed() {
        let cases: [(f32, Vec3); 3] = [
            // A small box against the middle of a big one's side.
            (1.0, Vec3::new(1.5, 0.0, 0.0)),
            // Against a corner, half on and half off.
            (1.0, Vec3::new(1.5, 0.7, 0.4)),
            // Two the same size, face to face.
            (2.0, Vec3::new(2.0, 0.0, 0.0)),
        ];
        for (size, at) in cases {
            let a = crate::primitives::cube(2.0);
            let mut b = crate::primitives::cube(size);
            for p in b.positions_mut() {
                *p += at;
            }
            let r = try_boolean(&a, &b, BooleanOp::Union).unwrap();
            assert_eq!(unmatched_edges(&r), 0, "{size} at {at}");
            let v = crate::measure::mass_properties(&r, |p| p).volume;
            let want = 8.0 + (size as f64).powi(3);
            assert!(
                (v - want).abs() < 1e-3,
                "{size} at {at}: {v} against {want}"
            );
        }
        // At a part's size: a round boss standing on a 3 mm plate, its foot
        // a hair off the plate's top, as positions typed in millimetres are.
        let mut plate = crate::primitives::cube(1.0);
        for p in plate.positions_mut() {
            *p *= Vec3::new(0.06, 0.04, 0.003);
        }
        for gap in [0.0f32, 1e-7, -1e-7] {
            let mut boss = crate::primitives::cylinder(0.004, 0.006, 32);
            let lo = boss
                .positions()
                .iter()
                .map(|p| p.z)
                .fold(f32::MAX, f32::min);
            for p in boss.positions_mut() {
                p.z += 0.0015 - lo + gap;
                p.x += 0.012;
            }
            let r = try_boolean(&plate, &boss, BooleanOp::Union).unwrap();
            assert_eq!(unmatched_edges(&r), 0, "boss off by {gap}");
        }
    }

    /// A weld by distance on a boolean's result keeps it closed: merging two
    /// vertices of one long face — round a threaded hole — pinched the face,
    /// and the merge threw the whole face away.
    #[test]
    fn welding_a_boolean_result_keeps_it_closed() {
        let mut plate = crate::primitives::cube(1.0);
        for p in plate.positions_mut() {
            *p *= Vec3::new(0.06, 0.04, 0.003);
        }
        let mut c = crate::mech::internal_thread_cutter(0.003, 0.0005, 0.01, 0.0);
        for p in c.positions_mut() {
            *p += Vec3::new(0.015, 0.01, -0.005);
        }
        let r = try_boolean(&plate, &c, BooleanOp::Difference).unwrap();
        assert_eq!(unmatched_edges(&r), 0);
        for d in [1e-6f32, 1e-5] {
            let mut bm = r.to_bmesh();
            let all: Vec<_> = bm.vert_ids().collect();
            let m = crate::edit::weld_by_distance(&mut bm, &all, d).unwrap();
            let out = bm.to_mesh_data();
            let _ = m;
            assert_eq!(unmatched_edges(&out), 0, "weld {d}");
            assert!(
                out.num_faces() > r.num_faces() - 40,
                "weld {d}: {} faces",
                out.num_faces()
            );
        }
    }

    /// A 3 mm plate cut by a modelled M3 thread, a plain hole and another
    /// thread, one after another, as a part is made: closed, every edge one
    /// face each way, after each. At this size the welding distance, as an
    /// absolute length, once shut the thread's slivers and opened the plate
    /// along them; and each later cut, rebuilding the faces round the
    /// earlier holes from pieces, folded a few of them over.
    #[test]
    fn a_plate_cut_by_threads_one_after_another_stays_closed() {
        let mut plate = crate::primitives::cube(1.0);
        for p in plate.positions_mut() {
            *p *= Vec3::new(0.06, 0.04, 0.003);
        }
        for (k, (x, y)) in [(0.015f32, 0.01f32), (-0.015, 0.01), (0.0, -0.01)]
            .iter()
            .enumerate()
        {
            let mut c = if k == 1 {
                crate::primitives::cylinder(0.0017, 0.01, 24)
            } else {
                crate::mech::internal_thread_cutter(0.003, 0.0005, 0.01, 0.0)
            };
            for p in c.positions_mut() {
                *p += Vec3::new(*x, *y, -0.005);
            }
            plate = try_boolean(&plate, &c, BooleanOp::Difference).expect("in time");
            assert_eq!(unmatched_edges(&plate), 0, "after cut {k}");
            assert!(plate.validate().is_empty(), "after cut {k}");
        }
    }

    #[test]
    fn a_ring_through_one_vertex_twice_becomes_two_faces() {
        // A bow tie through vertex 0, and a doubled corner.
        let mut loops = simple_loops(vec![vec![0, 1, 2, 0, 3, 4], vec![5, 6, 6, 7, 5]]);
        for l in &mut loops {
            let at = l.iter().position(|v| v == l.iter().min().unwrap()).unwrap();
            l.rotate_left(at);
        }
        loops.sort();
        assert_eq!(loops, vec![vec![0, 1, 2], vec![0, 3, 4], vec![5, 6, 7]]);
    }

    /// How long a boolean on a rounded mesh takes, and where the budget
    /// stops it: run by hand, in release.
    #[test]
    #[ignore]
    fn timing() {
        for (seg, rings) in [(64, 32), (128, 64), (256, 128), (512, 256)] {
            let s = crate::primitives::uv_sphere(1.0, seg, rings);
            let c = crate::primitives::cube(1.0);
            let t = std::time::Instant::now();
            let r = try_boolean(&s, &c, BooleanOp::Difference);
            println!(
                "sphere {} faces: {:?} in {:?}",
                s.num_faces(),
                r.as_ref().map(|m| m.num_faces()),
                t.elapsed()
            );
        }
        let thread = crate::mech::internal_thread_cutter(0.003, 0.0005, 0.01, 0.0);
        let plate = crate::primitives::cube(0.02);
        let t = std::time::Instant::now();
        let r = try_boolean(&plate, &thread, BooleanOp::Difference);
        println!(
            "M3 thread {} faces: {:?} in {:?}",
            thread.num_faces(),
            r.as_ref().map(|m| m.num_faces()),
            t.elapsed()
        );
    }

    use super::*;
    use crate::primitives;

    fn moved(mut m: MeshData, by: Vec3) -> MeshData {
        for p in m.positions_mut() {
            *p += by;
        }
        m
    }

    /// The volume a closed mesh encloses, by the divergence theorem.
    fn volume(m: &MeshData) -> f64 {
        let p = m.positions();
        let mut v = 0.0;
        for f in 0..m.num_faces() {
            let ring = m.face_verts(f);
            for i in 1..ring.len() - 1 {
                let (a, b, c) = (
                    p[ring[0] as usize].as_dvec3(),
                    p[ring[i] as usize].as_dvec3(),
                    p[ring[i + 1] as usize].as_dvec3(),
                );
                v += a.dot(b.cross(c)) / 6.0;
            }
        }
        v
    }

    /// Edges not matched by the same edge the other way round: holes and
    /// T-junctions. Zero for a closed mesh.
    fn open_edges(m: &MeshData) -> usize {
        let mut e: std::collections::HashMap<(u32, u32), i32> = std::collections::HashMap::new();
        for f in 0..m.num_faces() {
            let r = m.face_verts(f);
            for i in 0..r.len() {
                let (a, b) = (r[i], r[(i + 1) % r.len()]);
                *e.entry((a, b)).or_default() += 1;
                *e.entry((b, a)).or_default() -= 1;
            }
        }
        e.values().filter(|v| **v != 0).count() / 2
    }

    /// The volume of what the renderer would draw: through the real
    /// triangulation, so a merged non-convex face that ear clipping got wrong
    /// shows up here even though its polygon is right.
    fn drawn_volume(m: &MeshData) -> f64 {
        let p = m.positions();
        m.triangulate()
            .indices
            .chunks(3)
            .map(|t| {
                let (a, b, c) = (
                    p[t[0] as usize].as_dvec3(),
                    p[t[1] as usize].as_dvec3(),
                    p[t[2] as usize].as_dvec3(),
                );
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    fn rotated_to_x(mut m: MeshData) -> MeshData {
        for p in m.positions_mut() {
            *p = Vec3::new(p.z, p.y, -p.x);
        }
        m
    }

    #[test]
    fn the_result_is_closed_and_draws_the_volume_it_encloses() {
        let cube = primitives::cube(2.0);
        let rod = primitives::cylinder(0.5, 4.0, 32);
        let cases = [
            ("hole", boolean(&cube, &rod, BooleanOp::Difference)),
            (
                "sphere out",
                boolean(
                    &cube,
                    &primitives::uv_sphere(1.2, 32, 16),
                    BooleanOp::Difference,
                ),
            ),
            (
                "sphere in",
                boolean(
                    &cube,
                    &primitives::uv_sphere(1.2, 32, 16),
                    BooleanOp::Intersect,
                ),
            ),
            (
                "pocket",
                boolean(
                    &cube,
                    &moved(primitives::cube(1.0), Vec3::new(0.0, 0.0, 0.5)),
                    BooleanOp::Difference,
                ),
            ),
            (
                "stacked",
                boolean(
                    &cube,
                    &moved(primitives::cube(1.0), Vec3::new(0.0, 0.0, 1.5)),
                    BooleanOp::Union,
                ),
            ),
            (
                "torus",
                boolean(
                    &cube,
                    &primitives::torus(1.0, 0.3, 24, 12),
                    BooleanOp::Difference,
                ),
            ),
        ];
        for (name, m) in &cases {
            assert_eq!(open_edges(m), 0, "{name}: open edges");
            assert!(
                (drawn_volume(m) - volume(m)).abs() < 1e-3,
                "{name}: drawn {} encloses {}",
                drawn_volume(m),
                volume(m)
            );
        }
        // A second hole across the first, on the first's result: what a stack
        // of two booleans does.
        let twice = boolean(
            &cases[0].1,
            &rotated_to_x(rod.clone()),
            BooleanOp::Difference,
        );
        assert_eq!(open_edges(&twice), 0, "second hole: open edges");
        assert!((drawn_volume(&twice) - volume(&twice)).abs() < 1e-3);
    }

    #[test]
    fn faces_the_cut_does_not_reach_stay_whole() {
        // A rod through the top and bottom: the four sides are untouched and
        // come back as the four quads they were, not as strips cut along the
        // rod's facets.
        let out = boolean(
            &primitives::cube(2.0),
            &primitives::cylinder(0.5, 4.0, 32),
            BooleanOp::Difference,
        );
        let p = out.positions();
        let sides = (0..out.num_faces())
            .filter(|f| {
                let ring = out.face_verts(*f);
                let n = crate::tessellate::newell_normal(
                    &ring.iter().map(|v| p[*v as usize]).collect::<Vec<_>>(),
                );
                n.z.abs() < 0.5
                    && ring
                        .iter()
                        .all(|v| p[*v as usize].x.abs() > 0.999 || p[*v as usize].y.abs() > 0.999)
            })
            .count();
        assert_eq!(sides, 4, "the sides came back in pieces");
        // The top and bottom each have a hole, so each is a few polygons
        // rather than one — but a few, not the fan of 32 slivers it was.
        assert!(
            out.num_faces() < 6 + 32 + 2 * 12,
            "{} faces",
            out.num_faces()
        );
    }

    #[test]
    fn two_unit_overlapping_cubes_give_the_volumes_arithmetic_says() {
        // Two 2-unit cubes, the second moved one unit along X: they overlap in
        // a 1 x 2 x 2 box.
        let a = primitives::cube(2.0);
        let b = moved(primitives::cube(2.0), Vec3::X);
        let near = |got: f64, want: f64| (got - want).abs() < 1e-3;
        let difference = boolean(&a, &b, BooleanOp::Difference);
        assert!(
            near(volume(&difference), 4.0),
            "difference {}",
            volume(&difference)
        );
        let union = boolean(&a, &b, BooleanOp::Union);
        assert!(near(volume(&union), 12.0), "union {}", volume(&union));
        let intersect = boolean(&a, &b, BooleanOp::Intersect);
        assert!(
            near(volume(&intersect), 4.0),
            "intersect {}",
            volume(&intersect)
        );
        assert!(intersect.bounds().min.x > -0.01 && intersect.bounds().max.x < 1.01);
    }

    #[test]
    fn a_cutter_that_misses_leaves_the_mesh_as_it_was() {
        let a = primitives::cube(2.0);
        let b = moved(primitives::cube(1.0), Vec3::new(10.0, 0.0, 0.0));
        let out = boolean(&a, &b, BooleanOp::Difference);
        assert!((volume(&out) - 8.0).abs() < 1e-4);
        assert!((volume(&boolean(&a, &b, BooleanOp::Intersect))).abs() < 1e-6);
    }

    #[test]
    fn a_hole_through_a_cube_takes_out_what_it_should() {
        // A 0.5-wide square rod right through a 2-unit cube: 8 - 0.25 * 2.
        let a = primitives::cube(2.0);
        let mut rod = primitives::cube(1.0);
        for p in rod.positions_mut() {
            p.x *= 0.5;
            p.y *= 0.5;
            p.z *= 4.0;
        }
        let out = boolean(&a, &rod, BooleanOp::Difference);
        assert!((volume(&out) - 7.5).abs() < 1e-3, "{}", volume(&out));
    }

    /// A box whose top is warped: one corner raised, as moving a vertex in
    /// edit mode leaves it. The tree used to split such a face against its
    /// own plane without end, until the memory ran out.
    #[test]
    fn a_warped_face_is_cut_and_the_cut_ends() {
        let mut a = primitives::cube(2.0);
        let top = a
            .positions()
            .iter()
            .enumerate()
            .find(|(_, p)| p.x > 0.0 && p.y > 0.0 && p.z > 0.0)
            .unwrap()
            .0;
        a.positions_mut()[top].z += 0.3;
        let cutter = moved(
            primitives::cylinder(0.4, 4.0, 32),
            Vec3::new(0.3, 0.2, -1.0),
        );
        let start = std::time::Instant::now();
        let out = boolean(&a, &cutter, BooleanOp::Difference);
        assert!(start.elapsed().as_secs() < 5, "{:?}", start.elapsed());
        assert!(out.num_faces() > 0);
        assert_eq!(open_edges(&out), 0, "closed");
        assert!(volume(&out) < volume(&a), "something was taken out");
    }

    /// A concave prism, an L, cut through its corner: the cut takes what
    /// a cylinder there holds and leaves the rest closed.
    #[test]
    fn a_concave_face_is_cut_as_a_whole() {
        let outline = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        let mut positions = Vec::new();
        for z in [0.0f32, 1.0] {
            for (x, y) in outline {
                positions.push(Vec3::new(x, y, z));
            }
        }
        let n = outline.len() as u32;
        let mut faces: Vec<Vec<u32>> = vec![(0..n).rev().collect(), (n..2 * n).collect()];
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push(vec![i, j, n + j, n + i]);
        }
        let l = MeshData::from_faces(positions, &faces);
        assert!((volume(&l) - 3.0).abs() < 1e-6);
        let cutter = moved(
            primitives::cylinder(0.3, 4.0, 48),
            Vec3::new(0.5, 0.5, -1.0),
        );
        let out = boolean(&l, &cutter, BooleanOp::Difference);
        assert_eq!(open_edges(&out), 0, "closed");
        let hole = std::f64::consts::PI * 0.09;
        assert!(
            (volume(&out) - (3.0 - hole)).abs() < 0.01,
            "{}",
            volume(&out)
        );
    }

    #[test]
    fn two_overlapping_boxes_join_into_one_tidy_box() {
        // The top of each box lies in one plane with the other's; the union
        // is one box, and its sides are one face each rather than pieces.
        let a = primitives::cube(2.0);
        let b = moved(primitives::cube(2.0), Vec3::new(1.0, 0.0, 0.0));
        let out = boolean(&a, &b, BooleanOp::Union);
        assert!((volume(&out) - 12.0).abs() < 1e-3, "{}", volume(&out));
        assert_eq!(out.num_faces(), 6, "a box has six sides");
        assert_eq!(out.validate(), Vec::new());
    }
}
