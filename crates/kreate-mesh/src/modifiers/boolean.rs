//! The boolean modifier: one mesh cut by, joined with, or kept inside another.
//!
//! By binary space partitioning, the method of Thibault and Naylor that
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
//! Polygons split along the other solid's planes, their corners welded where
//! they meet. Correct, not tidy: the faces around a cut are fans of slivers,
//! which is what a modeller applying a boolean then cleans up. Attributes
//! (UVs, materials) are not carried; the result is geometry.
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
        Some(Plane { normal, w: normal.dot(a) })
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
}

impl Polygon {
    fn new(verts: Vec<DVec3>) -> Option<Polygon> {
        // The plane of the whole polygon, from its Newell normal, so a thin
        // first triangle does not decide it.
        let mut n = DVec3::ZERO;
        for i in 0..verts.len() {
            let (p, q) = (verts[i], verts[(i + 1) % verts.len()]);
            n += DVec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
        }
        let len = n.length();
        if verts.len() < 3 || len < 1e-12 {
            return Plane::from_points(verts[0], verts[1], *verts.get(2)?).map(|plane| Polygon { verts, plane });
        }
        let normal = n / len;
        let centre = verts.iter().fold(DVec3::ZERO, |s, p| s + *p) / verts.len() as f64;
        Some(Polygon { plane: Plane { normal, w: normal.dot(centre) }, verts })
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

/// Sorts `polygon` against `plane` into the four lists.
fn split(
    plane: &Plane,
    polygon: &Polygon,
    coplanar_front: &mut Vec<Polygon>,
    coplanar_back: &mut Vec<Polygon>,
    front: &mut Vec<Polygon>,
    back: &mut Vec<Polygon>,
) {
    let mut kind = 0u8;
    let types: Vec<u8> = polygon
        .verts
        .iter()
        .map(|v| {
            let t = plane.normal.dot(*v) - plane.w;
            let k = if t < -EPSILON {
                BACK
            } else if t > EPSILON {
                FRONT
            } else {
                COPLANAR
            };
            kind |= k;
            k
        })
        .collect();
    match kind {
        COPLANAR => {
            if plane.normal.dot(polygon.plane.normal) > 0.0 {
                coplanar_front.push(polygon.clone());
            } else {
                coplanar_back.push(polygon.clone());
            }
        }
        FRONT => front.push(polygon.clone()),
        BACK => back.push(polygon.clone()),
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
                front.push(Polygon { verts: f, plane: polygon.plane });
            }
            if b.len() >= 3 {
                back.push(Polygon { verts: b, plane: polygon.plane });
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

#[derive(Default)]
struct Node {
    plane: Option<Plane>,
    front: Option<usize>,
    back: Option<usize>,
    polygons: Vec<Polygon>,
}

impl Tree {
    fn new(polygons: Vec<Polygon>) -> Tree {
        let mut tree = Tree { nodes: vec![Node::default()] };
        tree.build(0, polygons);
        tree
    }

    fn build(&mut self, root: usize, polygons: Vec<Polygon>) {
        let mut work = vec![(root, polygons)];
        while let Some((at, polygons)) = work.pop() {
            if polygons.is_empty() {
                continue;
            }
            let plane = *self.nodes[at].plane.get_or_insert(polygons[0].plane);
            let (mut cf, mut cb, mut f, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for p in &polygons {
                split(&plane, p, &mut cf, &mut cb, &mut f, &mut b);
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
    }

    fn child(&mut self, at: usize, front: bool) -> usize {
        let existing = if front { self.nodes[at].front } else { self.nodes[at].back };
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
    fn clip_polygons(&self, polygons: Vec<Polygon>) -> Vec<Polygon> {
        let mut out = Vec::new();
        let mut work = vec![(0usize, polygons)];
        while let Some((at, polygons)) = work.pop() {
            let node = &self.nodes[at];
            let Some(plane) = node.plane else {
                out.extend(polygons);
                continue;
            };
            let (mut cf, mut cb, mut f, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for p in &polygons {
                split(&plane, p, &mut cf, &mut cb, &mut f, &mut b);
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
        out
    }

    /// Removes from this tree's polygons everything inside `other`.
    fn clip_to(&mut self, other: &Tree) {
        for i in 0..self.nodes.len() {
            let polygons = std::mem::take(&mut self.nodes[i].polygons);
            self.nodes[i].polygons = other.clip_polygons(polygons);
        }
    }

    fn all_polygons(&self) -> Vec<Polygon> {
        self.nodes.iter().flat_map(|n| n.polygons.iter().cloned()).collect()
    }
}

fn polygons_of(mesh: &MeshData) -> Vec<Polygon> {
    let positions = mesh.positions();
    (0..mesh.num_faces())
        .filter_map(|f| {
            let verts: Vec<DVec3> = mesh.face_verts(f).iter().map(|v| positions[*v as usize].as_dvec3()).collect();
            Polygon::new(verts)
        })
        .collect()
}

/// `a` combined with `b`, both in the same space.
pub fn boolean(a: &MeshData, b: &MeshData, op: BooleanOp) -> MeshData {
    if a.is_empty() {
        return match op {
            BooleanOp::Union => b.clone(),
            _ => a.clone(),
        };
    }
    if b.is_empty() {
        return match op {
            BooleanOp::Intersect => MeshData::from_faces(Vec::new(), &[]),
            _ => a.clone(),
        };
    }
    let mut ta = Tree::new(polygons_of(a));
    let mut tb = Tree::new(polygons_of(b));
    let polygons = match op {
        BooleanOp::Union => {
            ta.clip_to(&tb);
            tb.clip_to(&ta);
            tb.invert();
            tb.clip_to(&ta);
            tb.invert();
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            all
        }
        BooleanOp::Difference => {
            ta.invert();
            ta.clip_to(&tb);
            tb.clip_to(&ta);
            tb.invert();
            tb.clip_to(&ta);
            tb.invert();
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            let mut result = Tree::new(all);
            result.invert();
            result.all_polygons()
        }
        BooleanOp::Intersect => {
            ta.invert();
            tb.clip_to(&ta);
            tb.invert();
            ta.clip_to(&tb);
            tb.clip_to(&ta);
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            let mut result = Tree::new(all);
            result.invert();
            result.all_polygons()
        }
    };
    weld(polygons)
}

/// Polygons to a mesh, corners at the same place made one vertex.
fn weld(polygons: Vec<Polygon>) -> MeshData {
    let scale = 1e5;
    let mut index: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for p in polygons {
        let mut ring: Vec<u32> = Vec::with_capacity(p.verts.len());
        for v in &p.verts {
            let key = [(v.x * scale).round() as i64, (v.y * scale).round() as i64, (v.z * scale).round() as i64];
            let i = *index.entry(key).or_insert_with(|| {
                positions.push(v.as_vec3());
                (positions.len() - 1) as u32
            });
            if ring.last() != Some(&i) && ring.first() != Some(&i) {
                ring.push(i);
            }
        }
        if ring.len() >= 3 {
            faces.push(ring);
        }
    }
    MeshData::from_faces(positions, &faces)
}

#[cfg(test)]
mod tests {
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
                let (a, b, c) = (p[ring[0] as usize].as_dvec3(), p[ring[i] as usize].as_dvec3(), p[ring[i + 1] as usize].as_dvec3());
                v += a.dot(b.cross(c)) / 6.0;
            }
        }
        v
    }

    #[test]
    fn two_unit_overlapping_cubes_give_the_volumes_arithmetic_says() {
        // Two 2-unit cubes, the second moved one unit along X: they overlap in
        // a 1 x 2 x 2 box.
        let a = primitives::cube(2.0);
        let b = moved(primitives::cube(2.0), Vec3::X);
        let near = |got: f64, want: f64| (got - want).abs() < 1e-3;
        let difference = boolean(&a, &b, BooleanOp::Difference);
        assert!(near(volume(&difference), 4.0), "difference {}", volume(&difference));
        let union = boolean(&a, &b, BooleanOp::Union);
        assert!(near(volume(&union), 12.0), "union {}", volume(&union));
        let intersect = boolean(&a, &b, BooleanOp::Intersect);
        assert!(near(volume(&intersect), 4.0), "intersect {}", volume(&intersect));
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
}
