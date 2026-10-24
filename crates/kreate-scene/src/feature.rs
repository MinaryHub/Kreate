//! Features: what makes a body out of sketches, step by step, as Fusion's
//! timeline does — extrude this region so far, turn that one about a line —
//! each joining the body or cutting from it.
//!
//! A body is a mesh object whose features are not empty: what it shows is
//! its features run in order, not the mesh it holds. Each names its sketch
//! by identity and its regions by a point inside each, in the sketch's own
//! plane, so a sketch edited or re-dimensioned is picked up where its lines
//! now are and the body follows — the regions are found again every time
//! (see `kreate_mesh::regions`).

use kreate_math::{DAffine3, DVec2, Vec2};
use kreate_mesh::modifiers::boolean::{boolean, BooleanOp};
use kreate_mesh::regions::{self, Region};
use kreate_mesh::MeshData;
use kreate_reflect::Uid;

use crate::Document;

/// What a feature does to the body so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Starts the body, or adds to it — Fusion's New Body for the first, Join
    /// for the rest.
    #[default]
    Join,
    /// Takes away from it.
    Cut,
}

impl Operation {
    pub const ALL: [Operation; 2] = [Operation::Join, Operation::Cut];

    pub fn tag(self) -> &'static str {
        match self {
            Operation::Join => "join",
            Operation::Cut => "cut",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.tag() == tag)
    }
}

/// The solid a feature makes of its regions.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// Pushed out of the sketch's plane this far, in metres: negative goes
    /// the other way; symmetric goes half each way.
    Extrude { distance: f64, symmetric: bool },
    /// Turned about the line through two of the sketch's points, by this
    /// many radians — or about the sketch's own horizontal or vertical axis,
    /// when both are [`AXIS_X`] or [`AXIS_Y`].
    Revolve { axis: [u32; 2], angle: f64 },
    /// Every sharp edge of the body so far rounded to this radius, in
    /// metres — or cut back this far, for a chamfer. Names no sketch.
    Round { radius: f64, chamfer: bool },
    /// The body hollowed to a wall `thickness` thick; `open` is the side
    /// taken off — 0 none, then +X, −X, +Y, −Y, +Z, −Z — so a box becomes a
    /// tray. Fusion's Shell.
    Shell { thickness: f64, open: u32 },
    /// The feature before it repeated `count` times in all, each `step`
    /// further along in the body's space, joined or cut as it was.
    Pattern { count: u32, step: [f64; 3] },
    /// The feature before it repeated `count` times round the body's axis
    /// (0 X, 1 Y, 2 Z) through its origin, evenly.
    Circular { count: u32, axis: u32 },
    /// The feature before it reflected across the body's plane through its
    /// origin square to the axis given, joined or cut as it was.
    Mirror { axis: u32 },
}

/// The directions a shell's open side may face, as `Shape::Shell::open`
/// numbers them after none.
pub const SHELL_SIDES: [&str; 7] = ["none", "+x", "-x", "+y", "-y", "+z", "-z"];

fn side_direction(open: u32) -> Option<kreate_math::DVec3> {
    use kreate_math::DVec3;
    Some(match open {
        1 => DVec3::X,
        2 => -DVec3::X,
        3 => DVec3::Y,
        4 => -DVec3::Y,
        5 => DVec3::Z,
        6 => -DVec3::Z,
        _ => return None,
    })
}

/// One step of a body.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    pub shape: Shape,
    /// The sketch its regions are in.
    pub sketch: Uid,
    /// A point in each region it takes, in the sketch's plane.
    pub seeds: Vec<[f64; 2]>,
    pub operation: Operation,
    /// Where it stands on the document's timeline: features of every body
    /// in the order they were made, for the feature list.
    pub order: u64,
}

impl Feature {
    pub fn kind_tag(&self) -> &'static str {
        match self.shape {
            Shape::Extrude { .. } => "extrude",
            Shape::Revolve { .. } => "revolve",
            Shape::Round { chamfer: false, .. } => "fillet",
            Shape::Round { chamfer: true, .. } => "chamfer",
            Shape::Shell { .. } => "shell",
            Shape::Pattern { .. } => "pattern",
            Shape::Circular { .. } => "circular",
            Shape::Mirror { .. } => "mirror",
        }
    }

    /// Whether it works on the body so far rather than on a sketch.
    pub fn on_body(&self) -> bool {
        !matches!(self.shape, Shape::Extrude { .. } | Shape::Revolve { .. })
    }
}

/// A revolve's axis that is the sketch's own horizontal: through its origin
/// along its X. Twice, as `[AXIS_X, AXIS_X]`; no sketch has this many points.
pub const AXIS_X: u32 = u32::MAX;
/// The sketch's own vertical, as [`AXIS_X`] is its horizontal.
pub const AXIS_Y: u32 = u32::MAX - 1;

/// A revolve's axis in its sketch's plane: a point on it and its direction.
/// `None` when it names a point the sketch no longer has.
pub fn revolve_axis(
    doc: &Document,
    sketch: crate::document::ObjectId,
    axis: [u32; 2],
) -> Option<(Vec2, Vec2)> {
    match axis {
        [AXIS_X, AXIS_X] => return Some((Vec2::ZERO, Vec2::X)),
        [AXIS_Y, AXIS_Y] => return Some((Vec2::ZERO, Vec2::Y)),
        _ => {}
    }
    let mesh = doc
        .object(sketch)
        .and_then(|o| o.mesh())
        .and_then(|m| doc.mesh(m))?;
    let at = |v: u32| {
        mesh.positions()
            .get(v as usize)
            .map(|p| Vec2::new(p.x, p.y))
    };
    let (a, b) = (at(axis[0])?, at(axis[1])?);
    (a.distance(b) > 0.0).then_some((a, b - a))
}

/// The regions a feature of this shape picks from: a sketch's regions, and
/// for a revolve those with the loose ends closed against its axis — so a
/// profile drawn open beside the axis is one to turn.
pub fn shape_regions(
    doc: &Document,
    sketch: crate::document::ObjectId,
    shape: &Shape,
) -> Vec<Region> {
    let Shape::Revolve { axis, .. } = shape else {
        return sketch_regions(doc, sketch);
    };
    let Some((origin, dir)) = revolve_axis(doc, sketch, *axis) else {
        return sketch_regions(doc, sketch);
    };
    let Some((points, edges)) = sketch_drawing(doc, sketch) else {
        return Vec::new();
    };
    regions::regions_against_axis(&points, &edges, origin, dir)
}

/// The regions of a sketch a feature's seeds pick, each once.
pub fn picked(regions: &[Region], seeds: &[[f64; 2]]) -> Vec<usize> {
    let mut out: Vec<usize> = seeds
        .iter()
        .filter_map(|s| regions::region_at(regions, Vec2::new(s[0] as f32, s[1] as f32)))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// What an extrude takes from a sketch nobody has picked in: a point inside
/// each region it would take, in the sketch's plane.
///
/// Every region, less two kinds. A region bounded only by fixed points is
/// the outline of the face the sketch was made on, projected into it for
/// drawing against — extruding it is extruding the whole face. And a region
/// that fills another's hole is that hole: the counter of an `O`, the bore
/// of a washer, the circle cut out of a plate. So a rectangle on a face, a
/// plate with holes and a line of text each come out as drawn.
pub fn profile_seeds(doc: &Document, sketch: crate::document::ObjectId) -> Vec<[f64; 2]> {
    seeds_among(doc, sketch, &sketch_regions(doc, sketch))
}

/// The same for a feature of this shape: a revolve's, among the regions
/// closed against its axis.
pub fn shape_profile_seeds(
    doc: &Document,
    sketch: crate::document::ObjectId,
    shape: &Shape,
) -> Vec<[f64; 2]> {
    seeds_among(doc, sketch, &shape_regions(doc, sketch, shape))
}

fn seeds_among(
    doc: &Document,
    sketch: crate::document::ObjectId,
    regions: &[Region],
) -> Vec<[f64; 2]> {
    let (Some(object), Some(mesh)) = (
        doc.object(sketch),
        doc.object(sketch)
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh(m)),
    ) else {
        return Vec::new();
    };
    let fixed: Vec<Vec2> = object
        .sketch
        .iter()
        .filter_map(|c| match c {
            crate::sketch::Constraint::Fixed { a, .. } => mesh
                .positions()
                .get(*a as usize)
                .map(|p| Vec2::new(p.x, p.y)),
            _ => None,
        })
        .collect();
    let on_fixed = |p: &Vec2| fixed.iter().any(|f| f.distance(*p) < 1e-7);
    // A ring as its set of points, which way round it runs left out.
    let key = |ring: &[Vec2]| -> Vec<(u32, u32)> {
        let mut k: Vec<(u32, u32)> = ring
            .iter()
            .map(|p| (p.x.to_bits(), p.y.to_bits()))
            .collect();
        k.sort_unstable();
        k
    };
    let candidates: Vec<usize> = (0..regions.len())
        .filter(|&i| !regions[i].outline.iter().all(on_fixed))
        .collect();
    let holes: Vec<Vec<(u32, u32)>> = candidates
        .iter()
        .flat_map(|&i| regions[i].holes.iter().map(|h| key(h)))
        .collect();
    let mut out = Vec::new();
    for &i in &candidates {
        if holes.contains(&key(&regions[i].outline)) {
            continue;
        }
        if let Some(p) = inside_point(regions, i) {
            out.push(p);
        }
    }
    out
}

/// A point inside region `i` and no other: the middle of its largest
/// triangle, checked against the regions as a click would be.
fn inside_point(regions: &[Region], i: usize) -> Option<[f64; 2]> {
    let r = &regions[i];
    let mut rings: Vec<Vec<DVec2>> = vec![r.outline.iter().map(|p| p.as_dvec2()).collect()];
    rings.extend(
        r.holes
            .iter()
            .map(|h| h.iter().map(|p| p.as_dvec2()).collect()),
    );
    let all: Vec<DVec2> = rings.iter().flatten().copied().collect();
    let mut tris = kreate_mesh::earcut::triangulate(&rings);
    let area = |t: &[usize; 3]| {
        let (a, b, c) = (all[t[0]], all[t[1]], all[t[2]]);
        (b - a).perp_dot(c - a).abs()
    };
    tris.sort_by(|a, b| area(b).total_cmp(&area(a)));
    tris.iter().find_map(|t| {
        let m = (all[t[0]] + all[t[1]] + all[t[2]]) / 3.0;
        (regions::region_at(regions, m.as_vec2()) == Some(i)).then_some([m.x, m.y])
    })
}

/// A sketch's regions: its points are its mesh's, its lines the object's
/// own (see `Object::sketch_lines`) and its faces' edges, which is what a
/// sketch drawn before it had lines of its own was made of.
pub fn sketch_regions(doc: &Document, sketch: crate::document::ObjectId) -> Vec<Region> {
    sketch_drawing(doc, sketch)
        .map(|(points, edges)| regions::regions(&points, &edges))
        .unwrap_or_default()
}

/// A sketch's points in its plane and the lines that bound regions.
fn sketch_drawing(
    doc: &Document,
    sketch: crate::document::ObjectId,
) -> Option<(Vec<Vec2>, Vec<[u32; 2]>)> {
    let object = doc.object(sketch)?;
    let mesh = object.mesh().and_then(|m| doc.mesh(m))?;
    let points: Vec<Vec2> = mesh
        .positions()
        .iter()
        .map(|p| Vec2::new(p.x, p.y))
        .collect();
    let mut edges: Vec<[u32; 2]> = mesh.edges().to_vec();
    // Construction lines place things; they bound nothing.
    edges.extend(
        object
            .sketch_lines
            .iter()
            .copied()
            .filter(|l| !crate::sketch::Constraint::is_construction(&object.sketch, *l)),
    );
    Some((points, edges))
}

/// The solid one feature makes, in its sketch's own space; empty when its
/// sketch is gone or its seeds fall in no region.
pub fn solid(doc: &Document, feature: &Feature) -> MeshData {
    let Some(sketch) = doc.object_by_uid(feature.sketch) else {
        return MeshData::empty();
    };
    let all = shape_regions(doc, sketch, &feature.shape);
    let mut out = MeshData::empty();
    for i in picked(&all, &feature.seeds) {
        let piece = match feature.shape {
            Shape::Extrude {
                distance,
                symmetric,
            } => {
                let d = distance as f32;
                let (z0, z1) = if symmetric {
                    (-d.abs() * 0.5, d.abs() * 0.5)
                } else {
                    (0.0, d)
                };
                regions::extrude_region(&all[i], z0, z1)
            }
            Shape::Revolve { axis, angle } => {
                let Some((origin, dir)) = revolve_axis(doc, sketch, axis) else {
                    continue;
                };
                regions::revolve_region(&all[i], origin, dir, angle as f32, 64)
            }
            Shape::Round { .. }
            | Shape::Shell { .. }
            | Shape::Pattern { .. }
            | Shape::Circular { .. }
            | Shape::Mirror { .. } => continue,
        };
        out = if out.num_faces() == 0 {
            piece
        } else {
            boolean(&out, &piece, BooleanOp::Union)
        };
    }
    out
}

/// `piece` joined to or cut from `out`, as `op` says; the first piece
/// starts the body.
fn combine(out: MeshData, piece: MeshData, op: Operation) -> MeshData {
    if out.num_faces() == 0 {
        return match op {
            Operation::Join => piece,
            Operation::Cut => out,
        };
    }
    let op = match op {
        Operation::Join => BooleanOp::Union,
        Operation::Cut => BooleanOp::Difference,
    };
    boolean(&out, &piece, op)
}

/// `piece` with every position put through `f`, its faces turned round
/// when `f` turns the solid inside out.
fn mapped(
    piece: &MeshData,
    flip: bool,
    f: impl Fn(kreate_math::DVec3) -> kreate_math::DVec3,
) -> MeshData {
    let positions: Vec<kreate_math::Vec3> = piece
        .positions()
        .iter()
        .map(|p| f(p.as_dvec3()).as_vec3())
        .collect();
    let faces: Vec<Vec<u32>> = piece
        .faces()
        .map(|face| {
            if flip {
                face.iter().rev().copied().collect()
            } else {
                face.to_vec()
            }
        })
        .collect();
    MeshData::from_faces(positions, &faces)
}

/// The faces of `mesh` that face `dir` and lie furthest that way: a box's
/// top, for +Z, however many faces it is in.
fn faces_facing(mesh: &MeshData, dir: kreate_math::DVec3) -> Vec<u32> {
    let p = mesh.positions();
    let mut candidates: Vec<(u32, f64)> = Vec::new();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = kreate_math::DVec3::ZERO;
        let mut c = kreate_math::DVec3::ZERO;
        for i in 0..ring.len() {
            let (a, b) = (
                p[ring[i] as usize].as_dvec3(),
                p[ring[(i + 1) % ring.len()] as usize].as_dvec3(),
            );
            n += a.cross(b);
            c += a;
        }
        let n = n.normalize_or_zero();
        if n.dot(dir) > 0.999 {
            candidates.push((f, (c / ring.len() as f64).dot(dir)));
        }
    }
    let Some(top) = candidates
        .iter()
        .map(|c| c.1)
        .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| m.max(d))))
    else {
        return Vec::new();
    };
    let scale = p
        .iter()
        .map(|v| v.as_dvec3().length())
        .fold(0.0f64, f64::max)
        .max(1e-9);
    candidates
        .into_iter()
        .filter(|(_, d)| (top - d).abs() < 1e-6 * scale.max(1.0))
        .map(|(f, _)| f)
        .collect()
}

/// A body: its features run in order, in its own space. The first one
/// starts it whatever it says; after that, each joins or cuts.
pub fn body(doc: &Document, features: &[Feature], body_world: DAffine3) -> MeshData {
    let to_body = body_world.inverse();
    let mut out = MeshData::empty();
    // The last solid a feature made, in the body's space, with what it did:
    // what a pattern or a mirror repeats.
    let mut last: Option<(MeshData, Operation)> = None;
    for f in features {
        // These work on what the body is so far, not on a sketch.
        match f.shape {
            Shape::Round { radius, chamfer } => {
                if out.num_faces() > 0 && radius > 0.0 {
                    out = kreate_mesh::modifiers::bevel::round_edges(&out, radius as f32, chamfer);
                }
                continue;
            }
            Shape::Shell { thickness, open } => {
                if out.num_faces() > 0 && thickness > 0.0 {
                    let open: Vec<u32> = side_direction(open)
                        .map(|dir| faces_facing(&out, dir))
                        .unwrap_or_default();
                    out = kreate_mesh::mech::shell(&out, &open, thickness as f32);
                }
                continue;
            }
            Shape::Pattern { count, step } => {
                if let Some((piece, op)) = &last {
                    let step = kreate_math::DVec3::from_array(step);
                    for k in 1..count.max(1) {
                        let copy = mapped(piece, false, |p| p + step * k as f64);
                        out = combine(out, copy, *op);
                    }
                }
                continue;
            }
            Shape::Circular { count, axis } => {
                if let Some((piece, op)) = &last {
                    let axis = match axis {
                        0 => kreate_math::DVec3::X,
                        1 => kreate_math::DVec3::Y,
                        _ => kreate_math::DVec3::Z,
                    };
                    for k in 1..count.max(1) {
                        let turn = kreate_math::DQuat::from_axis_angle(
                            axis,
                            std::f64::consts::TAU * k as f64 / count as f64,
                        );
                        let copy = mapped(piece, false, |p| turn * p);
                        out = combine(out, copy, *op);
                    }
                }
                continue;
            }
            Shape::Mirror { axis } => {
                if let Some((piece, op)) = &last {
                    let a = (axis as usize).min(2);
                    let copy = mapped(piece, true, |mut p| {
                        p[a] = -p[a];
                        p
                    });
                    out = combine(out, copy, *op);
                }
                continue;
            }
            Shape::Extrude { .. } | Shape::Revolve { .. } => {}
        }
        let mut piece = solid(doc, f);
        if piece.num_faces() == 0 {
            continue;
        }
        let sketch_world = doc
            .object_by_uid(f.sketch)
            .map(|id| doc.world_transform(id))
            .unwrap_or(DAffine3::IDENTITY);
        let into = to_body * sketch_world;
        for p in piece.positions_mut() {
            *p = into.transform_point3(p.as_dvec3()).as_vec3();
        }
        last = Some((piece.clone(), f.operation));
        out = combine(out, piece, f.operation);
    }
    out
}

/// What a body's shape depends on, for the cache: its features, each
/// sketch's points and place, and its own place.
pub fn key(doc: &Document, features: &[Feature], body_world: DAffine3) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{features:?}").hash(&mut h);
    for v in body_world.to_cols_array() {
        v.to_bits().hash(&mut h);
    }
    for f in features {
        if let Some(id) = doc.object_by_uid(f.sketch) {
            for v in doc.world_transform(id).to_cols_array() {
                v.to_bits().hash(&mut h);
            }
            if let Some(o) = doc.object(id) {
                o.sketch_lines.hash(&mut h);
            }
            if let Some(mesh) = doc
                .object(id)
                .and_then(|o| o.mesh())
                .and_then(|m| doc.mesh_arc(m))
            {
                (std::sync::Arc::as_ptr(mesh) as usize).hash(&mut h);
            }
        }
    }
    h.finish()
}

/// The next place on the timeline.
pub fn next_order(doc: &Document) -> u64 {
    doc.objects()
        .flat_map(|(_, o)| o.features.iter().map(|f| f.order))
        .max()
        .map_or(1, |m| m + 1)
}

/// A seed point in sketch space, as stored.
pub fn seed(p: DVec2) -> [f64; 2] {
    p.to_array()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};
    use kreate_math::Vec3;
    use kreate_mesh::measure::mass_properties;

    /// A sketch object of the given loops, each closed, in the XY plane: its
    /// points a mesh of no faces, its lines its own.
    fn sketch(doc: &mut Document, loops: &[Vec<Vec2>]) -> Uid {
        let mut points = Vec::new();
        let mut lines: Vec<[u32; 2]> = Vec::new();
        for l in loops {
            let s = points.len() as u32;
            points.extend(l.iter().map(|p| Vec3::new(p.x, p.y, 0.0)));
            for i in 0..l.len() as u32 {
                lines.push([s + i, s + (i + 1) % l.len() as u32]);
            }
        }
        let m = doc.add_mesh(MeshData::from_faces(points, &[]));
        let mut object = Object::new("Sketch", ObjectData::Mesh(m));
        object.sketch_lines = lines;
        let id = doc.add_object(object);
        doc.object_uid(id).expect("uid")
    }

    fn rect(w: f32, h: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(w, 0.0),
            Vec2::new(w, h),
            Vec2::new(0.0, h),
        ]
    }

    fn circle(c: Vec2, r: f32) -> Vec<Vec2> {
        (0..48)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 48.0;
                c + Vec2::new(a.cos(), a.sin()) * r
            })
            .collect()
    }

    #[test]
    fn a_plate_extruded_and_its_hole_cut_has_the_volume_it_should() {
        let mut doc = Document::new();
        let s = sketch(
            &mut doc,
            &[rect(0.04, 0.02), circle(Vec2::new(0.02, 0.01), 0.005)],
        );
        // The whole plate, then the disc taken out of it again — the hole as
        // Fusion makes it, a cut of the inner region.
        let plate = Feature {
            shape: Shape::Extrude {
                distance: 0.01,
                symmetric: false,
            },
            sketch: s,
            seeds: vec![[0.002, 0.002], [0.02, 0.01]],
            operation: Operation::Join,
            order: 1,
        };
        let hole = Feature {
            seeds: vec![[0.02, 0.01]],
            operation: Operation::Cut,
            order: 2,
            ..plate.clone()
        };
        let solid = body(&doc, std::slice::from_ref(&plate), DAffine3::IDENTITY);
        let full = mass_properties(&solid, |p| p).volume;
        assert!((full - 0.04 * 0.02 * 0.01).abs() < 1e-8, "{full}");
        let holed = body(&doc, &[plate, hole], DAffine3::IDENTITY);
        let v = mass_properties(&holed, |p| p).volume;
        let disc = std::f64::consts::PI * 0.005f64.powi(2) * 0.01;
        assert!(
            (v - (full - disc)).abs() < full * 5e-3,
            "{v} against {}",
            full - disc
        );
    }

    #[test]
    fn a_body_lands_where_its_sketch_is() {
        let mut doc = Document::new();
        let s = sketch(&mut doc, &[rect(1.0, 1.0)]);
        let id = doc.object_by_uid(s).expect("sketch");
        doc.object_mut(id).expect("sketch").transform.translation =
            kreate_math::DVec3::new(5.0, 0.0, 2.0);
        let f = Feature {
            shape: Shape::Extrude {
                distance: 1.0,
                symmetric: false,
            },
            sketch: s,
            seeds: vec![[0.5, 0.5]],
            operation: Operation::Join,
            order: 1,
        };
        let solid = body(&doc, &[f], DAffine3::IDENTITY);
        let lo = solid
            .positions()
            .iter()
            .fold(Vec3::splat(f32::MAX), |m, p| m.min(*p));
        assert!((lo - Vec3::new(5.0, 0.0, 2.0)).length() < 1e-5, "{lo:?}");
    }

    #[test]
    fn a_seed_outside_every_region_makes_nothing() {
        let mut doc = Document::new();
        let s = sketch(&mut doc, &[rect(1.0, 1.0)]);
        let f = Feature {
            shape: Shape::Extrude {
                distance: 1.0,
                symmetric: true,
            },
            sketch: s,
            seeds: vec![[5.0, 5.0]],
            operation: Operation::Join,
            order: 1,
        };
        assert_eq!(body(&doc, &[f], DAffine3::IDENTITY).num_faces(), 0);
    }
}
