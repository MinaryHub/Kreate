//! A body built exactly, by the B-rep kernel (`kreate_occt`), where it can
//! be: see design/brep-kernel.md, P1.
//!
//! The same features in the same order make the same body; only the
//! arithmetic differs. A drawing's regions go to the kernel as the curves
//! they are — an arc as an arc, a circle as a circle — so a hole is a
//! cylinder and not a prism of many sides. Anything this does not yet build
//! (a fillet, a shell, a sweep…), or anything the kernel gives up on, makes
//! the whole body come from the mesh path instead, as it always has: the
//! answer is `None` and the caller carries on.

use kreate_geom2d::{Curve2D, DVec2};
use kreate_math::{DAffine3, DVec3, Vec3};
use kreate_mesh::MeshData;
use kreate_occt::{Boolean, Loop, Piece, Shape as Solid};

use super::{Feature, Operation, Shape};
use crate::Document;

/// Whether bodies are built by the kernel: it is linked in, and
/// `KREATE_BREP=0` has not turned it off.
pub fn enabled() -> bool {
    kreate_occt::available() && std::env::var("KREATE_BREP").map_or(true, |v| v != "0")
}

/// How far the screen's triangles stray from the true surface: a hundredth
/// of a millimetre, as the mesh path's curves.
const DEFLECTION: f64 = 1e-5;
/// …nor turn more than this from one to the next, so a large curve is
/// smooth however far it is from its chord.
const ANGLE: f64 = 0.2;

/// An affine map's rows, three of four.
fn rows(m: DAffine3) -> [[f64; 4]; 3] {
    let (c, t) = (m.matrix3, m.translation);
    [
        [c.x_axis.x, c.y_axis.x, c.z_axis.x, t.x],
        [c.x_axis.y, c.y_axis.y, c.z_axis.y, t.y],
        [c.x_axis.z, c.y_axis.z, c.z_axis.z, t.z],
    ]
}

/// A curve as the kernel's pieces: a polyline's segments one by one, each
/// straight or, with a bulge, an arc.
fn pieces(c: &Curve2D, out: &mut Loop) -> Option<()> {
    let p = |v: DVec2| [v.x, v.y];
    match c {
        Curve2D::Line { a, b } => out.push(Piece::Line { a: p(*a), b: p(*b) }),
        Curve2D::Circle { c, r } => out.push(Piece::Circle {
            centre: p(*c),
            radius: *r,
        }),
        Curve2D::Arc { c, r, start, sweep } => out.push(Piece::Arc {
            centre: p(*c),
            radius: *r,
            start: *start,
            sweep: *sweep,
        }),
        Curve2D::Ellipse {
            c,
            major,
            ratio,
            start,
            sweep,
        } => out.push(Piece::Ellipse {
            centre: p(*c),
            major: p(*major),
            ratio: *ratio,
            start: *start,
            sweep: *sweep,
        }),
        Curve2D::Spline(s) => out.push(Piece::Spline {
            degree: s.degree,
            ctrl: s.ctrl.iter().map(|q| p(*q)).collect(),
            knots: s.knots.clone(),
            weights: s.weights.clone(),
        }),
        Curve2D::Polyline { verts, closed } => {
            let n = verts.len();
            let segments = if *closed { n } else { n.saturating_sub(1) };
            for i in 0..segments {
                let (a, bulge) = verts[i];
                let b = verts[(i + 1) % n].0;
                if bulge.abs() < 1e-12 {
                    out.push(Piece::Line { a: p(a), b: p(b) });
                    continue;
                }
                // DXF's bulge: the tangent of a quarter of the arc's sweep,
                // positive anticlockwise.
                let sweep = 4.0 * bulge.atan();
                let chord = b - a;
                let half = chord.length() * 0.5;
                if half < 1e-15 {
                    return None;
                }
                let radius = half / (sweep * 0.5).sin().abs();
                let to_centre = half / (sweep * 0.5).tan();
                let centre = (a + b) * 0.5 + chord.perp().normalize() * to_centre;
                let start = (a - centre).y.atan2((a - centre).x);
                out.push(Piece::Arc {
                    centre: p(centre),
                    radius,
                    start,
                    sweep,
                });
            }
        }
    }
    Some(())
}

/// The faces a feature's seeds pick in its sketch, each in the sketch's
/// plane: a drawing's from its curves, an older sketch's from its lines.
/// `None` when a seed falls in no region, or a region does not make a face.
fn faces(doc: &Document, f: &Feature) -> Option<Vec<Solid>> {
    let sketch = doc.object_by_uid(f.sketch)?;
    let object = doc.object(sketch)?;
    let mut out = Vec::new();
    if let Some(drawing) = object.drawing.as_deref() {
        let regions = drawing.regions(super::DRAWING_CHORD);
        let mut taken: Vec<usize> = Vec::new();
        for seed in &f.seeds {
            let at = DVec2::new(seed[0], seed[1]);
            // The smallest region holding the seed: a hole's filling is in
            // its own region, not its outside's.
            let i = (0..regions.len())
                .filter(|i| regions[*i].contains(at))
                .min_by(|a, b| regions[*a].area.total_cmp(&regions[*b].area))?;
            if !taken.contains(&i) {
                taken.push(i);
            }
        }
        for i in taken {
            let r = &regions[i];
            let to_loop = |l: &kreate_geom2d::Loop| -> Option<Loop> {
                let mut out = Vec::new();
                for piece in l {
                    pieces(&piece.curve, &mut out)?;
                }
                Some(out)
            };
            let outer = to_loop(&r.outer)?;
            let holes: Vec<Loop> = r.holes.iter().map(to_loop).collect::<Option<_>>()?;
            out.push(Solid::face(&outer, &holes)?);
        }
    } else {
        let all = super::shape_regions(doc, sketch, &f.shape);
        for i in super::picked(&all, &f.seeds) {
            let ring = |r: &[kreate_math::Vec2]| -> Loop {
                (0..r.len())
                    .map(|k| {
                        let (a, b) = (r[k], r[(k + 1) % r.len()]);
                        Piece::Line {
                            a: [f64::from(a.x), f64::from(a.y)],
                            b: [f64::from(b.x), f64::from(b.y)],
                        }
                    })
                    .collect()
            };
            let holes: Vec<Loop> = all[i].holes.iter().map(|h| ring(h)).collect();
            out.push(Solid::face(&ring(&all[i].outline), &holes)?);
        }
    }
    (!out.is_empty()).then_some(out)
}

/// The solid an extrude or a revolve makes, in its sketch's space.
fn solid(doc: &Document, f: &Feature) -> Option<Solid> {
    let sketch = doc.object_by_uid(f.sketch)?;
    let mut made: Option<Solid> = None;
    for face in faces(doc, f)? {
        let piece = match f.shape {
            Shape::Extrude {
                distance,
                symmetric,
            } => {
                let (z0, z1) = if symmetric {
                    (-distance.abs() * 0.5, distance.abs() * 0.5)
                } else {
                    (0.0, distance)
                };
                let (lo, hi) = (z0.min(z1), z0.max(z1));
                let face = if lo != 0.0 {
                    face.transformed(rows(DAffine3::from_translation(DVec3::new(0.0, 0.0, lo))))?
                } else {
                    face
                };
                face.extrude([0.0, 0.0, hi - lo])?
            }
            Shape::Revolve { axis, angle } => {
                let (origin, dir) = super::revolve_axis(doc, sketch, axis)?;
                let full = angle.abs() >= std::f64::consts::TAU - 1e-4;
                let turn = if full { std::f64::consts::TAU } else { angle };
                face.revolve(
                    [f64::from(origin.x), f64::from(origin.y), 0.0],
                    [f64::from(dir.x), f64::from(dir.y), 0.0],
                    turn,
                )?
            }
            _ => return None,
        };
        made = Some(match made {
            None => piece,
            Some(m) => m.boolean(&piece, Boolean::Join)?,
        });
    }
    made
}

/// `piece` joined to or cut from `out`; the first piece starts the body.
fn combine(out: Option<Solid>, piece: &Solid, op: Operation) -> Option<Option<Solid>> {
    Some(match (out, op) {
        (None, Operation::Join) => Some(piece.transformed(rows(DAffine3::IDENTITY))?),
        (None, Operation::Cut) => None,
        (Some(o), Operation::Join) => Some(o.boolean(piece, Boolean::Join)?),
        (Some(o), Operation::Cut) => Some(o.boolean(piece, Boolean::Cut)?),
    })
}

/// A body as an exact solid, in its own space: its extrudes and revolves
/// joined and cut, its patterns and mirrors repeating the feature before.
/// `None` for a body with a feature this does not build yet, or one the
/// kernel gives up on.
pub fn body_solid(doc: &Document, features: &[Feature], body_world: DAffine3) -> Option<Solid> {
    let to_body = body_world.inverse();
    let mut out: Option<Solid> = None;
    let mut last: Option<(Solid, Operation)> = None;
    for f in features {
        match f.shape {
            Shape::Extrude { .. } | Shape::Revolve { .. } => {
                let piece = solid(doc, f)?;
                let sketch_world = doc
                    .object_by_uid(f.sketch)
                    .map(|id| doc.world_transform(id))
                    .unwrap_or(DAffine3::IDENTITY);
                let piece = piece.transformed(rows(to_body * sketch_world))?;
                out = combine(out, &piece, f.operation)?;
                last = Some((piece, f.operation));
            }
            Shape::Pattern { count, step } => {
                if let Some((piece, op)) = &last {
                    let step = DVec3::from_array(step);
                    for k in 1..count.max(1) {
                        let copy =
                            piece.transformed(rows(DAffine3::from_translation(step * k as f64)))?;
                        out = combine(out, &copy, *op)?;
                    }
                }
            }
            Shape::Circular { count, axis } => {
                if let Some((piece, op)) = &last {
                    let axis = match axis {
                        0 => DVec3::X,
                        1 => DVec3::Y,
                        _ => DVec3::Z,
                    };
                    for k in 1..count.max(1) {
                        let turn = DAffine3::from_axis_angle(
                            axis,
                            std::f64::consts::TAU * k as f64 / count as f64,
                        );
                        let copy = piece.transformed(rows(turn))?;
                        out = combine(out, &copy, *op)?;
                    }
                }
            }
            Shape::Mirror { axis } => {
                if let Some((piece, op)) = &last {
                    let mut scale = DVec3::ONE;
                    scale[(axis as usize).min(2)] = -1.0;
                    let copy = piece.transformed(rows(DAffine3::from_scale(scale)))?;
                    out = combine(out, &copy, *op)?;
                }
            }
            // Not yet built exactly: the mesh path builds the whole body.
            Shape::Round { .. }
            | Shape::Shell { .. }
            | Shape::Sweep { .. }
            | Shape::Loft { .. }
            | Shape::SheetMetal { .. } => return None,
        }
    }
    out.filter(|s| s.is_valid())
}

/// A body built exactly and meshed for the screen, its faces' shared
/// corners welded so the surface is closed where the solid is. `None` when
/// it cannot be (see [`body_solid`]).
pub fn body_mesh(doc: &Document, features: &[Feature], body_world: DAffine3) -> Option<MeshData> {
    if !enabled() {
        return None;
    }
    let solid = body_solid(doc, features, body_world)?;
    let mesh = solid.mesh(DEFLECTION, ANGLE)?;
    if mesh.triangles.is_empty() {
        return None;
    }
    // The kernel meshes each face on its own, sharing its edges' points with
    // the next: one vertex for each place.
    let mut index: std::collections::HashMap<[u64; 3], u32> = std::collections::HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let remap: Vec<u32> = mesh
        .positions
        .iter()
        .map(|p| {
            let key = p.map(|c| (c as f32).to_bits() as u64);
            *index.entry(key).or_insert_with(|| {
                positions.push(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32));
                positions.len() as u32 - 1
            })
        })
        .collect();
    let tris: Vec<[u32; 3]> = mesh
        .triangles
        .iter()
        .map(|t| t.map(|v| remap[v as usize]))
        .collect();
    // Each flat face of the solid one polygon, as the mesh path makes it —
    // a box is six faces to edit, not twelve triangles — when its triangles
    // run round one boundary; the rest stay triangles.
    let mut by_face: std::collections::BTreeMap<u32, Vec<[u32; 3]>> = Default::default();
    for (t, f) in tris.iter().zip(&mesh.faces) {
        if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
            by_face.entry(*f).or_default().push(*t);
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for group in by_face.values() {
        match flat_polygon(&positions, group) {
            Some(ring) => faces.push(ring),
            None => faces.extend(group.iter().map(|t| t.to_vec())),
        }
    }
    Some(MeshData::from_faces(positions, &faces))
}

/// A face's triangles as one polygon: when they lie in one plane and their
/// outside edges make a single loop. Its points in order, wound as the
/// triangles are.
fn flat_polygon(positions: &[Vec3], tris: &[[u32; 3]]) -> Option<Vec<u32>> {
    if tris.len() < 2 {
        return None;
    }
    let normal = |t: &[u32; 3]| {
        let [a, b, c] = t.map(|i| positions[i as usize].as_dvec3());
        (b - a).cross(c - a)
    };
    let n = tris.iter().map(normal).sum::<DVec3>().normalize_or_zero();
    if n == DVec3::ZERO
        || tris
            .iter()
            .any(|t| normal(t).normalize_or_zero().dot(n) < 1.0 - 1e-9)
    {
        return None;
    }
    // The edges only one triangle has, each from its start to its end.
    let mut count: std::collections::HashMap<(u32, u32), u32> = Default::default();
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut next: std::collections::HashMap<u32, u32> = Default::default();
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if count[&(a.min(b), a.max(b))] == 1 && next.insert(a, b).is_some() {
                // A point the boundary passes twice: not one simple loop.
                return None;
            }
        }
    }
    let &start = next.keys().min()?;
    let mut ring = vec![start];
    let mut at = start;
    loop {
        at = *next.get(&at)?;
        if at == start {
            break;
        }
        if ring.len() > next.len() {
            return None;
        }
        ring.push(at);
    }
    // Every boundary edge in the one loop: no hole, no second piece.
    (ring.len() == next.len() && ring.len() >= 3).then_some(ring)
}

#[cfg(all(test, feature = "occt"))]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};

    /// A 40 × 20 mm plate drawn with a 6 mm hole, extruded 10 mm, then a
    /// pattern of the extrude: the kernel's volume is the true one to the
    /// last digits — the hole a cylinder, not a prism of chords — and its
    /// screen mesh is closed, with each flat face one polygon.
    #[test]
    fn a_drawn_plate_with_a_hole_is_built_exactly() {
        let mut doc = Document::new();
        let mut d = crate::drawing::Drawing::new();
        let c = [[0.0, 0.0], [0.04, 0.0], [0.04, 0.02], [0.0, 0.02]];
        for i in 0..4 {
            d.add_curve(Curve2D::Line {
                a: DVec2::from(c[i]),
                b: DVec2::from(c[(i + 1) % 4]),
            });
        }
        d.add_curve(Curve2D::Circle {
            c: DVec2::new(0.01, 0.01),
            r: 0.003,
        });
        let mesh = doc.add_mesh(MeshData::empty());
        let mut sketch = Object::new("Sketch", ObjectData::Mesh(mesh));
        sketch.drawing = Some(Box::new(d));
        let id = doc.add_object(sketch);
        let uid = doc.object_uid(id).unwrap();
        let features = vec![
            Feature {
                shape: Shape::Extrude {
                    distance: 0.01,
                    symmetric: false,
                },
                sketch: uid,
                seeds: vec![[0.03, 0.01]],
                operation: Operation::Join,
                order: 1,
            },
            Feature {
                shape: Shape::Pattern {
                    count: 2,
                    step: [0.0, 0.0, 0.01],
                },
                sketch: uid,
                seeds: Vec::new(),
                operation: Operation::Join,
                order: 2,
            },
        ];
        let solid = body_solid(&doc, &features, DAffine3::IDENTITY).expect("built exactly");
        let want = (0.04 * 0.02 - std::f64::consts::PI * 0.003f64.powi(2)) * 0.02;
        assert!(
            (solid.volume() - want).abs() < want * 1e-9,
            "{} against {want}",
            solid.volume()
        );
        let m = body_mesh(&doc, &features, DAffine3::IDENTITY).expect("meshed");
        assert!(
            kreate_mesh::measure::is_closed(&m),
            "the screen's mesh is closed"
        );
        let v = kreate_mesh::measure::mass_properties(&m, |p| p).volume;
        assert!((v - want).abs() < want * 1e-3, "{v}");
        // The top and bottom, holed, stay triangles; the four sides are one
        // polygon each.
        let quads = m.faces().filter(|f| f.len() == 4).count();
        assert!(quads >= 4, "{quads} four-sided faces");
        // And `body` takes this path.
        let built = super::super::body(&doc, &features, DAffine3::IDENTITY);
        assert_eq!(built.num_faces(), m.num_faces());
    }

    /// What the kernel does not build yet makes the body come from meshes.
    #[test]
    fn a_body_with_a_fillet_falls_back_to_meshes() {
        let doc = Document::new();
        let features = vec![Feature {
            shape: Shape::Round {
                radius: 0.001,
                chamfer: false,
            },
            sketch: kreate_reflect::Uid::from_raw(1),
            seeds: Vec::new(),
            operation: Operation::Join,
            order: 1,
        }];
        assert!(body_solid(&doc, &features, DAffine3::IDENTITY).is_none());
    }
}
