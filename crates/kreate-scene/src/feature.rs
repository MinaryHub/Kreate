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

pub mod exact;
pub mod shapes;

/// What a feature does to the body so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Starts the body, or adds to it — Fusion's New Body for the first, Join
    /// for the rest.
    #[default]
    Join,
    /// Takes away from it.
    Cut,
    /// Keeps only what the body and it share — Fusion's Intersect,
    /// SolidWorks's Common.
    Intersect,
}

impl Operation {
    pub const ALL: [Operation; 3] = [Operation::Join, Operation::Cut, Operation::Intersect];

    pub fn tag(self) -> &'static str {
        match self {
            Operation::Join => "join",
            Operation::Cut => "cut",
            Operation::Intersect => "intersect",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.tag() == tag)
    }
}

/// The rest of how an extrude goes, beside how far: what Fusion's and
/// SolidWorks's extrudes offer under their distance. All nought, it is the
/// plain push.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExtrudeMore {
    /// How far it goes the other way too, in metres: two sides, each its
    /// own length.
    pub second: f64,
    /// Where it starts: this far off the sketch's plane, the way it goes.
    pub offset: f64,
    /// How its sides lean, in radians: positive, it widens as it goes;
    /// negative, it narrows — a mould's draft.
    pub taper: f64,
    /// Through all of the body in its way, however thick that is or
    /// becomes: the distance says only which way.
    pub through: bool,
}

impl ExtrudeMore {
    /// Where the extrude starts and ends along the sketch's normal: the
    /// start is the end that is the region as drawn, the other the one a
    /// taper has grown or shrunk.
    pub fn span(&self, distance: f64, symmetric: bool) -> (f64, f64) {
        if symmetric {
            let half = distance.abs() * 0.5;
            return (self.offset - half, self.offset + half);
        }
        let way = if distance < 0.0 { -1.0 } else { 1.0 };
        (
            way * (self.offset - self.second.abs()),
            way * self.offset + distance,
        )
    }

    /// The distance a through-all extrude goes: past `reach`, how far the
    /// body lies from the sketch's plane at its furthest, the way
    /// `distance` points.
    pub fn through_distance(&self, distance: f64, symmetric: bool, reach: f64) -> f64 {
        let way = if distance < 0.0 { -1.0 } else { 1.0 };
        let far = reach * 1.001 + 1e-4;
        way * if symmetric { far * 2.0 } else { far }
    }
}

/// The solid a feature makes of its regions.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// Pushed out of the sketch's plane this far, in metres: negative goes
    /// the other way; symmetric goes half each way. `more` is the rest of
    /// it: a second side, a start offset, a taper, through all.
    Extrude {
        distance: f64,
        symmetric: bool,
        more: ExtrudeMore,
    },
    /// Turned about the line through two of the sketch's points, by this
    /// many radians — or about the sketch's own horizontal or vertical axis,
    /// when both are [`AXIS_X`] or [`AXIS_Y`].
    Revolve { axis: [u32; 2], angle: f64 },
    /// The body's edges at `edges` — a point on each, in the body's space,
    /// with the chain of edges each runs on in — rounded to this radius, in
    /// metres, or cut back this far, for a chamfer; every sharp edge when
    /// there are none. Names no sketch.
    Round {
        radius: f64,
        chamfer: bool,
        edges: Vec<[f64; 3]>,
    },
    /// The body hollowed to a wall `thickness` thick; `open` is the side
    /// taken off — 0 none, then +X, −X, +Y, −Y, +Z, −Z — so a box becomes a
    /// tray. Fusion's Shell.
    Shell { thickness: f64, open: u32 },
    /// The feature before it repeated `count` times in all, each `step`
    /// further along in the body's space — and each of those `count2`
    /// times `step2` along a second direction, a grid; one is a row —
    /// joined or cut as it was.
    Pattern {
        count: u32,
        step: [f64; 3],
        count2: u32,
        step2: [f64; 3],
    },
    /// The feature before it repeated `count` times round an axis: the
    /// body's (0 X, 1 Y, 2 Z) through its origin, or the construction axis
    /// `around` when there is one. Spread over `angle` radians: evenly all
    /// round for a whole turn, end to end over less.
    Circular {
        count: u32,
        axis: u32,
        angle: f64,
        around: Option<Uid>,
    },
    /// The feature before it reflected across the body's plane through its
    /// origin square to the axis given, joined or cut as it was.
    Mirror { axis: u32 },
    /// The sketch's picked regions carried along another object's lines —
    /// a sketch's or a mesh's edges, by its identity — their size going
    /// from one to `scale_end`, the ends closed when `caps`. Fusion's Sweep.
    Sweep {
        path: Uid,
        scale_end: f64,
        caps: bool,
    },
    /// A solid through the sketch's largest region and each of these
    /// sketches', in order through space. Fusion's Loft.
    Loft { sections: Vec<Uid> },
    /// The sketch's picked region (its largest when none is) as a plate
    /// `thickness` thick, with a flange bent up from each straight edge
    /// nearest one of `edges` (points in the sketch's plane) — or from every
    /// edge that can take one, when there are none. Laid out flat, the
    /// blank. See `kreate_mesh::mech::plate`.
    SheetMetal {
        thickness: f64,
        radius: f64,
        k_factor: f64,
        angle: f64,
        height: f64,
        edges: Vec<[f64; 2]>,
        flat: bool,
    },
    /// A hole at every point of the sketch — every circle's centre, where
    /// it has no points — drilled down its Z from its plane: plain,
    /// counterbored, countersunk or tapped (`kind`, as
    /// `kreate_mesh::mech::HoleKind` in that order) for the ISO metric
    /// screw `size` (an index into `kreate_mesh::mech::METRIC`), `depth`
    /// deep or through all. `diameter` is a plain hole's, when not its
    /// size's clearance. Fusion's Hole, from sketch points.
    Hole {
        kind: u32,
        size: u32,
        diameter: f64,
        depth: f64,
        through: bool,
    },
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
    /// Left out of the body for now, and kept: Fusion's Suppress. What comes
    /// after it is built as if it were not there.
    pub suppressed: bool,
    /// What the timeline calls it; empty for its kind and its place.
    pub name: String,
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
            Shape::Sweep { .. } => "sweep",
            Shape::Loft { .. } => "loft",
            Shape::SheetMetal { .. } => "sheet_metal",
            Shape::Hole { .. } => "hole",
        }
    }

    /// Every object it is made from: its sketch, and a sweep's path or a
    /// loft's other sections.
    pub fn references(&self) -> Vec<Uid> {
        let mut out = vec![self.sketch];
        match &self.shape {
            Shape::Sweep { path, .. } => out.push(*path),
            Shape::Circular {
                around: Some(axis), ..
            } => out.push(*axis),
            Shape::Loft { sections } => out.extend(sections.iter().copied()),
            _ => {}
        }
        out
    }

    /// Whether it works on the body so far rather than on a sketch.
    pub fn on_body(&self) -> bool {
        !matches!(
            self.shape,
            Shape::Extrude { .. }
                | Shape::Revolve { .. }
                | Shape::Sweep { .. }
                | Shape::Loft { .. }
                | Shape::SheetMetal { .. }
                | Shape::Hole { .. }
        )
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
    // A drawing names its axis line by its id, twice.
    if let Some(drawing) = doc.object(sketch).and_then(|o| o.drawing.as_deref()) {
        if axis[0] != axis[1] {
            return None;
        }
        let kreate_geom2d::Curve2D::Line { a, b } =
            drawing.curve(crate::drawing::EntityId(axis[0]))?
        else {
            return None;
        };
        let (a, b) = (a.as_vec2(), b.as_vec2());
        return (a.distance(b) > 0.0).then_some((a, b - a));
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
    let Some(object) = doc.object(sketch) else {
        return Vec::new();
    };
    // Points held where they are — a face's projected outline — whose
    // regions alone are no profile.
    let fixed: Vec<Vec2> = match object.drawing.as_deref() {
        Some(drawing) => drawing
            .constraints
            .iter()
            .filter_map(|c| match c {
                crate::drawing::Constraint::Fixed(_, at) => Some(at.as_vec2()),
                _ => None,
            })
            .collect(),
        None => {
            let Some(mesh) = object.mesh().and_then(|m| doc.mesh(m)) else {
                return Vec::new();
            };
            object
                .sketch
                .iter()
                .filter_map(|c| match c {
                    crate::sketch::Constraint::Fixed { a, .. } => mesh
                        .positions()
                        .get(*a as usize)
                        .map(|p| Vec2::new(p.x, p.y)),
                    _ => None,
                })
                .collect()
        }
    };
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
    if let Some(drawing) = object.drawing.as_deref() {
        return Some(drawn_edges(drawing));
    }
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

/// How finely a drawing's curves are drawn for a feature: a hundredth of a
/// millimetre from the true curve, which no part shows and no machine
/// holds.
pub const DRAWING_CHORD: f64 = 1e-5;

/// A drawing's profile curves as points and the edges between them, for
/// the region finder and the sweeps that take points: each curve drawn to
/// [`DRAWING_CHORD`], its ends shared with whatever else ends there.
pub fn drawn_edges(drawing: &crate::drawing::Drawing) -> (Vec<Vec2>, Vec<[u32; 2]>) {
    let mut points: Vec<Vec2> = Vec::new();
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let index = |p: kreate_geom2d::DVec2, points: &mut Vec<Vec2>| -> u32 {
        let q = Vec2::new(p.x as f32, p.y as f32);
        if let Some(i) = points.iter().position(|o| o.distance(q) < 1e-7) {
            return i as u32;
        }
        points.push(q);
        (points.len() - 1) as u32
    };
    for (_, curve) in drawing.profile_curves() {
        let drawn = curve.tessellate(DRAWING_CHORD);
        let ids: Vec<u32> = drawn.iter().map(|p| index(*p, &mut points)).collect();
        for w in ids.windows(2) {
            if w[0] != w[1] {
                edges.push([w[0], w[1]]);
            }
        }
    }
    (points, edges)
}

/// The solid one feature makes, in its sketch's own space; empty when its
/// sketch is gone or its seeds fall in no region.
pub fn solid(doc: &Document, feature: &Feature) -> MeshData {
    solid_reaching(doc, feature, None)
}

/// [`solid`], for a feature in a body: `reach` is how far the body so far
/// lies from the feature's sketch plane at its furthest, which is how far a
/// through-all extrude goes.
pub fn solid_reaching(doc: &Document, feature: &Feature, reach: Option<f64>) -> MeshData {
    let Some(sketch) = doc.object_by_uid(feature.sketch) else {
        return MeshData::empty();
    };
    // A sweep and a loft are made in the world, from more than the one
    // sketch, and brought into the sketch's space as the rest are made.
    let world = match &feature.shape {
        Shape::Sweep {
            path,
            scale_end,
            caps,
        } => Some(
            doc.object_by_uid(*path)
                .and_then(|p| shapes::object_path(doc, p))
                .and_then(|(points, closed)| {
                    let profiles = shapes::picked_regions(doc, sketch, &feature.seeds);
                    shapes::sweep_world(&points, closed, &profiles, *scale_end, *caps).ok()
                }),
        ),
        Shape::Loft { sections } => Some({
            let all: Vec<_> = std::iter::once(sketch)
                .chain(sections.iter().filter_map(|u| doc.object_by_uid(*u)))
                .filter_map(|id| shapes::section_world(doc, id))
                .collect();
            shapes::loft_world(all).ok()
        }),
        _ => None,
    };
    // Holes: a cutter at each of the sketch's points, in its own space.
    if let Shape::Hole {
        kind,
        size,
        diameter,
        depth,
        through,
    } = feature.shape
    {
        return holes(doc, sketch, kind, size, diameter, depth, through, reach);
    }
    // A plate is made in the sketch's own plane, as an extrude is.
    if matches!(feature.shape, Shape::SheetMetal { .. }) {
        return sheet_plate(doc, feature)
            .map(|p| kreate_mesh::mech::plate(&p))
            .unwrap_or_else(MeshData::empty);
    }
    if let Some(world) = world {
        let Some(mesh) = world else {
            return MeshData::empty();
        };
        let to_sketch = doc.world_transform(sketch).inverse();
        return mapped(&mesh, false, |p| to_sketch.transform_point3(p));
    }
    let all = shape_regions(doc, sketch, &feature.shape);
    let mut out = MeshData::empty();
    for i in picked(&all, &feature.seeds) {
        let piece = match feature.shape {
            Shape::Extrude {
                distance,
                symmetric,
                more,
            } => {
                let distance = match (more.through, reach) {
                    (true, Some(reach)) => more.through_distance(distance, symmetric, reach),
                    _ => distance,
                };
                let (z0, z1) = more.span(distance, symmetric);
                let grow = more.taper.tan() * (z1 - z0).abs();
                regions::extrude_region_tapered(&all[i], z0 as f32, z1 as f32, grow as f32)
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
            | Shape::Mirror { .. }
            | Shape::Sweep { .. }
            | Shape::Loft { .. }
            | Shape::SheetMetal { .. }
            | Shape::Hole { .. } => continue,
        };
        out = if out.num_faces() == 0 {
            piece
        } else {
            boolean(&out, &piece, BooleanOp::Union)
        };
    }
    out
}

/// A sheet metal feature's plate, in its sketch's plane: the picked region
/// (or the largest), its outline made ready to bend, and a flange on each
/// edge nearest one of the feature's edge points — or on every edge that
/// can take one. `None` for another kind of feature or a sketch with no
/// region.
pub fn sheet_plate(doc: &Document, feature: &Feature) -> Option<kreate_mesh::mech::Plate> {
    let Shape::SheetMetal {
        thickness,
        radius,
        k_factor,
        angle,
        height,
        edges,
        flat,
    } = &feature.shape
    else {
        return None;
    };
    let sketch = doc.object_by_uid(feature.sketch)?;
    let (outline, holes) = shapes::picked_regions(doc, sketch, &feature.seeds)
        .into_iter()
        .next()
        .or_else(|| shapes::largest_region(doc, sketch))?;
    let outline = kreate_mesh::mech::plate_outline(&outline);
    let t = *thickness as f32;
    let chosen: Vec<usize> = if edges.is_empty() {
        // Every edge long enough to bend: four times the bend's reach.
        kreate_mesh::mech::flangeable_edges(&outline, 4.0 * (*radius as f32 + t))
    } else {
        let n = outline.len();
        let mut out: Vec<usize> = edges
            .iter()
            .filter_map(|e| {
                let q = kreate_math::Vec2::new(e[0] as f32, e[1] as f32);
                (0..n).min_by(|&i, &j| {
                    let d = |k: usize| {
                        let (a, b) = (outline[k], outline[(k + 1) % n]);
                        let ab = b - a;
                        let s = ((q - a).dot(ab) / ab.length_squared().max(1e-20)).clamp(0.0, 1.0);
                        (a + ab * s).distance(q)
                    };
                    d(i).total_cmp(&d(j))
                })
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    };
    Some(kreate_mesh::mech::Plate {
        outline,
        holes,
        thickness: t,
        radius: *radius as f32,
        k_factor: *k_factor as f32,
        flanges: chosen
            .into_iter()
            .map(|edge| kreate_mesh::mech::Flange {
                edge,
                angle: *angle as f32,
                height: *height as f32,
            })
            .collect(),
        flat: *flat,
    })
}

/// Where a pattern puts its copies, in the body's space: each copy's move
/// from the original, the original's own left out.
pub fn pattern_moves(doc: &Document, shape: &Shape, to_body: DAffine3) -> Vec<DAffine3> {
    use kreate_math::{DQuat, DVec3};
    match shape {
        Shape::Pattern {
            count,
            step,
            count2,
            step2,
        } => {
            let (a, b) = (DVec3::from_array(*step), DVec3::from_array(*step2));
            let mut out = Vec::new();
            for j in 0..(*count2).max(1) {
                for i in 0..(*count).max(1) {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    out.push(DAffine3::from_translation(a * i as f64 + b * j as f64));
                }
            }
            out
        }
        Shape::Circular {
            count,
            axis,
            angle,
            around,
        } => {
            // The axis: a construction axis's Z through its middle, in the
            // body's space; else the body's own.
            let (origin, dir) = match around.and_then(|u| doc.object_by_uid(u)) {
                Some(id) => {
                    let w = to_body * doc.world_transform(id);
                    (
                        w.translation,
                        w.transform_vector3(DVec3::Z).normalize_or_zero(),
                    )
                }
                None => (
                    DVec3::ZERO,
                    match axis {
                        0 => DVec3::X,
                        1 => DVec3::Y,
                        _ => DVec3::Z,
                    },
                ),
            };
            if dir == DVec3::ZERO {
                return Vec::new();
            }
            let n = (*count).max(1);
            let whole = angle.abs() >= std::f64::consts::TAU - 1e-6 || *angle == 0.0;
            let each = if whole {
                std::f64::consts::TAU / n as f64
            } else if n > 1 {
                angle / (n - 1) as f64
            } else {
                0.0
            };
            (1..n)
                .map(|k| {
                    DAffine3::from_translation(origin)
                        * DAffine3::from_quat(DQuat::from_axis_angle(dir, each * k as f64))
                        * DAffine3::from_translation(-origin)
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Where a sketch's holes go, in its own plane: its points, or with none
/// its circles' centres.
pub fn hole_points(doc: &Document, sketch: crate::document::ObjectId) -> Vec<DVec2> {
    let Some(d) = doc.object(sketch).and_then(|o| o.drawing.as_deref()) else {
        return Vec::new();
    };
    let points: Vec<DVec2> = d
        .entities
        .iter()
        .filter_map(|e| match &e.kind {
            crate::drawing::EntityKind::Point(p) => Some(*p),
            _ => None,
        })
        .collect();
    if !points.is_empty() {
        return points;
    }
    d.entities
        .iter()
        .filter(|e| !e.construction)
        .filter_map(|e| match &e.kind {
            crate::drawing::EntityKind::Curve(kreate_geom2d::Curve2D::Circle { c, .. }) => Some(*c),
            _ => None,
        })
        .collect()
}

/// The cutters of a hole feature, in the sketch's space: one at each of
/// its points, going down its Z — through all, past `reach`, when asked.
#[allow(clippy::too_many_arguments)]
fn holes(
    doc: &Document,
    sketch: crate::document::ObjectId,
    kind: u32,
    size: u32,
    diameter: f64,
    depth: f64,
    through: bool,
    reach: Option<f64>,
) -> MeshData {
    use kreate_mesh::mech::{hole_cutter, HoleKind, METRIC};
    let kind = match kind {
        1 => HoleKind::Counterbore,
        2 => HoleKind::Countersink,
        3 => HoleKind::Tapped,
        _ => HoleKind::Simple,
    };
    let Some(size) = METRIC.get(size as usize) else {
        return MeshData::empty();
    };
    let depth = match (through, reach) {
        (true, Some(reach)) => reach * 1.001 + 1e-4,
        (true, None) => depth.max(0.1),
        _ => depth,
    };
    if depth.is_nan() || depth <= 0.0 {
        return MeshData::empty();
    }
    let cutter = hole_cutter(
        kind,
        size,
        (diameter > 0.0).then_some(diameter),
        depth,
        through,
    );
    let points = hole_points(doc, sketch);
    let placed: Vec<MeshData> = points
        .iter()
        .map(|p| {
            mapped(&cutter, false, |q| {
                q + kreate_math::DVec3::new(p.x, p.y, 0.0)
            })
        })
        .collect();
    let parts: Vec<kreate_mesh::edit::join::Part<'_>> = placed
        .iter()
        .map(|m| kreate_mesh::edit::join::Part {
            mesh: m,
            transform: DAffine3::IDENTITY,
        })
        .collect();
    if parts.is_empty() {
        return MeshData::empty();
    }
    kreate_mesh::edit::join::join(&parts).0
}

/// `piece` joined to or cut from `out`, as `op` says; the first piece
/// starts the body.
fn combine(out: MeshData, piece: MeshData, op: Operation) -> MeshData {
    if out.num_faces() == 0 {
        return match op {
            Operation::Join => piece,
            // Nothing to take from, nothing to share with.
            Operation::Cut | Operation::Intersect => out,
        };
    }
    let op = match op {
        Operation::Join => BooleanOp::Union,
        Operation::Cut => BooleanOp::Difference,
        Operation::Intersect => BooleanOp::Intersect,
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
    body_with_status(doc, features, body_world).0
}

/// Why a feature made nothing of the body, as a catalogue key: its sketch
/// gone, its regions closed up or changed, its edges gone, nothing before
/// it to repeat. One for each of `features`; `None` for one that built, or
/// is left out.
pub type FeatureStatus = Option<&'static str>;

/// [`body`], with what became of each feature.
pub fn body_with_status(
    doc: &Document,
    features: &[Feature],
    body_world: DAffine3,
) -> (MeshData, Vec<FeatureStatus>) {
    let mut status: Vec<FeatureStatus> = vec![None; features.len()];
    // Past the rollback marker, left out as a suppressed one is.
    let rolled = |f: &Feature| doc.design.rollback.is_some_and(|r| f.order > r);
    let marked: Vec<Feature>;
    let features = if features.iter().any(&rolled) {
        marked = features
            .iter()
            .map(|f| Feature {
                suppressed: f.suppressed || rolled(f),
                ..f.clone()
            })
            .collect();
        &marked[..]
    } else {
        features
    };
    // Where each of what is built stands among all of them, for its status.
    let places: Vec<usize> = (0..features.len())
        .filter(|i| !features[*i].suppressed)
        .collect();
    // Without what is suppressed, for the kernel and the meshes alike.
    let live: Vec<Feature>;
    let features = if features.iter().any(|f| f.suppressed) {
        // A pattern, a ring or a mirror repeats the feature before it:
        // with that one left out there is nothing of its own to repeat, and
        // it goes too — taken as it stood, it repeated whatever came before
        // that instead, and a suppressed hole's pattern made five plates.
        let mut source_gone = false;
        live = features
            .iter()
            .filter(|f| {
                let repeats = matches!(
                    f.shape,
                    Shape::Pattern { .. } | Shape::Circular { .. } | Shape::Mirror { .. }
                );
                let from_sketch = matches!(
                    f.shape,
                    Shape::Extrude { .. }
                        | Shape::Revolve { .. }
                        | Shape::Sweep { .. }
                        | Shape::Loft { .. }
                        | Shape::SheetMetal { .. }
                        | Shape::Hole { .. }
                );
                if from_sketch {
                    source_gone = f.suppressed;
                }
                !f.suppressed && !(repeats && source_gone)
            })
            .cloned()
            .collect();
        &live[..]
    } else {
        features
    };
    // Exactly, by the kernel, where it is there and can; otherwise as
    // meshes, as below. The kernel builds a body whole or not at all, so
    // what it builds failed nowhere.
    if let Some(exact) = exact::body_mesh(doc, features, body_world) {
        return (exact, status);
    }
    let place = |i: usize| places.get(i).copied();
    let to_body = body_world.inverse();
    let mut out = MeshData::empty();
    // The last solid a feature made, in the body's space, with what it did:
    // what a pattern or a mirror repeats.
    let mut last: Option<(MeshData, Operation)> = None;
    for (i, f) in features.iter().enumerate() {
        let mut fail = |why: &'static str| {
            if let Some(at) = place(i) {
                status[at] = Some(why);
            }
        };
        // These work on what the body is so far, not on a sketch.
        if matches!(
            f.shape,
            Shape::Pattern { .. } | Shape::Circular { .. } | Shape::Mirror { .. }
        ) && last.is_none()
        {
            fail("feature-failed-source");
        }
        match f.shape {
            Shape::Round {
                radius,
                chamfer,
                ref edges,
            } => {
                if out.num_faces() > 0 && radius > 0.0 {
                    let points: Vec<kreate_math::Vec3> = edges
                        .iter()
                        .map(|p| kreate_math::DVec3::from_array(*p).as_vec3())
                        .collect();
                    if !points.is_empty()
                        && kreate_mesh::modifiers::bevel::chosen_edges(&out, &points).is_empty()
                    {
                        fail("feature-failed-edges");
                    }
                    out = kreate_mesh::modifiers::bevel::round_chosen(
                        &out,
                        radius as f32,
                        chamfer,
                        &points,
                    );
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
            Shape::Pattern { .. } | Shape::Circular { .. } => {
                if let Some((piece, op)) = &last {
                    for m in pattern_moves(doc, &f.shape, to_body) {
                        let copy = mapped(piece, false, |p| m.transform_point3(p));
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
            Shape::Extrude { .. }
            | Shape::Revolve { .. }
            | Shape::Sweep { .. }
            | Shape::Loft { .. }
            | Shape::SheetMetal { .. }
            | Shape::Hole { .. } => {}
        }
        let sketch_world = doc
            .object_by_uid(f.sketch)
            .map(|id| doc.world_transform(id))
            .unwrap_or(DAffine3::IDENTITY);
        // Through all: how far the body so far lies from the sketch's
        // plane, in the sketch's own space, which is what it has to pass.
        let reach = match f.shape {
            Shape::Hole { through: true, .. } if out.num_faces() > 0 => {
                let to_sketch = (to_body * sketch_world).inverse();
                Some(
                    out.positions()
                        .iter()
                        .map(|p| to_sketch.transform_point3(p.as_dvec3()).z.abs())
                        .fold(0.0f64, f64::max),
                )
            }
            Shape::Extrude { more, .. } if more.through && out.num_faces() > 0 => {
                let to_sketch = (to_body * sketch_world).inverse();
                Some(
                    out.positions()
                        .iter()
                        .map(|p| to_sketch.transform_point3(p.as_dvec3()).z.abs())
                        .fold(0.0f64, f64::max),
                )
            }
            _ => None,
        };
        let mut piece = solid_reaching(doc, f, reach);
        if piece.num_faces() == 0 {
            fail(if doc.object_by_uid(f.sketch).is_none() {
                "feature-failed-sketch"
            } else {
                "feature-failed-region"
            });
            // Nothing made — its sketch gone, its region closed up: what
            // follows has nothing of this one to repeat.
            last = None;
            continue;
        }
        let into = to_body * sketch_world;
        for p in piece.positions_mut() {
            *p = into.transform_point3(p.as_dvec3()).as_vec3();
        }
        last = Some((piece.clone(), f.operation));
        out = combine(out, piece, f.operation);
    }
    (out, status)
}

/// The order a feature made now takes: the next on the timeline — or,
/// rolled back, the one just after the marker, those after it moved one
/// on to make room (and what is held to them with them), and the marker
/// after the new one, as Fusion puts a feature in where the marker is.
pub fn take_order(doc: &mut Document) -> u64 {
    let Some(r) = doc.design.rollback else {
        return next_order(doc);
    };
    let ids: Vec<crate::document::ObjectId> = doc.objects().map(|(id, _)| id).collect();
    for id in ids {
        if let Some(o) = doc.object_mut(id) {
            for f in o.features.iter_mut().filter(|f| f.order > r) {
                f.order += 1;
            }
        }
    }
    for b in &mut doc.design.bindings {
        if let crate::design::Bound::FeatureAmount { order, .. } = &mut b.target {
            if *order > r {
                *order += 1;
            }
        }
    }
    doc.design.rollback = Some(r + 1);
    r + 1
}

/// What a body's shape depends on, for the cache: its features, each
/// sketch's points and place, and its own place.
pub fn key(doc: &Document, features: &[Feature], body_world: DAffine3) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{features:?}").hash(&mut h);
    doc.design.rollback.hash(&mut h);
    // Where each thing it is made from stands *from the body*, to a
    // nanometre: a body moved with its sketches is the same body, and
    // built again at every step of a drag it was a boolean per frame.
    let to_body = body_world.inverse();
    for uid in features.iter().flat_map(Feature::references) {
        if let Some(id) = doc.object_by_uid(uid) {
            for v in (to_body * doc.world_transform(id)).to_cols_array() {
                ((v * 1e9).round() as i64).hash(&mut h);
            }
            if let Some(o) = doc.object(id) {
                // A path that is a curve: its points are the curve's.
                if let Some(curve) = &o.curve {
                    curve.key().hash(&mut h);
                }
                o.sketch_lines.hash(&mut h);
                // Which of its lines are construction: they bound no region.
                format!("{:?}", o.sketch).hash(&mut h);
                // A drawing is its own geometry, not a mesh's.
                if let Some(d) = o.drawing.as_deref() {
                    format!("{d:?}").hash(&mut h);
                }
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
                more: Default::default(),
            },
            sketch: s,
            seeds: vec![[0.002, 0.002], [0.02, 0.01]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
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

    /// What an extrude has beside its distance: a start offset, a second
    /// side, through all of the body in its way, and keeping only what it
    /// shares with the body.
    #[test]
    fn an_extrude_starts_off_the_plane_goes_both_ways_and_through_all() {
        let mut doc = Document::new();
        let s = sketch(
            &mut doc,
            &[rect(0.04, 0.02), circle(Vec2::new(0.02, 0.01), 0.005)],
        );
        let extrude = |distance: f64, more: ExtrudeMore| Shape::Extrude {
            distance,
            symmetric: false,
            more,
        };
        let plate = Feature {
            shape: extrude(0.01, ExtrudeMore::default()),
            sketch: s,
            seeds: vec![[0.002, 0.002], [0.02, 0.01]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
        };
        let area = 0.04 * 0.02;
        let volume = |features: &[Feature]| {
            mass_properties(&body(&doc, features, DAffine3::IDENTITY), |p| p).volume
        };
        // From 2 mm off the plane, 10 mm on: the same slab, higher.
        let lifted = Feature {
            shape: extrude(
                0.01,
                ExtrudeMore {
                    offset: 0.002,
                    ..Default::default()
                },
            ),
            ..plate.clone()
        };
        let made = body(&doc, std::slice::from_ref(&lifted), DAffine3::IDENTITY);
        let b = made.bounds();
        assert!(
            (b.min.z - 0.002).abs() < 1e-6 && (b.max.z - 0.012).abs() < 1e-6,
            "{b:?}"
        );
        // 10 mm one way and 4 the other.
        let both = Feature {
            shape: extrude(
                0.01,
                ExtrudeMore {
                    second: 0.004,
                    ..Default::default()
                },
            ),
            ..plate.clone()
        };
        let v = volume(std::slice::from_ref(&both));
        assert!((v - area * 0.014).abs() < area * 0.014 * 1e-3, "{v}");
        // The disc cut through all, asked for only a hair deep: all the
        // way, however thick the plate.
        let disc = std::f64::consts::PI * 0.005f64.powi(2);
        let through = Feature {
            shape: extrude(
                0.0001,
                ExtrudeMore {
                    through: true,
                    ..Default::default()
                },
            ),
            seeds: vec![[0.02, 0.01]],
            operation: Operation::Cut,
            order: 2,
            ..plate.clone()
        };
        let v = volume(&[plate.clone(), through.clone()]);
        let want = (area - disc) * 0.01;
        assert!((v - want).abs() < want * 5e-3, "{v} against {want}");
        // The plate made three times as thick: still through.
        let thick = Feature {
            shape: extrude(0.03, ExtrudeMore::default()),
            ..plate.clone()
        };
        let v = volume(&[thick, through]);
        let want = (area - disc) * 0.03;
        assert!((v - want).abs() < want * 5e-3, "{v} against {want}");
        // Intersect: only the disc of the plate is left.
        let shared = Feature {
            shape: extrude(0.05, ExtrudeMore::default()),
            seeds: vec![[0.02, 0.01]],
            operation: Operation::Intersect,
            order: 2,
            ..plate.clone()
        };
        let v = volume(&[plate, shared]);
        let want = disc * 0.01;
        assert!((v - want).abs() < want * 2e-2, "{v} against {want}");
    }

    /// A pattern repeats the feature before it: with that one suppressed
    /// it has nothing of its own to repeat, and does not take up the one
    /// before that instead.
    #[test]
    fn a_pattern_goes_with_the_feature_it_repeats() {
        let mut doc = Document::new();
        let s = sketch(
            &mut doc,
            &[rect(0.04, 0.02), circle(Vec2::new(0.02, 0.01), 0.005)],
        );
        let plate = Feature {
            shape: Shape::Extrude {
                distance: 0.01,
                symmetric: false,
                more: Default::default(),
            },
            sketch: s,
            seeds: vec![[0.002, 0.002], [0.02, 0.01]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
        };
        let hole = Feature {
            seeds: vec![[0.02, 0.01]],
            operation: Operation::Cut,
            order: 2,
            suppressed: true,
            ..plate.clone()
        };
        let pattern = Feature {
            shape: Shape::Pattern {
                count: 3,
                step: [0.1, 0.0, 0.0],
                count2: 1,
                step2: [0.0; 3],
            },
            order: 3,
            ..plate.clone()
        };
        let made = body(&doc, &[plate, hole, pattern], DAffine3::IDENTITY);
        let v = mass_properties(&made, |p| p).volume;
        // One plate, whole: not three of them.
        let want = 0.04 * 0.02 * 0.01;
        assert!((v - want).abs() < want * 1e-3, "{v} against {want}");
    }

    /// A fillet on one edge rounds that edge only — less taken away than
    /// every edge; and when an edit takes the edge away from where it was
    /// picked, it rounds nothing rather than another edge.
    #[test]
    fn a_fillet_on_one_edge_rounds_that_edge_and_follows_an_edit() {
        let mut doc = Document::new();
        let s = sketch(&mut doc, &[rect(0.04, 0.02)]);
        let plate = |height: f64| Feature {
            shape: Shape::Extrude {
                distance: height,
                symmetric: false,
                more: Default::default(),
            },
            sketch: s,
            seeds: vec![[0.002, 0.002]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
        };
        let round = |edges: Vec<[f64; 3]>| Feature {
            shape: Shape::Round {
                radius: 0.003,
                chamfer: false,
                edges,
            },
            sketch: kreate_reflect::Uid::NONE,
            seeds: Vec::new(),
            operation: Operation::Join,
            order: 2,
            suppressed: false,
            name: String::new(),
        };
        let volume = |features: &[Feature]| {
            mass_properties(&body(&doc, features, DAffine3::IDENTITY), |p| p).volume
        };
        let whole = volume(&[plate(0.01)]);
        // The long top edge along X, at y = 0, z = 0.01.
        let one = volume(&[plate(0.01), round(vec![[0.02, 0.0, 0.01]])]);
        let all = volume(&[plate(0.01), round(Vec::new())]);
        // A quarter round of 3 mm along 40 mm takes (1 - π/4) r² l.
        let want = whole - (1.0 - std::f64::consts::FRAC_PI_4) * 0.003f64.powi(2) * 0.04;
        assert!((one - want).abs() < want * 2e-3, "{one} against {want}");
        assert!(all < one, "every edge takes more: {all} {one}");
        // Twice as thick: the top edge has gone 10 mm up from the point it
        // was kept by, out of reach, and the fillet finds nothing to round
        // — it does not round some other edge instead.
        let thick = volume(&[plate(0.02), round(vec![[0.02, 0.0, 0.01]])]);
        let thick_whole = volume(&[plate(0.02)]);
        assert!((thick - thick_whole).abs() < thick_whole * 1e-9);
    }

    /// A feature that makes nothing says why; one rolled past the marker
    /// is left out; one made while rolled back goes in at the marker, the
    /// later ones moved on to make room.
    #[test]
    fn a_failed_feature_says_why_and_the_marker_leaves_out_what_follows() {
        let mut doc = Document::new();
        let s = sketch(
            &mut doc,
            &[rect(0.04, 0.02), circle(Vec2::new(0.02, 0.01), 0.005)],
        );
        let plate = Feature {
            shape: Shape::Extrude {
                distance: 0.01,
                symmetric: false,
                more: Default::default(),
            },
            sketch: s,
            seeds: vec![[0.002, 0.002], [0.02, 0.01]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
        };
        // A hole whose seed is in no region; a pattern after a sketch gone.
        let lost = Feature {
            seeds: vec![[5.0, 5.0]],
            operation: Operation::Cut,
            order: 2,
            ..plate.clone()
        };
        let gone = Feature {
            sketch: kreate_reflect::Uid::from_raw(999_999),
            order: 3,
            ..plate.clone()
        };
        let pattern = Feature {
            shape: Shape::Pattern {
                count: 2,
                step: [0.1, 0.0, 0.0],
                count2: 1,
                step2: [0.0; 3],
            },
            order: 4,
            ..plate.clone()
        };
        let all = [plate.clone(), lost, gone, pattern];
        let (_, status) = body_with_status(&doc, &all, DAffine3::IDENTITY);
        assert_eq!(
            status,
            vec![
                None,
                Some("feature-failed-region"),
                Some("feature-failed-sketch"),
                Some("feature-failed-source"),
            ]
        );
        // Rolled back to the plate: the rest are left out, and fail nothing.
        doc.design.rollback = Some(1);
        let (made, status) = body_with_status(&doc, &all, DAffine3::IDENTITY);
        assert!(status.iter().all(Option::is_none), "{status:?}");
        let v = mass_properties(&made, |p| p).volume;
        assert!((v - 0.04 * 0.02 * 0.01).abs() < 1e-9, "{v}");
        // A feature made now goes in after the plate; those after move on.
        let mut body_object = crate::document::Object::new(
            "Body",
            crate::document::ObjectData::Mesh(doc.add_mesh(MeshData::empty())),
        );
        body_object.features = all.to_vec();
        let id = doc.add_object(body_object);
        assert_eq!(take_order(&mut doc), 2);
        assert_eq!(doc.design.rollback, Some(2));
        let orders: Vec<u64> = doc
            .object(id)
            .unwrap()
            .features
            .iter()
            .map(|f| f.order)
            .collect();
        assert_eq!(orders, vec![1, 3, 4, 5]);
    }

    /// A grid of three by two copies; a ring of four over a quarter turn,
    /// end to end; and a ring about a construction axis off the origin.
    #[test]
    fn patterns_go_in_grids_and_partial_rings() {
        let mut doc = Document::new();
        let grid = Shape::Pattern {
            count: 3,
            step: [0.1, 0.0, 0.0],
            count2: 2,
            step2: [0.0, 0.1, 0.0],
        };
        let moves = pattern_moves(&doc, &grid, DAffine3::IDENTITY);
        assert_eq!(moves.len(), 5);
        assert!(moves
            .iter()
            .any(|m| (m.translation - kreate_math::DVec3::new(0.2, 0.1, 0.0)).length() < 1e-12));
        let ring = Shape::Circular {
            count: 4,
            axis: 2,
            angle: std::f64::consts::FRAC_PI_2,
            around: None,
        };
        let moves = pattern_moves(&doc, &ring, DAffine3::IDENTITY);
        // The last of four is a quarter turn round: (1, 0) to (0, 1).
        let last = moves
            .last()
            .unwrap()
            .transform_point3(kreate_math::DVec3::X);
        assert!((last - kreate_math::DVec3::Y).length() < 1e-9, "{last:?}");
        // About an axis standing at (1, 0): a half turn takes the origin to (2, 0).
        let mut axis = crate::document::Object::new("Axis", crate::document::ObjectData::Empty);
        axis.transform.translation = kreate_math::DVec3::X;
        let id = doc.add_object(axis);
        let uid = doc.object_uid(id).unwrap();
        let ring = Shape::Circular {
            count: 2,
            axis: 2,
            angle: std::f64::consts::TAU,
            around: Some(uid),
        };
        let moves = pattern_moves(&doc, &ring, DAffine3::IDENTITY);
        let p = moves[0].transform_point3(kreate_math::DVec3::ZERO);
        assert!(
            (p - kreate_math::DVec3::new(2.0, 0.0, 0.0)).length() < 1e-9,
            "{p:?}"
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
                more: Default::default(),
            },
            sketch: s,
            seeds: vec![[0.5, 0.5]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
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
                more: Default::default(),
            },
            sketch: s,
            seeds: vec![[5.0, 5.0]],
            operation: Operation::Join,
            order: 1,
            suppressed: false,
            name: String::new(),
        };
        assert_eq!(body(&doc, &[f], DAffine3::IDENTITY).num_faces(), 0);
    }
}
