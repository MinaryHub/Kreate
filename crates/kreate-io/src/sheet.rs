//! A drawing sheet: the parts drawn on paper as a workshop wants them — the
//! three views a projection method places, an isometric view, the overall
//! sizes, a border, a title block and a parts list — at a standard scale.
//! See design/2d-drafting.md §9.
//!
//! Composed afresh from the parts as they stand, every time it is asked
//! for, so a sheet is never behind its model. What it makes is a
//! [`Drawing`] in paper space — a millimetre of paper a millimetre of the
//! drawing — which the DXF writer ([`crate::dxf_write`]) and the PDF writer
//! ([`crate::pdf`]) both take, and which the drafting mode can open.
//!
//! The sizes are written as numbers of the model (a 40 mm part drawn at
//! 1:2 is 20 mm on paper and says 40), drawn out as lines and text rather
//! than as the drawing's own dimensions, which would measure the paper.
//!
//! The paper and its margins are ISO 5457's (A4 to A0, 20 mm on the left
//! for binding, 10 elsewhere); the scales ISO 5455's; the projections ISO
//! 128-30's first angle (Europe) and third angle (Korea's KS B 0001, the
//! Americas, Japan).

use kreate_geom2d::{Curve2D, DVec2};
use kreate_math::{DAffine3, DVec3};
use kreate_mesh::MeshData;
use kreate_scene::drawing::{ColorRef, Drawing, EntityKind, LayerId, Lineweight, Text, TextAlign};

use crate::drawing::{project_drawing, Line2, View};

/// A sheet of the ISO A series.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Paper {
    A4,
    #[default]
    A3,
    A2,
    A1,
    A0,
}

impl Paper {
    pub const ALL: [Paper; 5] = [Paper::A4, Paper::A3, Paper::A2, Paper::A1, Paper::A0];

    pub fn name(self) -> &'static str {
        match self {
            Paper::A4 => "A4",
            Paper::A3 => "A3",
            Paper::A2 => "A2",
            Paper::A1 => "A1",
            Paper::A0 => "A0",
        }
    }

    /// Width and height upright, in millimetres.
    pub fn size_mm(self) -> (f64, f64) {
        match self {
            Paper::A4 => (210.0, 297.0),
            Paper::A3 => (297.0, 420.0),
            Paper::A2 => (420.0, 594.0),
            Paper::A1 => (594.0, 841.0),
            Paper::A0 => (841.0, 1189.0),
        }
    }
}

/// Where the views go about the front: ISO 128-30.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    /// The top view below the front, the view from the right on its left.
    First,
    /// The top view above the front, the view from the right on its right:
    /// KS B 0001's.
    #[default]
    Third,
}

/// One line of the parts list.
#[derive(Clone, Debug, PartialEq)]
pub struct PartRow {
    pub name: String,
    pub count: u32,
    pub material: String,
}

/// The title block's and the parts list's headings, in the reader's
/// language.
#[derive(Clone, Debug, PartialEq)]
pub struct Labels {
    pub title: String,
    pub number: String,
    pub scale: String,
    pub projection: String,
    pub first_angle: String,
    pub third_angle: String,
    pub material: String,
    pub drawn: String,
    pub date: String,
    pub unit: String,
    pub item: String,
    pub part: String,
    pub quantity: String,
    pub more: String,
}

impl Default for Labels {
    fn default() -> Self {
        let s = |t: &str| t.to_string();
        Labels {
            title: s("TITLE"),
            number: s("DWG NO."),
            scale: s("SCALE"),
            projection: s("PROJECTION"),
            first_angle: s("1st angle"),
            third_angle: s("3rd angle"),
            material: s("MATERIAL"),
            drawn: s("DRAWN"),
            date: s("DATE"),
            unit: s("UNIT"),
            item: s("NO."),
            part: s("PART"),
            quantity: s("QTY"),
            more: s("and {n} more"),
        }
    }
}

/// What the sheet is to be.
#[derive(Clone, Debug, PartialEq)]
pub struct SheetSpec {
    pub paper: Paper,
    pub landscape: bool,
    /// Paper length over model length (`0.5` is 1:2); `None` picks the
    /// largest standard scale the views fit at.
    pub scale: Option<f64>,
    pub projection: Projection,
    /// An isometric view beside the three.
    pub iso: bool,
    pub title: String,
    pub number: String,
    pub material: String,
    pub author: String,
    pub date: String,
    /// The unit sizes are written in: metres to one, and its symbol.
    pub metres_per_unit: f64,
    pub unit: String,
    pub parts: Vec<PartRow>,
    pub labels: Labels,
}

impl Default for SheetSpec {
    fn default() -> Self {
        SheetSpec {
            paper: Paper::A3,
            landscape: true,
            scale: None,
            projection: Projection::Third,
            iso: true,
            title: String::new(),
            number: String::new(),
            material: String::new(),
            author: String::new(),
            date: String::new(),
            metres_per_unit: 1e-3,
            unit: "mm".into(),
            parts: Vec::new(),
            labels: Labels::default(),
        }
    }
}

/// A composed sheet: the drawing in paper space (metres of paper), its
/// size, and the scale it was drawn at.
#[derive(Clone, Debug)]
pub struct Sheet {
    pub drawing: Drawing,
    /// Each view drawn and the box it takes on the paper, in millimetres:
    /// its lowest corner and its highest.
    pub views: Vec<(View, DVec2, DVec2)>,
    pub width_mm: f64,
    pub height_mm: f64,
    pub scale: f64,
}

/// ISO 5455's scales, largest first.
pub const SCALES: [f64; 16] = [
    50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005,
];

/// A scale as a drawing writes it: `1:2`, `5:1`, `1:1`.
pub fn scale_label(scale: f64) -> String {
    let trim = |v: f64| {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    if scale >= 1.0 {
        format!("{}:1", trim(scale))
    } else {
        format!("1:{}", trim(1.0 / scale))
    }
}

/// The largest standard scale not above `fit`; the smallest there is when
/// none is.
pub fn standard_scale(fit: f64) -> f64 {
    SCALES
        .iter()
        .copied()
        .find(|s| *s <= fit * (1.0 + 1e-9))
        .unwrap_or(SCALES[SCALES.len() - 1])
}

// Paper measures, in millimetres.
const MARGIN_LEFT: f64 = 20.0;
const MARGIN: f64 = 10.0;
const TITLE_W: f64 = 180.0;
const TITLE_H: f64 = 36.0;
const ROW_H: f64 = 7.0;
/// The most parts the list shows before it says how many more.
const MAX_ROWS: usize = 12;
/// Room about each view for its sizes.
const PAD: f64 = 16.0;
const TEXT: f64 = 3.5;
const SMALL: f64 = 2.5;

/// A view to place: which, its lines, their bounds, and its cell.
type Placed<'a> = (View, &'a [Line2], DVec2, DVec2, (usize, usize));

/// The most round things one view is given marks and sizes for, and the
/// most places across or up it is given dimensions to: a sheet is to be
/// read, and a plate of two hundred holes dimensioned each way is not.
const MAX_ROUNDS: usize = 40;
const MAX_PLACES: usize = 3;

/// A round edge of a part, in the world: the rim of a hole, of a boss or of
/// a shaft's end.
struct Round {
    centre: DVec3,
    /// Square to its plane, of length one.
    axis: DVec3,
    radius: f64,
}

/// The round edges of the parts: closed runs of sharp or open edges, eight
/// or more, lying in a plane at one distance from their middle. From the
/// meshes, which is what a sheet is made from — so a hole is found whether
/// a feature, a boolean or an import made it.
fn round_edges(parts: &[(&MeshData, DAffine3)]) -> Vec<Round> {
    use std::collections::BTreeMap;
    let mut out: Vec<Round> = Vec::new();
    for (mesh, place) in parts {
        // Points that are one point, one number: a mesh built from pieces
        // repeats them.
        let mut ids: BTreeMap<[i64; 3], u32> = BTreeMap::new();
        let mut points: Vec<DVec3> = Vec::new();
        let welded: Vec<u32> = mesh
            .positions()
            .iter()
            .map(|p| {
                let w = place.transform_point3(p.as_dvec3());
                let key = [w.x, w.y, w.z].map(|c| (c * 1e6).round() as i64);
                *ids.entry(key).or_insert_with(|| {
                    points.push(w);
                    points.len() as u32 - 1
                })
            })
            .collect();
        // Each edge with the faces on it, by their normals.
        let mut edges: BTreeMap<(u32, u32), Vec<DVec3>> = BTreeMap::new();
        for f in 0..mesh.num_faces() {
            let ring: Vec<u32> = mesh
                .face_verts(f)
                .iter()
                .map(|v| welded[*v as usize])
                .collect();
            let mut normal = DVec3::ZERO;
            for i in 0..ring.len() {
                let (a, b) = (
                    points[ring[i] as usize],
                    points[ring[(i + 1) % ring.len()] as usize],
                );
                normal += a.cross(b);
            }
            let normal = normal.normalize_or_zero();
            for i in 0..ring.len() {
                let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
                if a != b {
                    edges.entry((a.min(b), a.max(b))).or_default().push(normal);
                }
            }
        }
        // Sharp, or open: a rim's edges are; a cylinder's own facets, of
        // eight sides or more, are not.
        let mut next: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for ((a, b), normals) in &edges {
            let sharp = match normals.as_slice() {
                [_] => true,
                [n, m] => n.dot(*m) < 0.5,
                _ => false,
            };
            if sharp {
                next.entry(*a).or_default().push(*b);
                next.entry(*b).or_default().push(*a);
            }
        }
        // Closed runs through points with just two such edges.
        let mut used: std::collections::BTreeSet<u32> = Default::default();
        for (&start, ways) in &next {
            if ways.len() != 2 || used.contains(&start) {
                continue;
            }
            let mut ring = vec![start];
            let (mut before, mut at) = (start, ways[0]);
            let closed = loop {
                if at == start {
                    break true;
                }
                let Some(w) = next.get(&at).filter(|w| w.len() == 2) else {
                    break false;
                };
                if ring.len() > points.len() {
                    break false;
                }
                ring.push(at);
                let on = if w[0] == before { w[1] } else { w[0] };
                before = at;
                at = on;
            };
            used.extend(ring.iter().copied());
            if !closed || ring.len() < 8 {
                continue;
            }
            let centre =
                ring.iter().map(|v| points[*v as usize]).sum::<DVec3>() / ring.len() as f64;
            let radii: Vec<f64> = ring
                .iter()
                .map(|v| points[*v as usize].distance(centre))
                .collect();
            let radius = radii.iter().sum::<f64>() / radii.len() as f64;
            let (least, most) = radii.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), r| {
                (lo.min(*r), hi.max(*r))
            });
            if radius < 1e-6 || most - least > radius * 0.02 {
                continue;
            }
            let mut axis = DVec3::ZERO;
            for i in 0..ring.len() {
                let (a, b) = (
                    points[ring[i] as usize] - centre,
                    points[ring[(i + 1) % ring.len()] as usize] - centre,
                );
                axis += a.cross(b);
            }
            let axis = axis.normalize_or_zero();
            let flat = ring
                .iter()
                .all(|v| (points[*v as usize] - centre).dot(axis).abs() <= radius * 0.02);
            if axis == DVec3::ZERO || !flat {
                continue;
            }
            out.push(Round {
                centre,
                axis,
                radius,
            });
        }
    }
    out
}

struct Layers {
    border: LayerId,
    visible: LayerId,
    hidden: LayerId,
    dim: LayerId,
    text: LayerId,
}

/// The sheet of `parts` (meshes placed in the world) as `spec` asks.
pub fn compose(parts: &[(&MeshData, DAffine3)], spec: &SheetSpec) -> Sheet {
    let (mut w, mut h) = spec.paper.size_mm();
    if spec.landscape {
        std::mem::swap(&mut w, &mut h);
    }
    let mut d = Drawing::new();
    let layer = |d: &mut Drawing, name: &str, colour: u8, linetype: &str, weight: f32| {
        let id = d.add_layer(name);
        if let Some(l) = d.layers.iter_mut().find(|l| l.id == id) {
            l.color = ColorRef::Index(colour);
            l.linetype = linetype.into();
            l.lineweight = Lineweight::Mm(weight);
        }
        id
    };
    let layers = Layers {
        border: layer(&mut d, "BORDER", 7, "CONTINUOUS", 0.7),
        visible: layer(&mut d, "VISIBLE", 7, "CONTINUOUS", 0.5),
        hidden: layer(&mut d, "HIDDEN", 8, "DASHED", 0.25),
        dim: layer(&mut d, "DIM", 7, "CONTINUOUS", 0.18),
        text: layer(&mut d, "TEXT", 7, "CONTINUOUS", 0.25),
    };
    let mut pen = Pen { d: &mut d };

    // The frame, and the band at its foot for the title block and the list.
    let (x0, y0, x1, y1) = (MARGIN_LEFT, MARGIN, w - MARGIN, h - MARGIN);
    pen.rect(layers.border, [x0, y0], [x1, y1]);
    let rows = spec.parts.len().min(MAX_ROWS) + usize::from(spec.parts.len() > MAX_ROWS);
    let list_h = if spec.parts.is_empty() {
        0.0
    } else {
        (rows + 1) as f64 * ROW_H
    };

    // The views, each in model metres on its page.
    let mut views: Vec<(View, Vec<Line2>)> = [View::Front, View::Top, View::Right]
        .into_iter()
        .chain(spec.iso.then_some(View::Iso))
        .map(|v| (v, project_drawing(parts, v, 30f64.to_radians())))
        .collect();
    views.retain(|(_, lines)| !lines.is_empty());
    let bounds = |lines: &[Line2]| {
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for l in lines {
            for p in [l.a, l.b] {
                lo = lo.min(DVec2::from(p));
                hi = hi.max(DVec2::from(p));
            }
        }
        (lo, hi)
    };
    // Where each view sits: column and row, row 0 at the top of the page.
    let cell = |v: View| match (spec.projection, v) {
        (Projection::Third, View::Top) => (0, 0),
        (Projection::Third, View::Iso) => (1, 0),
        (Projection::Third, View::Front) => (0, 1),
        (Projection::Third, View::Right) => (1, 1),
        (Projection::First, View::Right) => (0, 0),
        (Projection::First, View::Front) => (1, 0),
        (Projection::First, View::Iso) => (0, 1),
        (Projection::First, View::Top) => (1, 1),
    };
    let mut col_w = [0.0f64; 2];
    let mut row_h = [0.0f64; 2];
    // The three views set the grid and the scale; the isometric view goes
    // in the corner they leave, at their scale if it fits there and at a
    // smaller one if not, so it never shrinks them.
    let iso = views
        .iter()
        .position(|(v, _)| *v == View::Iso)
        .map(|i| views.remove(i));
    let placed: Vec<Placed<'_>> = views
        .iter()
        .map(|(v, lines)| {
            let (lo, hi) = bounds(lines);
            let c = cell(*v);
            col_w[c.0] = col_w[c.0].max(hi.x - lo.x);
            row_h[c.1] = row_h[c.1].max(hi.y - lo.y);
            (*v, lines.as_slice(), lo, hi, c)
        })
        .collect();
    let cols = col_w.iter().filter(|w| **w > 0.0).count().max(1) as f64;
    let rows_n = row_h.iter().filter(|h| **h > 0.0).count().max(1) as f64;
    // The room for views: the frame less the band, in millimetres.
    let band = TITLE_H + list_h + 4.0;
    let (area_w, area_h) = (x1 - x0, y1 - y0 - band);
    let model_w: f64 = col_w.iter().sum::<f64>() * 1e3;
    let model_h: f64 = row_h.iter().sum::<f64>() * 1e3;
    let fit = ((area_w - cols * 2.0 * PAD) / model_w.max(1e-9))
        .min((area_h - rows_n * 2.0 * PAD) / model_h.max(1e-9));
    let scale = spec
        .scale
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or_else(|| standard_scale(fit.max(1e-9)));
    // Each cell's size on paper, and the grid centred in the room.
    let cw = col_w.map(|w| {
        if w > 0.0 {
            w * 1e3 * scale + 2.0 * PAD
        } else {
            0.0
        }
    });
    let ch = row_h.map(|h| {
        if h > 0.0 {
            h * 1e3 * scale + 2.0 * PAD
        } else {
            0.0
        }
    });
    let grid_w = cw[0] + cw[1];
    let grid_h = ch[0] + ch[1];
    let left = x0 + ((area_w - grid_w) * 0.5).max(0.0);
    let top = y1 - ((area_h - grid_h) * 0.5).max(0.0);
    let mut boxes = Vec::new();
    let rounds = round_edges(parts);
    let length = |metres: f64| {
        let v = metres / spec.metres_per_unit;
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    for (v, lines, lo, hi, (c, r)) in placed.iter().copied() {
        let size = (hi - lo) * 1e3 * scale;
        let cell_x = left + if c == 0 { 0.0 } else { cw[0] };
        let cell_top = top - if r == 0 { 0.0 } else { ch[0] };
        // Centred in its cell: a column's views share their width, a row's
        // their height, so the centring keeps them in line.
        let col_size = if c == 0 { cw[0] } else { cw[1] };
        let row_size = if r == 0 { ch[0] } else { ch[1] };
        let origin = DVec2::new(
            cell_x + (col_size - size.x) * 0.5,
            cell_top - row_size + (row_size - size.y) * 0.5,
        );
        let to_paper = |p: [f64; 2]| origin + (DVec2::from(p) - lo) * 1e3 * scale;
        boxes.push((v, origin, origin + size));
        // Hidden first, so the visible lines lie over them.
        for hidden in [true, false] {
            for l in lines.iter().filter(|l| l.hidden == hidden) {
                let layer = if hidden {
                    layers.hidden
                } else {
                    layers.visible
                };
                pen.line(layer, to_paper(l.a), to_paper(l.b));
            }
        }
        // The overall sizes: the front's width and height, the right
        // view's depth.
        let _ = r;
        let (bl, tr) = (origin, origin + size);
        match v {
            View::Front => {
                pen.dim_h(
                    layers.dim,
                    bl,
                    DVec2::new(tr.x, bl.y),
                    -10.0,
                    &length(hi.x - lo.x),
                );
                pen.dim_v(
                    layers.dim,
                    bl,
                    DVec2::new(bl.x, tr.y),
                    -10.0,
                    &length(hi.y - lo.y),
                );
            }
            View::Right => {
                pen.dim_h(
                    layers.dim,
                    bl,
                    DVec2::new(tr.x, bl.y),
                    -10.0,
                    &length(hi.x - lo.x),
                );
            }
            _ => {}
        }
        // What is round and seen end on in this view — a hole, a boss, a
        // shaft: its centre marked, its diameter written once for all of
        // that size, and where it is from the view's lower left corner.
        let mut seen: Vec<(DVec2, f64)> = Vec::new();
        for round in &rounds {
            // End on: its axis is the way this view looks.
            let along = DVec2::from(v.project(round.axis));
            if along.length() > 0.02 {
                continue;
            }
            let at = DVec2::from(v.project(round.centre));
            if !seen
                .iter()
                .any(|(c, r)| c.distance(at) < 1e-5 && (r - round.radius).abs() < 1e-5)
            {
                seen.push((at, round.radius));
            }
        }
        seen.truncate(MAX_ROUNDS);
        let mut sizes: Vec<f64> = Vec::new();
        for (at, radius) in &seen {
            let centre = to_paper((*at).into());
            let r = radius * 1e3 * scale;
            let reach = r + 2.0;
            pen.line(
                layers.dim,
                centre - DVec2::X * reach,
                centre + DVec2::X * reach,
            );
            pen.line(
                layers.dim,
                centre - DVec2::Y * reach,
                centre + DVec2::Y * reach,
            );
            if sizes.iter().any(|s| (s - radius).abs() < 1e-5) {
                continue;
            }
            sizes.push(*radius);
            let count = seen
                .iter()
                .filter(|(_, other)| (other - radius).abs() < 1e-5)
                .count();
            let diameter = format!("Ø{}", length(radius * 2.0));
            let note = if count > 1 {
                format!("{count}x {diameter}")
            } else {
                diameter
            };
            // A leader up and to the right from the rim, its note at the end.
            let out = DVec2::new(
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            );
            let (from, to) = (centre + out * r, centre + out * (r + 5.0));
            pen.line(layers.dim, from, to);
            pen.arrow(layers.dim, from, -out);
            pen.text(
                layers.dim,
                to + DVec2::new(1.0, 0.0),
                SMALL,
                &note,
                TextAlign::Left,
            );
        }
        // Where they are: each distinct place across, and each up, from the
        // corner — the nearest innermost. Not for what fills the view, a
        // shaft seen end on, which is where the view is.
        let fills = |at: DVec2, radius: f64| {
            let mid = (lo + hi) * 0.5;
            at.distance(mid) < 1e-5 && radius * 2.0 >= (hi.x - lo.x).min(hi.y - lo.y) * 0.98
        };
        let mut across: Vec<f64> = Vec::new();
        let mut up: Vec<f64> = Vec::new();
        for (at, radius) in seen.iter().filter(|(at, r)| !fills(*at, *r)) {
            let _ = radius;
            if !across.iter().any(|x| (x - at.x).abs() < 1e-5) {
                across.push(at.x);
            }
            if !up.iter().any(|y| (y - at.y).abs() < 1e-5) {
                up.push(at.y);
            }
        }
        across.sort_by(f64::total_cmp);
        up.sort_by(f64::total_cmp);
        for (i, x) in across.iter().take(MAX_PLACES).enumerate() {
            let to = to_paper([*x, hi.y]);
            pen.dim_h(
                layers.dim,
                DVec2::new(bl.x, tr.y),
                DVec2::new(to.x, tr.y),
                4.0 + 4.5 * i as f64,
                &length(x - lo.x),
            );
        }
        for (i, y) in up.iter().take(MAX_PLACES).enumerate() {
            let to = to_paper([hi.x, *y]);
            pen.dim_v(
                layers.dim,
                DVec2::new(tr.x, bl.y),
                DVec2::new(tr.x, to.y),
                4.0 + 4.5 * i as f64,
                &length(y - lo.y),
            );
        }
    }
    if let Some((_, lines)) = &iso {
        let (lo, hi) = bounds(lines);
        let model = (hi - lo) * 1e3;
        // The corner the grid leaves: beside the top view, toward the
        // frame's edge.
        let (rx, ry) = match spec.projection {
            Projection::Third => ([left + cw[0], x1], [top - ch[0], y1]),
            Projection::First => ([x0, left + cw[0]], [y0 + band, top - ch[0]]),
        };
        let room = DVec2::new(rx[1] - rx[0], ry[1] - ry[0]) - DVec2::splat(2.0 * PAD);
        let fits = (room.x / model.x.max(1e-9)).min(room.y / model.y.max(1e-9));
        let iso_scale = if fits >= scale {
            scale
        } else {
            standard_scale(fits.max(1e-9))
        };
        let size = model * iso_scale;
        let origin = DVec2::new(
            (rx[0] + rx[1] - size.x) * 0.5,
            (ry[0] + ry[1] - size.y) * 0.5,
        );
        boxes.push((View::Iso, origin, origin + size));
        // Its visible lines only, as an isometric view is drawn.
        for l in lines.iter().filter(|l| !l.hidden) {
            pen.line(
                layers.visible,
                origin + (DVec2::from(l.a) - lo) * 1e3 * iso_scale,
                origin + (DVec2::from(l.b) - lo) * 1e3 * iso_scale,
            );
        }
        if iso_scale != scale {
            pen.text(
                layers.text,
                DVec2::new(origin.x + size.x * 0.5, origin.y - 6.0),
                TEXT,
                &format!("({})", scale_label(iso_scale)),
                TextAlign::Center,
            );
        }
    }

    // The title block, at the frame's bottom right.
    let tb = [x1 - TITLE_W, y0];
    pen.rect(layers.border, tb, [x1, y0 + TITLE_H]);
    let l = &spec.labels;
    let split = 110.0;
    pen.line(
        layers.border,
        DVec2::new(tb[0], y0 + 12.0),
        DVec2::new(x1, y0 + 12.0),
    );
    pen.line(
        layers.border,
        DVec2::new(tb[0] + split, y0 + 12.0),
        DVec2::new(tb[0] + split, y0 + TITLE_H),
    );
    pen.line(
        layers.border,
        DVec2::new(tb[0] + split, y0 + 24.0),
        DVec2::new(x1, y0 + 24.0),
    );
    for x in [60.0, 120.0] {
        pen.line(
            layers.border,
            DVec2::new(tb[0] + x, y0),
            DVec2::new(tb[0] + x, y0 + 12.0),
        );
    }
    let field = |pen: &mut Pen, at: [f64; 2], label: &str, value: &str, big: f64| {
        pen.text(
            layers.text,
            DVec2::new(at[0] + 1.5, at[1] - 3.5),
            SMALL,
            label,
            TextAlign::Left,
        );
        if !value.is_empty() {
            pen.text(
                layers.text,
                DVec2::new(at[0] + 3.0, at[1] - 4.5 - big - 1.0),
                big,
                value,
                TextAlign::Left,
            );
        }
    };
    field(&mut pen, [tb[0], y0 + TITLE_H], &l.title, &spec.title, 5.0);
    let projection = match spec.projection {
        Projection::First => &l.first_angle,
        Projection::Third => &l.third_angle,
    };
    field(
        &mut pen,
        [tb[0] + split, y0 + TITLE_H],
        &l.scale,
        &format!("{}   ({} {})", scale_label(scale), l.unit, spec.unit),
        TEXT,
    );
    field(
        &mut pen,
        [tb[0] + split, y0 + 24.0],
        &l.projection,
        projection,
        TEXT,
    );
    pen.projection_symbol(
        layers.text,
        DVec2::new(x1 - 14.0, y0 + 17.5),
        spec.projection,
    );
    field(&mut pen, [tb[0], y0 + 12.0], &l.number, &spec.number, TEXT);
    field(
        &mut pen,
        [tb[0] + 60.0, y0 + 12.0],
        &l.material,
        &spec.material,
        TEXT,
    );
    let drawn = match (spec.author.is_empty(), spec.date.is_empty()) {
        (false, false) => format!("{}  {}", spec.author, spec.date),
        _ => format!("{}{}", spec.author, spec.date),
    };
    field(&mut pen, [tb[0] + 120.0, y0 + 12.0], &l.drawn, &drawn, TEXT);

    // The parts list, above the title block, its heading at the foot as
    // ISO 7573 has it.
    if !spec.parts.is_empty() {
        let base = y0 + TITLE_H;
        let cols = [0.0, 15.0, 120.0, 140.0, TITLE_W];
        let heads = [&l.item, &l.part, &l.quantity, &l.material];
        let mut shown: Vec<[String; 4]> = spec
            .parts
            .iter()
            .take(MAX_ROWS)
            .enumerate()
            .map(|(i, p)| {
                [
                    (i + 1).to_string(),
                    p.name.clone(),
                    p.count.to_string(),
                    p.material.clone(),
                ]
            })
            .collect();
        if spec.parts.len() > MAX_ROWS {
            let n = (spec.parts.len() - MAX_ROWS).to_string();
            shown.push([
                String::new(),
                l.more.replace("{n}", &n),
                String::new(),
                String::new(),
            ]);
        }
        let top_y = base + list_h;
        pen.rect(layers.border, [tb[0], base], [x1, top_y]);
        for (k, row) in std::iter::once(heads.map(|s| s.to_string()))
            .chain(shown.into_iter().rev())
            .enumerate()
        {
            let y = base + k as f64 * ROW_H;
            if k > 0 {
                pen.line(layers.border, DVec2::new(tb[0], y), DVec2::new(x1, y));
            }
            for (i, value) in row.iter().enumerate() {
                pen.text(
                    layers.text,
                    DVec2::new(tb[0] + cols[i] + 1.5, y + (ROW_H - SMALL) * 0.5),
                    SMALL,
                    value,
                    TextAlign::Left,
                );
            }
        }
        for x in &cols[1..4] {
            pen.line(
                layers.border,
                DVec2::new(tb[0] + x, base),
                DVec2::new(tb[0] + x, top_y),
            );
        }
    }

    d.current_layer = LayerId(0);
    Sheet {
        drawing: d,
        views: boxes,
        width_mm: w,
        height_mm: h,
        scale,
    }
}

/// Draws on a sheet in millimetres of paper; the drawing holds metres.
struct Pen<'a> {
    d: &'a mut Drawing,
}

const MM: f64 = 1e-3;

impl Pen<'_> {
    fn line(&mut self, layer: LayerId, a: DVec2, b: DVec2) {
        if (a - b).length() < 1e-9 {
            return;
        }
        self.d.current_layer = layer;
        self.d.add_curve(Curve2D::Line {
            a: a * MM,
            b: b * MM,
        });
    }

    fn rect(&mut self, layer: LayerId, a: [f64; 2], b: [f64; 2]) {
        let (a, b) = (DVec2::from(a), DVec2::from(b));
        let c = [a, DVec2::new(b.x, a.y), b, DVec2::new(a.x, b.y)];
        for i in 0..4 {
            self.line(layer, c[i], c[(i + 1) % 4]);
        }
    }

    fn text(&mut self, layer: LayerId, at: DVec2, height: f64, content: &str, align: TextAlign) {
        self.text_turned(layer, at, height, content, align, 0.0);
    }

    fn text_turned(
        &mut self,
        layer: LayerId,
        at: DVec2,
        height: f64,
        content: &str,
        align: TextAlign,
        rotation: f64,
    ) {
        if content.trim().is_empty() {
            return;
        }
        self.d.current_layer = layer;
        self.d.add(EntityKind::Text(Text {
            at: at * MM,
            height: height * MM,
            rotation,
            content: content.to_string(),
            align,
            font: String::new(),
        }));
    }

    /// An open arrowhead at `tip`, pointing along `dir`.
    fn arrow(&mut self, layer: LayerId, tip: DVec2, dir: DVec2) {
        let back = -dir * 3.0;
        let side = dir.perp() * 0.9;
        self.line(layer, tip, tip + back + side);
        self.line(layer, tip, tip + back - side);
    }

    /// A horizontal size from `a` to `b`, its line `offset` mm below them
    /// (negative) or above, its number over the middle.
    fn dim_h(&mut self, layer: LayerId, a: DVec2, b: DVec2, offset: f64, text: &str) {
        let y = a.y.min(b.y) + offset;
        let (pa, pb) = (DVec2::new(a.x, y), DVec2::new(b.x, y));
        let out = offset.signum() * 2.0;
        self.line(
            layer,
            a + DVec2::new(0.0, offset.signum() * 1.0),
            pa + DVec2::new(0.0, out),
        );
        self.line(
            layer,
            b + DVec2::new(0.0, offset.signum() * 1.0),
            pb + DVec2::new(0.0, out),
        );
        self.line(layer, pa, pb);
        self.arrow(layer, pa, DVec2::NEG_X);
        self.arrow(layer, pb, DVec2::X);
        self.text(
            layer,
            DVec2::new((a.x + b.x) * 0.5, y + 1.0),
            TEXT,
            text,
            TextAlign::Center,
        );
    }

    /// A vertical size from `a` to `b`, its line `offset` mm to the left
    /// (negative) or right, its number along it, read from the right.
    fn dim_v(&mut self, layer: LayerId, a: DVec2, b: DVec2, offset: f64, text: &str) {
        let x = a.x.min(b.x) + offset;
        let (pa, pb) = (DVec2::new(x, a.y), DVec2::new(x, b.y));
        let out = offset.signum() * 2.0;
        self.line(
            layer,
            a + DVec2::new(offset.signum() * 1.0, 0.0),
            pa + DVec2::new(out, 0.0),
        );
        self.line(
            layer,
            b + DVec2::new(offset.signum() * 1.0, 0.0),
            pb + DVec2::new(out, 0.0),
        );
        self.line(layer, pa, pb);
        self.arrow(layer, pa, DVec2::NEG_Y);
        self.arrow(layer, pb, DVec2::Y);
        self.text_turned(
            layer,
            DVec2::new(x - 1.0, (a.y + b.y) * 0.5),
            TEXT,
            text,
            TextAlign::Center,
            std::f64::consts::FRAC_PI_2,
        );
    }

    /// ISO 128-30's projection symbol: a truncated cone seen from its side
    /// and from its end, the side view where the method puts it.
    fn projection_symbol(&mut self, layer: LayerId, centre: DVec2, projection: Projection) {
        let (big, small, long) = (4.0, 2.0, 7.0);
        let side_x = if projection == Projection::Third {
            -6.0
        } else {
            6.0
        };
        let end_x = -side_x;
        let s = centre + DVec2::new(side_x, 0.0);
        // The cone from its side, narrow end to the left in either method;
        // only where the end view goes tells them apart.
        let wide = long * 0.5;
        let a = s + DVec2::new(wide, big);
        let b = s + DVec2::new(wide, -big);
        let c = s + DVec2::new(-wide, -small);
        let e = s + DVec2::new(-wide, small);
        for (p, q) in [(a, b), (b, c), (c, e), (e, a)] {
            self.line(layer, p, q);
        }
        self.d.current_layer = layer;
        for r in [big, small] {
            self.d.add_curve(Curve2D::Circle {
                c: (centre + DVec2::new(end_x, 0.0)) * MM,
                r: r * MM,
            });
        }
    }
}

/// A sheet metal plate's blank as a drawing to cut it by: the outline and
/// the holes on a CUT layer, each bend's line on a BEND layer, dashed, and
/// beside it which way, how far and round what radius it is bent — in the
/// plate's own plane, a metre a metre, its sizes written in `unit`.
pub fn flat_pattern(p: &kreate_mesh::mech::Plate, metres_per_unit: f64, unit: &str) -> Drawing {
    let (cut, bends) = kreate_mesh::mech::plate_flat_pattern(p);
    let mut d = Drawing::new();
    let layer = |d: &mut Drawing, name: &str, colour: u8, linetype: &str| {
        let id = d.add_layer(name);
        if let Some(l) = d.layers.iter_mut().find(|l| l.id == id) {
            l.color = ColorRef::Index(colour);
            l.linetype = linetype.into();
        }
        id
    };
    let cut_layer = layer(&mut d, "CUT", 7, "CONTINUOUS");
    let bend_layer = layer(&mut d, "BEND", 1, "DASHED");
    let note_layer = layer(&mut d, "NOTE", 3, "CONTINUOUS");
    let v = |q: kreate_math::Vec2| DVec2::new(f64::from(q.x), f64::from(q.y));
    d.current_layer = cut_layer;
    for [a, b] in cut {
        d.add_curve(Curve2D::Line { a: v(a), b: v(b) });
    }
    let size = |m: f64| {
        let s = format!("{:.2}", m / metres_per_unit);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    let height = (f64::from(p.thickness) * 2.0).max(0.002);
    for b in bends {
        let (a, e) = (v(b.a), v(b.b));
        d.current_layer = bend_layer;
        d.add_curve(Curve2D::Line { a, b: e });
        let along = (e - a).normalize_or(DVec2::X);
        let turn = along.y.atan2(along.x);
        // Readable from the bottom or the right.
        let turn = if turn.abs() > std::f64::consts::FRAC_PI_2 + 1e-9 {
            turn + std::f64::consts::PI
        } else {
            turn
        };
        d.current_layer = note_layer;
        d.add(EntityKind::Text(Text {
            at: (a + e) * 0.5 + along.perp() * height * 0.6,
            height,
            rotation: turn,
            content: format!(
                "UP {:.0}° R{} {}",
                f64::from(b.angle).to_degrees(),
                size(f64::from(b.radius)),
                unit
            ),
            align: TextAlign::Center,
            font: String::new(),
        }));
    }
    d.current_layer = LayerId(0);
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(x: f64, y: f64, z: f64) -> (MeshData, DAffine3) {
        (
            kreate_mesh::primitives::cube(1.0),
            DAffine3::from_scale(kreate_math::DVec3::new(x, y, z)),
        )
    }

    #[test]
    fn a_scale_is_the_largest_standard_one_that_fits() {
        assert_eq!(standard_scale(1.3), 1.0);
        assert_eq!(standard_scale(0.37), 0.2);
        assert_eq!(standard_scale(7.0), 5.0);
        assert_eq!(scale_label(0.5), "1:2");
        assert_eq!(scale_label(5.0), "5:1");
        assert_eq!(scale_label(1.0), "1:1");
    }

    fn texts(sheet: &Sheet) -> Vec<String> {
        sheet
            .drawing
            .entities
            .iter()
            .filter_map(|e| match &e.kind {
                EntityKind::Text(t) => Some(t.content.clone()),
                _ => None,
            })
            .collect()
    }

    /// A 120 × 60 × 40 mm block on A3: drawn at a standard scale, every
    /// line inside the frame, its sizes written as the model's, and the
    /// title block filled.
    #[test]
    fn a_block_is_drawn_inside_the_frame_with_its_own_sizes() {
        let (mesh, at) = block(0.12, 0.06, 0.04);
        let spec = SheetSpec {
            title: "브래킷".into(),
            number: "KR-001".into(),
            parts: vec![PartRow {
                name: "Block".into(),
                count: 1,
                material: "AL6061".into(),
            }],
            ..Default::default()
        };
        let sheet = compose(&[(&mesh, at)], &spec);
        assert_eq!((sheet.width_mm, sheet.height_mm), (420.0, 297.0));
        assert!(SCALES.contains(&sheet.scale));
        let (lo, hi) = sheet.drawing.extents().expect("something drawn");
        assert!(lo.x >= MARGIN_LEFT * MM - 1e-9 && lo.y >= MARGIN * MM - 1e-9);
        assert!(hi.x <= (420.0 - MARGIN) * MM + 1e-9 && hi.y <= (297.0 - MARGIN) * MM + 1e-9);
        let t = texts(&sheet);
        for size in ["120", "60", "40"] {
            assert!(t.iter().any(|s| s == size), "{size} not written: {t:?}");
        }
        assert!(t.iter().any(|s| s == "브래킷"));
        assert!(t.iter().any(|s| s.starts_with(&scale_label(sheet.scale))));
        assert!(t.iter().any(|s| s == "AL6061"));
    }

    /// A plate with two posts standing on it: the top view marks each
    /// post's centre, writes their diameter once for the two, and says
    /// where they are from the corner; the front view, which sees them from
    /// the side, writes no diameter.
    #[test]
    fn round_things_are_marked_sized_and_placed_where_they_are_seen_end_on() {
        let plate = block(0.12, 0.06, 0.01);
        let post = kreate_mesh::primitives::cylinder(0.005, 0.02, 32);
        let stand = |x: f64, y: f64| DAffine3::from_translation(DVec3::new(x, y, 0.015));
        let parts = [
            (&plate.0, plate.1),
            (&post, stand(-0.03, 0.0)),
            (&post, stand(0.03, 0.01)),
        ];
        let rounds = round_edges(&parts);
        // Each post's two rims; the plate, all corners, has none.
        assert_eq!(rounds.len(), 4, "{}", rounds.len());
        assert!(rounds
            .iter()
            .all(|r| (r.radius - 0.005).abs() < 1e-5 && r.axis.cross(DVec3::Z).length() < 1e-6));
        let sheet = compose(&parts, &SheetSpec::default());
        let t = texts(&sheet);
        assert_eq!(
            t.iter().filter(|s| s.contains('Ø')).collect::<Vec<_>>(),
            ["2x Ø10"],
            "{t:?}"
        );
        // From the plate's left edge at x = -60: 30 and 90 across; from its
        // near edge at y = -30: 30 and 40 up.
        for place in ["30", "90", "40"] {
            assert!(t.iter().any(|s| s == place), "{place} not written: {t:?}");
        }
        let (lo, hi) = sheet.drawing.extents().expect("something drawn");
        assert!(lo.x >= MARGIN_LEFT * MM - 1e-9 && lo.y >= MARGIN * MM - 1e-9);
        assert!(hi.x <= (420.0 - MARGIN) * MM + 1e-9 && hi.y <= (297.0 - MARGIN) * MM + 1e-9);
        // To look at: `KREATE_SHEET_OUT=sheet.pdf cargo test -p kreate-io round_things`.
        if let Some(path) = std::env::var_os("KREATE_SHEET_OUT") {
            let pdf = crate::pdf::write(&sheet.drawing, sheet.width_mm * MM, sheet.height_mm * MM);
            std::fs::write(&path, pdf).expect("written");
            // And its lines and words as plain text, for a script to draw.
            let mut plain = String::new();
            for e in &sheet.drawing.entities {
                match &e.kind {
                    EntityKind::Curve(Curve2D::Line { a, b }) => {
                        plain += &format!(
                            "L {} {} {} {}
",
                            a.x, a.y, b.x, b.y
                        );
                    }
                    EntityKind::Text(t) => {
                        plain += &format!(
                            "T {} {} {} {} {}
",
                            t.at.x, t.at.y, t.height, t.rotation, t.content
                        );
                    }
                    _ => {}
                }
            }
            let mut txt = std::path::PathBuf::from(path);
            txt.set_extension("txt");
            std::fs::write(txt, plain).expect("written");
        }
    }

    /// In third angle the top view is above the front and the right view
    /// to its right; in first angle, below and to its left — each in line
    /// with the front, and none overlapping another.
    #[test]
    fn the_views_go_where_the_projection_puts_them() {
        let (mesh, at) = block(0.1, 0.05, 0.03);
        for projection in [Projection::Third, Projection::First] {
            let spec = SheetSpec {
                projection,
                ..Default::default()
            };
            let sheet = compose(&[(&mesh, at)], &spec);
            let get = |v: View| {
                sheet
                    .views
                    .iter()
                    .find(|(w, _, _)| *w == v)
                    .map(|(_, lo, hi)| (*lo, *hi))
                    .unwrap_or_else(|| panic!("no {v:?} view"))
            };
            let (front, top, right) = (get(View::Front), get(View::Top), get(View::Right));
            let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
            // In line: the top view over the front's width, the right view
            // across the front's height.
            assert!(near(top.0.x, front.0.x) && near(top.1.x, front.1.x));
            assert!(near(right.0.y, front.0.y) && near(right.1.y, front.1.y));
            match projection {
                Projection::Third => {
                    assert!(top.0.y > front.1.y, "the top view above the front");
                    assert!(right.0.x > front.1.x, "the right view to its right");
                }
                Projection::First => {
                    assert!(top.1.y < front.0.y, "the top view below the front");
                    assert!(right.1.x < front.0.x, "the right view to its left");
                }
            }
            let iso = get(View::Iso);
            for other in [front, top, right] {
                let apart = iso.1.x < other.0.x
                    || iso.0.x > other.1.x
                    || iso.1.y < other.0.y
                    || iso.0.y > other.1.y;
                assert!(apart, "the isometric view overlaps another");
            }
        }
    }
    /// A square plate with two flanges: its blank has the plate's two bare
    /// edges, each flange unrolled, two bend lines and a note on each.
    #[test]
    fn a_flat_pattern_has_its_cut_its_bends_and_their_notes() {
        use kreate_math::Vec2;
        let p = kreate_mesh::mech::Plate {
            outline: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(0.1, 0.0),
                Vec2::new(0.1, 0.1),
                Vec2::new(0.0, 0.1),
            ],
            holes: Vec::new(),
            thickness: 0.002,
            radius: 0.002,
            k_factor: 0.5,
            flanges: [0, 2]
                .map(|edge| kreate_mesh::mech::Flange {
                    edge,
                    angle: std::f32::consts::FRAC_PI_2,
                    height: 0.02,
                })
                .to_vec(),
            flat: true,
        };
        let d = flat_pattern(&p, 1e-3, "mm");
        let on = |name: &str| {
            let id = d.layers.iter().find(|l| l.name == name).unwrap().id;
            d.entities.iter().filter(move |e| e.layer == id)
        };
        // Two bare edges and three lines for each flange.
        assert_eq!(on("CUT").count(), 2 + 3 + 3);
        assert_eq!(on("BEND").count(), 2);
        let notes: Vec<String> = on("NOTE")
            .filter_map(|e| match &e.kind {
                EntityKind::Text(t) => Some(t.content.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(notes, ["UP 90° R2 mm", "UP 90° R2 mm"]);
        // The blank is the plate and each flange's allowance and height
        // long: 100 + 2 × (π/2 × 3 + 20) mm.
        let (lo, hi) = d.extents().unwrap();
        let want = 0.1 + 2.0 * (std::f64::consts::FRAC_PI_2 * 0.003 + 0.02);
        assert!(((hi.y - lo.y) - want).abs() < 1e-6, "{}", hi.y - lo.y);
    }
}
