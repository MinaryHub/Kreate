//! Building elements: walls, doors, windows, stairs, columns, railings,
//! slabs and roofs, as solids made of plain boxes and prisms.
//!
//! Each is one mesh of several closed pieces — a door's frame, leaf and
//! handles — with each face's material slot set, so a door is wood with a
//! metal handle and a window's panes are glass, without a step after. Sizes
//! are in metres; every element stands on its origin (z = 0 is the floor it
//! stands on), lies along X, and faces −Y, the way it is first seen.

use std::sync::Arc;

use crate::attribute::{names, AttrData, Domain};
use crate::mesh_data::MeshData;
use kreate_math::{Vec2, Vec3};

/// A mesh put together from closed pieces, each face with its slot.
#[derive(Default)]
pub struct Pieces {
    positions: Vec<Vec3>,
    faces: Vec<Vec<u32>>,
    slots: Vec<i32>,
}

impl Pieces {
    /// A box between two corners.
    pub fn cuboid(&mut self, a: Vec3, b: Vec3, slot: i32) {
        let (lo, hi) = (a.min(b), a.max(b));
        if (hi - lo).min_element() <= 0.0 {
            return;
        }
        let s = self.positions.len() as u32;
        for z in [lo.z, hi.z] {
            self.positions.push(Vec3::new(lo.x, lo.y, z));
            self.positions.push(Vec3::new(hi.x, lo.y, z));
            self.positions.push(Vec3::new(hi.x, hi.y, z));
            self.positions.push(Vec3::new(lo.x, hi.y, z));
        }
        for f in [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ] {
            self.faces.push(f.iter().map(|i| s + i).collect());
            self.slots.push(slot);
        }
    }

    /// A prism: the polygon `xz` (in the XZ plane) run along Y from `y0` to
    /// `y1`. Either winding is taken.
    pub fn prism_y(&mut self, xz: &[Vec2], y0: f32, y1: f32, slot: i32) {
        if xz.len() < 3 || y1 <= y0 {
            return;
        }
        let mut p = xz.to_vec();
        let area: f32 = (0..p.len())
            .map(|i| {
                let (a, b) = (p[i], p[(i + 1) % p.len()]);
                a.x * b.y - b.x * a.y
            })
            .sum();
        // Anticlockwise seen from −Y (x right, z up): the front face, at
        // y0, then faces out of it.
        if area < 0.0 {
            p.reverse();
        }
        let n = p.len() as u32;
        let s = self.positions.len() as u32;
        self.positions
            .extend(p.iter().map(|q| Vec3::new(q.x, y0, q.y)));
        self.positions
            .extend(p.iter().map(|q| Vec3::new(q.x, y1, q.y)));
        self.faces.push((0..n).map(|i| s + i).collect());
        self.slots.push(slot);
        self.faces.push((0..n).rev().map(|i| s + n + i).collect());
        self.slots.push(slot);
        for i in 0..n {
            let j = (i + 1) % n;
            self.faces.push(vec![s + i, s + n + i, s + n + j, s + j]);
            self.slots.push(slot);
        }
    }

    /// Another mesh's faces, moved by `offset`.
    pub fn mesh(&mut self, m: &MeshData, offset: Vec3, slot: i32) {
        let s = self.positions.len() as u32;
        self.positions
            .extend(m.positions().iter().map(|p| *p + offset));
        for f in m.faces() {
            self.faces.push(f.iter().map(|v| s + v).collect());
            self.slots.push(slot);
        }
    }

    pub fn finish(self) -> MeshData {
        let mut m = MeshData::from_faces(self.positions, &self.faces);
        m.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(Arc::new(self.slots)),
        );
        m
    }
}

/// `x` kept between `lo` and `hi`, never failing: parameters come from
/// files and scripts, and a bound worked out from them — a handle's height
/// under a door 5 cm tall — can fall below the other. `f32::clamp` stops the
/// program then; this gives `lo`, as it does for a value that is not a
/// number.
fn within(x: f32, lo: f32, hi: f32) -> f32 {
    if hi >= lo {
        x.max(lo).min(hi)
    } else {
        lo
    }
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// A wall `length` long, `height` high and `thickness` thick, centred on
/// its origin along X.
pub fn wall(length: f32, height: f32, thickness: f32) -> MeshData {
    let mut p = Pieces::default();
    let (l, t) = (length * 0.5, thickness * 0.5);
    p.cuboid(v(-l, -t, 0.0), v(l, t, height), 0);
    p.finish()
}

/// Slots of a door: 0 the leaf, 1 the frame, 2 the handles.
pub const DOOR_SLOTS: [&str; 3] = ["Door Wood", "Door Frame", "Door Metal"];

/// A door, as its parameters say it.
#[derive(Clone, Copy, Debug)]
pub struct Door {
    /// The opening.
    pub width: f32,
    pub height: f32,
    /// The frame's depth (the wall's thickness) and width round the opening.
    pub depth: f32,
    pub frame: f32,
    /// The leaf's own thickness.
    pub leaf: f32,
    /// Raised panels on each face, stacked up the leaf; zero for a flush door.
    pub panels: u32,
    /// The handles' height above the floor.
    pub handle_height: f32,
    /// The handles on the right (seen from the front), or the left.
    pub handle_right: bool,
}

impl Default for Door {
    fn default() -> Self {
        Self {
            width: 0.9,
            height: 2.1,
            depth: 0.2,
            frame: 0.06,
            leaf: 0.04,
            panels: 2,
            handle_height: 1.0,
            handle_right: true,
        }
    }
}

/// A door in its frame: a leaf with raised panels on both faces and a lever
/// handle each side.
pub fn door(d: &Door) -> MeshData {
    let mut p = Pieces::default();
    let (w, dd, f, height) = (d.width * 0.5, d.depth * 0.5, d.frame, d.height);
    // The frame: two jambs and a head.
    p.cuboid(v(-w - f, -dd, 0.0), v(-w, dd, height + f), 1);
    p.cuboid(v(w, -dd, 0.0), v(w + f, dd, height + f), 1);
    p.cuboid(v(-w, -dd, height), v(w, dd, height + f), 1);
    // The leaf, a few millimetres clear of the frame and the floor.
    let gap = 0.003;
    let l = within(d.leaf, 0.005, d.depth) * 0.5;
    p.cuboid(v(-w + gap, -l, gap), v(w - gap, l, height - gap), 0);
    // Raised panels, both faces, stacked with rails between them; two are
    // a short one below and a tall one above, as a panelled door has.
    let n = d.panels.min(8);
    if n > 0 {
        let inset = (d.width * 0.14).max(0.05).min(w * 0.8);
        let raise = 0.006;
        let (bottom, top) = (height * 0.08, height * 0.92);
        let rail = height * 0.03;
        let bands: Vec<(f32, f32)> = if n == 2 {
            let split = height * 0.38;
            vec![(bottom, split - rail), (split + rail, top)]
        } else {
            let step = (top - bottom) / n as f32;
            (0..n)
                .map(|i| {
                    let z0 = bottom + step * i as f32 + if i > 0 { rail } else { 0.0 };
                    let z1 = bottom + step * (i + 1) as f32 - if i + 1 < n { rail } else { 0.0 };
                    (z0, z1)
                })
                .collect()
        };
        for (z0, z1) in bands {
            for (y0, y1) in [(-l - raise, -l), (l, l + raise)] {
                p.cuboid(v(-w + inset, y0, z0), v(w - inset, y1, z1), 0);
            }
        }
    }
    // A lever each side, near the edge away from the hinges, pointing in.
    let side = if d.handle_right { 1.0 } else { -1.0 };
    let hx = side * (w - 0.07).max(0.0);
    let hz = within(d.handle_height, 0.05, height - 0.05);
    for s in [-1.0f32, 1.0] {
        let rose_y = s * (l + 0.012);
        p.cuboid(
            v(hx - 0.025, rose_y.min(s * l), hz - 0.025),
            v(hx + 0.025, rose_y.max(s * l), hz + 0.025),
            2,
        );
        let lever_y = s * (l + 0.045);
        p.cuboid(
            v(hx - 0.012, rose_y.min(lever_y), hz - 0.01),
            v(hx + 0.012, rose_y.max(lever_y), hz + 0.01),
            2,
        );
        p.cuboid(
            v(hx - side * 0.13, lever_y - 0.01, hz - 0.011),
            v(hx + side * 0.012, lever_y + 0.01, hz + 0.011),
            2,
        );
    }
    p.finish()
}

/// Slots of a window: 0 the frame, 1 the glass.
pub const WINDOW_SLOTS: [&str; 2] = ["Window Frame", "Glass"];

/// A window, as its parameters say it.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub width: f32,
    pub height: f32,
    /// The frame's depth and its width.
    pub depth: f32,
    pub frame: f32,
    /// Its bottom's height above the floor.
    pub sill_height: f32,
    /// Panes across and up.
    pub columns: u32,
    pub rows: u32,
    /// The mullions' and transoms' width.
    pub bar: f32,
    /// The glass's thickness.
    pub glass: f32,
    /// How far the sill stands out in front; zero for none.
    pub sill: f32,
}

impl Default for Window {
    fn default() -> Self {
        Self {
            width: 1.2,
            height: 1.4,
            depth: 0.15,
            frame: 0.06,
            sill_height: 0.9,
            columns: 2,
            rows: 1,
            bar: 0.036,
            glass: 0.008,
            sill: 0.04,
        }
    }
}

/// A window: a frame divided into panes of glass, on a sill.
pub fn window(s: &Window) -> MeshData {
    let mut p = Pieces::default();
    let (w, d, f) = (s.width * 0.5, s.depth * 0.5, s.frame);
    let (z0, z1) = (s.sill_height, s.sill_height + s.height);
    // The outer frame.
    p.cuboid(v(-w, -d, z0), v(-w + f, d, z1), 0);
    p.cuboid(v(w - f, -d, z0), v(w, d, z1), 0);
    p.cuboid(v(-w + f, -d, z0), v(w - f, d, z0 + f), 0);
    p.cuboid(v(-w + f, -d, z1 - f), v(w - f, d, z1), 0);
    // Mullions and transoms, shallower than the frame.
    let (cols, rows) = (s.columns.clamp(1, 12), s.rows.clamp(1, 12));
    let bar = s.bar.max(0.001);
    let bd = d * 0.6;
    let (ix0, ix1, iz0, iz1) = (-w + f, w - f, z0 + f, z1 - f);
    let col_w = (ix1 - ix0) / cols as f32;
    let row_h = (iz1 - iz0) / rows as f32;
    for c in 1..cols {
        let x = ix0 + col_w * c as f32;
        p.cuboid(v(x - bar * 0.5, -bd, iz0), v(x + bar * 0.5, bd, iz1), 0);
    }
    for r in 1..rows {
        let z = iz0 + row_h * r as f32;
        p.cuboid(v(ix0, -bd, z - bar * 0.5), v(ix1, bd, z + bar * 0.5), 0);
    }
    // The panes: one sheet each, between the bars.
    let g = within(s.glass * 0.5, 0.0005, bd);
    for c in 0..cols {
        for r in 0..rows {
            let x0 = ix0 + col_w * c as f32 + if c > 0 { bar * 0.5 } else { 0.0 };
            let x1 = ix0 + col_w * (c + 1) as f32 - if c + 1 < cols { bar * 0.5 } else { 0.0 };
            let zz0 = iz0 + row_h * r as f32 + if r > 0 { bar * 0.5 } else { 0.0 };
            let zz1 = iz0 + row_h * (r + 1) as f32 - if r + 1 < rows { bar * 0.5 } else { 0.0 };
            p.cuboid(v(x0, -g, zz0), v(x1, g, zz1), 1);
        }
    }
    // The sill, standing proud of the wall in front.
    if s.sill > 0.0 && s.sill_height > 0.03 {
        p.cuboid(v(-w - 0.03, -d - s.sill, z0 - 0.03), v(w + 0.03, d, z0), 0);
    }
    p.finish()
}

/// A straight flight of stairs, as its parameters say it.
#[derive(Clone, Copy, Debug)]
pub struct Stairs {
    pub width: f32,
    /// The height it climbs and how far it runs, along +Y.
    pub height: f32,
    pub length: f32,
    pub steps: u32,
    /// A solid stack under the treads, or open treads alone.
    pub solid: bool,
    /// An open tread's thickness.
    pub tread: f32,
}

impl Default for Stairs {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 2.7,
            length: 3.6,
            steps: 15,
            solid: true,
            tread: 0.04,
        }
    }
}

pub fn stairs(s: &Stairs) -> MeshData {
    let mut p = Pieces::default();
    let n = s.steps.clamp(1, 200);
    let (w, run, rise) = (s.width * 0.5, s.length / n as f32, s.height / n as f32);
    for i in 0..n {
        let y0 = run * i as f32;
        let top = rise * (i + 1) as f32;
        let bottom = if s.solid {
            0.0
        } else {
            (top - within(s.tread, 0.005, rise)).max(0.0)
        };
        p.cuboid(v(-w, y0, bottom), v(w, y0 + run, top), 0);
    }
    p.finish()
}

/// A column, as its parameters say it.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    pub diameter: f32,
    pub height: f32,
    pub round: bool,
    /// A square base and capital, and how much wider than the shaft.
    pub base: bool,
    pub base_scale: f32,
    /// Sides round a round shaft.
    pub segments: u32,
}

impl Default for Column {
    fn default() -> Self {
        Self {
            diameter: 0.3,
            height: 2.7,
            round: true,
            base: true,
            base_scale: 1.35,
            segments: 32,
        }
    }
}

pub fn column(c: &Column) -> MeshData {
    let mut p = Pieces::default();
    let r = c.diameter * 0.5;
    let plinth = if c.base {
        (c.diameter * 0.25).min(c.height * 0.1)
    } else {
        0.0
    };
    if c.base {
        let b = r * c.base_scale.max(1.0);
        p.cuboid(v(-b, -b, 0.0), v(b, b, plinth), 0);
        p.cuboid(v(-b, -b, c.height - plinth), v(b, b, c.height), 0);
    }
    let shaft = c.height - 2.0 * plinth;
    if shaft > 0.0 {
        if c.round {
            let m = crate::primitives::cylinder(r, shaft, c.segments.clamp(3, 256));
            p.mesh(&m, v(0.0, 0.0, plinth + shaft * 0.5), 0);
        } else {
            p.cuboid(v(-r, -r, plinth), v(r, r, c.height - plinth), 0);
        }
    }
    p.finish()
}

/// A railing, as its parameters say it.
#[derive(Clone, Copy, Debug)]
pub struct Railing {
    pub length: f32,
    pub height: f32,
    /// About how far apart the balusters stand.
    pub spacing: f32,
    /// The end posts', the balusters' and the top rail's widths.
    pub post: f32,
    pub baluster: f32,
    pub rail: f32,
}

impl Default for Railing {
    fn default() -> Self {
        Self {
            length: 3.0,
            height: 1.0,
            spacing: 0.12,
            post: 0.06,
            baluster: 0.02,
            rail: 0.07,
        }
    }
}

pub fn railing(r: &Railing) -> MeshData {
    let mut p = Pieces::default();
    let l = r.length * 0.5;
    let post = within(r.post, 0.005, r.length * 0.4);
    let rail_h = 0.05f32.min(r.height * 0.2);
    let bal = r.baluster.max(0.002);
    let rail = r.rail.max(0.01) * 0.5;
    for x in [-l, l - post] {
        p.cuboid(v(x, -post * 0.5, 0.0), v(x + post, post * 0.5, r.height), 0);
    }
    p.cuboid(
        v(-l + post, -rail, r.height - rail_h),
        v(l - post, rail, r.height),
        0,
    );
    let bottom = 0.08f32.min(r.height * 0.2);
    p.cuboid(
        v(-l + post, -0.02, bottom),
        v(l - post, 0.02, bottom + 0.04),
        0,
    );
    let inner = r.length - 2.0 * post;
    let count = ((inner / r.spacing.max(0.03)).floor() as u32)
        .saturating_sub(1)
        .min(400);
    for i in 1..=count {
        let x = -l + post + inner * i as f32 / (count + 1) as f32;
        p.cuboid(
            v(x - bal * 0.5, -bal * 0.5, bottom + 0.04),
            v(x + bal * 0.5, bal * 0.5, r.height - rail_h),
            0,
        );
    }
    p.finish()
}

/// A floor slab `width` × `length`, `thickness` thick, its top at the
/// origin — the level it is the floor of.
pub fn slab(width: f32, length: f32, thickness: f32) -> MeshData {
    let mut p = Pieces::default();
    p.cuboid(
        v(-width * 0.5, -length * 0.5, -thickness),
        v(width * 0.5, length * 0.5, 0.0),
        0,
    );
    p.finish()
}

/// What shape of roof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoofKind {
    /// Two slopes to a ridge along Y, gable ends.
    Gable,
    /// Four slopes, a short ridge (or a point over a square plan).
    Hip,
    /// One slope, rising across X.
    Shed,
    /// Level.
    Flat,
}

/// A roof, as its parameters say it.
#[derive(Clone, Copy, Debug)]
pub struct Roof {
    pub kind: RoofKind,
    /// The plan across the slopes (X) and along the ridge (Y).
    pub width: f32,
    pub length: f32,
    /// From the eaves line up to the ridge.
    pub rise: f32,
    pub thickness: f32,
    /// How far it reaches past the walls.
    pub overhang: f32,
}

impl Default for Roof {
    fn default() -> Self {
        Self {
            kind: RoofKind::Gable,
            width: 6.0,
            length: 8.0,
            rise: 2.0,
            thickness: 0.15,
            overhang: 0.3,
        }
    }
}

/// A roof over a `width` (X) × `length` (Y) plan, its eaves line — where
/// the slopes cross the walls — at the origin's z.
pub fn roof(r: &Roof) -> MeshData {
    let mut p = Pieces::default();
    let half = r.width * 0.5;
    if half <= 0.0 || r.length <= 0.0 {
        return p.finish();
    }
    let oh = r.overhang.max(0.0);
    let (y0, y1) = (-r.length * 0.5 - oh, r.length * 0.5 + oh);
    let t = r.thickness.max(0.001);
    match r.kind {
        RoofKind::Gable => {
            let slope = r.rise / half;
            for s in [-1.0f32, 1.0] {
                let eave = Vec2::new(s * (half + oh), -oh * slope);
                let ridge = Vec2::new(0.0, r.rise);
                let down = Vec2::new(0.0, t);
                p.prism_y(&[eave - down, ridge - down, ridge, eave], y0, y1, 0);
            }
        }
        RoofKind::Shed => {
            // Rising from the eaves at −X to `rise` at +X.
            let slope = r.rise / r.width;
            let z = |x: f32| (x + half) * slope;
            let (a, b) = (-half - oh, half + oh);
            p.prism_y(
                &[
                    Vec2::new(a, z(a) - t),
                    Vec2::new(b, z(b) - t),
                    Vec2::new(b, z(b)),
                    Vec2::new(a, z(a)),
                ],
                y0,
                y1,
                0,
            );
        }
        RoofKind::Flat => {
            p.cuboid(v(-half - oh, y0, 0.0), v(half + oh, y1, t), 0);
        }
        RoofKind::Hip => {
            // A solid: the eaves rectangle, and slopes of one pitch on all
            // four sides up to a ridge as long as the plan is longer than
            // it is wide — a point over a square.
            let slope = r.rise / half;
            let (x, y) = (half + oh, r.length * 0.5 + oh);
            let ze = -oh * slope;
            let ridge = (y - x).max(0.0);
            let s = p.positions.len() as u32;
            p.positions.extend([
                v(-x, -y, ze),
                v(x, -y, ze),
                v(x, y, ze),
                v(-x, y, ze),
                v(0.0, -ridge, r.rise),
                v(0.0, ridge, r.rise),
            ]);
            let (c0, c1, c2, c3, r0, r1) = (s, s + 1, s + 2, s + 3, s + 4, s + 5);
            let mut faces = vec![vec![c0, c3, c2, c1], vec![c0, c1, r0], vec![c2, c3, r1]];
            if ridge > 1e-6 {
                faces.push(vec![c1, c2, r1, r0]);
                faces.push(vec![c3, c0, r0, r1]);
            } else {
                // The two ridge ends are one point: triangles to it, and
                // the second, unused, is left out of the mesh.
                p.positions.pop();
                faces[2] = vec![c2, c3, r0];
                faces.push(vec![c1, c2, r0]);
                faces.push(vec![c3, c0, r0]);
            }
            for f in faces {
                p.faces.push(f);
                p.slots.push(0);
            }
        }
    }
    p.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::mass_properties;

    fn closed_with_volume(m: &MeshData) -> f64 {
        let mp = mass_properties(m, |p| p);
        assert!(mp.closed, "every piece closed");
        assert!(mp.volume > 0.0, "faces out: {}", mp.volume);
        mp.volume
    }

    /// Sizes that leave no room — a door lower than its handle, a window
    /// thinner than its glass, a railing shorter than its posts — and values
    /// that are not numbers make some shape or none, never a crash: they can
    /// come from a file.
    #[test]
    fn sizes_that_leave_no_room_do_not_crash() {
        let nan = f32::NAN;
        door(&Door {
            height: 0.05,
            ..Door::default()
        });
        door(&Door {
            height: 0.01,
            depth: 0.001,
            leaf: nan,
            handle_height: nan,
            ..Door::default()
        });
        window(&Window {
            depth: 0.0001,
            glass: 1.0,
            ..Window::default()
        });
        window(&Window {
            depth: nan,
            frame: nan,
            bar: nan,
            ..Window::default()
        });
        railing(&Railing {
            length: 0.001,
            post: 1.0,
            ..Railing::default()
        });
        railing(&Railing {
            length: nan,
            post: nan,
            spacing: nan,
            ..Railing::default()
        });
        stairs(&Stairs {
            height: 0.0001,
            solid: false,
            tread: 1.0,
            ..Stairs::default()
        });
        column(&Column {
            height: nan,
            diameter: nan,
            base_scale: nan,
            ..Column::default()
        });
        roof(&Roof {
            width: nan,
            rise: nan,
            ..Roof::default()
        });
        assert_eq!(within(nan, 1.0, 2.0), 1.0);
        assert_eq!(within(5.0, 1.0, 0.5), 1.0);
        assert_eq!(within(1.5, 1.0, 2.0), 1.5);
    }

    #[test]
    fn a_wall_is_its_box() {
        let m = wall(4.0, 2.7, 0.2);
        let vol = closed_with_volume(&m);
        assert!((vol - 4.0 * 2.7 * 0.2).abs() < 1e-4, "{vol}");
        let b = m.bounds();
        assert!(b.min.z.abs() < 1e-6 && (b.max.z - 2.7).abs() < 1e-5);
    }

    #[test]
    fn a_door_fits_its_opening_and_has_three_materials() {
        let m = door(&Door::default());
        closed_with_volume(&m);
        let b = m.bounds();
        assert!((b.max.x - b.min.x - (0.9 + 0.12)).abs() < 1e-4);
        assert!((b.max.z - 2.16).abs() < 1e-4 && b.min.z.abs() < 1e-6);
        let slots = m.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
        for s in 0..3 {
            assert!(slots.contains(&s), "slot {s}");
        }
    }

    /// The handles change sides, the panels change in number, and a flush
    /// door has none.
    #[test]
    fn a_doors_details_follow_its_parameters() {
        let handle_x = |d: &Door| {
            let m = door(d);
            let slots = m.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
            let p = m.positions();
            let (mut sum, mut n) = (0.0, 0);
            for (f, corners) in m.faces().enumerate() {
                if slots[f] == 2 {
                    for c in corners {
                        sum += p[*c as usize].x;
                        n += 1;
                    }
                }
            }
            sum / n as f32
        };
        assert!(handle_x(&Door::default()) > 0.2);
        let left = Door {
            handle_right: false,
            ..Door::default()
        };
        assert!(handle_x(&left) < -0.2);
        let faces = |n| {
            door(&Door {
                panels: n,
                ..Door::default()
            })
            .num_faces()
        };
        assert_eq!(
            faces(4) - faces(0),
            4 * 2 * 6,
            "four panels a face, six faces each"
        );
        assert!(
            closed_with_volume(&door(&Door {
                panels: 5,
                ..Door::default()
            })) > 0.0
        );
    }

    #[test]
    fn a_window_has_its_panes_and_sits_on_its_sill() {
        let m = window(&Window {
            rows: 2,
            ..Window::default()
        });
        closed_with_volume(&m);
        let slots = m.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
        // Four panes, six faces each.
        assert_eq!(slots.iter().filter(|s| **s == 1).count(), 4 * 6);
        let b = m.bounds();
        assert!((b.max.z - 2.3).abs() < 1e-4, "{}", b.max.z);
        // No sill: nothing stands out in front of the frame.
        let bare = window(&Window {
            sill: 0.0,
            ..Window::default()
        });
        assert!(bare.bounds().min.y > b.min.y + 0.03);
    }

    #[test]
    fn stairs_rise_as_asked_solid_or_open() {
        let m = stairs(&Stairs::default());
        let vol = closed_with_volume(&m);
        // A staircase of n steps fills (n + 1) / 2n of its bounding box.
        let full = 1.0 * 2.7 * 3.6;
        assert!((vol - full * 16.0 / 30.0).abs() < 1e-3, "{vol}");
        assert!((m.bounds().max.z - 2.7).abs() < 1e-5);
        let open = stairs(&Stairs {
            solid: false,
            ..Stairs::default()
        });
        let treads = closed_with_volume(&open);
        assert!((treads - 15.0 * 1.0 * 0.24 * 0.04).abs() < 1e-4, "{treads}");
    }

    #[test]
    fn every_roof_kind_is_a_solid_under_its_ridge() {
        for kind in [
            RoofKind::Gable,
            RoofKind::Hip,
            RoofKind::Shed,
            RoofKind::Flat,
        ] {
            let r = roof(&Roof {
                kind,
                ..Roof::default()
            });
            closed_with_volume(&r);
            let top = r.bounds().max.z;
            let want = if kind == RoofKind::Flat { 0.15 } else { 2.0 };
            assert!(top <= want + 0.3 && top >= want - 1e-3, "{kind:?}: {top}");
        }
        // A hip roof over a square plan comes to a point.
        let point = roof(&Roof {
            kind: RoofKind::Hip,
            width: 6.0,
            length: 6.0,
            ..Roof::default()
        });
        closed_with_volume(&point);
        assert_eq!(point.num_verts(), 5);
    }

    #[test]
    fn columns_railings_and_slabs_are_solids() {
        for m in [
            column(&Column::default()),
            column(&Column {
                round: false,
                base: false,
                ..Column::default()
            }),
            railing(&Railing::default()),
            slab(4.0, 5.0, 0.2),
        ] {
            closed_with_volume(&m);
        }
        assert!(
            (slab(4.0, 5.0, 0.2).bounds().max.z).abs() < 1e-6,
            "top at 0"
        );
    }
}
