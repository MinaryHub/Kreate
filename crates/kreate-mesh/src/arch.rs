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

/// A door in its frame: an opening `width` × `height`, the frame `depth`
/// deep (the wall's thickness) and `frame` wide, the leaf with a raised
/// panel above and below on both faces, and a lever handle each side.
pub fn door(width: f32, height: f32, depth: f32, frame: f32) -> MeshData {
    let mut p = Pieces::default();
    let (w, d, f) = (width * 0.5, depth * 0.5, frame);
    // The frame: two jambs and a head.
    p.cuboid(v(-w - f, -d, 0.0), v(-w, d, height + f), 1);
    p.cuboid(v(w, -d, 0.0), v(w + f, d, height + f), 1);
    p.cuboid(v(-w, -d, height), v(w, d, height + f), 1);
    // The leaf, a few millimetres clear of the frame and the floor.
    let gap = 0.003;
    let leaf = (0.04f32).min(depth * 0.5);
    let l = leaf * 0.5;
    p.cuboid(v(-w + gap, -l, gap), v(w - gap, l, height - gap), 0);
    // Raised panels, both faces: a tall one above, a shorter one below.
    let inset = (width * 0.14).max(0.05);
    let raise = 0.006;
    let lower_top = height * 0.38;
    for (z0, z1) in [
        (height * 0.08, lower_top - height * 0.03),
        (lower_top + height * 0.03, height * 0.92),
    ] {
        for (y0, y1) in [(-l - raise, -l), (l, l + raise)] {
            p.cuboid(v(-w + inset, y0, z0), v(w - inset, y1, z1), 0);
        }
    }
    // A lever each side, at the usual height, near the opening edge.
    let hx = w - 0.07;
    let hz = 1.0f32.min(height * 0.48);
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
            v(hx - 0.13, lever_y - 0.01 * s.signum(), hz - 0.011),
            v(hx + 0.012, lever_y + 0.01 * s.signum(), hz + 0.011),
            2,
        );
    }
    p.finish()
}

/// Slots of a window: 0 the frame, 1 the glass.
pub const WINDOW_SLOTS: [&str; 2] = ["Window Frame", "Glass"];

/// A window `width` × `height`, its bottom `sill` above the origin, `depth`
/// deep and its frame `frame` wide, divided into `columns` × `rows` panes,
/// with a sill ledge in front.
pub fn window(
    width: f32,
    height: f32,
    depth: f32,
    frame: f32,
    sill: f32,
    columns: u32,
    rows: u32,
) -> MeshData {
    let mut p = Pieces::default();
    let (w, d, f) = (width * 0.5, depth * 0.5, frame);
    let (z0, z1) = (sill, sill + height);
    // The outer frame.
    p.cuboid(v(-w, -d, z0), v(-w + f, d, z1), 0);
    p.cuboid(v(w - f, -d, z0), v(w, d, z1), 0);
    p.cuboid(v(-w + f, -d, z0), v(w - f, d, z0 + f), 0);
    p.cuboid(v(-w + f, -d, z1 - f), v(w - f, d, z1), 0);
    // Mullions and transoms, thinner and shallower than the frame.
    let (cols, rows) = (columns.clamp(1, 12), rows.clamp(1, 12));
    let bar = f * 0.6;
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
    let glass = 0.004;
    for c in 0..cols {
        for r in 0..rows {
            let x0 = ix0 + col_w * c as f32 + if c > 0 { bar * 0.5 } else { 0.0 };
            let x1 = ix0 + col_w * (c + 1) as f32 - if c + 1 < cols { bar * 0.5 } else { 0.0 };
            let zz0 = iz0 + row_h * r as f32 + if r > 0 { bar * 0.5 } else { 0.0 };
            let zz1 = iz0 + row_h * (r + 1) as f32 - if r + 1 < rows { bar * 0.5 } else { 0.0 };
            p.cuboid(v(x0, -glass, zz0), v(x1, glass, zz1), 1);
        }
    }
    // The sill, standing proud of the wall in front.
    if sill > 0.03 {
        p.cuboid(v(-w - 0.03, -d - 0.04, z0 - 0.03), v(w + 0.03, d, z0), 0);
    }
    p.finish()
}

/// A straight flight: `steps` steps rising `height` over `length` along +Y,
/// `width` wide, as a solid stack under the treads.
pub fn stairs(width: f32, height: f32, length: f32, steps: u32) -> MeshData {
    let mut p = Pieces::default();
    let n = steps.clamp(1, 200);
    let (w, run, rise) = (width * 0.5, length / n as f32, height / n as f32);
    for i in 0..n {
        let y0 = run * i as f32;
        p.cuboid(v(-w, y0, 0.0), v(w, y0 + run, rise * (i + 1) as f32), 0);
    }
    p.finish()
}

/// A column `height` tall and `diameter` across, round or square, with a
/// square base and capital.
pub fn column(diameter: f32, height: f32, round: bool) -> MeshData {
    let mut p = Pieces::default();
    let r = diameter * 0.5;
    let plinth = (diameter * 0.25).min(height * 0.1);
    let b = r * 1.35;
    p.cuboid(v(-b, -b, 0.0), v(b, b, plinth), 0);
    p.cuboid(v(-b, -b, height - plinth), v(b, b, height), 0);
    let shaft = height - 2.0 * plinth;
    if shaft > 0.0 {
        if round {
            let c = crate::primitives::cylinder(r, shaft, 32);
            p.mesh(&c, v(0.0, 0.0, plinth + shaft * 0.5), 0);
        } else {
            p.cuboid(v(-r, -r, plinth), v(r, r, height - plinth), 0);
        }
    }
    p.finish()
}

/// A railing `length` long along X and `height` high: a post at each end,
/// a top and a bottom rail, and balusters about `spacing` apart.
pub fn railing(length: f32, height: f32, spacing: f32) -> MeshData {
    let mut p = Pieces::default();
    let l = length * 0.5;
    let post = 0.06f32;
    let rail_h = 0.05f32;
    let bal = 0.02f32;
    for x in [-l, l - post] {
        p.cuboid(v(x, -post * 0.5, 0.0), v(x + post, post * 0.5, height), 0);
    }
    p.cuboid(
        v(-l + post, -0.035, height - rail_h),
        v(l - post, 0.035, height),
        0,
    );
    let bottom = 0.08f32.min(height * 0.2);
    p.cuboid(
        v(-l + post, -0.02, bottom),
        v(l - post, 0.02, bottom + 0.04),
        0,
    );
    let inner = length - 2.0 * post;
    let count = ((inner / spacing.max(0.03)).floor() as u32)
        .saturating_sub(1)
        .min(400);
    for i in 1..=count {
        let x = -l + post + inner * i as f32 / (count + 1) as f32;
        p.cuboid(
            v(x - bal * 0.5, -bal * 0.5, bottom + 0.04),
            v(x + bal * 0.5, bal * 0.5, height - rail_h),
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

/// A gable roof over a `width` (X) × `length` (Y) plan: two slopes rising
/// `rise` to a ridge along Y, `thickness` thick, reaching `overhang` past
/// the walls all round. Its eaves line, at the walls, is the origin's z.
pub fn roof(width: f32, length: f32, rise: f32, thickness: f32, overhang: f32) -> MeshData {
    let mut p = Pieces::default();
    let half = width * 0.5;
    if half <= 0.0 {
        return p.finish();
    }
    let slope = rise / half;
    let (y0, y1) = (-length * 0.5 - overhang, length * 0.5 + overhang);
    for s in [-1.0f32, 1.0] {
        let eave = Vec2::new(s * (half + overhang), -overhang * slope);
        let ridge = Vec2::new(0.0, rise);
        let down = Vec2::new(0.0, thickness);
        p.prism_y(&[eave - down, ridge - down, ridge, eave], y0, y1, 0);
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
        let m = door(0.9, 2.1, 0.2, 0.06);
        closed_with_volume(&m);
        let b = m.bounds();
        assert!((b.max.x - b.min.x - (0.9 + 0.12)).abs() < 1e-4);
        assert!((b.max.z - 2.16).abs() < 1e-4 && b.min.z.abs() < 1e-6);
        let slots = m.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
        for s in 0..3 {
            assert!(slots.contains(&s), "slot {s}");
        }
    }

    #[test]
    fn a_window_has_its_panes_and_sits_on_its_sill() {
        let m = window(1.2, 1.4, 0.15, 0.06, 0.9, 2, 2);
        closed_with_volume(&m);
        let slots = m.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
        // Four panes, six faces each.
        assert_eq!(slots.iter().filter(|s| **s == 1).count(), 4 * 6);
        let b = m.bounds();
        assert!((b.max.z - 2.3).abs() < 1e-4, "{}", b.max.z);
    }

    #[test]
    fn stairs_rise_as_asked() {
        let m = stairs(1.0, 2.7, 3.6, 15);
        let vol = closed_with_volume(&m);
        // A staircase of n steps fills (n + 1) / 2n of its bounding box.
        let full = 1.0 * 2.7 * 3.6;
        assert!((vol - full * 16.0 / 30.0).abs() < 1e-3, "{vol}");
        assert!((m.bounds().max.z - 2.7).abs() < 1e-5);
    }

    #[test]
    fn columns_railings_slabs_and_roofs_are_solids() {
        for m in [
            column(0.3, 2.7, true),
            column(0.3, 2.7, false),
            railing(3.0, 1.0, 0.12),
            slab(4.0, 5.0, 0.2),
            roof(6.0, 8.0, 2.0, 0.15, 0.3),
        ] {
            closed_with_volume(&m);
        }
        assert!(
            (slab(4.0, 5.0, 0.2).bounds().max.z).abs() < 1e-6,
            "top at 0"
        );
        let r = roof(6.0, 8.0, 2.0, 0.15, 0.3);
        assert!((r.bounds().max.z - 2.0).abs() < 1e-4);
        assert!((r.bounds().max.x - 3.3).abs() < 1e-4);
    }
}
