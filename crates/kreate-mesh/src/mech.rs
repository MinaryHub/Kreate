//! Machine parts and the cutters that make room for them, to the standards a
//! workshop orders by: ISO metric threads (ISO 261, ISO 68-1), clearance
//! holes (ISO 273), counterbores (DIN 974) and countersinks (ISO 10642) for
//! socket head screws.
//!
//! Every length here is in metres. The tables are in millimetres, as the
//! standards print them, and converted where they are read.

use std::f64::consts::TAU;

use std::f64::consts::PI;

use kreate_math::{DVec2, Vec3};

use crate::MeshData;

/// One ISO metric size: its name and what a hole for it measures, in mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricSize {
    pub name: &'static str,
    /// The nominal (major) diameter.
    pub d: f64,
    /// The coarse pitch (ISO 261).
    pub pitch: f64,
    /// The medium clearance hole (ISO 273).
    pub clearance: f64,
    /// A socket head cap screw's counterbore: diameter and depth (DIN 974).
    pub cbore_d: f64,
    pub cbore_depth: f64,
    /// A countersunk screw's head diameter, at 90° (ISO 10642).
    pub csink_d: f64,
}

/// M2 to M24, the sizes a machine is mostly held together with.
pub const METRIC: [MetricSize; 13] = [
    MetricSize {
        name: "M2",
        d: 2.0,
        pitch: 0.4,
        clearance: 2.4,
        cbore_d: 4.3,
        cbore_depth: 2.4,
        csink_d: 4.4,
    },
    MetricSize {
        name: "M2.5",
        d: 2.5,
        pitch: 0.45,
        clearance: 2.9,
        cbore_d: 5.0,
        cbore_depth: 2.9,
        csink_d: 5.5,
    },
    MetricSize {
        name: "M3",
        d: 3.0,
        pitch: 0.5,
        clearance: 3.4,
        cbore_d: 6.5,
        cbore_depth: 3.4,
        csink_d: 6.72,
    },
    MetricSize {
        name: "M4",
        d: 4.0,
        pitch: 0.7,
        clearance: 4.5,
        cbore_d: 8.0,
        cbore_depth: 4.4,
        csink_d: 8.96,
    },
    MetricSize {
        name: "M5",
        d: 5.0,
        pitch: 0.8,
        clearance: 5.5,
        cbore_d: 10.0,
        cbore_depth: 5.4,
        csink_d: 11.2,
    },
    MetricSize {
        name: "M6",
        d: 6.0,
        pitch: 1.0,
        clearance: 6.6,
        cbore_d: 11.0,
        cbore_depth: 6.4,
        csink_d: 13.44,
    },
    MetricSize {
        name: "M8",
        d: 8.0,
        pitch: 1.25,
        clearance: 9.0,
        cbore_d: 15.0,
        cbore_depth: 8.6,
        csink_d: 17.92,
    },
    MetricSize {
        name: "M10",
        d: 10.0,
        pitch: 1.5,
        clearance: 11.0,
        cbore_d: 18.0,
        cbore_depth: 10.6,
        csink_d: 22.4,
    },
    MetricSize {
        name: "M12",
        d: 12.0,
        pitch: 1.75,
        clearance: 13.5,
        cbore_d: 20.0,
        cbore_depth: 12.6,
        csink_d: 26.88,
    },
    MetricSize {
        name: "M14",
        d: 14.0,
        pitch: 2.0,
        clearance: 15.5,
        cbore_d: 24.0,
        cbore_depth: 14.6,
        csink_d: 30.8,
    },
    MetricSize {
        name: "M16",
        d: 16.0,
        pitch: 2.0,
        clearance: 17.5,
        cbore_d: 26.0,
        cbore_depth: 16.6,
        csink_d: 33.6,
    },
    MetricSize {
        name: "M20",
        d: 20.0,
        pitch: 2.5,
        clearance: 22.0,
        cbore_d: 33.0,
        cbore_depth: 20.6,
        csink_d: 40.32,
    },
    MetricSize {
        name: "M24",
        d: 24.0,
        pitch: 3.0,
        clearance: 26.0,
        cbore_d: 40.0,
        cbore_depth: 24.8,
        csink_d: 48.0,
    },
];

impl MetricSize {
    /// The tap drill: the major diameter less a pitch, which leaves the
    /// usual 75 % or so of thread.
    pub fn tap_drill(&self) -> f64 {
        self.d - self.pitch
    }
}

/// What kind of hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoleKind {
    /// A plain hole: the clearance for its size, or a diameter given.
    Simple,
    /// A clearance hole with a counterbore for a socket head cap screw.
    Counterbore,
    /// A clearance hole with a 90° countersink for a countersunk screw.
    Countersink,
    /// A tap drill, threaded if asked.
    Tapped,
}

/// A hole as a cutter: a solid of revolution about Z, the hole going down
/// −Z from z = 0 — the surface it is cut into — `depth` deep, and standing a
/// little above that surface so the cut does not share it. A blind hole
/// ends in a drill's 118° point.
pub fn hole_cutter(
    kind: HoleKind,
    size: &MetricSize,
    diameter: Option<f64>,
    depth: f64,
    through: bool,
) -> MeshData {
    let mm = 0.001;
    let r = 0.5
        * mm
        * match kind {
            HoleKind::Tapped => size.tap_drill(),
            _ => diameter
                .filter(|d| *d > 0.0)
                .map(|d| d / mm)
                .unwrap_or(size.clearance),
        };
    let above = (r * 0.5).max(0.0005);
    let depth = depth.max(1e-5);
    // Bottom to top, so the faces point out (see `primitives::lathe`).
    let mut profile: Vec<(f32, f32)> = Vec::new();
    if through {
        profile.push((0.0, -depth as f32));
    } else {
        // The drill's point: 118° included, so its sides fall at 31° from
        // the hole's floor.
        let tip = r * (31f64.to_radians()).tan();
        profile.push((0.0, (-depth - tip) as f32));
    }
    profile.push((r as f32, -depth as f32));
    match kind {
        HoleKind::Counterbore => {
            let (cr, cd) = (size.cbore_d * 0.5 * mm, size.cbore_depth * mm);
            profile.push((r as f32, -cd as f32));
            profile.push((cr as f32, -cd as f32));
            profile.push((cr as f32, above as f32));
        }
        HoleKind::Countersink => {
            // 90°: the cone falls as far as it narrows.
            let cr = size.csink_d * 0.5 * mm;
            profile.push((r as f32, -(cr - r) as f32));
            profile.push((cr as f32, 0.0));
            profile.push(((cr + above) as f32, above as f32));
        }
        _ => profile.push((r as f32, above as f32)),
    }
    let top = profile.last().map(|p| p.1).unwrap_or(0.0);
    profile.push((0.0, top));
    crate::primitives::lathe(&profile, 48)
}

/// A helical thread round Z from z = 0 to `length`: an axial section that
/// is `tip` from the axis over `tip_half` either side of each crest's
/// middle, `root` over `root_half` either side of each root's, straight
/// between — a trapezoid, as ISO 68-1's and the basic rack's both are —
/// wound up by `lead` a turn; a crest's middle on the +X side at
/// z = `phase`. Every corner a vertex: flanks flat, crests and roots clean
/// helices, the ends cut off flat; a bore of `bore` (radius) through.
#[allow(clippy::too_many_arguments)]
pub fn thread_solid(
    length: f64,
    pitch: f64,
    lead: f64,
    tip: f64,
    root: f64,
    tip_half: f64,
    root_half: f64,
    phase: f64,
    bore: f64,
) -> MeshData {
    let radius = |u: f64| -> f64 {
        let u = (u.rem_euclid(pitch) + pitch * 0.5).rem_euclid(pitch) - pitch * 0.5;
        let d = u.abs();
        if d <= tip_half {
            tip
        } else if d >= root_half {
            root
        } else {
            tip + (d - tip_half) / (root_half - tip_half) * (root - tip)
        }
    };
    let corners = [-root_half, -tip_half, tip_half, root_half];
    let around = 96usize;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut columns: Vec<Vec<(f64, u32)>> = Vec::with_capacity(around);
    for i in 0..around {
        let phi = i as f64 * TAU / around as f64;
        let shift = lead * phi / TAU + phase;
        let mut zs: Vec<f64> = vec![0.0, length];
        let first = ((-shift - pitch) / pitch).floor() as i64;
        let last = ((length - shift + pitch) / pitch).ceil() as i64;
        for k in first..=last {
            for c in corners {
                let z = c + k as f64 * pitch + shift;
                if z > 1e-9 && z < length - 1e-9 {
                    zs.push(z);
                }
            }
        }
        zs.sort_by(f64::total_cmp);
        zs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let (sn, c) = phi.sin_cos();
        let column = zs
            .into_iter()
            .map(|z| {
                let rr = radius(z - shift);
                positions.push(Vec3::new((rr * c) as f32, (rr * sn) as f32, z as f32));
                (z, positions.len() as u32 - 1)
            })
            .collect();
        columns.push(column);
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for i in 0..around {
        let a = &columns[i];
        let b = &columns[(i + 1) % around];
        let (mut p, mut q) = (0usize, 0usize);
        while p + 1 < a.len() || q + 1 < b.len() {
            let take_a = q + 1 >= b.len() || (p + 1 < a.len() && a[p + 1].0 <= b[q + 1].0);
            if take_a {
                faces.push(vec![a[p].1, b[q].1, a[p + 1].1]);
                p += 1;
            } else {
                faces.push(vec![a[p].1, b[q].1, b[q + 1].1]);
                q += 1;
            }
        }
    }
    let holes = if bore > 0.0 { 48 } else { 0 };
    let mut bore_rings = [Vec::new(), Vec::new()];
    for (k, z) in [0.0, length].into_iter().enumerate() {
        for i in 0..holes {
            let a = -(i as f64) * TAU / holes as f64;
            positions.push(Vec3::new(
                (bore * a.cos()) as f32,
                (bore * a.sin()) as f32,
                z as f32,
            ));
            bore_rings[k].push(positions.len() as u32 - 1);
        }
    }
    for (k, top) in [false, true].into_iter().enumerate() {
        let ends: Vec<u32> = columns
            .iter()
            .map(|c| if top { c.last().unwrap().1 } else { c[0].1 })
            .collect();
        let outline: Vec<DVec2> = ends
            .iter()
            .map(|v| {
                let p = positions[*v as usize];
                DVec2::new(p.x as f64, p.y as f64)
            })
            .collect();
        let mut rings = vec![outline];
        if holes > 0 {
            rings.push(
                (0..holes)
                    .map(|i| {
                        let a = i as f64 * TAU / holes as f64;
                        DVec2::new(bore * a.cos(), bore * a.sin())
                    })
                    .collect(),
            );
        }
        let index = |v: usize| -> u32 {
            if v < around {
                ends[v]
            } else {
                bore_rings[k][(holes - (v - around)) % holes]
            }
        };
        let tris = crate::earcut::triangulate(&rings);
        let cap = crate::regions::merged(
            tris.iter()
                .map(|[a, b, c]| vec![index(*a), index(*b), index(*c)])
                .collect(),
        );
        for f in cap {
            if top {
                faces.push(f);
            } else {
                faces.push(f.into_iter().rev().collect());
            }
        }
    }
    for i in 0..holes {
        let j = (i + 1) % holes;
        let (lo, hi) = (&bore_rings[0], &bore_rings[1]);
        faces.push(vec![lo[i], lo[j], hi[j], hi[i]]);
    }
    MeshData::from_faces(positions, &faces)
}

/// An ISO 68-1 external thread, `length` long up Z from 0: major diameter
/// `d`, pitch `p`, its 60° flanks, a crest flat of P/8 and a root of P/4 at
/// the basic minor diameter. In metres.
pub fn metric_thread(d: f64, p: f64, length: f64) -> MeshData {
    let h = 3f64.sqrt() * 0.5 * p;
    let tip = d * 0.5;
    let root = tip - 5.0 / 8.0 * h;
    // The crest is P/8 wide, the root P/4: measured from a crest's middle,
    // the flank runs from P/16 out to P/2 − P/8.
    thread_solid(
        length,
        p,
        p,
        tip,
        root,
        p / 16.0,
        p * 0.5 - p / 8.0,
        0.0,
        0.0,
    )
}

/// The cutter for a tapped hole's thread: an external thread a little over
/// size, so what it cuts takes a bolt of size `d` with play.
pub fn internal_thread_cutter(d: f64, p: f64, length: f64, play: f64) -> MeshData {
    metric_thread(d + 2.0 * play, p, length)
}

// --- fasteners ------------------------------------------------------------------

/// One size's fastener dimensions, in mm: ISO 4762 socket head cap screws
/// (head diameter, socket across flats and depth), ISO 4017 hex bolts' head
/// height, ISO 4032 nuts (across flats, height) and ISO 7089 washers (bore,
/// outside, thickness).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastenerSize {
    pub cap_head: f64,
    pub socket: f64,
    pub socket_depth: f64,
    pub hex_head: f64,
    pub across_flats: f64,
    pub nut_height: f64,
    pub washer_bore: f64,
    pub washer_od: f64,
    pub washer_thickness: f64,
}

#[allow(clippy::too_many_arguments)]
const fn fs(
    cap_head: f64,
    socket: f64,
    socket_depth: f64,
    hex_head: f64,
    across_flats: f64,
    nut_height: f64,
    washer_bore: f64,
    washer_od: f64,
    washer_thickness: f64,
) -> FastenerSize {
    FastenerSize {
        cap_head,
        socket,
        socket_depth,
        hex_head,
        across_flats,
        nut_height,
        washer_bore,
        washer_od,
        washer_thickness,
    }
}

/// In `METRIC`'s order.
pub const FASTENERS: [FastenerSize; 13] = [
    fs(3.8, 1.5, 1.0, 1.4, 4.0, 1.6, 2.2, 5.0, 0.3),
    fs(4.5, 2.0, 1.1, 1.7, 5.0, 2.0, 2.7, 6.0, 0.5),
    fs(5.5, 2.5, 1.3, 2.0, 5.5, 2.4, 3.2, 7.0, 0.5),
    fs(7.0, 3.0, 2.0, 2.8, 7.0, 3.2, 4.3, 9.0, 0.8),
    fs(8.5, 4.0, 2.5, 3.5, 8.0, 4.7, 5.3, 10.0, 1.0),
    fs(10.0, 5.0, 3.0, 4.0, 10.0, 5.2, 6.4, 12.0, 1.6),
    fs(13.0, 6.0, 4.0, 5.3, 13.0, 6.8, 8.4, 16.0, 1.6),
    fs(16.0, 8.0, 5.0, 6.4, 16.0, 8.4, 10.5, 20.0, 2.0),
    fs(18.0, 10.0, 6.0, 7.5, 18.0, 10.8, 13.0, 24.0, 2.5),
    fs(21.0, 12.0, 7.0, 8.8, 21.0, 12.8, 15.0, 28.0, 2.5),
    fs(24.0, 14.0, 8.0, 10.0, 24.0, 14.8, 17.0, 30.0, 3.0),
    fs(30.0, 17.0, 10.0, 12.5, 30.0, 18.0, 21.0, 37.0, 3.0),
    fs(36.0, 19.0, 12.0, 15.0, 36.0, 21.5, 25.0, 44.0, 4.0),
];

/// What kind of fastener.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastenerKind {
    /// ISO 4762: a round head with a hex socket.
    CapScrew,
    /// ISO 4017: a hex head, threaded to it.
    HexBolt,
    /// ISO 4032.
    Nut,
    /// ISO 7089.
    Washer,
}

/// A regular hexagon `across` its flats, anticlockwise, a corner on +X.
pub fn hexagon(across: f64) -> Vec<DVec2> {
    let r = across / 3f64.sqrt();
    (0..6)
        .map(|i| {
            let a = i as f64 * TAU / 6.0;
            DVec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// A circle of `n` points, anticlockwise.
pub fn round(r: f64, n: usize) -> Vec<DVec2> {
    (0..n)
        .map(|i| {
            let a = i as f64 * TAU / n as f64;
            DVec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// A prism from z0 to z1: `outline` with `hole` through, both anticlockwise.
pub fn prism(outline: &[DVec2], hole: &[DVec2], z0: f64, z1: f64) -> MeshData {
    let mut m = crate::gear::sweep(outline, hole, z1 - z0, 1, |q, _| q, |q, _| q);
    for p in m.positions_mut() {
        p.z += z0 as f32;
    }
    m
}

/// Closed solids side by side as one mesh.
pub fn together(parts: &[MeshData]) -> MeshData {
    let parts: Vec<crate::edit::join::Part<'_>> = parts
        .iter()
        .map(|m| crate::edit::join::Part {
            mesh: m,
            transform: kreate_math::DAffine3::IDENTITY,
        })
        .collect();
    crate::edit::join::join(&parts).0
}

/// A fastener of `size` (an index into [`METRIC`]), as it sits on a surface
/// at z = 0: a screw's head on it and its shank down −Z for `length`, a nut
/// or a washer on it going up. The thread modelled to ISO 68-1 when
/// `thread`, plain at its major diameter otherwise; a nut's thread is shown
/// as its bore at the minor diameter.
pub fn fastener(kind: FastenerKind, size: usize, length: f64, thread: bool) -> MeshData {
    let mm = 0.001;
    let s = METRIC[size.min(METRIC.len() - 1)];
    let f = FASTENERS[size.min(FASTENERS.len() - 1)];
    let d = s.d * mm;
    let shank = |length: f64| -> MeshData {
        let length = length.max(d);
        let mut m = if thread {
            metric_thread(d, s.pitch * mm, length)
        } else {
            prism(&round(d * 0.5, 48), &[], 0.0, length)
        };
        for p in m.positions_mut() {
            p.z -= length as f32;
        }
        m
    };
    match kind {
        FastenerKind::CapScrew => {
            // The head, k = d high, its hex socket sunk into the top.
            let k = s.d * mm;
            let t = (f.socket_depth * mm).min(k * 0.8);
            let solid = prism(&round(f.cap_head * 0.5 * mm, 48), &[], 0.0, k - t);
            let ring = prism(
                &round(f.cap_head * 0.5 * mm, 48),
                &hexagon(f.socket * mm),
                k - t,
                k,
            );
            together(&[shank(length), solid, ring])
        }
        FastenerKind::HexBolt => together(&[
            shank(length),
            prism(&hexagon(f.across_flats * mm), &[], 0.0, f.hex_head * mm),
        ]),
        FastenerKind::Nut => {
            let minor = (s.d - 1.0825 * s.pitch) * mm;
            let height = f.nut_height * mm;
            let body = prism(
                &hexagon(f.across_flats * mm),
                &round(minor * 0.5, 48),
                0.0,
                height,
            );
            if !thread {
                return body;
            }
            // The thread cut into the bore: the screw's own, a pitch longer
            // each end so it runs out through both faces, with a twentieth
            // of a millimetre of play for a printed nut to take its screw.
            let p = s.pitch * mm;
            let mut cutter = internal_thread_cutter(d, p, height + 2.0 * p, 0.00005);
            for q in cutter.positions_mut() {
                q.z -= p as f32;
            }
            crate::modifiers::boolean::boolean(
                &body,
                &cutter,
                crate::modifiers::BooleanOp::Difference,
            )
        }
        FastenerKind::Washer => prism(
            &round(f.washer_od * 0.5 * mm, 64),
            &round(f.washer_bore * 0.5 * mm, 48),
            0.0,
            f.washer_thickness * mm,
        ),
    }
}

// --- bearings -------------------------------------------------------------------

/// A deep groove ball bearing's designation and size, in mm (ISO 15):
/// bore, outside diameter, width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingSize {
    pub name: &'static str,
    pub bore: f64,
    pub outside: f64,
    pub width: f64,
}

const fn b(name: &'static str, bore: f64, outside: f64, width: f64) -> BearingSize {
    BearingSize {
        name,
        bore,
        outside,
        width,
    }
}

/// The miniature 6xx, and the 60 and 62 series to 50 mm bore.
pub const BEARINGS: [BearingSize; 27] = [
    b("625", 5.0, 16.0, 5.0),
    b("626", 6.0, 19.0, 6.0),
    b("608", 8.0, 22.0, 7.0),
    b("6000", 10.0, 26.0, 8.0),
    b("6001", 12.0, 28.0, 8.0),
    b("6002", 15.0, 32.0, 9.0),
    b("6003", 17.0, 35.0, 10.0),
    b("6004", 20.0, 42.0, 12.0),
    b("6005", 25.0, 47.0, 12.0),
    b("6006", 30.0, 55.0, 13.0),
    b("6007", 35.0, 62.0, 14.0),
    b("6008", 40.0, 68.0, 15.0),
    b("6009", 45.0, 75.0, 16.0),
    b("6010", 50.0, 80.0, 16.0),
    b("6200", 10.0, 30.0, 9.0),
    b("6201", 12.0, 32.0, 10.0),
    b("6202", 15.0, 35.0, 11.0),
    b("6203", 17.0, 40.0, 12.0),
    b("6204", 20.0, 47.0, 14.0),
    b("6205", 25.0, 52.0, 15.0),
    b("6206", 30.0, 62.0, 16.0),
    b("6207", 35.0, 72.0, 17.0),
    b("6208", 40.0, 80.0, 18.0),
    b("6209", 45.0, 85.0, 19.0),
    b("6210", 50.0, 90.0, 20.0),
    b("6300", 10.0, 35.0, 11.0),
    b("6301", 12.0, 37.0, 12.0),
];

/// A deep groove ball bearing standing on z = 0 about Z: its outer ring,
/// its inner ring — each with its raceway groove — and its balls between,
/// three solids touching nothing. The rings are what a drawing and a fit
/// need; the balls are there to be seen.
pub fn bearing(size: &BearingSize) -> MeshData {
    bearing_shielded(size, 0)
}

/// [`bearing`] with `shields` metal shields over the balls: 0 open, 1 on
/// the top face (a "Z"), 2 on both ("2Z"). Each a thin ring set just in from
/// the face, held by the outer ring and clear of the inner.
pub fn bearing_shielded(size: &BearingSize, shields: u32) -> MeshData {
    let mm = 0.001;
    let (d, big, w) = (size.bore * mm, size.outside * mm, size.width * mm);
    let ball = 0.3 * (big - d);
    let pitch = (big + d) * 0.25;
    let gap = ball * 0.1;
    // The rings' faces either side of the balls, and the raceway each has.
    let inner_out = pitch - ball * 0.32;
    let outer_in = pitch + ball * 0.32;
    let groove = ball * 0.52;
    let race = |from: f64, to: f64, n: usize| -> Vec<(f32, f32)> {
        // A circular groove of radius `groove` about (pitch, w/2), from the
        // ring face at radius `from` across to `to`: the arc's points
        // where it is inside the ring.
        let mut pts = Vec::new();
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let z = w * 0.5 + (t - 0.5) * 2.0 * ball * 0.42;
            let dz = z - w * 0.5;
            let dr = (groove * groove - dz * dz).max(0.0).sqrt();
            let r = if to > from { pitch + dr } else { pitch - dr };
            let r = if to > from { r.max(from) } else { r.min(from) };
            pts.push((r as f32, z as f32));
        }
        pts
    };
    // The outer ring, bottom to top round its section: outside, top, the
    // bore with its groove, bottom.
    let mut outer: Vec<(f32, f32)> = vec![
        ((outer_in) as f32, 0.0),
        ((big * 0.5) as f32, 0.0),
        ((big * 0.5) as f32, w as f32),
        (outer_in as f32, w as f32),
    ];
    let mut groove_out = race(outer_in, big, 8);
    groove_out.reverse();
    outer.extend(groove_out);
    let mut inner: Vec<(f32, f32)> = vec![((d * 0.5) as f32, 0.0), (inner_out as f32, 0.0)];
    inner.extend(race(inner_out, 0.0, 8));
    inner.push((inner_out as f32, w as f32));
    inner.push(((d * 0.5) as f32, w as f32));
    let outer = ring_of(&outer, 96);
    let inner = ring_of(&inner, 96);
    let count = ((PI * 2.0 * pitch) / (ball * 1.25)).floor().max(3.0) as usize;
    let mut parts = vec![outer, inner];
    for k in 0..count {
        let a = k as f64 * TAU / count as f64;
        let mut s = crate::primitives::uv_sphere(((ball - gap) * 0.5) as f32, 16, 8);
        for p in s.positions_mut() {
            p.x += (pitch * a.cos()) as f32;
            p.y += (pitch * a.sin()) as f32;
            p.z += (w * 0.5) as f32;
        }
        parts.push(s);
    }
    let thick = (w * 0.04).max(0.0002);
    let recess = (w * 0.03).max(0.0001);
    let clear = (ball * 0.04).max(0.00005);
    for k in 0..shields.min(2) {
        // Top first; the second at the bottom.
        let (z0, z1) = if k == 0 {
            (w - recess - thick, w - recess)
        } else {
            (recess, recess + thick)
        };
        let section = [
            ((inner_out + clear) as f32, z0 as f32),
            (outer_in as f32, z0 as f32),
            (outer_in as f32, z1 as f32),
            ((inner_out + clear) as f32, z1 as f32),
        ];
        parts.push(ring_of(&section, 96));
    }
    together(&parts)
}

/// A closed section `(r, z)`, anticlockwise in the (r, z) plane, turned
/// once round Z: a ring.
fn ring_of(section: &[(f32, f32)], segments: usize) -> MeshData {
    let n = section.len();
    let mut positions: Vec<Vec3> = Vec::with_capacity(n * segments);
    for s in 0..segments {
        let a = s as f64 * TAU / segments as f64;
        let (sn, c) = a.sin_cos();
        for (r, z) in section {
            positions.push(Vec3::new(r * c as f32, r * sn as f32, *z));
        }
    }
    let mut faces = Vec::new();
    for s in 0..segments {
        let t = (s + 1) % segments;
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = ((s * n + i) as u32, (s * n + j) as u32);
            let (c, d) = ((t * n + j) as u32, (t * n + i) as u32);
            faces.push(vec![a, d, c, b]);
        }
    }
    MeshData::from_faces(positions, &faces)
}

// --- shafts -----------------------------------------------------------------------

/// A parallel key and its keyway, by the shaft it suits (DIN 6885 A), in
/// mm: shaft diameters over and up to, key width and height, keyway depth
/// in the shaft and in the hub.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeySize {
    pub over: f64,
    pub upto: f64,
    pub width: f64,
    pub height: f64,
    pub shaft_depth: f64,
    pub hub_depth: f64,
}

const fn k(
    over: f64,
    upto: f64,
    width: f64,
    height: f64,
    shaft_depth: f64,
    hub_depth: f64,
) -> KeySize {
    KeySize {
        over,
        upto,
        width,
        height,
        shaft_depth,
        hub_depth,
    }
}

pub const KEYS: [KeySize; 11] = [
    k(6.0, 8.0, 2.0, 2.0, 1.2, 1.0),
    k(8.0, 10.0, 3.0, 3.0, 1.8, 1.4),
    k(10.0, 12.0, 4.0, 4.0, 2.5, 1.8),
    k(12.0, 17.0, 5.0, 5.0, 3.0, 2.3),
    k(17.0, 22.0, 6.0, 6.0, 3.5, 2.8),
    k(22.0, 30.0, 8.0, 7.0, 4.0, 3.3),
    k(30.0, 38.0, 10.0, 8.0, 5.0, 3.3),
    k(38.0, 44.0, 12.0, 8.0, 5.0, 3.3),
    k(44.0, 50.0, 14.0, 9.0, 5.5, 3.8),
    k(50.0, 58.0, 16.0, 10.0, 6.0, 4.3),
    k(58.0, 65.0, 18.0, 11.0, 7.0, 4.4),
];

/// The key a shaft of `d` mm takes.
pub fn key_for(d: f64) -> Option<KeySize> {
    KEYS.iter().copied().find(|k| d > k.over && d <= k.upto)
}

/// A retaining ring groove for a shaft of `d` mm (DIN 471): the groove's
/// diameter and width, from the nearest size not larger.
pub fn ring_groove_for(d: f64) -> (f64, f64) {
    const TABLE: [(f64, f64, f64); 11] = [
        (10.0, 9.6, 1.1),
        (12.0, 11.5, 1.1),
        (15.0, 14.3, 1.1),
        (17.0, 16.2, 1.1),
        (20.0, 19.0, 1.3),
        (25.0, 23.9, 1.3),
        (30.0, 28.6, 1.6),
        (35.0, 33.0, 1.6),
        (40.0, 37.5, 1.85),
        (45.0, 42.5, 1.85),
        (50.0, 47.0, 2.15),
    ];
    let row = TABLE
        .iter()
        .rev()
        .find(|r| r.0 <= d + 1e-9)
        .copied()
        .unwrap_or(TABLE[0]);
    // Scaled to the diameter between the table's sizes.
    (d * row.1 / row.0, row.2)
}

/// A key's outline: `length` long along X, `width` across, its ends round
/// (DIN 6885 form A), centred on the origin.
pub fn key_outline(length: f64, width: f64) -> Vec<DVec2> {
    let r = width * 0.5;
    let straight = (length * 0.5 - r).max(0.0);
    let mut out = Vec::new();
    for i in 0..=12 {
        let a = -PI * 0.5 + PI * i as f64 / 12.0;
        out.push(DVec2::new(straight + r * a.cos(), r * a.sin()));
    }
    for i in 0..=12 {
        let a = PI * 0.5 + PI * i as f64 / 12.0;
        out.push(DVec2::new(-straight + r * a.cos(), r * a.sin()));
    }
    out
}

/// A shaft standing up Z from 0: `d` across, `length` long, with a DIN 471
/// retaining ring groove `groove_at` in from each end (none at zero). Its
/// keyway is a separate cutter — [`keyway_cutter`] — for a boolean to take
/// out, as a mill would.
pub fn shaft(d: f64, length: f64, groove_at: f64) -> MeshData {
    let r = d * 0.5;
    let mut profile: Vec<(f32, f32)> = vec![(0.0, 0.0), (r as f32, 0.0)];
    let mm = 0.001;
    if groove_at > 0.0 && groove_at * 2.0 < length {
        let (gd, gw) = ring_groove_for(d / mm);
        let (gr, gw) = (gd * 0.5 * mm, gw * mm);
        for at in [groove_at, length - groove_at - gw] {
            profile.push((r as f32, at as f32));
            profile.push((gr as f32, at as f32));
            profile.push((gr as f32, (at + gw) as f32));
            profile.push((r as f32, (at + gw) as f32));
        }
    }
    profile.push((r as f32, length as f32));
    profile.push((0.0, length as f32));
    crate::primitives::lathe(&profile, 64)
}

/// A keyway's cutter on a shaft of `d` standing up Z: the key's outline
/// `length` long up the shaft, centred at height `at`, sunk `depth` into
/// the +Y side, and standing proud of it so the cut is clean.
pub fn keyway_cutter(d: f64, key: &KeySize, length: f64, at: f64) -> MeshData {
    let mm = 0.001;
    let outline = key_outline(length, key.width * mm);
    let depth = key.shaft_depth * mm;
    // The key's outline lies in the shaft's X–Z plane, swept along +Y from
    // the keyway's floor to past the surface.
    let mut m = prism(&outline, &[], d * 0.5 - depth, d * 0.5 + key.height * mm);
    for p in m.positions_mut() {
        // Its outline's X up the shaft, its Z out along +Y.
        let (x, y, z) = (p.x, p.y, p.z);
        *p = Vec3::new(y, z, x + at as f32);
    }
    // A cyclic turn of the axes: a rotation, the faces still outward.
    m
}

/// The key that sits in that keyway, `length` long, standing in it.
pub fn key(d: f64, key: &KeySize, length: f64, at: f64) -> MeshData {
    let mm = 0.001;
    let outline = key_outline(length, key.width * mm);
    let floor = d * 0.5 - key.shaft_depth * mm;
    let mut m = prism(&outline, &[], floor, floor + key.height * mm);
    for p in m.positions_mut() {
        let (x, y, z) = (p.x, p.y, p.z);
        *p = Vec3::new(y, z, x + at as f32);
    }
    m
}

// --- pulleys and sprockets -------------------------------------------------------

/// A timing belt profile: its pitch, the pitch line's distance outside the
/// pulley, and the tooth gap's depth and radius, in mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeltProfile {
    pub name: &'static str,
    pub pitch: f64,
    pub pitch_line: f64,
    pub gap_depth: f64,
    pub gap_radius: f64,
}

pub const BELTS: [BeltProfile; 3] = [
    BeltProfile {
        name: "GT2",
        pitch: 2.0,
        pitch_line: 0.254,
        gap_depth: 0.75,
        gap_radius: 0.555,
    },
    BeltProfile {
        name: "GT3",
        pitch: 3.0,
        pitch_line: 0.381,
        gap_depth: 1.14,
        gap_radius: 0.85,
    },
    BeltProfile {
        name: "HTD 5M",
        pitch: 5.0,
        pitch_line: 0.5715,
        gap_depth: 2.06,
        gap_radius: 1.49,
    },
];

/// A timing pulley's outline, anticlockwise: `teeth` round gaps of the
/// belt's, its outside the pitch circle less twice the pitch line.
pub fn pulley_outline(belt: &BeltProfile, teeth: u32) -> Vec<DVec2> {
    let mm = 0.001;
    let z = teeth.max(8) as f64;
    let pitch_d = z * belt.pitch / PI * mm;
    let ro = pitch_d * 0.5 - belt.pitch_line * mm;
    let (gr, depth) = (belt.gap_radius * mm, belt.gap_depth * mm);
    let centre = ro - depth + gr;
    // Where the gap's circle meets the outside circle.
    let half = {
        let x = (ro * ro + centre * centre - gr * gr) / (2.0 * ro * centre);
        x.clamp(-1.0, 1.0).acos()
    };
    let step = TAU / z;
    let mut out = Vec::new();
    for t in 0..teeth.max(8) {
        let c = t as f64 * step;
        // The land between gaps, round the outside.
        for i in 0..=4 {
            let a = c + half + (step - 2.0 * half) * i as f64 / 4.0;
            out.push(DVec2::new(ro * a.cos(), ro * a.sin()));
        }
        // Down into the next gap and out again, round its circle.
        let g = c + step;
        let cp = DVec2::new(centre * g.cos(), centre * g.sin());
        let enter = DVec2::new(ro * (g - half).cos(), ro * (g - half).sin());
        let leave = DVec2::new(ro * (g + half).cos(), ro * (g + half).sin());
        let (a0, a1) = (
            (enter - cp).y.atan2((enter - cp).x),
            (leave - cp).y.atan2((leave - cp).x),
        );
        // The long way round, through the gap's bottom: clockwise about cp.
        let mut sweep = a1 - a0;
        while sweep > 0.0 {
            sweep -= TAU;
        }
        for i in 1..12 {
            let a = a0 + sweep * i as f64 / 12.0;
            out.push(cp + DVec2::new(gr * a.cos(), gr * a.sin()));
        }
    }
    out.dedup_by(|a, b| a.distance(*b) < 1e-9);
    if out.len() > 1 && out[0].distance(*out.last().unwrap()) < 1e-9 {
        out.pop();
    }
    out
}

/// A timing pulley standing on z = 0: the toothed part `width` high, a
/// flange each side a millimetre thick standing out past the teeth, and a
/// bore.
pub fn pulley(belt: &BeltProfile, teeth: u32, width: f64, bore: f64, flanges: bool) -> MeshData {
    let mm = 0.001;
    let outline = pulley_outline(belt, teeth);
    let hole = if bore > 0.0 {
        round(bore * 0.5, 48)
    } else {
        Vec::new()
    };
    let mut parts = Vec::new();
    let (z0, f) = if flanges { (mm, mm) } else { (0.0, 0.0) };
    parts.push(prism(&outline, &hole, z0, z0 + width));
    if flanges {
        let ro = outline.iter().map(|q| q.length()).fold(0.0, f64::max) + 1.5 * mm;
        let disc = round(ro, 96);
        parts.push(prism(&disc, &hole, 0.0, f));
        parts.push(prism(&disc, &hole, z0 + width, z0 + width + f));
    }
    together(&parts)
}

/// A roller chain (ISO 606): its pitch, roller diameter and inner width, mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainSize {
    pub name: &'static str,
    pub pitch: f64,
    pub roller: f64,
    pub inner_width: f64,
}

pub const CHAINS: [ChainSize; 7] = [
    ChainSize {
        name: "04B",
        pitch: 6.0,
        roller: 4.0,
        inner_width: 2.8,
    },
    ChainSize {
        name: "05B",
        pitch: 8.0,
        roller: 5.0,
        inner_width: 3.0,
    },
    ChainSize {
        name: "06B",
        pitch: 9.525,
        roller: 6.35,
        inner_width: 5.72,
    },
    ChainSize {
        name: "08B",
        pitch: 12.7,
        roller: 8.51,
        inner_width: 7.75,
    },
    ChainSize {
        name: "10B",
        pitch: 15.875,
        roller: 10.16,
        inner_width: 9.65,
    },
    ChainSize {
        name: "25 (ANSI)",
        pitch: 6.35,
        roller: 3.3,
        inner_width: 3.18,
    },
    ChainSize {
        name: "40 (ANSI)",
        pitch: 12.7,
        roller: 7.92,
        inner_width: 7.85,
    },
];

/// A sprocket's pitch diameter, in metres: what its rollers' centres sit on.
pub fn sprocket_pitch_diameter(chain: &ChainSize, teeth: u32) -> f64 {
    chain.pitch * 0.001 / (PI / teeth.max(6) as f64).sin()
}

/// A sprocket's outline, anticlockwise, to ISO 606's tooth form: a roller
/// seat of radius 0.505 d1 at each pitch point, its flanks running out to
/// the tip diameter half way between ISO's least and most.
pub fn sprocket_outline(chain: &ChainSize, teeth: u32) -> Vec<DVec2> {
    let mm = 0.001;
    let z = teeth.max(6) as f64;
    let (p, d1) = (chain.pitch * mm, chain.roller * mm);
    let d = sprocket_pitch_diameter(chain, teeth);
    let ri = 0.505 * d1;
    let da_max = d + 1.25 * p - d1;
    let da_min = d + p * (1.0 - 1.6 / z) - d1;
    let ra = (da_max + da_min) * 0.25;
    let step = TAU / z;
    // The seat's arc: ISO 606's roller seating angle, 140° − 90°/z.
    let seat = (140.0 - 90.0 / z).to_radians();
    let wrap = |a: f64| (a + PI).rem_euclid(TAU) - PI;
    let mut out = Vec::new();
    for t in 0..teeth.max(6) {
        let g = t as f64 * step;
        let c = DVec2::new(d * 0.5 * g.cos(), d * 0.5 * g.sin());
        // Round the seat, the side of the roller toward the centre: from its
        // end the smaller angle round the sprocket to the larger.
        let inward = g + PI;
        for i in 0..=10 {
            let a = inward + seat * 0.5 - seat * i as f64 / 10.0;
            out.push(c + DVec2::new(ri * a.cos(), ri * a.sin()));
        }
        // The tooth after it: out along the flank to the tip circle, across
        // the tip, and down to the next seat.
        let e = *out.last().unwrap();
        let delta = wrap(e.y.atan2(e.x) - g).abs();
        let mid = g + step * 0.5;
        let tip_half = (step * 0.5 - delta).max(0.0) * 0.35;
        for a in [mid - tip_half, mid + tip_half] {
            out.push(DVec2::new(ra * a.cos(), ra * a.sin()));
        }
    }
    out.dedup_by(|a, b| a.distance(*b) < 1e-9);
    out
}

/// A sprocket standing on z = 0, as thick as its chain allows (0.93 of its
/// inner width), with a bore.
pub fn sprocket(chain: &ChainSize, teeth: u32, bore: f64) -> MeshData {
    let mm = 0.001;
    let outline = sprocket_outline(chain, teeth);
    let hole = if bore > 0.0 {
        round(bore * 0.5, 48)
    } else {
        Vec::new()
    };
    prism(&outline, &hole, 0.0, 0.93 * chain.inner_width * mm)
}

// --- springs ----------------------------------------------------------------------

/// What kind of spring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpringKind {
    /// Open coils with closed, ground ends.
    Compression,
    /// Close-wound coils with a hook at each end.
    Extension,
    /// Close-wound coils with a straight leg at each end.
    Torsion,
}

/// A spring of round wire `wire` across, coiled `mean` across its middle,
/// standing up Z from 0: `coils` turns over `length` (a compression
/// spring's free length; the others are close-wound and ignore it).
pub fn spring(kind: SpringKind, wire: f64, mean: f64, coils: f64, length: f64) -> MeshData {
    let r = mean * 0.5;
    let coils = coils.max(1.0);
    let mut path: Vec<Vec3> = Vec::new();
    let per_turn = 24usize;
    let total = (coils * per_turn as f64).ceil() as usize;
    let height = |t: f64| -> f64 {
        match kind {
            SpringKind::Compression => {
                // A closed coil at each end — pitch of one wire — and the
                // rest spread over what is left of the length.
                let ends = 1.0f64.min(coils * 0.25);
                let body = (coils - 2.0 * ends).max(0.1);
                let body_pitch = ((length - wire) - 2.0 * ends * wire).max(0.0) / body;
                let turn = t;
                let z = if turn < ends {
                    turn * wire
                } else if turn > coils - ends {
                    ends * wire + body * body_pitch + (turn - (coils - ends)) * wire
                } else {
                    ends * wire + (turn - ends) * body_pitch
                };
                z + wire * 0.5
            }
            _ => t * wire * 1.02 + wire * 0.5,
        }
    };
    let leg = mean * 0.8;
    if kind == SpringKind::Torsion {
        path.push(Vec3::new((r + leg) as f32, 0.0, height(0.0) as f32));
    }
    for i in 0..=total {
        let t = i as f64 / per_turn as f64;
        let t = t.min(coils);
        let a = t * TAU;
        path.push(Vec3::new(
            (r * a.cos()) as f32,
            (r * a.sin()) as f32,
            height(t) as f32,
        ));
    }
    let end = coils * TAU;
    match kind {
        SpringKind::Torsion => {
            let e = Vec3::new(
                (r * end.cos()) as f32,
                (r * end.sin()) as f32,
                height(coils) as f32,
            );
            let out = Vec3::new(end.cos() as f32, end.sin() as f32, 0.0) * leg as f32;
            path.push(e + out);
        }
        SpringKind::Extension => {
            // A hook at each end: a loop the coil's size, standing up.
            let top = height(coils);
            let start_hook: Vec<Vec3> = (1..=12)
                .map(|i| {
                    let a = PI * i as f64 / 12.0;
                    Vec3::new(
                        (r * a.cos()) as f32,
                        0.0,
                        (-(r * a.sin()) + height(0.0)) as f32,
                    )
                })
                .collect();
            let mut with = start_hook.into_iter().rev().collect::<Vec<_>>();
            with.extend(path);
            path = with;
            let e = Vec3::new((r * end.cos()) as f32, (r * end.sin()) as f32, top as f32);
            for i in 1..=12 {
                let a = PI * i as f64 / 12.0;
                path.push(Vec3::new(
                    (e.x as f64 * a.cos()) as f32,
                    (e.y as f64 * a.cos()) as f32,
                    (top + r * a.sin()) as f32,
                ));
            }
        }
        SpringKind::Compression => {}
    }
    let profile = crate::primitives::circle_profile((wire * 0.5) as f32, 12);
    let mut m = crate::primitives::sweep(&crate::primitives::Sweep {
        path: &path,
        profile: &profile,
        resolution: 1,
        closed: false,
        scale: (1.0, 1.0),
        caps: true,
    });
    if kind == SpringKind::Compression {
        // Ground flat at each end, to stand square.
        for p in m.positions_mut() {
            p.z = p.z.clamp(0.0, length as f32);
        }
    }
    m
}

// --- shell and draft --------------------------------------------------------------

/// A solid hollowed: the faces given taken away — the open side — and what
/// is left given a wall `thickness` thick, inside. What Fusion's Shell does
/// to a box with its top picked: a tray.
pub fn shell(mesh: &MeshData, open: &[u32], thickness: f32) -> MeshData {
    let keep: Vec<Vec<u32>> = (0..mesh.num_faces())
        .filter(|f| !open.contains(f))
        .map(|f| mesh.face_verts(f).to_vec())
        .collect();
    let opened = MeshData::from_faces(mesh.positions().to_vec(), &keep);
    thicken_inward(&opened, thickness)
}

/// A surface given a wall `thickness` thick on its inside, the same
/// thickness everywhere: each vertex moved in along its faces' mean normal
/// far enough that every face's plane moves the whole thickness — at a box's
/// corner √3 times it, along the diagonal. A plain normal offset would
/// leave the wall thinner wherever faces meet.
pub fn thicken_inward(surface: &MeshData, thickness: f32) -> MeshData {
    let n = surface.num_verts() as usize;
    let positions = surface.positions();
    let normals = surface.compute_face_normals();
    let mut around: Vec<Vec<Vec3>> = vec![Vec::new(); n];
    for f in 0..surface.num_faces() {
        let u = normals[f as usize].normalize_or_zero();
        if u == Vec3::ZERO {
            continue;
        }
        for v in surface.face_verts(f) {
            // Each plane once: two triangles of one face lean no harder.
            let list = &mut around[*v as usize];
            if !list.iter().any(|w| w.dot(u) > 0.9999) {
                list.push(u);
            }
        }
    }
    let mut out = positions.to_vec();
    for (v, planes) in around.iter().enumerate() {
        let dir = planes.iter().copied().sum::<Vec3>().normalize_or_zero();
        let cos = if planes.is_empty() {
            1.0
        } else {
            planes.iter().map(|u| u.dot(dir)).sum::<f32>() / planes.len() as f32
        };
        out.push(positions[v] - dir * (thickness / cos.max(0.25)));
    }
    let inner = |v: u32| v + n as u32;
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for f in 0..surface.num_faces() {
        faces.push(surface.face_verts(f).to_vec());
        faces.push(
            surface
                .face_verts(f)
                .iter()
                .rev()
                .map(|v| inner(*v))
                .collect(),
        );
    }
    // The rim: every edge on one face only, joined across to the inside.
    let mut uses: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for f in 0..surface.num_faces() {
        let ring = surface.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    for f in 0..surface.num_faces() {
        let ring = surface.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if uses.get(&(a.min(b), a.max(b))) == Some(&1) {
                faces.push(vec![b, a, inner(a), inner(b)]);
            }
        }
    }
    MeshData::from_faces(out, &faces)
}

/// Faces given a draft: each vertex of `faces` moved out from the part
/// along its faces' normals, across the pull direction, by its height above
/// the neutral plane — the lowest of them along `pull` — times the tangent
/// of `angle`. A mould's walls tapered so the part comes out.
pub fn draft(mesh: &mut MeshData, faces: &[u32], pull: Vec3, angle: f32) {
    let pull = pull.normalize_or_zero();
    if pull == Vec3::ZERO || faces.is_empty() {
        return;
    }
    let normals = mesh.compute_face_normals();
    // Each vertex's walls, each plane once, across the pull only.
    let mut walls: Vec<Vec<Vec3>> = vec![Vec::new(); mesh.num_verts() as usize];
    let mut used = vec![false; mesh.num_verts() as usize];
    for f in faces {
        let n = normals[*f as usize].normalize_or_zero();
        let across = (n - pull * n.dot(pull)).normalize_or_zero();
        for v in mesh.face_verts(*f) {
            used[*v as usize] = true;
            let list = &mut walls[*v as usize];
            if across != Vec3::ZERO && !list.iter().any(|w| w.dot(across) > 0.9999) {
                list.push(across);
            }
        }
    }
    let positions = mesh.positions();
    let neutral = (0..positions.len())
        .filter(|v| used[*v])
        .map(|v| positions[v].dot(pull))
        .fold(f32::MAX, f32::min);
    let t = angle.tan();
    let moves: Vec<(usize, Vec3)> = (0..positions.len())
        .filter(|v| used[*v])
        .map(|v| {
            let h = positions[v].dot(pull) - neutral;
            let dir = walls[v].iter().copied().sum::<Vec3>().normalize_or_zero();
            // Every wall's plane in by h tan(angle): at a corner further,
            // along the diagonal, as a shell's wall is.
            let cos = if walls[v].is_empty() {
                1.0
            } else {
                walls[v].iter().map(|u| u.dot(dir)).sum::<f32>() / walls[v].len() as f32
            };
            (v, positions[v] - dir * (h * t / cos.max(0.25)))
        })
        .collect();
    let p = mesh.positions_mut();
    for (v, q) in moves {
        p[v] = q;
    }
}

// --- loft --------------------------------------------------------------------------

/// A closed outline given `n` points: every corner of its own kept, the rest
/// spread along its sides by their lengths — so a square lofted to a circle
/// keeps its corners rather than being rounded by an even resampling.
pub fn resample_loop(ring: &[Vec3], n: usize) -> Vec<Vec3> {
    let k = ring.len();
    if k == 0 {
        return Vec::new();
    }
    if n <= k {
        return ring[..n.max(1)].to_vec();
    }
    let lengths: Vec<f32> = (0..k)
        .map(|i| ring[i].distance(ring[(i + 1) % k]))
        .collect();
    let total: f32 = lengths.iter().sum::<f32>().max(f32::EPSILON);
    // One sample a side at least, the rest by length, largest remainder.
    let spare = n - k;
    let share: Vec<f32> = lengths.iter().map(|l| l / total * spare as f32).collect();
    let mut extra: Vec<usize> = share.iter().map(|s| s.floor() as usize).collect();
    let mut left = spare - extra.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| {
        (share[*b] - share[*b].floor())
            .partial_cmp(&(share[*a] - share[*a].floor()))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in order {
        if left == 0 {
            break;
        }
        extra[i] += 1;
        left -= 1;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..k {
        let (a, b) = (ring[i], ring[(i + 1) % k]);
        let steps = extra[i] + 1;
        for s in 0..steps {
            out.push(a.lerp(b, s as f32 / steps as f32));
        }
    }
    out
}

/// A solid through closed outlines, in order: each outline given the same
/// count of points, wound the same way about the way the loft goes, and
/// turned so its start lies by the last one's, then the sides joined with
/// quads and the two ends closed. A square base to a round top, a duct, a
/// bottle's shoulder.
pub fn loft(sections: &[Vec<Vec3>]) -> MeshData {
    let sections: Vec<&Vec<Vec3>> = sections.iter().filter(|s| s.len() >= 3).collect();
    if sections.len() < 2 {
        return MeshData::empty();
    }
    let n = sections
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(3)
        .clamp(32, 512);
    let centre = |s: &[Vec3]| s.iter().copied().sum::<Vec3>() / s.len() as f32;
    let way = (centre(sections[sections.len() - 1]) - centre(sections[0])).normalize_or_zero();
    let mut rings: Vec<Vec<Vec3>> = Vec::new();
    for s in &sections {
        let mut ring: Vec<Vec3> = s.to_vec();
        let c = centre(&ring);
        let turn: Vec3 = (0..ring.len())
            .map(|i| (ring[i] - c).cross(ring[(i + 1) % ring.len()] - c))
            .sum();
        // Counter-clockwise about the way it goes, so the sides face out.
        let about = if way == Vec3::ZERO { turn } else { way };
        if turn.dot(about) < 0.0 {
            ring.reverse();
        }
        let mut ring = resample_loop(&ring, n);
        if let Some(last) = rings.last() {
            // The start that puts each point nearest its partner, centres
            // laid over each other: no twist from where the outline began.
            let (cl, cr) = (centre(last), centre(&ring));
            let best = (0..n)
                .min_by(|a, b| {
                    let cost = |s: usize| -> f32 {
                        (0..n)
                            .map(|i| ((last[i] - cl) - (ring[(i + s) % n] - cr)).length_squared())
                            .sum()
                    };
                    cost(*a)
                        .partial_cmp(&cost(*b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            ring.rotate_left(best);
        }
        rings.push(ring);
    }
    let mut positions: Vec<Vec3> = Vec::with_capacity(n * rings.len());
    for r in &rings {
        positions.extend_from_slice(r);
    }
    let at = |ring: usize, i: usize| (ring * n + i % n) as u32;
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for r in 0..rings.len() - 1 {
        for i in 0..n {
            faces.push(vec![at(r, i), at(r, i + 1), at(r + 1, i + 1), at(r + 1, i)]);
        }
    }
    faces.push((0..n).rev().map(|i| at(0, i)).collect());
    faces.push((0..n).map(|i| at(rings.len() - 1, i)).collect());
    MeshData::from_faces(positions, &faces)
}

// --- sheet metal -------------------------------------------------------------------

/// A sheet metal part: a base plate and a flange on any of its sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheetMetal {
    /// The base's flat between the bends, across x and along y.
    pub width: f32,
    pub length: f32,
    pub thickness: f32,
    /// The bends' inside radius.
    pub radius: f32,
    /// Where the neutral line lies through the thickness, from the inside:
    /// 0.5 the middle; 0.3 to 0.45 for most steel bent in a press brake.
    pub k_factor: f32,
    /// The flanges' angle from flat, radians: π/2 square up.
    pub angle: f32,
    /// The flanges' straight length past their bend.
    pub height: f32,
    /// Which sides have a flange: −y, +x, +y, −x.
    pub flanges: [bool; 4],
    /// Laid out flat: each bend unrolled to its bend allowance — the
    /// blank to cut before it is bent.
    pub flat: bool,
}

impl SheetMetal {
    /// The length of sheet a bend takes up, along the neutral line.
    pub fn bend_allowance(&self) -> f32 {
        self.angle * (self.radius + self.k_factor * self.thickness)
    }

    /// The blank's size: the base and each flange unrolled.
    pub fn flat_size(&self) -> (f32, f32) {
        let leg = self.bend_allowance() + self.height;
        let [front, right, back, left] = self.flanges.map(|f| if f { leg } else { 0.0 });
        (self.width + left + right, self.length + front + back)
    }
}

/// A sheet metal part as one closed solid, base on z = 0 to the thickness,
/// the flanges bent up. The corners are left open — the relief a real
/// part has there, so the flanges do not run into each other.
pub fn sheet_metal(p: &SheetMetal) -> MeshData {
    use kreate_math::Vec2;
    use std::f32::consts::PI;
    let t = p.thickness;
    let r = p.radius.max(t * 0.01);
    let (hw, hl) = (p.width * 0.5, p.length * 0.5);
    // The section of a flange across its edge: inside and outside points,
    // u out from the edge and v up.
    let section: Vec<(Vec2, Vec2)> = if p.flat {
        let u = p.bend_allowance() + p.height;
        vec![
            (Vec2::new(0.0, t), Vec2::ZERO),
            (Vec2::new(u, t), Vec2::new(u, 0.0)),
        ]
    } else {
        let steps = 2 + (p.angle / (PI / 16.0)).ceil() as usize;
        let at = |radius: f32, phi: f32| Vec2::new(radius * phi.sin(), t + r - radius * phi.cos());
        let mut s: Vec<(Vec2, Vec2)> = (0..=steps)
            .map(|i| {
                let phi = p.angle * i as f32 / steps as f32;
                (at(r, phi), at(r + t, phi))
            })
            .collect();
        let d = Vec2::new(p.angle.cos(), p.angle.sin()) * p.height;
        let (i, o) = *s.last().expect("a bend");
        if p.height > 0.0 {
            s.push((i + d, o + d));
        }
        s
    };
    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let add = |p: Vec3, positions: &mut Vec<Vec3>| {
        positions.push(p);
        positions.len() as u32 - 1
    };
    // The base, as a box.
    let corners: Vec<u32> = (0..8)
        .map(|i| {
            let x = if i & 1 == 0 { -hw } else { hw };
            let y = if i & 2 == 0 { -hl } else { hl };
            let z = if i & 4 == 0 { 0.0 } else { t };
            add(Vec3::new(x, y, z), &mut positions)
        })
        .collect();
    for q in [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [3, 2, 6, 7],
        [2, 0, 4, 6],
    ] {
        faces.push(q.iter().map(|i| corners[*i]).collect());
    }
    // Each side: where its edge's middle is, out, along (out × along = z).
    let sides = [
        (Vec3::new(0.0, -hl, 0.0), -Vec3::Y, Vec3::X, p.width),
        (Vec3::new(hw, 0.0, 0.0), Vec3::X, Vec3::Y, p.length),
        (Vec3::new(0.0, hl, 0.0), Vec3::Y, -Vec3::X, p.width),
        (Vec3::new(-hw, 0.0, 0.0), -Vec3::X, -Vec3::Y, p.length),
    ];
    for (k, (mid, out, along, span)) in sides.iter().enumerate() {
        if !p.flanges[k] {
            continue;
        }
        let place = |q: Vec2, s: f32| *mid + *out * q.x + Vec3::Z * q.y + *along * s;
        let ends = [-span * 0.5, span * 0.5];
        let ring: Vec<[[u32; 2]; 2]> = section
            .iter()
            .map(|(i, o)| {
                let mut v = [[0; 2]; 2];
                for (e, s) in ends.iter().enumerate() {
                    v[0][e] = add(place(*i, *s), &mut positions);
                    v[1][e] = add(place(*o, *s), &mut positions);
                }
                v
            })
            .collect();
        let (inner, outer, a, b) = (0, 1, 0, 1);
        for j in 0..ring.len() - 1 {
            let (c, n) = (ring[j], ring[j + 1]);
            faces.push(vec![c[inner][a], n[inner][a], n[inner][b], c[inner][b]]);
            faces.push(vec![c[outer][a], c[outer][b], n[outer][b], n[outer][a]]);
            faces.push(vec![c[inner][a], c[outer][a], n[outer][a], n[inner][a]]);
            faces.push(vec![c[inner][b], n[inner][b], n[outer][b], c[outer][b]]);
        }
        let (first, last) = (ring[0], ring[ring.len() - 1]);
        faces.push(vec![
            last[inner][a],
            last[outer][a],
            last[outer][b],
            last[inner][b],
        ]);
        // Where it meets the base: the base's side, the other way round —
        // the two cancel below and the flange grows out of the plate.
        faces.push(vec![
            first[inner][a],
            first[inner][b],
            first[outer][b],
            first[outer][a],
        ]);
    }
    // Weld what lies in one place, then drop the faces met by their own
    // reverse: the base's sides that a flange starts from.
    let mut index: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();
    let mut welded: Vec<Vec3> = Vec::new();
    let remap: Vec<u32> = positions
        .iter()
        .map(|q| {
            let key = (*q * 1e7).round().as_i64vec3().to_array();
            *index.entry(key).or_insert_with(|| {
                welded.push(*q);
                welded.len() as u32 - 1
            })
        })
        .collect();
    let faces: Vec<Vec<u32>> = faces
        .into_iter()
        .map(|f| f.iter().map(|v| remap[*v as usize]).collect())
        .collect();
    let key = |f: &Vec<u32>| {
        let mut k = f.clone();
        k.sort_unstable();
        k
    };
    let mut count: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();
    for f in &faces {
        *count.entry(key(f)).or_default() += 1;
    }
    let kept: Vec<Vec<u32>> = faces.into_iter().filter(|f| count[&key(f)] == 1).collect();
    MeshData::from_faces(welded, &kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{is_closed, mass_properties};
    use std::f64::consts::PI;

    fn solid(m: &MeshData) -> f64 {
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert!(is_closed(m), "not closed");
        let v = mass_properties(m, |q| q).volume;
        assert!(v > 0.0, "inside out: {v}");
        v
    }

    fn m6() -> MetricSize {
        *METRIC.iter().find(|s| s.name == "M6").unwrap()
    }

    #[test]
    fn the_table_is_iso_s() {
        let s = m6();
        assert_eq!((s.d, s.pitch, s.clearance), (6.0, 1.0, 6.6));
        assert!((s.tap_drill() - 5.0).abs() < 1e-12);
        let m10 = METRIC.iter().find(|s| s.name == "M10").unwrap();
        assert!((m10.tap_drill() - 8.5).abs() < 1e-12);
    }

    #[test]
    fn a_hole_cutter_is_as_big_as_its_size_says() {
        let s = m6();
        let through = hole_cutter(HoleKind::Simple, &s, None, 0.01, true);
        let v = solid(&through);
        // 6.6 mm across, 10 mm deep and the little above: a 48-sided prism
        // a hair under the true cylinder.
        let r = 0.0033;
        let above = (r * 0.5f64).max(0.0005);
        let want = PI * r * r * (0.01 + above) * (48.0 / TAU * (TAU / 48.0).sin());
        assert!((v - want).abs() < want * 1e-3, "{v} against {want}");
        // A counterbore is bigger, a tapped hole smaller.
        let cb = solid(&hole_cutter(HoleKind::Counterbore, &s, None, 0.01, true));
        let cs = solid(&hole_cutter(HoleKind::Countersink, &s, None, 0.01, true));
        let tap = solid(&hole_cutter(HoleKind::Tapped, &s, None, 0.01, true));
        assert!(cb > v && cs > v && tap < v);
        // A blind hole has its point below the depth.
        let blind = hole_cutter(HoleKind::Simple, &s, None, 0.01, false);
        solid(&blind);
        assert!(blind.bounds().min.z < -0.0101);
    }

    /// A threaded nut is the plain one with the thread cut into its bore:
    /// closed, as tall, and lighter.
    #[test]
    fn a_threaded_nut_has_its_thread_cut_in() {
        let m6 = 5;
        let plain = fastener(FastenerKind::Nut, m6, 0.0, false);
        let threaded = fastener(FastenerKind::Nut, m6, 0.0, true);
        assert!(threaded.validate().is_empty());
        assert!(is_closed(&threaded), "closed");
        assert!((threaded.bounds().size().z - plain.bounds().size().z).abs() < 1e-6);
        let (vp, vt) = (solid(&plain), solid(&threaded));
        assert!(vt < vp * 0.99 && vt > vp * 0.8, "{vt} against {vp}");
        // The bore now reaches out to the thread's root, near the major size.
        let widest = threaded
            .positions()
            .iter()
            .filter(|q| q.truncate().length() < 0.004)
            .map(|q| q.truncate().length())
            .fold(0.0f32, f32::max);
        assert!(widest > 0.0029, "{widest}");
    }

    #[test]
    fn a_bearing_is_its_bore_outside_and_width() {
        let b6204 = BEARINGS.iter().find(|b| b.name == "6204").unwrap();
        let m = bearing(b6204);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert!(is_closed(&m));
        let b = m.bounds();
        assert!((b.size().x as f64 - 0.047).abs() < 1e-5, "{b:?}");
        assert!((b.size().z as f64 - 0.014).abs() < 1e-6, "{b:?}");
        // The bore is clear: nothing nearer the axis than 10 mm.
        let nearest = m
            .positions()
            .iter()
            .map(|p| p.truncate().length())
            .fold(f32::MAX, f32::min);
        assert!((nearest as f64 - 0.010).abs() < 1e-6, "{nearest}");
        assert!(mass_properties(&m, |q| q).volume > 0.0);
        // Shielded: each shield a closed ring, inside the width, more metal.
        let open = mass_properties(&m, |q| q).volume;
        let one = bearing_shielded(b6204, 1);
        let two = bearing_shielded(b6204, 2);
        for s in [&one, &two] {
            assert!(s.validate().is_empty());
            assert!(is_closed(s));
            assert!((s.bounds().size().z as f64 - 0.014).abs() < 1e-6);
        }
        let (v1, v2) = (
            mass_properties(&one, |q| q).volume,
            mass_properties(&two, |q| q).volume,
        );
        assert!(
            v1 > open && (v2 - open - 2.0 * (v1 - open)).abs() < 1e-12,
            "{open} {v1} {v2}"
        );
    }

    #[test]
    fn a_shaft_takes_its_key_and_its_ring_grooves() {
        let k = key_for(20.0).unwrap();
        assert_eq!((k.width, k.height, k.shaft_depth), (6.0, 6.0, 3.5));
        assert!(key_for(70.0).is_none());
        assert_eq!(ring_groove_for(20.0), (19.0, 1.3));
        let plain = solid(&shaft(0.02, 0.1, 0.0));
        let grooved = solid(&shaft(0.02, 0.1, 0.005));
        // Two grooves, each 1.3 mm wide from 10 down to 9.5 mm radius.
        let ring = std::f64::consts::PI * (0.01f64.powi(2) - 0.0095f64.powi(2)) * 0.0013 * 2.0;
        assert!(
            ((plain - grooved) - ring).abs() < ring * 0.05,
            "{} against {ring}",
            plain - grooved
        );
        // The key: its outline's area by its height.
        let key_solid = solid(&key(0.02, &k, 0.03, 0.05));
        let outline = key_outline(0.03, 0.006);
        let area: f64 = (0..outline.len())
            .map(|i| outline[i].perp_dot(outline[(i + 1) % outline.len()]))
            .sum::<f64>()
            * 0.5;
        assert!((key_solid - area * 0.006).abs() < area * 0.006 * 1e-3);
        // Lying up the shaft, standing on the keyway's floor, on +Y.
        let kb = key(0.02, &k, 0.03, 0.05).bounds();
        assert!((kb.min.y as f64 - (0.01 - 0.0035)).abs() < 1e-6, "{kb:?}");
        assert!((kb.size().z as f64 - 0.03).abs() < 1e-6, "{kb:?}");
        solid(&keyway_cutter(0.02, &k, 0.03, 0.05));
    }

    #[test]
    fn pulleys_and_sprockets_are_their_standard_sizes() {
        // A 20-tooth GT2 pulley: 12.73 mm pitch diameter, 12.22 outside.
        let gt2 = BELTS[0];
        let o = pulley_outline(&gt2, 20);
        let max = o.iter().map(|q| q.length()).fold(0.0, f64::max);
        assert!(
            (max * 2.0 - (40.0 / std::f64::consts::PI - 0.508) * 0.001).abs() < 1e-6,
            "{max}"
        );
        let min = o.iter().map(|q| q.length()).fold(f64::MAX, f64::min);
        assert!(
            (max - min - 0.00075).abs() < 2e-5,
            "a gap {} deep",
            max - min
        );
        solid(&pulley(&gt2, 20, 0.006, 0.005, true));
        // A 17-tooth 08B sprocket: pitch diameter p / sin(180°/17).
        let chain = CHAINS.iter().find(|c| c.name == "08B").unwrap();
        let d = sprocket_pitch_diameter(chain, 17);
        assert!((d - 0.0127 / (std::f64::consts::PI / 17.0).sin()).abs() < 1e-12);
        let s = sprocket_outline(chain, 17);
        let root = s.iter().map(|q| q.length()).fold(f64::MAX, f64::min);
        // Its root circle: the pitch circle less the roller seat's
        // diameter, about.
        assert!((root * 2.0 - (d - 1.01 * 0.00851)).abs() < 1e-4, "{root}");
        solid(&sprocket(chain, 17, 0.008));
    }

    #[test]
    fn springs_stand_their_length() {
        let c = spring(SpringKind::Compression, 0.001, 0.01, 8.0, 0.03);
        assert!(c.validate().is_empty());
        let b = c.bounds();
        assert!(
            (b.max.z as f64 - 0.03).abs() < 1e-6 && b.min.z.abs() < 1e-6,
            "{b:?}"
        );
        assert!((b.size().x as f64 - 0.011).abs() < 2e-4, "{b:?}");
        for kind in [SpringKind::Extension, SpringKind::Torsion] {
            let m = spring(kind, 0.001, 0.01, 6.0, 0.0);
            assert!(m.validate().is_empty());
            assert!(m.num_faces() > 100);
        }
    }

    #[test]
    fn a_square_lofted_to_a_circle_is_closed_and_keeps_its_corners() {
        let square: Vec<Vec3> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .iter()
            .map(|(x, y)| Vec3::new(*x, *y, 0.0))
            .collect();
        // Wound the other way, and started elsewhere: the loft sorts it out.
        let circle: Vec<Vec3> = (0..48)
            .rev()
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / 48.0 + 1.0;
                Vec3::new(a.cos(), a.sin(), 2.0)
            })
            .collect();
        let m = loft(&[square.clone(), circle]);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        let v = solid(&m);
        // Between the square's 4 and the circle's π, two high: about 7.
        assert!(v > 6.0 && v < 8.0, "{v}");
        for c in &square {
            assert!(
                m.positions().iter().any(|p| p.distance(*c) < 1e-6),
                "corner {c}"
            );
        }
        // Two prisms the same: the volume of one.
        let m = loft(&[
            square.clone(),
            square.iter().map(|p| *p + Vec3::Z).collect(),
        ]);
        assert!((solid(&m) - 4.0).abs() < 1e-4, "{}", solid(&m));
    }

    #[test]
    fn sheet_metal_bends_up_and_lays_out_to_its_bend_allowance() {
        let mut p = SheetMetal {
            width: 0.1,
            length: 0.06,
            thickness: 0.002,
            radius: 0.002,
            k_factor: 0.5,
            angle: std::f32::consts::FRAC_PI_2,
            height: 0.02,
            flanges: [true, true, true, false],
            flat: false,
        };
        let folded = sheet_metal(&p);
        let v = solid(&folded);
        // Square up: the flanges reach R + t + the plate out, and up to
        // the plate, the bend and the leg.
        let b = folded.bounds();
        assert!((b.size().z - (0.002 + 0.002 + 0.02)).abs() < 1e-6, "{b:?}");
        assert!((b.size().x - (0.1 + 0.004)).abs() < 1e-6, "{b:?}");
        // Unrolled at the neutral line in the middle, the volume stays.
        p.flat = true;
        let flat = sheet_metal(&p);
        let vf = solid(&flat);
        assert!((v - vf).abs() < vf * 0.002, "{v} against {vf}");
        let (w, l) = p.flat_size();
        let b = flat.bounds();
        assert!(
            (b.size().x - w).abs() < 1e-6 && (b.size().y - l).abs() < 1e-6,
            "{b:?}"
        );
        assert!(
            (w - (0.1 + std::f32::consts::FRAC_PI_2 * 0.003 + 0.02)).abs() < 1e-6,
            "{w}"
        );
        // No flanges: the plate.
        p.flanges = [false; 4];
        assert!((solid(&sheet_metal(&p)) - 0.1 * 0.06 * 0.002).abs() < 1e-10);
    }

    #[test]
    fn a_box_shelled_with_its_top_open_is_a_tray_and_a_draft_leans_its_walls() {
        let cube = crate::primitives::cube(0.02);
        let top = (0..cube.num_faces())
            .find(|f| {
                cube.compute_face_normals()[*f as usize]
                    .normalize_or_zero()
                    .z
                    > 0.9
            })
            .unwrap();
        let tray = shell(&cube, &[top], 0.002);
        let v = solid(&tray);
        let want = 0.02f64.powi(3) - 0.016 * 0.016 * 0.018;
        assert!((v - want).abs() < want * 0.01, "{v} against {want}");
        // A draft of 5° on the four walls: the top edge in by 20 tan 5°.
        let mut m = crate::primitives::cube(0.02);
        let normals = m.compute_face_normals();
        let walls: Vec<u32> = (0..m.num_faces())
            .filter(|f| normals[*f as usize].normalize_or_zero().z.abs() < 0.1)
            .collect();
        draft(&mut m, &walls, Vec3::Z, 5f32.to_radians());
        let b = m.bounds();
        assert!((b.size().x - 0.02).abs() < 1e-6, "the bottom stays: {b:?}");
        let top_x = m
            .positions()
            .iter()
            .filter(|p| p.z > 0.009)
            .map(|p| p.x)
            .fold(f32::MIN, f32::max);
        let want = 0.01 - 0.02 * 5f32.to_radians().tan();
        assert!((top_x - want).abs() < 1e-6, "{top_x} against {want}");
    }

    #[test]
    fn fasteners_are_their_standard_sizes() {
        let m6 = 5;
        // A socket head cap screw: a 10 mm head 6 mm high, and the shank.
        let screw = fastener(FastenerKind::CapScrew, m6, 0.02, true);
        assert!(screw.validate().is_empty());
        let b = screw.bounds();
        assert!((b.size().x - 0.010).abs() < 1e-5, "{b:?}");
        assert!(
            (b.max.z - 0.006).abs() < 1e-6 && (b.min.z + 0.02).abs() < 1e-6,
            "{b:?}"
        );
        // A nut: 10 across its flats, 5.2 high, and closed; its thread cut
        // in, so it holds less than a hex with a bore at the thread's minor
        // diameter and more than one bored at its major — the grooves take
        // about half the difference.
        let nut = fastener(FastenerKind::Nut, m6, 0.0, true);
        let v = solid(&nut);
        let hex = 3f64.sqrt() / 2.0 * 0.010f64.powi(2);
        let minor = (6.0 - 1.0825) * 0.0005;
        let prism = 48.0 / std::f64::consts::TAU * (std::f64::consts::TAU / 48.0).sin();
        let bored_minor = (hex - PI * minor * minor * prism) * 0.0052;
        let bored_major = (hex - PI * 0.003f64.powi(2)) * 0.0052;
        let middle = (bored_minor + bored_major) * 0.5;
        assert!(
            v < bored_minor
                && v > bored_major
                && (v - middle).abs() < (bored_minor - bored_major) * 0.25,
            "{v} between {bored_major} and {bored_minor}"
        );
        assert!((nut.bounds().size().y - 0.010).abs() < 1e-6);
        // A washer.
        let w = solid(&fastener(FastenerKind::Washer, m6, 0.0, true));
        let want = (PI
            * 0.006f64.powi(2)
            * (64.0 / std::f64::consts::TAU * (std::f64::consts::TAU / 64.0).sin())
            - PI * 0.0032f64.powi(2) * prism)
            * 0.0016;
        assert!((w - want).abs() < want * 0.01, "{w} against {want}");
        // A hex bolt's head is a hexagon 10 across its flats.
        let bolt = fastener(FastenerKind::HexBolt, m6, 0.02, false);
        assert!((bolt.bounds().size().y - 0.010).abs() < 1e-6);
    }

    #[test]
    fn a_metric_thread_is_its_major_and_minor_diameters_round() {
        let t = metric_thread(0.006, 0.001, 0.012);
        solid(&t);
        let r: Vec<f32> = t
            .positions()
            .iter()
            .map(|q| q.truncate().length())
            .collect();
        let max = r.iter().cloned().fold(0.0, f32::max) as f64;
        let min = r.iter().cloned().fold(f32::MAX, f32::min) as f64;
        assert!((max - 0.003).abs() < 1e-7, "{max}");
        let h = 3f64.sqrt() * 0.5 * 0.001;
        assert!((min - (0.003 - 5.0 / 8.0 * h)).abs() < 1e-7, "{min}");
    }
}
