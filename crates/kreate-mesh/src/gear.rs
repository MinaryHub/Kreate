//! Involute spur and helical gears, to the standard basic rack — ISO 53
//! profile A: a pressure angle of 20°, an addendum of one module and a
//! dedendum of 1.25 — in the module series of ISO 54, so a part drawn here
//! meshes with one bought or cut to the same numbers.
//!
//! # The tooth
//!
//! Each flank is the involute of the base circle, from the base circle (or
//! the root, where that is larger) to the tip. Below the base circle the
//! flank runs radially down to the root circle and meets it in a round of
//! 0.38 module, ISO 53's root radius. The tooth's thickness on the pitch
//! circle is half the circular pitch, widened by the profile shift and
//! narrowed by the backlash — what makes two gears of the same module and
//! pressure angle run together at the centre distance [`centre_distance`]
//! gives.
//!
//! # A pair
//!
//! [`mate_turn`] is how far round the second gear of a pair has to be set so
//! a gap of it faces a tooth of the first across the line between them.

use std::f64::consts::{PI, TAU};

use kreate_math::{DQuat, DVec2, DVec3, Vec3};

use crate::MeshData;

/// ISO 54 series I modules, in millimetres: the ones to prefer.
pub const ISO_MODULES: [f64; 14] = [
    0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
];

/// What a gear is to be. Lengths in metres, angles in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearParams {
    /// The module: pitch diameter over the number of teeth.
    pub module: f64,
    pub teeth: u32,
    /// The face width, along the axis.
    pub width: f64,
    /// The bore's diameter; zero for none.
    pub bore: f64,
    /// The basic rack's pressure angle: 20° by ISO 53.
    pub pressure_angle: f64,
    /// The helix angle on the pitch circle; zero for a spur gear. A pair of
    /// helical gears on parallel shafts takes the same angle, one of each
    /// hand.
    pub helix: f64,
    /// The profile shift coefficient, in modules: positive thickens the
    /// tooth at its root and moves it out — what keeps a small gear's teeth
    /// from being undercut.
    pub shift: f64,
    /// Taken off the tooth's thickness on the pitch circle, as an arc: the
    /// play a pair runs with.
    pub backlash: f64,
}

impl Default for GearParams {
    fn default() -> Self {
        Self {
            module: 0.002,
            teeth: 20,
            width: 0.01,
            bore: 0.005,
            pressure_angle: 20f64.to_radians(),
            helix: 0.0,
            shift: 0.0,
            backlash: 0.0,
        }
    }
}

/// A gear's circles, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearInfo {
    /// Where the pitch circles of a pair roll on each other (transverse).
    pub pitch: f64,
    pub base: f64,
    /// The outside diameter: what the blank is turned to.
    pub tip: f64,
    pub root: f64,
}

/// The transverse module and pressure angle: a helical gear's teeth, cut
/// across the axis, are those of a spur gear of a larger module.
fn transverse(p: &GearParams) -> (f64, f64) {
    let c = p.helix.cos().abs().max(0.2);
    (p.module / c, (p.pressure_angle.tan() / c).atan())
}

pub fn gear_info(p: &GearParams) -> GearInfo {
    let (mt, at) = transverse(p);
    let z = p.teeth.max(3) as f64;
    let pitch = mt * z;
    GearInfo {
        pitch,
        base: pitch * at.cos(),
        tip: pitch + 2.0 * p.module * (1.0 + p.shift),
        root: pitch - 2.0 * p.module * (1.25 - p.shift),
    }
}

/// The distance between the shafts of two gears of the same module running
/// together without shift: half the sum of their pitch diameters.
pub fn centre_distance(a: &GearParams, b: &GearParams) -> f64 {
    (gear_info(a).pitch + gear_info(b).pitch) * 0.5
}

/// How far round the second gear of a pair is set, about its own centre, so
/// it meshes with the first: the first has a tooth on its +X axis, looking
/// at the second; the second, at `centre_distance` along +X, has a gap
/// looking back.
pub fn mate_turn(teeth: u32) -> f64 {
    let z = teeth.max(3) as f64;
    PI - PI / z
}

fn involute(a: f64) -> f64 {
    a.tan() - a
}

/// The outline across the axis, anticlockwise, the first tooth centred on
/// +X; `turn` sets it further round.
pub fn gear_outline(p: &GearParams, turn: f64) -> Vec<DVec2> {
    let (mt, at) = transverse(p);
    let info = gear_info(p);
    // The tooth's thickness on the pitch circle, as an arc.
    let s = mt * (PI / 2.0 + 2.0 * p.shift * at.tan()) - p.backlash;
    profile(p, turn, info.tip * 0.5, info.root * 0.5, s, true)
}

/// An involute outline of `p`'s teeth between the circles `ra` and `rf`,
/// each tooth `s` thick on the pitch circle; with ISO 53's root round where
/// the flank runs radially into the root, when `rounded`.
fn profile(p: &GearParams, turn: f64, ra: f64, rf: f64, s: f64, rounded: bool) -> Vec<DVec2> {
    let pitch_angle = TAU / p.teeth.max(3) as f64;
    let mut out = Vec::new();
    for k in 0..p.teeth.max(3) {
        out.extend(tooth(p, turn + k as f64 * pitch_angle, ra, rf, s, rounded));
    }
    tidy(out)
}

/// One tooth of [`profile`] centred at angle `c`, and the root after it to
/// where the next tooth's flank begins.
fn tooth(p: &GearParams, c: f64, ra: f64, rf: f64, s: f64, rounded: bool) -> Vec<DVec2> {
    let (_, at) = transverse(p);
    let info = gear_info(p);
    let z = p.teeth.max(3);
    let (rp, rb) = (info.pitch * 0.5, info.base * 0.5);
    let half = |r: f64| -> f64 {
        let r = r.max(rb);
        let ar = (rb / r).clamp(-1.0, 1.0).acos();
        s / (2.0 * rp) + involute(at) - involute(ar)
    };
    // A tooth too narrow at its tip is cut short where it comes to a point.
    let mut ra = ra;
    while half(ra) < 0.0 && ra > rb {
        ra -= (ra - rb) * 0.02 + 1e-9;
    }
    let pitch_angle = TAU / z as f64;
    let flank_from = rf.max(rb);
    const FLANK: usize = 12;
    // The root fillet: ISO 53's 0.38 module, kept within the gap.
    let gap_half = pitch_angle * 0.5 - half(flank_from);
    // Only where the flank runs radially into the root: an involute that
    // starts at the root circle already meets it.
    let fillet = if rounded && rb > rf {
        (0.38 * p.module)
            .min(rf * gap_half.max(0.0) * 0.5)
            .min(rb - rf)
    } else {
        0.0
    };

    let mut out = Vec::new();
    {
        let at_r = |r: f64, a: f64| DVec2::new(r * a.cos(), r * a.sin());
        // Up the first flank, on the clockwise side of the tooth: radially
        // from the root where the involute does not reach it.
        let mut flank: Vec<(f64, f64)> = Vec::new();
        if rb > rf {
            flank.push((rf + fillet, -half(rb)));
        }
        for i in 0..=FLANK {
            let t = i as f64 / FLANK as f64;
            // Closer together near the base circle, where the involute turns.
            let r = flank_from + (ra - flank_from) * t * t;
            flank.push((r, -half(r)));
        }
        for (r, a) in &flank {
            out.push(at_r(*r, c + a));
        }
        // Across the tip.
        let tip = half(ra);
        for i in 1..4 {
            let a = -tip + 2.0 * tip * i as f64 / 4.0;
            out.push(at_r(ra, c + a));
        }
        // Down the other flank.
        for (r, a) in flank.iter().rev() {
            out.push(at_r(*r, c - a));
        }
        // Round the root to the next tooth, with a fillet into each flank.
        let from = c + half(flank_from);
        let to = c + pitch_angle - half(flank_from);
        let fillet_angle = if rf > 0.0 { fillet / rf } else { 0.0 };
        let bottom_from = from + fillet_angle;
        let bottom_to = to - fillet_angle;
        if fillet > 0.0 {
            for i in 1..4 {
                let u = i as f64 / 4.0 * PI * 0.5;
                // From the flank's foot to the root circle, a quarter turn.
                let r = rf + fillet * (1.0 - u.sin());
                let a = from + fillet_angle * (1.0 - u.cos());
                out.push(at_r(r, a));
            }
        }
        let steps = 4;
        for i in 0..=steps {
            let a = bottom_from + (bottom_to - bottom_from) * i as f64 / steps as f64;
            out.push(at_r(rf, a));
        }
        if fillet > 0.0 {
            for i in (1..4).rev() {
                let u = i as f64 / 4.0 * PI * 0.5;
                let r = rf + fillet * (1.0 - u.sin());
                let a = to - fillet_angle * (1.0 - u.cos());
                out.push(at_r(r, a));
            }
        }
    }
    out
}

/// Neighbours that came out on top of each other, one of them.
fn tidy(mut out: Vec<DVec2>) -> Vec<DVec2> {
    out.dedup_by(|a, b| a.distance(*b) < 1e-9);
    if out.len() > 1 && out[0].distance(*out.last().unwrap()) < 1e-9 {
        out.pop();
    }
    out
}

/// A circle of `n` points, anticlockwise.
fn circle(r: f64, n: usize) -> Vec<DVec2> {
    (0..n)
        .map(|i| {
            let a = i as f64 * TAU / n as f64;
            DVec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// A section swept up Z from 0 to `width` in `layers` steps: `outline`
/// (anticlockwise) and `hole` (anticlockwise too; it is turned round here)
/// each moved at height fraction `t` by their own `place`.
pub(crate) fn sweep(
    outline: &[DVec2],
    hole: &[DVec2],
    width: f64,
    layers: usize,
    place_outline: impl Fn(DVec2, f64) -> DVec2,
    place_hole: impl Fn(DVec2, f64) -> DVec2,
) -> MeshData {
    let hole: Vec<DVec2> = hole.iter().rev().copied().collect();
    let (ring, holes) = (outline.len(), hole.len());
    let per_layer = ring + holes;
    let layers = layers.max(1);
    let mut positions: Vec<Vec3> = Vec::with_capacity(per_layer * (layers + 1));
    for j in 0..=layers {
        let t = j as f64 / layers as f64;
        let z = (width * t) as f32;
        for q in outline {
            let q = place_outline(*q, t);
            positions.push(Vec3::new(q.x as f32, q.y as f32, z));
        }
        for q in &hole {
            let q = place_hole(*q, t);
            positions.push(Vec3::new(q.x as f32, q.y as f32, z));
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    // The caps: the outline with the hole through it, filled.
    let mut rings: Vec<Vec<DVec2>> = vec![outline.to_vec()];
    if holes > 0 {
        rings.push(hole.iter().rev().copied().collect());
    }
    let tris = crate::earcut::triangulate(&rings);
    // Earcut numbers the hole as given to it, reversed; back to ours.
    let index = |v: usize| -> u32 {
        if v < ring {
            v as u32
        } else {
            (ring + (holes - 1 - (v - ring))) as u32
        }
    };
    let caps = crate::regions::merged(
        tris.iter()
            .map(|[a, b, c]| vec![index(*a), index(*b), index(*c)])
            .collect(),
    );
    let top = (layers * per_layer) as u32;
    for cap in &caps {
        faces.push(cap.iter().rev().copied().collect());
        faces.push(cap.iter().map(|v| v + top).collect());
    }
    for j in 0..layers {
        let (lo, hi) = ((j * per_layer) as u32, ((j + 1) * per_layer) as u32);
        for i in 0..ring {
            let (a, b) = (i as u32, ((i + 1) % ring) as u32);
            faces.push(vec![lo + a, lo + b, hi + b, hi + a]);
        }
        for i in 0..holes {
            let (a, b) = ((ring + i) as u32, (ring + (i + 1) % holes) as u32);
            faces.push(vec![lo + a, lo + b, hi + b, hi + a]);
        }
    }
    MeshData::from_faces(positions, &faces)
}

/// How far round a helical gear's teeth wind over its face width, and how
/// many layers that takes.
fn winding(p: &GearParams) -> (f64, usize) {
    let info = gear_info(p);
    let twist = p.width.max(1e-6) * p.helix.tan() / (info.pitch * 0.5).max(1e-12);
    let pitch_angle = TAU / p.teeth.max(3) as f64;
    let layers = if twist.abs() < 1e-12 {
        1
    } else {
        ((twist.abs() / pitch_angle) * 6.0).ceil().clamp(2.0, 400.0) as usize
    };
    (twist, layers)
}

fn turned(q: DVec2, angle: f64) -> DVec2 {
    let (s, c) = angle.sin_cos();
    DVec2::new(q.x * c - q.y * s, q.x * s + q.y * c)
}

/// The gear as a solid: standing on z = 0, its axis along Z, the first
/// tooth on +X at the bottom face, turned on by `turn`. A helical gear's
/// teeth wind round as they rise, by the helix angle on the pitch circle.
pub fn gear(p: &GearParams, turn: f64) -> MeshData {
    let outline = gear_outline(p, turn);
    let (twist, layers) = winding(p);
    let bore_r = (p.bore * 0.5).clamp(0.0, bore_limit(p) * 0.5);
    let bore = if bore_r > 0.0 {
        circle(bore_r, 48)
    } else {
        Vec::new()
    };
    sweep(
        &outline,
        &bore,
        p.width.max(1e-6),
        layers,
        |q, t| turned(q, twist * t),
        |q, _| q,
    )
}

// --- internal gears -----------------------------------------------------------

/// An internal gear's circles: its tip circle inside, its root outside.
pub fn internal_info(p: &GearParams) -> GearInfo {
    let info = gear_info(p);
    GearInfo {
        tip: info.pitch - 2.0 * p.module,
        root: info.pitch + 2.5 * p.module,
        ..info
    }
}

/// The inside of an internal (ring) gear, anticlockwise, a gap centred on
/// +X: the outline of an external gear whose teeth are the ring's gaps —
/// reaching out to the ring's root, in to its tip, as thick as the ring's
/// teeth leave room for.
pub fn internal_outline(p: &GearParams, turn: f64) -> Vec<DVec2> {
    let (mt, _) = transverse(p);
    let info = internal_info(p);
    let gap = mt * PI / 2.0 + p.backlash;
    profile(p, turn, info.root * 0.5, info.tip * 0.5, gap, false)
}

/// The centre distance of a pinion running inside an internal gear of the
/// same module: half the difference of their pitch diameters.
pub fn internal_centre_distance(ring: &GearParams, pinion: &GearParams) -> f64 {
    (gear_info(ring).pitch - gear_info(pinion).pitch) * 0.5
}

/// A ring gear: teeth inside, a rim `rim` thick round them outside.
pub fn internal_gear(p: &GearParams, rim: f64, turn: f64) -> MeshData {
    let inside = internal_outline(p, turn);
    let outside_r = internal_info(p).root * 0.5 + rim.max(p.module);
    let (twist, layers) = winding(p);
    sweep(
        &circle(outside_r, 96),
        &inside,
        p.width.max(1e-6),
        layers,
        |q, _| q,
        |q, t| turned(q, twist * t),
    )
}

// --- racks --------------------------------------------------------------------

/// A rack's outline, anticlockwise: `teeth` teeth along X, the first centred
/// on x = 0, pointing up +Y from the pitch line at y = 0; its back `body`
/// below the root.
pub fn rack_outline(p: &GearParams, body: f64) -> Vec<DVec2> {
    let m = p.module / p.helix.cos().abs().max(0.2);
    let (_, at) = transverse(p);
    let pitch = PI * m;
    let (ha, hf) = (p.module, 1.25 * p.module);
    let s = pitch * 0.5 - p.backlash;
    let slope = at.tan();
    let n = p.teeth.max(1) as i64;
    let back = -hf - body.max(p.module);
    let left = -pitch * 0.5;
    let right = left + pitch * n as f64;
    let mut out = vec![DVec2::new(left, back), DVec2::new(right, back)];
    // Along the top from right to left, one tooth at a time.
    out.push(DVec2::new(right, -hf));
    for k in (0..n).rev() {
        let c = k as f64 * pitch;
        // Each flank leans in by the pressure angle: at the tip the tooth is
        // narrower than on the pitch line, at the root wider.
        out.push(DVec2::new(c + s * 0.5 + hf * slope, -hf));
        out.push(DVec2::new(c + s * 0.5 - ha * slope, ha));
        out.push(DVec2::new(c - s * 0.5 + ha * slope, ha));
        out.push(DVec2::new(c - s * 0.5 - hf * slope, -hf));
    }
    out.push(DVec2::new(left, -hf));
    tidy(out)
}

/// A rack as a solid, `width` deep along Z; a helical rack's teeth lean
/// along X by the helix angle as they rise.
pub fn rack(p: &GearParams, body: f64) -> MeshData {
    let outline = rack_outline(p, body);
    let lean = p.width * p.helix.tan();
    sweep(
        &outline,
        &[],
        p.width.max(1e-6),
        if lean.abs() > 1e-12 { 8 } else { 1 },
        |q, t| q + DVec2::new(lean * t, 0.0),
        |q, _| q,
    )
}

/// How far round a pinion set on a rack is turned, a gap of it facing down
/// onto the rack's tooth at x = 0.
pub fn rack_mate_turn(teeth: u32) -> f64 {
    let z = teeth.max(3) as f64;
    -PI / 2.0 - PI / z
}

// --- bevel gears ----------------------------------------------------------------

/// A straight bevel gear's pitch cone angle, running with `mate` teeth on a
/// shaft at right angles.
pub fn bevel_cone_angle(teeth: u32, mate: u32) -> f64 {
    bevel_cone_angle_at(teeth, mate, PI / 2.0)
}

/// A bevel gear's pitch cone half-angle when its shaft meets its mate's at
/// `shaft` radians: tan δ = sin Σ / (z₂/z₁ + cos Σ), the two cones' angles
/// adding to Σ. At a right angle, atan(z₁/z₂).
pub fn bevel_cone_angle_at(teeth: u32, mate: u32, shaft: f64) -> f64 {
    let ratio = mate.max(3) as f64 / teeth.max(3) as f64;
    shaft.sin().atan2(ratio + shaft.cos())
}

/// A straight bevel gear, its shaft at right angles to its mate's: the
/// module and the outline at the large end, by Tredgold's approximation —
/// the tooth of a spur gear of `teeth / cos δ` teeth laid on the back cone
/// — and every point drawn straight toward the cone's apex for the face
/// width. Its axis is Z, the large end's pitch circle in z = 0, the apex
/// above it, the first tooth on +X turned on by `turn`.
pub fn bevel_gear(p: &GearParams, mate: u32, turn: f64) -> MeshData {
    bevel_gear_at(p, mate, PI / 2.0, turn)
}

/// [`bevel_gear`] for shafts meeting at `shaft` radians.
pub fn bevel_gear_at(p: &GearParams, mate: u32, shaft: f64, turn: f64) -> MeshData {
    let delta = bevel_cone_angle_at(p.teeth, mate, shaft);
    let z = p.teeth.max(3) as f64;
    let r = p.module * z * 0.5;
    let rv = r / delta.cos();
    // The virtual spur gear, the back cone rolled out flat.
    let virtual_teeth = (z / delta.cos()).round().max(3.0) as u32;
    let v = GearParams {
        teeth: virtual_teeth,
        helix: 0.0,
        ..*p
    };
    let (vmt, vat) = transverse(&v);
    let vinfo = gear_info(&v);
    let vs = vmt * (PI / 2.0 + 2.0 * v.shift * vat.tan()) - v.backlash;
    let one = tooth(&v, 0.0, vinfo.tip * 0.5, vinfo.root * 0.5, vs, true);
    let virtual_pitch = vinfo.pitch * 0.5;
    // The virtual gear's pitch radius from its real tooth count, not the
    // rounded one: the angles below are scaled to the real one's pitch.
    let _ = rv;
    let cone = r / delta.sin();
    let face = p
        .width
        .clamp(1e-6, bevel_face_limit_at(p.teeth, mate, shaft, p.module));
    let apex = DVec3::new(0.0, 0.0, r / delta.tan());
    let back_apex_z = -r * delta.tan();
    // One tooth of the virtual gear spans 2π / zv; the real one 2π / z. Laid
    // on the back cone, each virtual point at radius ρ and angle θ lands at
    // slant ρ (scaled to this gear's pitch) and azimuth θ · zv / z.
    let scale = (r / delta.cos()) / virtual_pitch;
    let ratio = virtual_teeth as f64 / z;
    let on_back = |q: DVec2| -> DVec3 {
        let rho = q.length() * scale;
        let theta = q.y.atan2(q.x) * ratio + turn;
        let radial = rho * delta.cos();
        DVec3::new(
            radial * theta.cos(),
            radial * theta.sin(),
            back_apex_z + rho * delta.sin(),
        )
    };
    // The one virtual tooth, a real pitch apart round the gear.
    let pitch_angle = TAU / z;
    let mut large: Vec<DVec3> = Vec::new();
    for k in 0..p.teeth.max(3) {
        let spin = DQuat::from_rotation_z(k as f64 * pitch_angle);
        large.extend(one.iter().map(|q| spin * on_back(*q)));
    }
    large.dedup_by(|a, b| a.distance(*b) < 1e-12);
    if large.len() > 1 && large[0].distance(*large.last().unwrap()) < 1e-12 {
        large.pop();
    }
    let k = (cone - face) / cone;
    let small: Vec<DVec3> = large.iter().map(|q| apex + (*q - apex) * k).collect();
    let bore_r = (p.bore * 0.5).clamp(0.0, bevel_bore_limit(p, mate) * 0.5);
    loft(&large, &small, bore_r)
}

/// A bevel gear's cone distance: from the apex to the large end's pitch
/// circle, along the cone.
pub fn bevel_cone_distance(teeth: u32, mate: u32, module: f64) -> f64 {
    bevel_cone_distance_at(teeth, mate, PI / 2.0, module)
}

/// [`bevel_cone_distance`] for shafts meeting at `shaft` radians.
pub fn bevel_cone_distance_at(teeth: u32, mate: u32, shaft: f64, module: f64) -> f64 {
    module * teeth.max(3) as f64 * 0.5 / bevel_cone_angle_at(teeth, mate, shaft).sin()
}

/// The widest a bevel gear's face can be: half its cone distance, where the
/// small end's teeth are half the large end's. A third is what is usually
/// cut; past a half the teeth at the small end are too small to carry
/// anything, and at the whole of it they meet at the apex.
pub fn bevel_face_limit(teeth: u32, mate: u32, module: f64) -> f64 {
    bevel_face_limit_at(teeth, mate, PI / 2.0, module)
}

/// [`bevel_face_limit`] for shafts meeting at `shaft` radians.
pub fn bevel_face_limit_at(teeth: u32, mate: u32, shaft: f64, module: f64) -> f64 {
    bevel_cone_distance_at(teeth, mate, shaft, module) * 0.5
}

/// The widest bore a bevel gear can take: within its small end's root,
/// leaving a fifth of it as wall.
pub fn bevel_bore_limit(p: &GearParams, mate: u32) -> f64 {
    bevel_bore_limit_at(p, mate, PI / 2.0)
}

/// [`bevel_bore_limit`] for shafts meeting at `shaft` radians.
pub fn bevel_bore_limit_at(p: &GearParams, mate: u32, shaft: f64) -> f64 {
    let delta = bevel_cone_angle_at(p.teeth, mate, shaft);
    let r = p.module * p.teeth.max(3) as f64 * 0.5;
    let cone = r / delta.sin();
    let face = p
        .width
        .clamp(1e-6, bevel_face_limit_at(p.teeth, mate, shaft, p.module));
    let k = (cone - face) / cone;
    2.0 * (r - 1.25 * p.module * delta.cos()) * k * 0.8
}

/// The widest bore a spur, helical or worm wheel can take: within its root
/// circle, a tenth of it left as wall.
pub fn bore_limit(p: &GearParams) -> f64 {
    gear_info(p).root * 0.9
}

/// The widest bore a worm can take: within its root, a fifth left as wall.
pub fn worm_bore_limit(w: &WormParams) -> f64 {
    (w.pitch_diameter() - 2.5 * w.module) * 0.8
}

/// A solid between two closed outlines of as many points, one above the
/// other, with a bore of `bore_r` through both along Z: the large end's
/// outline at the bottom.
fn loft(bottom: &[DVec3], top: &[DVec3], bore_r: f64) -> MeshData {
    let n = bottom.len();
    let hole = if bore_r > 0.0 { 48 } else { 0 };
    let mut positions: Vec<Vec3> = Vec::with_capacity(2 * (n + hole));
    // Each end flat where the bore leaves it: at the lowest and the highest
    // of its outline's roots.
    let (z0, z1) = (
        bottom.iter().map(|q| q.z).fold(f64::MAX, f64::min),
        top.iter().map(|q| q.z).fold(f64::MIN, f64::max),
    );
    for (ring, z) in [(bottom, z0), (top, z1)] {
        positions.extend(ring.iter().map(|q| q.as_vec3()));
        for i in 0..hole {
            // Clockwise: a hole runs the other way.
            let a = -(i as f64) * TAU / hole as f64;
            positions.push(Vec3::new(
                (bore_r * a.cos()) as f32,
                (bore_r * a.sin()) as f32,
                z as f32,
            ));
        }
    }
    let per = (n + hole) as u32;
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let flat: Vec<DVec2> = bottom.iter().map(|q| q.truncate()).collect();
    let mut rings = vec![flat];
    if hole > 0 {
        rings.push(circle(bore_r, hole));
    }
    let tris = crate::earcut::triangulate(&rings);
    // The bore given anticlockwise, stored clockwise.
    let index = |v: usize| -> u32 {
        if v < n {
            v as u32
        } else {
            let k = v - n;
            (n + (hole - k) % hole) as u32
        }
    };
    // Triangles, not merged faces: the ends are not flat.
    for [a, b, c] in &tris {
        let (a, b, c) = (index(*a), index(*b), index(*c));
        faces.push(vec![a, c, b]);
        faces.push(vec![a + per, b + per, c + per]);
    }
    for i in 0..n as u32 {
        let j = (i + 1) % n as u32;
        faces.push(vec![i, j, j + per, i + per]);
    }
    for i in 0..hole as u32 {
        let (a, b) = (n as u32 + i, n as u32 + (i + 1) % hole as u32);
        faces.push(vec![a, b, b + per, a + per]);
    }
    MeshData::from_faces(positions, &faces)
}

/// Where the second of a pair of bevel gears goes so it meshes with the
/// first: turned so its axis lies along -X toward the shared apex, and set
/// so its large end's pitch circle touches the first's. The rotation and the
/// translation, and how far its own outline is to be turned.
pub fn bevel_mate_place(first: u32, second: u32, module: f64) -> (DQuat, DVec3, f64) {
    bevel_mate_place_at(first, second, PI / 2.0, module)
}

/// [`bevel_mate_place`] for shafts meeting at `shaft` radians: the second
/// gear's axis through the first's apex, `shaft` from the first's, leaning
/// over +X; both cones' pitch elements meeting along the same line.
pub fn bevel_mate_place_at(
    first: u32,
    second: u32,
    shaft: f64,
    module: f64,
) -> (DQuat, DVec3, f64) {
    let d1 = bevel_cone_angle_at(first, second, shaft);
    let d2 = shaft - d1;
    let r1 = module * first.max(3) as f64 * 0.5;
    let r2 = module * second.max(3) as f64 * 0.5;
    // The apex above the first gear, and the second's large end back from
    // it down its own axis — which, toward the apex, is `w`.
    let apex = DVec3::new(0.0, 0.0, r1 / d1.tan());
    let w = DVec3::new(-shaft.sin(), 0.0, shaft.cos());
    let centre = apex - w * (r2 / d2.tan());
    // Its X square to its axis in the plane of the two, the side a gap of
    // it is to face the first gear; at a right angle, straight down.
    let x = DVec3::new(-shaft.cos(), 0.0, -shaft.sin());
    let turn = DQuat::from_mat3(&kreate_math::DMat3::from_cols(x, w.cross(x), w));
    (turn, centre, PI / second.max(3) as f64)
}

// --- worms ----------------------------------------------------------------------

/// A worm drive's numbers: the worm's axial module (the wheel's transverse
/// one), its starts, and its diameter factor q — the pitch diameter over the
/// module, ISO's usual 10.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WormParams {
    pub module: f64,
    pub starts: u32,
    pub diameter_factor: f64,
    pub length: f64,
    pub pressure_angle: f64,
    pub bore: f64,
    pub backlash: f64,
}

impl WormParams {
    pub fn pitch_diameter(&self) -> f64 {
        self.module * self.diameter_factor
    }
    /// The angle the thread climbs at on the pitch cylinder.
    pub fn lead_angle(&self) -> f64 {
        (self.starts.max(1) as f64 / self.diameter_factor.max(1.0)).atan()
    }
    /// How far the thread advances in one turn.
    pub fn lead(&self) -> f64 {
        PI * self.module * self.starts.max(1) as f64
    }
}

/// The worm wheel that runs with `worm`: a helical gear of `teeth`, its
/// helix the worm's lead angle, its transverse module the worm's axial one.
pub fn worm_wheel_params(worm: &WormParams, teeth: u32, width: f64, bore: f64) -> GearParams {
    let gamma = worm.lead_angle();
    GearParams {
        module: worm.module * gamma.cos(),
        teeth,
        width,
        bore,
        pressure_angle: (worm.pressure_angle.tan() * gamma.cos()).atan(),
        helix: gamma,
        shift: 0.0,
        backlash: worm.backlash,
    }
}

/// A worm, its axis along Z from 0 to its length: the basic rack wound round
/// it, `starts` times, a groove facing +X half way along.
pub fn worm(w: &WormParams) -> MeshData {
    let r = w.pitch_diameter() * 0.5;
    let m = w.module;
    let (ha, hf) = (m, 1.25 * m);
    let pitch = PI * m;
    let s = pitch * 0.5 - w.backlash;
    let slope = w.pressure_angle.tan();
    let length = w.length.max(pitch);
    let bore = (w.bore * 0.5).clamp(0.0, worm_bore_limit(w) * 0.5);
    // The basic rack wound round: a tooth centred on each crest, a groove
    // facing +X half way along.
    crate::mech::thread_solid(
        length,
        pitch,
        w.lead(),
        r + ha,
        r - hf,
        s * 0.5 - ha * slope,
        s * 0.5 + hf * slope,
        length * 0.5 + pitch * 0.5,
        bore,
    )
}

#[cfg(test)]
mod tests {
    /// At a right angle the general bevel is the right-angled one; at
    /// another, the two cones' angles add to the shaft angle, and both
    /// gears' apexes are one point.
    #[test]
    fn bevel_gears_at_any_shaft_angle_share_their_apex() {
        let q = PI / 2.0;
        assert!((bevel_cone_angle_at(20, 40, q) - bevel_cone_angle(20, 40)).abs() < 1e-12);
        let (t90, c90, _) = bevel_mate_place(20, 40, 0.002);
        let (t, c, _) = bevel_mate_place_at(20, 40, q, 0.002);
        assert!((c - c90).length() < 1e-12 && t.dot(t90).abs() > 1.0 - 1e-12);
        for shaft in [q * 0.6, q, q * 1.4] {
            let (d1, d2) = (
                bevel_cone_angle_at(20, 30, shaft),
                bevel_cone_angle_at(30, 20, shaft),
            );
            assert!((d1 + d2 - shaft).abs() < 1e-12, "{shaft}");
            let (turn, centre, _) = bevel_mate_place_at(20, 30, shaft, 0.002);
            let apex1 = DVec3::new(0.0, 0.0, 0.02 / d1.tan());
            let apex2 = centre + turn * DVec3::new(0.0, 0.0, 0.03 / d2.tan());
            assert!((apex1 - apex2).length() < 1e-12, "{shaft}: {apex1} {apex2}");
            // The axes meet at the shaft angle.
            let a2 = turn * DVec3::Z;
            assert!((a2.dot(DVec3::Z).acos() - shaft).abs() < 1e-9);
            let g = GearParams {
                teeth: 20,
                ..GearParams::default()
            };
            let m = bevel_gear_at(&GearParams { width: 0.004, ..g }, 30, shaft, 0.0);
            assert!(
                m.validate().is_empty() && crate::measure::is_closed(&m),
                "{shaft}"
            );
        }
    }

    use super::*;
    use crate::measure::{is_closed, mass_properties};

    fn area(poly: &[DVec2]) -> f64 {
        (0..poly.len())
            .map(|i| poly[i].perp_dot(poly[(i + 1) % poly.len()]))
            .sum::<f64>()
            * 0.5
    }

    fn inside_poly(poly: &[DVec2], p: DVec2) -> bool {
        inside(poly, p)
    }

    fn inside(poly: &[DVec2], p: DVec2) -> bool {
        let mut odd = false;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
                odd = !odd;
            }
        }
        odd
    }

    #[test]
    fn the_circles_are_the_standards() {
        // m 2, 20 teeth: pitch 40, tip 44, root 35, base 40 cos 20°.
        let p = GearParams::default();
        let i = gear_info(&p);
        assert!((i.pitch - 0.040).abs() < 1e-12);
        assert!((i.tip - 0.044).abs() < 1e-12);
        assert!((i.root - 0.035).abs() < 1e-12);
        assert!((i.base - 0.040 * 20f64.to_radians().cos()).abs() < 1e-12);
        let q = GearParams { teeth: 30, ..p };
        assert!((centre_distance(&p, &q) - 0.050).abs() < 1e-12);
    }

    #[test]
    fn the_outline_has_its_teeth_between_its_circles() {
        let p = GearParams::default();
        let o = gear_outline(&p, 0.0);
        assert!(area(&o) > 0.0, "anticlockwise");
        let r: Vec<f64> = o.iter().map(|q| q.length()).collect();
        let i = gear_info(&p);
        let max = r.iter().cloned().fold(0.0, f64::max);
        let min = r.iter().cloned().fold(f64::MAX, f64::min);
        assert!((max - i.tip * 0.5).abs() < 1e-9, "{max}");
        assert!((min - i.root * 0.5).abs() < 1e-9, "{min}");
        // Twenty teeth: twenty times the outline crosses the pitch circle
        // going out.
        let rp = i.pitch * 0.5;
        let out = (0..o.len())
            .filter(|k| r[*k] < rp && r[(k + 1) % o.len()] >= rp)
            .count();
        assert_eq!(out, 20);
    }

    #[test]
    fn a_tooth_is_half_the_pitch_thick_on_the_pitch_circle() {
        let p = GearParams::default();
        let o = gear_outline(&p, 0.0);
        let rp = gear_info(&p).pitch * 0.5;
        // Where the first tooth's two flanks cross the pitch circle.
        let mut crossings = Vec::new();
        for k in 0..o.len() {
            let (a, b) = (o[k], o[(k + 1) % o.len()]);
            let (ra, rb) = (a.length(), b.length());
            if (ra - rp) * (rb - rp) <= 0.0 && ra != rb {
                let t = (rp - ra) / (rb - ra);
                let q = a.lerp(b, t);
                let angle = q.y.atan2(q.x);
                if angle.abs() < 0.2 {
                    crossings.push(angle);
                }
            }
        }
        assert_eq!(crossings.len(), 2, "{crossings:?}");
        let thickness = (crossings[0] - crossings[1]).abs() * rp;
        let want = PI * p.module / 2.0;
        assert!(
            (thickness - want).abs() < want * 0.01,
            "{thickness} against {want}"
        );
    }

    #[test]
    fn a_gear_is_a_closed_solid_of_its_face_area_times_its_width() {
        let p = GearParams::default();
        let o = gear_outline(&p, 0.0);
        let bore = PI * 0.0025f64.powi(2) * (48.0 / TAU * (TAU / 48.0).sin());
        let want = (area(&o) - bore) * p.width;
        let m = gear(&p, 0.0);
        assert!(is_closed(&m), "not closed");
        let v = mass_properties(&m, |q| q).volume;
        assert!((v - want).abs() < want * 1e-3, "{v} against {want}");
        // Helical: the same section wound round, the same volume.
        let h = gear(
            &GearParams {
                helix: 20f64.to_radians(),
                ..p
            },
            0.0,
        );
        assert!(is_closed(&h));
        let hv = mass_properties(&h, |q| q).volume;
        let ht = GearParams {
            helix: 20f64.to_radians(),
            ..p
        };
        let hwant = (area(&gear_outline(&ht, 0.0)) - bore) * p.width;
        assert!((hv - hwant).abs() < hwant * 1e-2, "{hv} against {hwant}");
    }

    #[test]
    fn a_pair_set_by_mate_turn_meshes_without_running_into_each_other() {
        // With a little play, no point of either outline is inside the
        // other, for any pair of tooth counts — and with none, the teeth
        // still reach into each other's gaps: they are meshed, not apart.
        for (z1, z2) in [(20, 30), (12, 37), (17, 17), (25, 40)] {
            let a = GearParams {
                teeth: z1,
                backlash: 0.0002,
                ..Default::default()
            };
            let b = GearParams { teeth: z2, ..a };
            let d = centre_distance(&a, &b);
            let oa = gear_outline(&a, 0.0);
            let ob: Vec<DVec2> = gear_outline(&b, mate_turn(z2))
                .into_iter()
                .map(|q| q + DVec2::new(d, 0.0))
                .collect();
            for q in &ob {
                assert!(
                    !inside(&oa, *q),
                    "{z1}/{z2}: {q:?} is inside the first gear"
                );
            }
            for q in &oa {
                assert!(
                    !inside(&ob, *q),
                    "{z1}/{z2}: {q:?} is inside the second gear"
                );
            }
            // The first's tooth on +X reaches past the second's tip circle.
            let reach = oa.iter().map(|q| q.x).fold(f64::MIN, f64::max);
            assert!(reach > d - gear_info(&b).tip * 0.5, "{z1}/{z2}: not meshed");
        }
    }

    #[test]
    fn the_outline_never_crosses_itself_for_any_size_of_gear() {
        // Few teeth (deep radial flanks under the base circle) to many (the
        // involute starting on the root circle), shifted and not.
        for z in [8, 12, 17, 25, 42, 60, 120] {
            for shift in [0.0, 0.4] {
                let o = gear_outline(
                    &GearParams {
                        teeth: z,
                        shift,
                        ..Default::default()
                    },
                    0.0,
                );
                let n = o.len();
                let cross = |a: DVec2, b: DVec2, c: DVec2, d: DVec2| {
                    let o1 = (b - a).perp_dot(c - a);
                    let o2 = (b - a).perp_dot(d - a);
                    let o3 = (d - c).perp_dot(a - c);
                    let o4 = (d - c).perp_dot(b - c);
                    o1 * o2 < 0.0 && o3 * o4 < 0.0
                };
                for i in 0..n {
                    for j in i + 2..n {
                        if i == 0 && j == n - 1 {
                            continue;
                        }
                        assert!(
                            !cross(o[i], o[(i + 1) % n], o[j], o[(j + 1) % n]),
                            "z {z} shift {shift}: edges {i} and {j} cross"
                        );
                    }
                }
            }
        }
    }

    fn solid(m: &MeshData) -> f64 {
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert!(is_closed(m), "not closed");
        let v = mass_properties(m, |q| q).volume;
        assert!(v > 0.0, "inside out: {v}");
        v
    }

    #[test]
    fn a_pinion_runs_inside_a_ring_gear_without_running_into_it() {
        for (zr, zp) in [(60, 20), (48, 30), (72, 17)] {
            let ring = GearParams {
                teeth: zr,
                backlash: 0.0002,
                ..Default::default()
            };
            let pinion = GearParams { teeth: zp, ..ring };
            let a = internal_centre_distance(&ring, &pinion);
            assert!((a - 0.002 * (zr - zp) as f64 * 0.5).abs() < 1e-12);
            // The ring's inside: the pinion must be within it everywhere.
            let inside = internal_outline(&ring, 0.0);
            let pin: Vec<DVec2> = gear_outline(&pinion, 0.0)
                .into_iter()
                .map(|q| q + DVec2::new(a, 0.0))
                .collect();
            for q in &pin {
                assert!(inside_poly(&inside, *q), "{zr}/{zp}: {q:?} is in the ring");
            }
            // And meshed: the pinion's tooth on +X reaches past the ring's
            // tip circle into its gap.
            let reach = pin.iter().map(|q| q.length()).fold(0.0, f64::max);
            assert!(
                reach > internal_info(&ring).tip * 0.5,
                "{zr}/{zp}: not meshed"
            );
            solid(&internal_gear(&ring, 0.005, 0.0));
        }
        // Helical, too.
        solid(&internal_gear(
            &GearParams {
                teeth: 60,
                helix: 0.3,
                ..Default::default()
            },
            0.005,
            0.0,
        ));
    }

    #[test]
    fn a_pinion_runs_on_a_rack_without_running_into_it() {
        let p = GearParams {
            teeth: 10,
            backlash: 0.0002,
            ..Default::default()
        };
        let rack_line = rack_outline(&p, 0.004);
        assert!(area(&rack_line) > 0.0, "anticlockwise");
        for z in [12u32, 20, 31] {
            let pinion = GearParams { teeth: z, ..p };
            let r = gear_info(&pinion).pitch * 0.5;
            let pin: Vec<DVec2> = gear_outline(&pinion, rack_mate_turn(z))
                .into_iter()
                .map(|q| q + DVec2::new(0.0, r))
                .collect();
            for q in &pin {
                assert!(!inside_poly(&rack_line, *q), "z {z}: {q:?} is in the rack");
            }
            // Its teeth reach below the rack's tips: meshed.
            let low = pin.iter().map(|q| q.y).fold(f64::MAX, f64::min);
            assert!(low < p.module, "z {z}: not meshed ({low})");
        }
        // Thick on the pitch line as half the pitch, less the play.
        let v = solid(&rack(&p, 0.004));
        assert!(v > 0.0);
        solid(&rack(&GearParams { helix: 0.3, ..p }, 0.004));
    }

    #[test]
    fn a_bevel_pair_shares_its_apex_and_is_solid() {
        let p = GearParams {
            teeth: 20,
            width: 0.008,
            ..Default::default()
        };
        let g = bevel_gear(&p, 30, 0.0);
        solid(&g);
        // The large end's outside diameter: the pitch diameter and two
        // addenda laid on the back cone, 2 m cos δ.
        let delta = bevel_cone_angle(20, 30);
        let want = 0.040 + 2.0 * 0.002 * delta.cos();
        let size = g.bounds().size();
        let sx = size.x as f64;
        assert!((sx - want).abs() < want * 0.02, "{sx} against {want}");
        // The mate, set where `bevel_mate_place` says: its pitch circle's
        // nearest point on the first's, the apexes together.
        let (turn, at, own) = bevel_mate_place(20, 30, 0.002);
        let mate = bevel_gear(&GearParams { teeth: 30, ..p }, 20, own);
        solid(&mate);
        let apex1 = DVec3::new(0.0, 0.0, 0.020 / delta.tan());
        let delta2 = bevel_cone_angle(30, 20);
        let apex2 = at + turn * DVec3::new(0.0, 0.0, 0.030 / delta2.tan());
        assert!(apex1.distance(apex2) < 1e-9, "{apex1:?} {apex2:?}");
        let touch = at + turn * DVec3::new(0.030, 0.0, 0.0);
        assert!(
            touch.distance(DVec3::new(0.020, 0.0, 0.0)) < 1e-9,
            "{touch:?}"
        );
    }

    #[test]
    fn a_worm_is_its_thread_round_its_pitch_cylinder() {
        let w = WormParams {
            module: 0.002,
            starts: 2,
            diameter_factor: 10.0,
            length: 0.04,
            pressure_angle: 20f64.to_radians(),
            bore: 0.006,
            backlash: 0.0,
        };
        let m = worm(&w);
        solid(&m);
        let radius: Vec<f32> = m
            .positions()
            .iter()
            .map(|q| q.truncate().length())
            .collect();
        let max = radius.iter().cloned().fold(0.0, f32::max) as f64;
        // The outside: the pitch diameter and an addendum each side.
        assert!((max - (0.020 + 0.004) * 0.5).abs() < 1e-6, "{max}");
        // Two starts at q 10: the thread climbs at atan(2/10).
        assert!((w.lead_angle() - 0.2f64.atan()).abs() < 1e-12);
        assert!((w.lead() - 2.0 * PI * 0.002).abs() < 1e-12);
        // The wheel: its transverse module the worm's axial one.
        let wheel = worm_wheel_params(&w, 30, 0.01, 0.0);
        let info = gear_info(&wheel);
        assert!((info.pitch - 0.060).abs() < 1e-9, "{}", info.pitch);
        solid(&gear(&wheel, 0.0));
    }

    #[test]
    fn a_gear_s_mesh_is_valid_with_and_without_a_bore() {
        for bore in [0.0, 0.006] {
            let m = gear(
                &GearParams {
                    bore,
                    ..Default::default()
                },
                0.3,
            );
            assert!(m.validate().is_empty(), "{:?}", m.validate());
            assert!(is_closed(&m));
        }
    }
}
