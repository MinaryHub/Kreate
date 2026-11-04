//! A photographic studio to light a model in: the light a scene with no
//! lamps of its own is shown under, and the material preview's.
//!
//! An environment, not a handful of lamps, because a lamp lights a surface
//! but is nowhere to be seen in it: a metal under three lamps and a flat sky
//! reflects a flat grey, and reads as plastic. The studio is what a product
//! photographer builds — a cool, bright overhead falling to a darker floor,
//! one large softbox for the key, a strip for the rim and a dimmer fill —
//! so a surface's shape reads from its shading and a metal from what it
//! reflects. Made here rather than shipped as an image: a few kilobytes of
//! arithmetic, no licence, and the same picture on every machine.
//!
//! Kreate is Z-up, and the studio is laid out for a viewer standing on the
//! `-Y` side looking towards `+Y`, the default camera's view: the key is
//! over that viewer's left shoulder. [`key_direction`] is where it is, for
//! the shadow-casting lamp that goes with it.

use glam::Vec3;

/// The image's size: twice as wide as tall, as an equirectangular map is.
pub const WIDTH: u32 = 512;
pub const HEIGHT: u32 = 256;

/// One softbox: where it is, how large and how bright.
struct Softbox {
    /// Towards its middle, from the subject.
    towards: Vec3,
    /// Half its width and height, as tangents of the angle.
    half: (f32, f32),
    radiance: [f32; 3],
}

/// The rooms a part can be shown in, as Fusion's environment library offers
/// them: what lights it, what its metal reflects, and what is seen behind it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    /// A product photographer's room: a large key, a fill, a rim, and strips
    /// above and to the sides for a metal to show its shape in.
    #[default]
    Studio,
    /// The same room brighter and whiter, for a part to read on paper white.
    Bright,
    /// An even grey room with one soft overhead: the flattest light, for
    /// judging a shape rather than showing it off.
    GreyRoom,
    /// A clear midday sky and its sun: hard shadows and a blue reflection.
    Outdoor,
    /// The sun low over the horizon: long shadows, warm light, a sky
    /// paling to orange where the sun is.
    Sunset,
}

impl Preset {
    pub const ALL: [Preset; 5] = [
        Preset::Studio,
        Preset::Bright,
        Preset::GreyRoom,
        Preset::Outdoor,
        Preset::Sunset,
    ];

    /// The name the menus and the settings use.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Studio => "studio",
            Preset::Bright => "bright",
            Preset::GreyRoom => "grey_room",
            Preset::Outdoor => "outdoor",
            Preset::Sunset => "sunset",
        }
    }

    /// The sky, for a room that is one: how hazy (Preetham's turbidity —
    /// two is very clear, six hazy) and the colour of the ground under it.
    fn sky(self) -> Option<(f32, Vec3)> {
        match self {
            Preset::Outdoor => Some((3.0, Vec3::new(0.2, 0.18, 0.15))),
            Preset::Sunset => Some((4.0, Vec3::new(0.12, 0.1, 0.08))),
            _ => None,
        }
    }

    pub fn from_name(name: &str) -> Option<Preset> {
        Preset::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The room: its ceiling, its horizon and its floor.
    fn room(self) -> [Vec3; 3] {
        match self {
            Preset::Studio => [
                Vec3::new(0.52, 0.54, 0.58),
                Vec3::new(0.44, 0.44, 0.45),
                Vec3::new(0.24, 0.235, 0.23),
            ],
            Preset::Bright => [
                Vec3::new(0.95, 0.96, 0.98),
                Vec3::new(0.85, 0.85, 0.86),
                Vec3::new(0.45, 0.44, 0.43),
            ],
            Preset::GreyRoom => [
                Vec3::new(0.62, 0.62, 0.63),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(0.32, 0.32, 0.32),
            ],
            // Skies are their model's, not a gradient: see `sky`. These are
            // only what a caller asking for the room's three tones gets.
            Preset::Outdoor | Preset::Sunset => [
                Vec3::new(0.18, 0.32, 0.75),
                Vec3::new(0.62, 0.7, 0.82),
                Vec3::new(0.2, 0.18, 0.15),
            ],
        }
    }

    fn softboxes(self) -> Vec<Softbox> {
        let key = Softbox {
            towards: self.key_direction(),
            half: (0.55, 0.38),
            radiance: [2.2, 2.17, 2.1],
        };
        let fill = Softbox {
            towards: Vec3::new(0.75, -0.55, 0.25).normalize(),
            half: (0.45, 0.3),
            radiance: [1.1, 1.13, 1.17],
        };
        let rim = Softbox {
            towards: Vec3::new(0.25, 0.85, 0.45).normalize(),
            half: (0.18, 0.6),
            radiance: [2.8, 2.9, 3.0],
        };
        // Long thin strips: what a polished surface shows as clean bright
        // lines along its curves, the look a product shot is lit for.
        let overhead = Softbox {
            towards: Vec3::new(0.0, -0.15, 1.0).normalize(),
            half: (0.9, 0.12),
            radiance: [3.0, 3.0, 3.05],
        };
        let side = |x: f32| Softbox {
            towards: Vec3::new(x, 0.1, 0.25).normalize(),
            half: (0.08, 0.7),
            radiance: [2.4, 2.4, 2.45],
        };
        match self {
            Preset::Studio => vec![key, fill, rim, overhead, side(1.0), side(-1.0)],
            Preset::Bright => vec![
                Softbox {
                    radiance: [3.0, 3.0, 2.95],
                    ..key
                },
                Softbox {
                    radiance: [1.8, 1.8, 1.82],
                    ..fill
                },
                rim,
                overhead,
            ],
            Preset::GreyRoom => vec![Softbox {
                towards: Vec3::Z,
                half: (1.2, 1.2),
                radiance: [1.2, 1.2, 1.2],
            }],
            // The sun: small and very bright, its colour what the air
            // leaves it.
            Preset::Outdoor | Preset::Sunset => {
                let c = self.key_color();
                let strength = if self == Preset::Sunset { 400.0 } else { 900.0 };
                vec![Softbox {
                    towards: self.key_direction(),
                    half: (0.03, 0.03),
                    radiance: [c[0] * strength, c[1] * strength, c[2] * strength],
                }]
            }
        }
    }

    /// Where the key light is, from the subject: the direction the
    /// shadow-casting lamp that goes with the room shines *from*.
    pub fn key_direction(self) -> Vec3 {
        match self {
            // To the viewer's left and a little in front, half way up: the
            // shadow falls to the right of a part, where it is seen. Behind
            // the viewer it fell behind the part, hidden by it.
            Preset::Studio | Preset::Bright => Vec3::new(-0.8, -0.3, 0.75).normalize(),
            Preset::GreyRoom => Vec3::new(-0.2, -0.2, 1.0).normalize(),
            Preset::Outdoor => Vec3::new(-0.6, -0.45, 0.65).normalize(),
            // Six degrees up, off to the viewer's left.
            Preset::Sunset => Vec3::new(-0.8, -0.59, 0.105).normalize(),
        }
    }

    /// The colour of the key light, white for a room's lamps; for a sky, the
    /// sun's after the air it came through — warmer the lower it is.
    pub fn key_color(self) -> [f32; 3] {
        match self.sky() {
            Some((turbidity, _)) => sun_color(self.key_direction(), turbidity),
            None => [1.0; 3],
        }
    }

    /// How bright the shadow-casting lamp is, in lux: the share of the
    /// room's light that casts a shadow.
    pub fn key_lux(self) -> f32 {
        match self {
            Preset::Studio => 10.0,
            Preset::Bright => 14.0,
            Preset::GreyRoom => 4.0,
            Preset::Outdoor => 40.0,
            Preset::Sunset => 12.0,
        }
    }

    /// What is seen behind a part, top of the frame to bottom, in the
    /// renderer's linear units: a backdrop rather than the room, as a product
    /// shot has a sweep of paper behind it. `None` for a room that is its own
    /// backdrop — a sky is.
    pub fn backdrop(self) -> Option<[[f32; 3]; 2]> {
        match self {
            Preset::Studio => Some([[1.05, 1.07, 1.1], [0.62, 0.62, 0.64]]),
            Preset::Bright => Some([[1.6, 1.6, 1.62], [1.15, 1.15, 1.16]]),
            Preset::GreyRoom => Some([[0.8, 0.8, 0.8], [0.5, 0.5, 0.5]]),
            Preset::Outdoor | Preset::Sunset => None,
        }
    }
}

// --- the sky ---------------------------------------------------------------------
//
// Preetham, Shirley and Smits's analytic daylight (1999): the sky's luminance
// and chromaticity across it as Perez's five-parameter function of the angle
// from the zenith and the angle from the sun, each parameter linear in the
// turbidity. Not the newest model — Hosek and Wilkie's fits measured skies
// better near the horizon at a low sun — but closed-form, a few dozen
// numbers, and right where a midday product shot needs it.

/// Perez's function for one coefficient set at `(θ, γ)`.
fn perez(c: [f32; 5], cos_theta: f32, gamma: f32) -> f32 {
    let [a, b, cc, d, e] = c;
    let cg = gamma.cos();
    (1.0 + a * (b / cos_theta.max(0.01)).exp()) * (1.0 + cc * (d * gamma).exp() + e * cg * cg)
}

/// The luminance's coefficients, and the two chromaticities', at `t`.
fn perez_coefficients(t: f32) -> [[f32; 5]; 3] {
    [
        [
            0.1787 * t - 1.4630,
            -0.3554 * t + 0.4275,
            -0.0227 * t + 5.3251,
            0.1206 * t - 2.5771,
            -0.0670 * t + 0.3703,
        ],
        [
            -0.0193 * t - 0.2592,
            -0.0665 * t + 0.0008,
            -0.0004 * t + 0.2125,
            -0.0641 * t - 0.8989,
            -0.0033 * t + 0.0452,
        ],
        [
            -0.0167 * t - 0.2608,
            -0.0950 * t + 0.0092,
            -0.0079 * t + 0.2102,
            -0.0441 * t - 1.6537,
            -0.0109 * t + 0.0529,
        ],
    ]
}

/// The zenith's luminance (kcd/m²) and chromaticity, for a sun `theta_s`
/// from the zenith.
fn zenith(theta_s: f32, t: f32) -> (f32, f32, f32) {
    let chi = (4.0 / 9.0 - t / 120.0) * (std::f32::consts::PI - 2.0 * theta_s);
    let y = ((4.0453 * t - 4.9710) * chi.tan() - 0.2155 * t + 2.4192).max(0.0);
    let (s1, s2, s3) = (theta_s, theta_s * theta_s, theta_s * theta_s * theta_s);
    let cx = t * t * (0.00166 * s3 - 0.00375 * s2 + 0.00209 * s1)
        + t * (-0.02903 * s3 + 0.06377 * s2 - 0.03202 * s1 + 0.00394)
        + (0.11693 * s3 - 0.21196 * s2 + 0.06052 * s1 + 0.25886);
    let cy = t * t * (0.00275 * s3 - 0.00610 * s2 + 0.00317 * s1)
        + t * (-0.04214 * s3 + 0.08970 * s2 - 0.04153 * s1 + 0.00516)
        + (0.15346 * s3 - 0.26756 * s2 + 0.06670 * s1 + 0.26688);
    (y, cx, cy)
}

/// CIE xyY to linear Rec. 709 — the renderer's working space.
fn xyy_to_rgb(x: f32, y: f32, lum: f32) -> Vec3 {
    let y = y.max(1e-4);
    let (cx, cz) = (x / y * lum, (1.0 - x - y) / y * lum);
    Vec3::new(
        3.2406 * cx - 1.5372 * lum - 0.4986 * cz,
        -0.9689 * cx + 1.8758 * lum + 0.0415 * cz,
        0.0557 * cx - 0.2040 * lum + 1.0570 * cz,
    )
    .max(Vec3::ZERO)
}

/// How the sky's kilocandela per square metre become the renderer's units:
/// a midday zenith of about eight comes out near the old painted sky's.
const SKY_SCALE: f32 = 0.05;

/// The clear sky's light from `dir`, with the sun toward `sun` and the air's
/// turbidity `t`, in the renderer's units; the sun's disc is not in it.
pub fn sky_radiance(dir: Vec3, sun: Vec3, t: f32) -> Vec3 {
    let d = dir.normalize_or_zero();
    let sun = sun.normalize_or_zero();
    let theta_s = sun.z.clamp(-1.0, 1.0).acos().min(1.55);
    let gamma = d.dot(sun).clamp(-1.0, 1.0).acos();
    let cos_theta = d.z.max(0.0);
    let [cy, cx, cyy] = perez_coefficients(t);
    let (zy, zx, zyy) = zenith(theta_s, t);
    let ratio = |c: [f32; 5]| perez(c, cos_theta, gamma) / perez(c, 1.0, theta_s).max(1e-6);
    let lum = zy * ratio(cy);
    let x = zx * ratio(cx);
    let y = zyy * ratio(cyy);
    xyy_to_rgb(x, y, lum) * SKY_SCALE
}

/// The sun's colour after the air it comes through: what scattering out of
/// its beam leaves at the red, green and blue ends, over the air mass of its
/// height (Kasten and Young's), brought to a luminance of one. Rayleigh's
/// optical depths at sea level, and the aerosol's from the turbidity by
/// Ångström's law, as Preetham's paper takes it — what makes a low sun orange
/// and a high one white.
pub fn sun_color(sun: Vec3, t: f32) -> [f32; 3] {
    let elevation = sun
        .normalize_or_zero()
        .z
        .clamp(0.0, 1.0)
        .asin()
        .to_degrees();
    let air_mass =
        1.0 / (elevation.to_radians().sin() + 0.50572 * (elevation + 6.07995).powf(-1.6364));
    // At 680, 550 and 440 nm.
    let rayleigh = Vec3::new(0.044, 0.098, 0.24);
    let beta = (0.04608 * t - 0.04586).max(0.0);
    let aerosol = Vec3::new(0.68f32.powf(-1.3), 0.55f32.powf(-1.3), 0.44f32.powf(-1.3)) * beta;
    let kept = (-(rayleigh + aerosol) * air_mass).exp();
    let lum = 0.2126 * kept.x + 0.7152 * kept.y + 0.0722 * kept.z;
    if lum <= 1e-6 {
        return [1.0, 0.4, 0.1];
    }
    (kept / lum).to_array()
}

/// Where the default studio's key softbox is: see [`Preset::key_direction`].
pub fn key_direction() -> Vec3 {
    Preset::Studio.key_direction()
}

/// The default studio's light arriving from `dir`: see [`radiance_of`].
pub fn radiance(dir: Vec3) -> [f32; 3] {
    radiance_of(Preset::Studio, dir)
}

/// A room's light arriving from `dir`, in the scene-linear units the
/// renderer shades in.
pub fn radiance_of(preset: Preset, dir: Vec3) -> [f32; 3] {
    let d = dir.normalize_or_zero();
    // The room: a ceiling, a horizon, a floor a step darker — a pale
    // cyclorama, so what is seen behind a part is a light studio grey and
    // not a dark pit.
    let [top, horizon, floor] = preset.room();
    let mut c = match preset.sky() {
        // A sky: the model above the horizon; below it, the ground, lit by
        // the sky overhead and the sun.
        Some((turbidity, ground)) => {
            let sun = preset.key_direction();
            if d.z >= 0.0 {
                sky_radiance(Vec3::new(d.x, d.y, d.z.max(0.002)), sun, turbidity)
            } else {
                let overhead = sky_radiance(Vec3::Z, sun, turbidity);
                ground * (overhead * 2.0 + Vec3::from(preset.key_color()) * sun.z.max(0.0) * 2.0)
            }
        }
        None => {
            if d.z >= 0.0 {
                horizon.lerp(top, smooth(0.0, 0.9, d.z))
            } else {
                horizon.lerp(floor, smooth(0.0, 0.25, -d.z))
            }
        }
    };
    for b in preset.softboxes() {
        let f = b.towards;
        let ahead = d.dot(f);
        if ahead <= 0.05 {
            continue;
        }
        let right = f.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(f);
        let (x, y) = (d.dot(right) / ahead, d.dot(up) / ahead);
        // Soft edges, as diffusion cloth has.
        let inside = (1.0 - smooth(b.half.0 * 0.8, b.half.0, x.abs()))
            * (1.0 - smooth(b.half.1 * 0.8, b.half.1, y.abs()));
        if inside > 0.0 {
            c += Vec3::from(b.radiance) * inside;
        }
    }
    c.to_array()
}

fn smooth(from: f32, to: f32, x: f32) -> f32 {
    let t = ((x - from) / (to - from)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The direction an equirectangular texel at `(u, v)` looks along: the
/// inverse of [`crate::environment::equirect_uv`].
pub fn direction_of(u: f32, v: f32) -> Vec3 {
    let azimuth = (u - 0.5) * std::f32::consts::TAU;
    let polar = v * std::f32::consts::PI;
    let s = polar.sin();
    Vec3::new(azimuth.sin() * s, -azimuth.cos() * s, polar.cos())
}

/// The default studio as an image: see [`equirect_of`].
pub fn equirect() -> Vec<u8> {
    equirect_of(Preset::Studio)
}

/// A room as an `Rgba16Float` equirectangular image, row-major, top row
/// straight up: what [`crate::viewport::Viewport::set_environment`] takes.
pub fn equirect_of(preset: Preset) -> Vec<u8> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT * 8) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let dir = direction_of(
                (x as f32 + 0.5) / WIDTH as f32,
                (y as f32 + 0.5) / HEIGHT as f32,
            );
            let [r, g, b] = radiance_of(preset, dir);
            for v in [r, g, b, 1.0] {
                out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_direction_is_the_inverse_of_the_mapping() {
        for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.8, 0.7), (0.33, 0.9)] {
            let d = direction_of(u, v);
            let back = crate::environment::equirect_uv(d);
            assert!(
                (back.x - u).abs() < 1e-4 && (back.y - v).abs() < 1e-4,
                "{u} {v} {back:?}"
            );
        }
    }

    #[test]
    fn the_key_is_the_brightest_place_and_the_floor_the_darkest() {
        let key = radiance(key_direction());
        let up = radiance(Vec3::Z);
        let down = radiance(-Vec3::Z);
        assert!(key[0] > up[0] * 3.0, "{key:?} {up:?}");
        assert!(down[0] < up[0] * 0.6, "{down:?} {up:?}");
        // Away from the boxes the walls are the room's own.
        let side = radiance(Vec3::new(-1.0, 0.6, 0.0));
        assert!(side[0] < 1.0, "{side:?}");
    }

    #[test]
    fn the_image_is_the_size_it_says() {
        assert_eq!(equirect().len(), (WIDTH * HEIGHT * 8) as usize);
    }

    /// The clear sky is bluer overhead than at the horizon, brighter near
    /// the sun than away from it; a low sun is warm and a high one white.
    #[test]
    fn the_sky_is_blue_brightens_toward_the_sun_and_reddens_with_it() {
        let high = Vec3::new(-0.6, -0.45, 0.65).normalize();
        let zenith = sky_radiance(Vec3::Z, high, 3.0);
        assert!(
            zenith.z > zenith.x * 1.3,
            "the zenith is not blue: {zenith}"
        );
        let horizon = sky_radiance(Vec3::new(1.0, 0.0, 0.02), high, 3.0);
        assert!(
            horizon.z / horizon.x < zenith.z / zenith.x,
            "paler at the horizon: {horizon} against {zenith}"
        );
        let near = sky_radiance((high + Vec3::new(0.05, 0.0, 0.0)).normalize(), high, 3.0);
        let away = sky_radiance((-high + Vec3::Z * 1.5).normalize(), high, 3.0);
        assert!(
            near.length() > away.length() * 2.0,
            "{near} near the sun, {away} away"
        );
        let midday = sun_color(high, 3.0);
        let evening = sun_color(Preset::Sunset.key_direction(), 4.0);
        assert!(
            evening[0] / evening[2] > midday[0] / midday[2] * 1.2,
            "{evening:?} against {midday:?}"
        );
        assert!(zenith.is_finite() && horizon.is_finite());
    }

    /// Every room has its key where it says, brighter than the room round
    /// it; and every name comes back to its room.
    #[test]
    fn every_room_is_lit_from_its_key() {
        for p in Preset::ALL {
            let key = radiance_of(p, p.key_direction());
            let away = radiance_of(p, -p.key_direction());
            assert!(key[0] > away[0], "{p:?}: {key:?} {away:?}");
            assert_eq!(Preset::from_name(p.name()), Some(p));
        }
    }
}
