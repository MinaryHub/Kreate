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
    /// A sky and a sun: hard shadows and a blue reflection.
    Outdoor,
}

impl Preset {
    pub const ALL: [Preset; 4] = [
        Preset::Studio,
        Preset::Bright,
        Preset::GreyRoom,
        Preset::Outdoor,
    ];

    /// The name the menus and the settings use.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Studio => "studio",
            Preset::Bright => "bright",
            Preset::GreyRoom => "grey_room",
            Preset::Outdoor => "outdoor",
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
            // A blue sky paling to the horizon, a warm grey ground.
            Preset::Outdoor => [
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
            // The sun: small and very bright.
            Preset::Outdoor => vec![Softbox {
                towards: self.key_direction(),
                half: (0.03, 0.03),
                radiance: [900.0, 860.0, 780.0],
            }],
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
            Preset::Outdoor => None,
        }
    }
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
    let mut c = if d.z >= 0.0 {
        horizon.lerp(top, smooth(0.0, 0.9, d.z))
    } else {
        horizon.lerp(floor, smooth(0.0, 0.25, -d.z))
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
