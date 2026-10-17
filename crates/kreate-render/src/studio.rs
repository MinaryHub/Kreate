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

fn softboxes() -> [Softbox; 3] {
    [
        // The key: large, high, front left.
        Softbox {
            towards: key_direction(),
            half: (0.55, 0.38),
            radiance: [2.2, 2.17, 2.1],
        },
        // The fill: lower, front right, a third as bright.
        Softbox {
            towards: Vec3::new(0.75, -0.55, 0.25).normalize(),
            half: (0.45, 0.3),
            radiance: [1.1, 1.13, 1.17],
        },
        // The rim: a tall strip behind, for the edge that lifts a part off
        // the background.
        Softbox {
            towards: Vec3::new(0.25, 0.85, 0.45).normalize(),
            half: (0.18, 0.6),
            radiance: [2.8, 2.9, 3.0],
        },
    ]
}

/// Where the key softbox is, from the subject: the direction the
/// shadow-casting lamp that goes with the studio shines *from*.
pub fn key_direction() -> Vec3 {
    // To the viewer's left and a little in front, half way up: the shadow
    // falls to the right of a part, where it is seen. Behind the viewer it
    // fell behind the part, hidden by it.
    Vec3::new(-0.8, -0.3, 0.75).normalize()
}

/// The studio's light arriving from `dir`, in the scene-linear units the
/// renderer shades in.
pub fn radiance(dir: Vec3) -> [f32; 3] {
    let d = dir.normalize_or_zero();
    // The room: a cool bright ceiling, a neutral horizon, a floor a step
    // darker — a pale cyclorama, so what is seen behind a part is a light
    // studio grey and not a dark pit.
    let top = Vec3::new(0.46, 0.48, 0.53);
    let horizon = Vec3::new(0.4, 0.4, 0.41);
    let floor = Vec3::new(0.24, 0.235, 0.23);
    let mut c = if d.z >= 0.0 {
        horizon.lerp(top, smooth(0.0, 0.9, d.z))
    } else {
        horizon.lerp(floor, smooth(0.0, 0.25, -d.z))
    };
    for b in softboxes() {
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

/// The studio as an `Rgba16Float` equirectangular image, row-major, top row
/// straight up: what [`crate::viewport::Viewport::set_environment`] takes.
pub fn equirect() -> Vec<u8> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT * 8) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let dir = direction_of(
                (x as f32 + 0.5) / WIDTH as f32,
                (y as f32 + 0.5) / HEIGHT as f32,
            );
            let [r, g, b] = radiance(dir);
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
}
