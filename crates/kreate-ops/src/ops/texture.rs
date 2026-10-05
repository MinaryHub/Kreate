//! Procedural textures, made into images.
//!
//! Noise, wood, marble, cells, checks and bricks — the patterns a material
//! needs when there is no photograph of it. Made as image datablocks rather
//! than as live shader nodes, for one reason: an image is something every
//! consumer already reads. The viewport samples it, the path tracer samples
//! it, glTF exports it, and none of them has to learn a noise function. The
//! cost is that changing the pattern means making it again, which is one line.
//!
//! # Seamless
//!
//! Every pattern repeats exactly at the image's edges: the noise lattice wraps
//! with the image, cells and bricks are laid on whole counts across it. So a
//! texture tiled across a floor shows no seam, and `scale` is rounded to the
//! whole number of repeats that makes that true.

use kreate_scene::DocMut;

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub const KINDS: &[&str] = &["noise", "voronoi", "wood", "marble", "checker", "bricks"];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct GenerateParams {
    /// The pattern.
    #[param(variants("noise", "voronoi", "wood", "marble", "checker", "bricks"))]
    pub kind: u32,
    /// What the image holds: the pattern in colour, its height as grey (for a
    /// roughness map), or a normal map made from that height (for bumps).
    #[param(variants("color", "height", "normal"))]
    pub output: u32,
    /// The image's width and height in pixels.
    #[param(min = 16, max = 4096, unit = "count")]
    pub size: u32,
    /// How many times the pattern's features repeat across the image: cells,
    /// checks, rings, bricks along a row. Rounded to a whole number so the
    /// image tiles.
    #[param(min = 1.0, soft_max = 64.0)]
    pub scale: f64,
    /// Octaves of noise: more is finer detail.
    #[param(min = 1, max = 8, unit = "count")]
    pub detail: u32,
    /// How far noise bends the pattern: wood's grain, marble's veins.
    #[param(min = 0.0, soft_max = 8.0)]
    pub distortion: f64,
    /// The pattern's two colours, sRGB: low and high, or brick and mortar.
    #[param(color)]
    pub color_a: [f64; 3],
    #[param(color)]
    pub color_b: [f64; 3],
    /// A different number, a different pattern of the same kind.
    pub seed: i64,
    /// A normal map's depth: how steep the bumps look.
    #[param(min = 0.0, soft_max = 10.0)]
    pub strength: f64,
    /// The image's name. Empty: the pattern's.
    pub name: String,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            kind: 0,
            output: 0,
            size: 512,
            scale: 4.0,
            detail: 4,
            distortion: 1.0,
            color_a: [0.1, 0.1, 0.1],
            color_b: [0.9, 0.9, 0.9],
            seed: 0,
            strength: 2.0,
            name: String::new(),
        }
    }
}

/// Makes a procedural texture as an image datablock, for `material.set_image`
/// (its colour, normal or metallic-roughness map), `node.set_image` or a world.
pub struct Generate;

impl Operator for Generate {
    type Params = GenerateParams;

    fn id() -> &'static str {
        "image.generate"
    }
    fn label() -> &'static str {
        "Generate Texture"
    }

    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let size = params.size.clamp(16, 4096);
        let kind = KINDS.get(params.kind as usize).copied().unwrap_or("noise");
        let pattern = Pattern::new(kind, params);
        let height: Vec<f32> = (0..size * size)
            .map(|i| pattern.value((i % size) as f32 / size as f32, (i / size) as f32 / size as f32))
            .collect();
        let srgb = |c: [f64; 3]| c.map(|v| v.clamp(0.0, 1.0) as f32);
        let (a, b) = (srgb(params.color_a), srgb(params.color_b));
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let mut pixels = Vec::with_capacity((size * size * 4) as usize);
        match params.output {
            1 => {
                for h in &height {
                    let g = byte(*h);
                    pixels.extend_from_slice(&[g, g, g, 255]);
                }
            }
            2 => {
                let s = size as usize;
                let at = |x: usize, y: usize| height[(y % s) * s + (x % s)];
                let k = params.strength as f32;
                for y in 0..s {
                    for x in 0..s {
                        // Central differences, wrapping: the normal map tiles
                        // as the height does.
                        let dx = (at(x + 1, y) - at(x + s - 1, y)) * 0.5 * s as f32 / 64.0;
                        let dy = (at(x, y + 1) - at(x, y + s - 1)) * 0.5 * s as f32 / 64.0;
                        let n = glam::Vec3::new(-dx * k, dy * k, 1.0).normalize();
                        pixels.extend_from_slice(&[byte(n.x * 0.5 + 0.5), byte(n.y * 0.5 + 0.5), byte(n.z * 0.5 + 0.5), 255]);
                    }
                }
            }
            _ => {
                for h in &height {
                    let t = *h;
                    pixels.extend_from_slice(&[
                        byte(a[0] + (b[0] - a[0]) * t),
                        byte(a[1] + (b[1] - a[1]) * t),
                        byte(a[2] + (b[2] - a[2]) * t),
                        255,
                    ]);
                }
            }
        }
        let name = match params.name.trim() {
            "" => format!("{kind}-{}", ["color", "height", "normal"][params.output.min(2) as usize]),
            name => name.to_string(),
        };
        let space = if params.output == 0 {
            kreate_scene::ColorSpace::Srgb
        } else {
            kreate_scene::ColorSpace::Raw
        };
        let image = kreate_io::image_import::image_from_rgba8(&name, size, size, &pixels, space)
            .map_err(|e| OpError::Failed(e.to_string()))?;
        doc.document_mut().add_image(image);
        Ok(OpResult::Finished)
    }
}

/// One pattern, ready to be asked for its value anywhere on the unit square.
struct Pattern {
    kind: &'static str,
    scale: u32,
    detail: u32,
    distortion: f32,
    seed: u32,
}

impl Pattern {
    fn new(kind: &'static str, p: &GenerateParams) -> Self {
        Self {
            kind,
            scale: (p.scale.round() as u32).clamp(1, 512),
            detail: p.detail.clamp(1, 8),
            distortion: p.distortion as f32,
            seed: p.seed as u32,
        }
    }

    /// The pattern at `(u, v)` in [0, 1)², as a value in [0, 1].
    fn value(&self, u: f32, v: f32) -> f32 {
        let s = self.scale as f32;
        match self.kind {
            "voronoi" => self.cells(u * s, v * s, self.scale),
            "wood" => {
                // Rings across V, bent by noise stretched along the grain.
                let bend = self.fbm(u * 2.0, v * 8.0, 2, 8) * self.distortion * 0.35;
                let ring = (v * s + bend).fract();
                let fine = self.fbm(u * 16.0, v * 64.0, 16, 64) * 0.15;
                (smooth(0.0, 0.6, ring) * (1.0 - smooth(0.6, 1.0, ring)) * 0.85 + fine).clamp(0.0, 1.0)
            }
            "marble" => {
                // Bands bent by noise, then pushed towards the light colour so
                // what is left of the dark is thin veins rather than stripes.
                let bend = (self.fbm(u * 3.0, v * 3.0, 3, 3) - 0.5) * 2.0 * self.distortion * 1.5;
                // Slanted by one whole turn over V, so it still tiles.
                let band = 0.5 + 0.5 * ((u * s + v) * std::f32::consts::TAU + bend).sin();
                band.powf(0.3)
            }
            "checker" => {
                let (x, y) = ((u * s).floor() as i64, (v * s).floor() as i64);
                ((x + y).rem_euclid(2)) as f32
            }
            "bricks" => self.bricks(u, v),
            _ => self.fbm(u * s, v * s, self.scale, self.scale),
        }
    }

    /// Fractal noise, wrapping every `px` by `py` lattice cells at the first
    /// octave, in [0, 1].
    fn fbm(&self, x: f32, y: f32, px: u32, py: u32) -> f32 {
        let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, 1u32);
        for octave in 0..self.detail {
            sum += amp * gradient_noise(x * f as f32, y * f as f32, px * f, py * f, self.seed.wrapping_add(octave * 131));
            norm += amp;
            amp *= 0.5;
            f *= 2;
        }
        (sum / norm * 0.5 + 0.5).clamp(0.0, 1.0)
    }

    /// Cells: the distance to the nearest of one jittered point per cell, on a
    /// `period`-wrapped grid, with some of the noise laid over for texture.
    fn cells(&self, x: f32, y: f32, period: u32) -> f32 {
        let (cx, cy) = (x.floor() as i64, y.floor() as i64);
        let mut nearest = f32::MAX;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (gx, gy) = (cx + dx, cy + dy);
                let h = hash(gx.rem_euclid(period as i64) as u32, gy.rem_euclid(period as i64) as u32, self.seed);
                let jitter = (h as f32 / u32::MAX as f32, hash(h, 7, self.seed) as f32 / u32::MAX as f32);
                let p = (gx as f32 + jitter.0, gy as f32 + jitter.1);
                nearest = nearest.min(((p.0 - x).powi(2) + (p.1 - y).powi(2)).sqrt());
            }
        }
        let grain = self.fbm(x * 2.0, y * 2.0, period * 2, period * 2) * 0.2 * self.distortion.min(1.0);
        (nearest.min(1.0) + grain).clamp(0.0, 1.0)
    }

    /// Bricks: `scale` along a row, half as many rows, every other row offset
    /// by half a brick, mortar between. Mortar is 1, brick around 0 with a
    /// tone of its own each.
    fn bricks(&self, u: f32, v: f32) -> f32 {
        let across = self.scale.max(1) as f32;
        let rows = (self.scale.max(2) * 2) as f32;
        let y = v * rows;
        let row = y.floor();
        let offset = if (row as i64).rem_euclid(2) == 1 { 0.5 } else { 0.0 };
        let x = u * across + offset;
        let (fx, fy) = (x.fract(), y.fract());
        let mortar = 0.06;
        let edge = |f: f32, w: f32| smooth(0.0, w, f) * (1.0 - smooth(1.0 - w, 1.0, f));
        let inside = edge(fx, mortar * across / rows * 2.0) * edge(fy, mortar);
        let brick = (x.floor() as i64).rem_euclid(across as i64) as u32;
        let tone = hash(brick, row as u32, self.seed) as f32 / u32::MAX as f32 * 0.35;
        let grain = self.fbm(u * across * 4.0, v * rows * 2.0, (across as u32) * 4, (rows as u32) * 2) * 0.15;
        (1.0 - inside * (1.0 - tone - grain)).clamp(0.0, 1.0)
    }
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A well-mixed 32-bit hash of a lattice point and a seed.
fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

/// Perlin's gradient noise at `(x, y)`, its lattice wrapping every `px` by
/// `py` cells, in about [-1, 1].
fn gradient_noise(x: f32, y: f32, px: u32, py: u32, seed: u32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let corner = |ix: i64, iy: i64, dx: f32, dy: f32| {
        let h = hash(ix.rem_euclid(px.max(1) as i64) as u32, iy.rem_euclid(py.max(1) as i64) as u32, seed);
        let angle = h as f32 / u32::MAX as f32 * std::f32::consts::TAU;
        angle.cos() * dx + angle.sin() * dy
    };
    let (ix, iy) = (x0 as i64, y0 as i64);
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(fx), fade(fy));
    let a = corner(ix, iy, fx, fy) + (corner(ix + 1, iy, fx - 1.0, fy) - corner(ix, iy, fx, fy)) * u;
    let b = corner(ix, iy + 1, fx, fy - 1.0) + (corner(ix + 1, iy + 1, fx - 1.0, fy - 1.0) - corner(ix, iy + 1, fx, fy - 1.0)) * u;
    (a + (b - a) * v) * std::f32::consts::SQRT_2
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Generate>();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(kind: &'static str) -> Pattern {
        Pattern::new(kind, &GenerateParams { scale: 3.0, ..GenerateParams::default() })
    }

    #[test]
    fn every_pattern_tiles_without_a_seam() {
        // The left edge continues the right and the top the bottom: the value a
        // hair inside one edge is close to the value a hair inside the other.
        for kind in KINDS {
            let p = pattern(kind);
            let mut worst: f32 = 0.0;
            for i in 0..64 {
                let t = i as f32 / 64.0 + 0.003;
                worst = worst.max((p.value(0.0005, t) - p.value(0.9995, t)).abs());
                worst = worst.max((p.value(t, 0.0005) - p.value(t, 0.9995)).abs());
            }
            // A checker has an edge at the border by design; it still lines up.
            let allowed = if *kind == "checker" { 1.0 } else { 0.15 };
            assert!(worst <= allowed, "{kind} has a seam of {worst}");
        }
    }

    #[test]
    fn every_pattern_stays_in_range_and_varies() {
        for kind in KINDS {
            let p = pattern(kind);
            let values: Vec<f32> = (0..400).map(|i| p.value((i % 20) as f32 / 20.0, (i / 20) as f32 / 20.0)).collect();
            assert!(values.iter().all(|v| (0.0..=1.0).contains(v)), "{kind} left [0, 1]");
            let (lo, hi) = values.iter().fold((1.0f32, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            assert!(hi - lo > 0.2, "{kind} is flat: {lo}..{hi}");
        }
    }

    #[test]
    fn a_seed_changes_the_pattern() {
        let a = Pattern::new("noise", &GenerateParams::default());
        let b = Pattern::new("noise", &GenerateParams { seed: 9, ..GenerateParams::default() });
        assert!((0..50).any(|i| (a.value(i as f32 / 50.0, 0.3) - b.value(i as f32 / 50.0, 0.3)).abs() > 0.05));
    }
}

