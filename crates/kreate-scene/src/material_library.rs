//! Physical materials: what a part is made of, as Fusion's library has them.
//!
//! Each is a look and a density together — oak is brown, fairly rough and
//! 750 kg/m³ — so choosing one paints a part and weighs it in one step. The
//! densities are typical handbook values, good for a mass estimate and not
//! for a certificate.
//!
//! The look is the standard surface's numbers and, over them, a [`Finish`]:
//! what makes the stuff read as itself — oak's growth rings, marble's veins,
//! the hairlines of brushed steel, the pits of a casting. Each is a shading
//! graph of procedural textures read at the point's place in its object, in
//! metres: no images and no UVs, so any shape takes it without a seam, the
//! same in the viewport and the render, and a ring is millimetres apart on a
//! cup and on a table alike.
//!
//! Ids are stable — they are what a material remembers it came from, and what
//! a script names — and only ever appended. Names are the catalogue's, as
//! `physical-<id>`, so the list reads in the user's language.

use kreate_math::{SceneRgb, Srgb};

use crate::document::Material;
use crate::shader::{Link, Node, NodeKind, RampMode, ShaderGraph, WaveKind, RAMP_STOPS};

/// Which shelf of the library a material is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    Metal,
    Plastic,
    Wood,
    Glass,
    Stone,
    Rubber,
}

impl Category {
    pub const ALL: [Category; 6] = [
        Category::Metal,
        Category::Plastic,
        Category::Wood,
        Category::Glass,
        Category::Stone,
        Category::Rubber,
    ];

    /// The catalogue key of its name.
    pub fn name_key(self) -> &'static str {
        match self {
            Category::Metal => "physical-category-metal",
            Category::Plastic => "physical-category-plastic",
            Category::Wood => "physical-category-wood",
            Category::Glass => "physical-category-glass",
            Category::Stone => "physical-category-stone",
            Category::Rubber => "physical-category-rubber",
        }
    }
}

/// What the surface of a material looks like close to: the pattern laid
/// over its colour, roughness and normal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// Nothing over the numbers: a mirror, clear glass, a glaze.
    Plain,
    /// The roughness wandering a little across the surface, as no real
    /// polish is even: what keeps a machined metal or a gloss from looking
    /// like a render.
    Satin,
    /// Hairlines along the object's X: brushed or ground metal.
    Brushed,
    /// Fine pits and a blotchy colour: a casting, a fibre board.
    Cast,
    /// A fine texture moulded in: most plastic parts, rubber.
    Molded,
    /// Growth rings round a line along X, bent and figured: timber.
    Wood,
    /// Thin dark veins wandering through it.
    Marble,
    /// Speckled grains of three tones.
    Granite,
    /// Blotched, with pits and aggregate.
    Concrete,
    /// A pebbled grain.
    Leather,
    /// Fine bumps that scatter what is seen through: frosted glass.
    Frosted,
}

/// One entry in the library.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalMaterial {
    pub id: &'static str,
    pub category: Category,
    /// The base colour as authored, `0xRRGGBB` in sRGB.
    pub color: u32,
    pub metallic: f32,
    pub roughness: f32,
    /// Zero for an opaque solid; near one for clear glass and plastics.
    pub transmission: f32,
    pub ior: f32,
    /// Kilograms per cubic metre.
    pub density: f64,
    pub finish: Finish,
}

impl PhysicalMaterial {
    /// The catalogue key of its name.
    pub fn name_key(&self) -> String {
        format!("physical-{}", self.id.replace('_', "-"))
    }

    /// The material this makes, called `name`: the numbers, and its finish
    /// as the material's graph. Turning the graph off (Use Nodes) leaves the
    /// plain numbers, which are the same stuff without the close-up.
    pub fn material(&self, name: &str) -> Material {
        let mut material = Material {
            name: name.to_string(),
            base_color: Srgb::from_hex(self.color).as_reflectance(),
            metallic: self.metallic,
            roughness: self.roughness,
            transmission: self.transmission,
            ior: self.ior,
            density: Some(self.density),
            physical: Some(self.id.to_string()),
            ..Material::default()
        };
        material.graph = finish_graph(&material, self.finish);
        material
    }

    const fn with(mut self, finish: Finish) -> Self {
        self.finish = finish;
        self
    }
}

/// A graph being built: nodes added and wired, each by what this file
/// knows of their sockets — so a failure is a mistake here, not the caller's.
struct Builder(ShaderGraph);

/// The surface's inputs, by number.
const BASE: usize = 0;
const ROUGHNESS: usize = 2;
const NORMAL: usize = 3;

impl Builder {
    fn add(&mut self, name: &str, kind: NodeKind) -> usize {
        self.0
            .add_node(Node::new(name, kind))
            .expect("only a surface is refused")
    }

    fn wire(&mut self, from: usize, from_socket: usize, to: usize, to_socket: usize) {
        self.0
            .link(Link {
                from_node: from,
                from_socket,
                to_node: to,
                to_socket,
            })
            .expect("the sockets agree");
    }

    /// Fractal noise read at the point (or at `at`, a coordinate), its
    /// number out of socket 0.
    fn noise(&mut self, name: &str, scale: f32, detail: f32, at: Option<usize>) -> usize {
        let n = self.add(
            name,
            NodeKind::Noise {
                scale,
                detail,
                roughness: 0.55,
                distortion: 0.0,
            },
        );
        if let Some(at) = at {
            self.wire(at, 0, n, 4);
        }
        n
    }

    /// The point stretched: `scale` per axis, so a texture read at it runs
    /// long along the axis scaled least.
    fn stretched(&mut self, name: &str, scale: [f32; 3], location: [f32; 3]) -> usize {
        self.add(
            name,
            NodeKind::Mapping {
                location,
                rotation: [0.0; 3],
                scale,
            },
        )
    }

    /// The roughness, from `lo` to `hi` as `value`'s socket 0 goes from
    /// `from.0` to `from.1`.
    fn roughness_by(&mut self, value: usize, socket: usize, from: (f32, f32), lo: f32, hi: f32) {
        let r = self.add(
            "Roughness",
            NodeKind::MapRange {
                value: 0.5,
                from_min: from.0,
                from_max: from.1,
                to_min: lo.clamp(0.0, 1.0),
                to_max: hi.clamp(0.0, 1.0),
            },
        );
        self.wire(value, socket, r, 0);
        self.wire(r, 0, 0, ROUGHNESS);
    }

    /// The surface bumped by `height`'s number: `depth` metres from its
    /// lowest to its highest.
    fn bump(&mut self, height: usize, socket: usize, strength: f32, depth: f32) {
        let b = self.add(
            "Bump",
            NodeKind::Bump {
                strength,
                distance: depth,
            },
        );
        self.wire(height, socket, b, 2);
        self.wire(b, 0, 0, NORMAL);
    }

    /// A colour ramp of up to eight stops, from `value`'s number.
    fn ramp(
        &mut self,
        name: &str,
        value: usize,
        socket: usize,
        mode: RampMode,
        stops: &[(f32, SceneRgb)],
    ) -> usize {
        let mut all = [(1.0, SceneRgb::new(0.0, 0.0, 0.0)); RAMP_STOPS];
        for (slot, stop) in all.iter_mut().zip(stops) {
            *slot = *stop;
        }
        let r = self.add(
            name,
            NodeKind::ColorRamp {
                interpolation: mode,
                count: stops.len().min(RAMP_STOPS) as u8,
                stops: all,
            },
        );
        self.wire(value, socket, r, 0);
        r
    }

    /// The base colour blotched: `low` to `high` by slow noise.
    fn blotched(&mut self, scale: f32, low: SceneRgb, high: SceneRgb) -> usize {
        let n = self.noise("Blotches", scale, 3.0, None);
        let mix = self.add("Colour", NodeKind::MixColor { factor: 0.5 });
        let (a, b) = (
            self.add("Dark", NodeKind::Color(low)),
            self.add("Light", NodeKind::Color(high)),
        );
        self.wire(n, 0, mix, 0);
        self.wire(a, 0, mix, 1);
        self.wire(b, 0, mix, 2);
        mix
    }
}

/// `c` times `k`, held below one: a shade of a colour.
fn shade(c: SceneRgb, k: f32) -> SceneRgb {
    SceneRgb::new(
        (c.r * k).min(0.95),
        (c.g * k).min(0.95),
        (c.b * k).min(0.95),
    )
}

/// The graph that gives `material` its finish; `None` for a plain one.
///
/// Scales are in metres, as the textures read the point in its object:
/// a noise of scale 1000 has its bumps a millimetre apart. Depths are kept
/// to fractions of a millimetre — a finish is felt more than seen.
fn finish_graph(material: &Material, finish: Finish) -> Option<ShaderGraph> {
    let c = material.base_color;
    let r = material.roughness;
    let mut g = Builder(ShaderGraph::for_material(material));
    match finish {
        Finish::Plain => return None,
        Finish::Satin => {
            let n = g.noise("Polish", 70.0, 3.0, None);
            g.roughness_by(n, 0, (0.3, 0.7), r * 0.85, r * 1.15);
        }
        Finish::Brushed => {
            // Long along X, under a millimetre across: finer, and a screen
            // shows it as sand rather than lines.
            let at = g.stretched("Along X", [6.0, 1200.0, 1200.0], [0.0; 3]);
            let n = g.noise("Hairlines", 1.0, 2.0, Some(at));
            g.bump(n, 0, 0.3, 0.00003);
            g.roughness_by(n, 0, (0.2, 0.8), r * 0.7, r * 1.3);
        }
        Finish::Cast => {
            let colour = g.blotched(12.0, shade(c, 0.82), shade(c, 1.1));
            g.wire(colour, 0, 0, BASE);
            let pits = g.noise("Pits", 900.0, 3.0, None);
            g.bump(pits, 0, 0.5, 0.00015);
            let wander = g.noise("Wear", 25.0, 2.0, None);
            g.roughness_by(wander, 0, (0.3, 0.7), r * 0.85, r * 1.1);
        }
        Finish::Molded => {
            let grain = g.noise("Texture", 1800.0, 2.0, None);
            g.bump(grain, 0, 0.25, 0.00004);
            let wander = g.noise("Sheen", 30.0, 2.0, None);
            g.roughness_by(wander, 0, (0.3, 0.7), r * 0.9, r * 1.1);
        }
        Finish::Wood => {
            // Rings round a line along X, a little way off the part — as a
            // board is sawn from beside a trunk's heart — a few mm apart,
            // bent by slow noise so no two are alike.
            let at = g.stretched("Trunk", [0.08, 1.0, 1.0], [0.0, 0.07, 0.045]);
            let rings = g.add(
                "Rings",
                NodeKind::Wave {
                    kind: WaveKind::Rings,
                    scale: 55.0,
                    distortion: 7.0,
                    detail: 3.0,
                    detail_scale: 0.5,
                },
            );
            g.wire(at, 0, rings, 4);
            let wood = g.ramp(
                "Early and late wood",
                rings,
                1,
                RampMode::Ease,
                &[
                    (0.0, shade(c, 1.08)),
                    (0.6, c),
                    (0.88, shade(c, 0.74)),
                    (1.0, shade(c, 0.68)),
                ],
            );
            // Figure along the grain: streaks long in X.
            let streak_at = g.stretched("Along the grain", [6.0, 260.0, 260.0], [0.0; 3]);
            let streaks = g.noise("Figure", 1.0, 3.0, Some(streak_at));
            let lift = g.add(
                "Figure strength",
                NodeKind::MapRange {
                    value: 0.5,
                    from_min: 0.25,
                    from_max: 0.75,
                    to_min: 0.82,
                    to_max: 1.06,
                },
            );
            g.wire(streaks, 0, lift, 0);
            let grey = g.add("Figure tint", NodeKind::CombineRgb([1.0; 3]));
            for channel in 0..3 {
                g.wire(lift, 0, grey, channel);
            }
            let figured = g.add("Wood", NodeKind::Multiply);
            g.wire(wood, 0, figured, 0);
            g.wire(grey, 0, figured, 1);
            g.wire(figured, 0, 0, BASE);
            // Open pores along the grain, and the late wood a little glossier.
            let pore_at = g.stretched("Pores", [40.0, 3000.0, 3000.0], [0.0; 3]);
            let pores = g.noise("Pores", 1.0, 2.0, Some(pore_at));
            g.bump(pores, 0, 0.3, 0.00005);
            g.roughness_by(rings, 1, (0.0, 1.0), r * 1.1, r * 0.8);
        }
        Finish::Marble => {
            let veins = g.add(
                "Veins",
                NodeKind::Wave {
                    kind: WaveKind::Bands,
                    scale: 7.0,
                    distortion: 16.0,
                    detail: 5.0,
                    detail_scale: 1.5,
                },
            );
            let stone = g.ramp(
                "Marble",
                veins,
                1,
                RampMode::Linear,
                &[
                    (0.0, shade(c, 0.42)),
                    (0.05, shade(c, 0.78)),
                    (0.18, shade(c, 0.97)),
                    (1.0, c),
                ],
            );
            g.wire(stone, 0, 0, BASE);
            let wander = g.noise("Polish", 20.0, 2.0, None);
            g.roughness_by(wander, 0, (0.3, 0.7), r * 0.8, r * 1.3);
        }
        Finish::Granite => {
            let grains = g.noise("Grains", 450.0, 2.0, None);
            let stone = g.ramp(
                "Granite",
                grains,
                0,
                RampMode::Constant,
                &[
                    (0.0, shade(c, 0.35)),
                    (0.4, c),
                    (0.58, shade(c, 1.7)),
                    (0.66, shade(c, 0.55)),
                ],
            );
            g.wire(stone, 0, 0, BASE);
            g.roughness_by(grains, 0, (0.3, 0.7), r * 0.7, r * 1.3);
        }
        Finish::Concrete => {
            let colour = g.blotched(30.0, shade(c, 0.72), shade(c, 1.15));
            g.wire(colour, 0, 0, BASE);
            let pits = g.add(
                "Pores",
                NodeKind::Voronoi {
                    scale: 600.0,
                    randomness: 1.0,
                },
            );
            g.bump(pits, 0, 0.6, 0.0003);
            let wander = g.noise("Trowel", 8.0, 3.0, None);
            g.roughness_by(wander, 0, (0.3, 0.7), r * 0.85, (r * 1.1).min(1.0));
        }
        Finish::Leather => {
            let pebbles = g.add(
                "Grain",
                NodeKind::Voronoi {
                    scale: 900.0,
                    randomness: 1.0,
                },
            );
            g.bump(pebbles, 0, 0.6, 0.0001);
            let colour = g.blotched(30.0, shade(c, 0.85), shade(c, 1.12));
            g.wire(colour, 0, 0, BASE);
            g.roughness_by(pebbles, 0, (0.0, 0.6), r * 0.8, r * 1.15);
        }
        Finish::Frosted => {
            let frost = g.noise("Frost", 1500.0, 2.0, None);
            g.bump(frost, 0, 0.8, 0.00005);
        }
    }
    g.0.arrange();
    Some(g.0)
}

const fn solid(
    id: &'static str,
    category: Category,
    color: u32,
    metallic: f32,
    roughness: f32,
    density: f64,
) -> PhysicalMaterial {
    PhysicalMaterial {
        id,
        category,
        color,
        metallic,
        roughness,
        transmission: 0.0,
        ior: 1.5,
        density,
        finish: Finish::Satin,
    }
}

const fn clear(
    id: &'static str,
    category: Category,
    color: u32,
    roughness: f32,
    transmission: f32,
    ior: f32,
    density: f64,
) -> PhysicalMaterial {
    PhysicalMaterial {
        id,
        category,
        color,
        metallic: 0.0,
        roughness,
        transmission,
        ior,
        density,
        finish: Finish::Plain,
    }
}

use Category::*;

/// The library, shelf by shelf.
pub const LIBRARY: &[PhysicalMaterial] = &[
    // Metals: the colour is the reflection's, as a metal has no other.
    solid("steel", Metal, 0x9A9DA1, 1.0, 0.35, 7850.0),
    solid("stainless_steel", Metal, 0xC4C6C8, 1.0, 0.22, 8000.0),
    solid("brushed_steel", Metal, 0xAEB1B4, 1.0, 0.45, 7850.0).with(Finish::Brushed),
    solid("aluminum", Metal, 0xD6D8DA, 1.0, 0.3, 2700.0),
    solid("anodized_aluminum_black", Metal, 0x2A2B2E, 1.0, 0.4, 2700.0),
    solid("cast_iron", Metal, 0x5E5F61, 1.0, 0.6, 7200.0).with(Finish::Cast),
    solid("chrome", Metal, 0xE4E6E8, 1.0, 0.05, 7190.0).with(Finish::Plain),
    solid("brass", Metal, 0xD9B467, 1.0, 0.28, 8500.0),
    solid("copper", Metal, 0xE0946A, 1.0, 0.25, 8960.0),
    solid("bronze", Metal, 0xB8874F, 1.0, 0.35, 8800.0),
    solid("gold", Metal, 0xF3C766, 1.0, 0.2, 19300.0).with(Finish::Plain),
    solid("titanium", Metal, 0xA7A39C, 1.0, 0.35, 4500.0).with(Finish::Brushed),
    // Plastics.
    solid("abs_white", Plastic, 0xEDEBE6, 0.0, 0.45, 1050.0).with(Finish::Molded),
    solid("abs_black", Plastic, 0x1E1F21, 0.0, 0.45, 1050.0).with(Finish::Molded),
    solid("pla", Plastic, 0xD8D3C8, 0.0, 0.4, 1240.0).with(Finish::Molded),
    solid("nylon", Plastic, 0xE6E1D3, 0.0, 0.55, 1150.0).with(Finish::Molded),
    solid("polypropylene", Plastic, 0xE9E7E1, 0.0, 0.5, 905.0).with(Finish::Molded),
    solid("plastic_glossy_red", Plastic, 0xC2272D, 0.0, 0.15, 1050.0),
    solid("plastic_matte_blue", Plastic, 0x2F5DA8, 0.0, 0.6, 1050.0).with(Finish::Molded),
    clear("acrylic_clear", Plastic, 0xFFFFFF, 0.02, 0.95, 1.49, 1180.0),
    clear(
        "polycarbonate_clear",
        Plastic,
        0xF4F7F8,
        0.05,
        0.9,
        1.58,
        1200.0,
    ),
    // Woods.
    solid("oak", Wood, 0xA97C50, 0.0, 0.6, 750.0).with(Finish::Wood),
    solid("walnut", Wood, 0x5D3F2B, 0.0, 0.55, 650.0).with(Finish::Wood),
    solid("maple", Wood, 0xD9B98B, 0.0, 0.5, 700.0).with(Finish::Wood),
    solid("pine", Wood, 0xD6B27A, 0.0, 0.65, 500.0).with(Finish::Wood),
    solid("cherry", Wood, 0x9B5A3C, 0.0, 0.5, 600.0).with(Finish::Wood),
    solid("plywood", Wood, 0xCFAF80, 0.0, 0.7, 600.0).with(Finish::Wood),
    solid("mdf", Wood, 0x9C8366, 0.0, 0.8, 750.0).with(Finish::Cast),
    // Glass.
    clear("glass_clear", Glass, 0xFFFFFF, 0.0, 1.0, 1.52, 2500.0),
    clear("glass_frosted", Glass, 0xF2F5F5, 0.35, 0.9, 1.52, 2500.0).with(Finish::Frosted),
    clear("glass_tinted", Glass, 0x7FA39A, 0.0, 0.95, 1.52, 2500.0),
    // Stone and ceramic.
    solid("concrete", Stone, 0x9E9C97, 0.0, 0.9, 2400.0).with(Finish::Concrete),
    solid("marble_white", Stone, 0xE8E6E1, 0.0, 0.2, 2700.0).with(Finish::Marble),
    solid("granite", Stone, 0x5F5E5C, 0.0, 0.45, 2750.0).with(Finish::Granite),
    solid("ceramic_white", Stone, 0xF1F0EC, 0.0, 0.1, 2400.0).with(Finish::Plain),
    // Rubber and soft.
    solid("rubber_black", Rubber, 0x18191A, 0.0, 0.85, 1100.0).with(Finish::Molded),
    solid("silicone", Rubber, 0xDCDCD8, 0.0, 0.6, 1100.0).with(Finish::Molded),
    solid("leather_brown", Rubber, 0x6B4630, 0.0, 0.7, 860.0).with(Finish::Leather),
];

/// The entry with this id.
pub fn find(id: &str) -> Option<&'static PhysicalMaterial> {
    LIBRARY.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_every_shelf_has_something() {
        let mut ids: Vec<&str> = LIBRARY.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "an id twice");
        for c in Category::ALL {
            assert!(LIBRARY.iter().any(|m| m.category == c), "{c:?} is empty");
        }
        for m in LIBRARY {
            assert!(
                m.density > 0.0 && (0.0..=1.0).contains(&m.roughness),
                "{}",
                m.id
            );
        }
    }

    #[test]
    fn a_physical_material_paints_and_weighs() {
        let oak = find("oak").unwrap().material("Oak");
        assert_eq!(oak.density, Some(750.0));
        assert_eq!(oak.physical.as_deref(), Some("oak"));
        assert_eq!(oak.metallic, 0.0);
        let glass = find("glass_clear").unwrap().material("Glass");
        assert!(glass.transmission > 0.9);
        // Clear glass is plain; oak has its grain as a graph.
        assert!(glass.graph.is_none());
        assert!(oak.graph.is_some());
    }

    #[test]
    fn every_finish_is_a_graph_that_drives_the_surface() {
        for m in LIBRARY {
            let material = m.material(m.id);
            let Some(graph) = &material.graph else {
                assert_eq!(m.finish, Finish::Plain, "{}", m.id);
                continue;
            };
            // Something reaches the surface — a colour, a roughness or a
            // normal — or the finish draws as nothing.
            assert!(
                [BASE, ROUGHNESS, NORMAL]
                    .iter()
                    .any(|socket| graph.incoming(0, *socket).is_some()),
                "{}",
                m.id
            );
            // And the numbers the plain material has, it keeps.
            let NodeKind::Surface { transmission, .. } = graph.nodes[0].kind else {
                panic!("the surface first");
            };
            assert_eq!(transmission, m.transmission, "{}", m.id);
        }
    }
}
