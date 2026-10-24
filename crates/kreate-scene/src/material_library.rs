//! Physical materials: what a part is made of, as Fusion's library has them.
//!
//! Each is a look and a density together — oak is brown, fairly rough and
//! 750 kg/m³ — so choosing one paints a part and weighs it in one step. The
//! looks are the standard surface's numbers (no images: they need no UVs and
//! read the same on any shape); the densities are typical handbook values,
//! good for a mass estimate and not for a certificate.
//!
//! Ids are stable — they are what a material remembers it came from, and what
//! a script names — and only ever appended. Names are the catalogue's, as
//! `physical-<id>`, so the list reads in the user's language.

use kreate_math::Srgb;

use crate::document::Material;

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
}

impl PhysicalMaterial {
    /// The catalogue key of its name.
    pub fn name_key(&self) -> String {
        format!("physical-{}", self.id.replace('_', "-"))
    }

    /// The material this makes, called `name`.
    pub fn material(&self, name: &str) -> Material {
        Material {
            name: name.to_string(),
            base_color: Srgb::from_hex(self.color).as_reflectance(),
            metallic: self.metallic,
            roughness: self.roughness,
            transmission: self.transmission,
            ior: self.ior,
            density: Some(self.density),
            physical: Some(self.id.to_string()),
            ..Material::default()
        }
    }
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
    }
}

use Category::*;

/// The library, shelf by shelf.
pub const LIBRARY: &[PhysicalMaterial] = &[
    // Metals: the colour is the reflection's, as a metal has no other.
    solid("steel", Metal, 0x9A9DA1, 1.0, 0.35, 7850.0),
    solid("stainless_steel", Metal, 0xC4C6C8, 1.0, 0.22, 8000.0),
    solid("brushed_steel", Metal, 0xAEB1B4, 1.0, 0.45, 7850.0),
    solid("aluminum", Metal, 0xD6D8DA, 1.0, 0.3, 2700.0),
    solid("anodized_aluminum_black", Metal, 0x2A2B2E, 1.0, 0.4, 2700.0),
    solid("cast_iron", Metal, 0x5E5F61, 1.0, 0.6, 7200.0),
    solid("chrome", Metal, 0xE4E6E8, 1.0, 0.05, 7190.0),
    solid("brass", Metal, 0xD9B467, 1.0, 0.28, 8500.0),
    solid("copper", Metal, 0xE0946A, 1.0, 0.25, 8960.0),
    solid("bronze", Metal, 0xB8874F, 1.0, 0.35, 8800.0),
    solid("gold", Metal, 0xF3C766, 1.0, 0.2, 19300.0),
    solid("titanium", Metal, 0xA7A39C, 1.0, 0.35, 4500.0),
    // Plastics.
    solid("abs_white", Plastic, 0xEDEBE6, 0.0, 0.45, 1050.0),
    solid("abs_black", Plastic, 0x1E1F21, 0.0, 0.45, 1050.0),
    solid("pla", Plastic, 0xD8D3C8, 0.0, 0.4, 1240.0),
    solid("nylon", Plastic, 0xE6E1D3, 0.0, 0.55, 1150.0),
    solid("polypropylene", Plastic, 0xE9E7E1, 0.0, 0.5, 905.0),
    solid("plastic_glossy_red", Plastic, 0xC2272D, 0.0, 0.15, 1050.0),
    solid("plastic_matte_blue", Plastic, 0x2F5DA8, 0.0, 0.6, 1050.0),
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
    solid("oak", Wood, 0xA97C50, 0.0, 0.6, 750.0),
    solid("walnut", Wood, 0x5D3F2B, 0.0, 0.55, 650.0),
    solid("maple", Wood, 0xD9B98B, 0.0, 0.5, 700.0),
    solid("pine", Wood, 0xD6B27A, 0.0, 0.65, 500.0),
    solid("cherry", Wood, 0x9B5A3C, 0.0, 0.5, 600.0),
    solid("plywood", Wood, 0xCFAF80, 0.0, 0.7, 600.0),
    solid("mdf", Wood, 0x9C8366, 0.0, 0.8, 750.0),
    // Glass.
    clear("glass_clear", Glass, 0xFFFFFF, 0.0, 1.0, 1.52, 2500.0),
    clear("glass_frosted", Glass, 0xF2F5F5, 0.35, 0.9, 1.52, 2500.0),
    clear("glass_tinted", Glass, 0x7FA39A, 0.0, 0.95, 1.52, 2500.0),
    // Stone and ceramic.
    solid("concrete", Stone, 0x9E9C97, 0.0, 0.9, 2400.0),
    solid("marble_white", Stone, 0xE8E6E1, 0.0, 0.2, 2700.0),
    solid("granite", Stone, 0x5F5E5C, 0.0, 0.45, 2750.0),
    solid("ceramic_white", Stone, 0xF1F0EC, 0.0, 0.1, 2400.0),
    // Rubber and soft.
    solid("rubber_black", Rubber, 0x18191A, 0.0, 0.85, 1100.0),
    solid("silicone", Rubber, 0xDCDCD8, 0.0, 0.6, 1100.0),
    solid("leather_brown", Rubber, 0x6B4630, 0.0, 0.7, 860.0),
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
    }
}
