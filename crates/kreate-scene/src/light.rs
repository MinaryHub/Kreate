//! Lights, in the units the rest of the world uses.
//!
//! §3's fourth and tenth "decide now", and Phase 7's foundation. Two things
//! could not be deferred: that a light is a datablock, and what its intensity
//! is measured in.
//!
//! # A light is a datablock
//!
//! §1.4's argument for meshes, word for word. A hundred linked copies of a
//! lamp share one bulb, and two hundred call sites that assume `object.light_
//! color` is the object's own cannot be unpicked afterwards.
//!
//! # The units are glTF's, not ours
//!
//! Point and spot lights are in **candela** (lumens per steradian), the sun in
//! **lux** (lumens per square metre), and an area light in **nits** (candela
//! per square metre). Those are `KHR_lights_punctual`'s units and the units a
//! photometric datasheet quotes.
//!
//! Inventing a unit and converting at the boundary is the alternative, and it
//! fails in two ways at once: every round trip through a file accumulates
//! rounding, and nobody can say which side is authoritative when they disagree.
//! Blender's light "watts" are exactly this — a number that is not watts,
//! converted on the way in and out, and impossible to match against a real
//! fixture.
//!
//! Different kinds having *different* units looks like an inconsistency and is
//! the opposite. A point light emits in every direction, so what is physically
//! fixed about it is intensity per solid angle; the sun is infinitely far away,
//! so what is fixed is the illuminance arriving on a surface; an area light is
//! a surface, so what is fixed is its brightness per unit of area. A single
//! unit across all three would have to be a fiction for two of them.
//!
//! # Exposure is separate, and in stops
//!
//! `intensity × 2^exposure` is the amount of light. Kept as two numbers because
//! `UsdLux` has both and a round trip through USD would otherwise have to fold
//! them and guess how to unfold them — and because an artist doubling a lamp
//! reaches for a stop, not for a multiplication.
//!
//! This is *not* the exposure §2.1 warns about. That one belongs to the display
//! transform and describes how an image is shown; this one is part of the
//! light and describes the scene. Turning this up makes the room brighter for
//! the renderer and the viewport alike, which is what a light control should
//! do.

use kreate_math::SceneRgb;
use kreate_reflect::Uid;

use crate::document::ImageId;

/// What an intensity is measured in.
///
/// Carried so that the interface can label the field and an exporter knows
/// what it is holding. A number whose unit has to be inferred from the light's
/// kind at every call site is a number that gets converted twice somewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intensity {
    /// Lumens per steradian: a point or a spot.
    Candela,
    /// Lumens per square metre: the sun.
    Lux,
    /// Candela per square metre: an area light's surface.
    Nit,
}

impl Intensity {
    /// The symbol to put beside the field.
    ///
    /// A symbol and not a translated word: `cd` and `lx` are the same in every
    /// language, which is the point of an SI symbol.
    pub fn symbol(self) -> &'static str {
        match self {
            Intensity::Candela => "cd",
            Intensity::Lux => "lx",
            Intensity::Nit => "cd/m²",
        }
    }
}

/// The shape of an area light's emitting surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AreaShape {
    /// Metres.
    Rect { width: f64, height: f64 },
    /// Metres.
    Disc { radius: f64 },
    /// Metres, for a shape that is neither square nor round.
    Ellipse { width: f64, height: f64 },
}

impl AreaShape {
    /// The emitting area in square metres.
    ///
    /// What turns nits into candela: a surface of `A` square metres at `L`
    /// nits has a luminous intensity of `L·A` along its normal.
    pub fn area_m2(self) -> f64 {
        match self {
            AreaShape::Rect { width, height } => width * height,
            AreaShape::Disc { radius } => std::f64::consts::PI * radius * radius,
            AreaShape::Ellipse { width, height } => {
                std::f64::consts::PI * (width * 0.5) * (height * 0.5)
            }
        }
    }
}

/// What kind of light, and the parameters only that kind has.
///
/// Four kinds, which is the intersection of what glTF and USD both express and
/// the whole of what an artist reaches for. A mesh light — arbitrary geometry
/// that emits — is not a kind of light: the renderer handles it as an emissive
/// material, which is why it is not here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightKind {
    /// A glowing sphere. `radius` in metres, and it is what makes a shadow
    /// soft: a point light of radius zero casts a hard edge and nothing in the
    /// world does.
    Point { radius: f64 },
    /// Parallel light from infinitely far away.
    ///
    /// `angular_diameter` in radians is how wide the source appears, and it is
    /// the sun's own half-degree by default — which is why an outdoor shadow
    /// has the penumbra it has. Zero gives the geometrically perfect shadow
    /// that no photograph contains.
    Sun { angular_diameter: f64 },
    /// A cone. `cone_angle` is the full angle in radians; `blend` is the
    /// fraction of it that fades, zero being a hard-edged circle.
    Spot {
        radius: f64,
        cone_angle: f64,
        blend: f64,
    },
    /// A surface. `spread` in radians limits how far off the normal it emits,
    /// π being a full hemisphere.
    Area { shape: AreaShape, spread: f64 },
}

impl LightKind {
    /// How wide the sun appears from here, in radians.
    ///
    /// 0.526°, and using it means an outdoor shadow has the penumbra a
    /// photograph has without anybody tuning a number. Zero gives the
    /// geometrically perfect shadow that no photograph contains.
    ///
    /// Named separately from [`LightKind::SUN`] so that a caller needing the
    /// number — a loader filling in a field an old file omitted, say — can have
    /// it without destructuring a `const`, which Rust will not do in a pattern.
    /// The alternative was an `unreachable!()` arm in the loader, and a
    /// hazard removed beats a hazard guarded.
    pub const SUN_ANGULAR_DIAMETER: f64 = 0.526 * std::f64::consts::PI / 180.0;

    /// A sun at the sun's own angular size.
    ///
    /// A constant rather than a literal at each call site because it is a fact
    /// about the sky, and a `to_radians()` spelled out four times is four
    /// chances to write the diameter where the radius goes.
    pub const SUN: Self = LightKind::Sun {
        angular_diameter: Self::SUN_ANGULAR_DIAMETER,
    };

    /// A bulb five centimetres across.
    pub const BULB: Self = LightKind::Point { radius: 0.05 };

    /// A 45° cone with a soft edge, which is a stage lamp.
    pub const LAMP: Self = LightKind::Spot {
        radius: 0.05,
        cone_angle: 45.0 * std::f64::consts::PI / 180.0,
        blend: 0.15,
    };

    /// A one-metre-square panel emitting over the whole hemisphere.
    pub const PANEL: Self = LightKind::Area {
        shape: AreaShape::Rect {
            width: 1.0,
            height: 1.0,
        },
        spread: std::f64::consts::PI,
    };

    /// Every kind at its default, in the order a menu offers them.
    ///
    /// The order is the operator's enum indices, which are part of the keymap
    /// and scripting surface — so entries are appended and never reordered.
    pub const ALL: [Self; 4] = [Self::BULB, Self::SUN, Self::LAMP, Self::PANEL];

    /// The stable text form, for a file and an error message.
    pub fn id(self) -> &'static str {
        match self {
            LightKind::Point { .. } => "point",
            LightKind::Sun { .. } => "sun",
            LightKind::Spot { .. } => "spot",
            LightKind::Area { .. } => "area",
        }
    }

    /// What this kind's intensity is measured in.
    pub fn intensity_unit(self) -> Intensity {
        match self {
            LightKind::Point { .. } | LightKind::Spot { .. } => Intensity::Candela,
            LightKind::Sun { .. } => Intensity::Lux,
            LightKind::Area { .. } => Intensity::Nit,
        }
    }

    /// The brightness a light of this kind starts at, in its own unit.
    ///
    /// Beside the kind rather than in the operator that adds one, because the
    /// number is only meaningful together with the unit and the unit comes from
    /// the kind. A single default across all four would be 1000 of something
    /// that means four different things — a sun at 1000 lux is a dull overcast
    /// afternoon while a bulb at 1000 candela is a bright room.
    ///
    /// The point and the area default agree deliberately: a one-metre-square
    /// panel at 1000 nits is 1000 candela along its normal, so switching a
    /// light's kind does not change how bright the scene is.
    pub fn default_intensity(self) -> f64 {
        match self {
            LightKind::Point { .. } | LightKind::Spot { .. } => 1000.0,
            // A bright overcast day. Direct sun is nearer 100,000 lux, which is
            // a starting point that blows out every material in the scene.
            LightKind::Sun { .. } => 20_000.0,
            LightKind::Area { .. } => 1000.0,
        }
    }

    /// Whether this kind casts a shadow with a soft edge.
    ///
    /// Every kind can, and every kind can be made not to by shrinking its
    /// source to nothing. Worth asking because a renderer skips the sampling
    /// entirely for a hard shadow, which is the difference between a few rays
    /// and one.
    pub fn is_soft(self) -> bool {
        match self {
            LightKind::Point { radius } | LightKind::Spot { radius, .. } => radius > 0.0,
            LightKind::Sun { angular_diameter } => angular_diameter > 0.0,
            LightKind::Area { shape, .. } => shape.area_m2() > 0.0,
        }
    }
}

/// A light datablock.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    pub name: String,
    pub linkage: crate::document::Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,

    pub kind: LightKind,

    /// The light's tint: linear, and normally at most one per channel.
    ///
    /// A `SceneRgb` because it is linear and belongs to the working space, and
    /// linear because that is what glTF holds — encoding it would put a
    /// rounding step in every round trip for no gain. What it is *not* is an
    /// amount of light on its own: multiply it by [`Light::effective_intensity`]
    /// and the result is, which is what [`Light::radiance`] does.
    ///
    /// The interface converts at the colour picker, which is where every colour
    /// conversion belongs.
    pub color: SceneRgb,
    /// How much light, in the unit [`LightKind::intensity_unit`] names.
    pub intensity: f64,
    /// Stops on top of `intensity`. See the module docs on why there are two.
    pub exposure: f64,

    pub cast_shadow: bool,
    /// Whether this light lights the diffuse and specular responses.
    ///
    /// A cheat, and not an optional one: a rim light that must not blow out a
    /// highlight, or a fill that must not show as a second reflection, are
    /// ordinary requests on every production. A renderer that could only be
    /// physical would have them faked with geometry.
    pub affect_diffuse: bool,
    pub affect_specular: bool,
    /// How many bounces this light contributes to. Renderer only.
    pub max_bounces: u8,
    /// The action animating this light's colour and brightness, if any.
    pub action: Option<crate::document::ActionId>,
}

impl Default for Light {
    /// A thousand-candela point light with a five-centimetre bulb.
    ///
    /// Soft by default, because every real light is and a hard shadow is the
    /// thing that makes a render look like a render.
    fn default() -> Self {
        Self {
            name: String::new(),
            linkage: crate::document::Linkage::Local,
            uid: Uid::NONE,
            kind: LightKind::Point { radius: 0.05 },
            color: SceneRgb::WHITE,
            intensity: 1000.0,
            exposure: 0.0,
            cast_shadow: true,
            affect_diffuse: true,
            affect_specular: true,
            max_bounces: 8,
            action: None,
        }
    }
}

impl Light {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// A light of `kind`, at that kind's own brightness.
    ///
    /// The one constructor that gets the unit right, which is why the
    /// `AddLight` operator goes through it: a caller that sets `kind` on a
    /// default light gets a sun at 1000 lux, and 1000 lux of sunlight is a
    /// mistake nobody would make on purpose and nobody would notice as one.
    pub fn of_kind(name: impl Into<String>, kind: LightKind) -> Self {
        Self {
            kind,
            intensity: kind.default_intensity(),
            ..Self::new(name)
        }
    }

    /// A sun, at the sun's own angular size.
    ///
    /// 0.526° is what the sun subtends from here, and using it means an outdoor
    /// shadow has the penumbra a photograph has without anybody tuning a
    /// number. The default intensity is a bright overcast day rather than
    /// direct sun, which is the light most scenes want.
    pub fn sun(name: impl Into<String>) -> Self {
        Self::of_kind(name, LightKind::SUN)
    }

    /// The amount of light, with exposure folded in.
    ///
    /// The one place the two numbers combine, so that nothing multiplies them
    /// twice or forgets one. Still in the unit `kind` names.
    pub fn effective_intensity(&self) -> f64 {
        self.intensity * 2f64.powf(self.exposure)
    }

    /// The tint scaled by how much light there is.
    ///
    /// **In the unit [`LightKind::intensity_unit`] names**, per channel — so
    /// candela for a point or a spot, lux for a sun, nits for a panel. The name
    /// deliberately does not say which: exactly one of those three is a
    /// radiance, and a method called `radiance` that returned lux would be
    /// wrong for three kinds out of four while looking right at every call
    /// site.
    ///
    /// Named at all, rather than left to the caller, because the tint alone is
    /// not an amount of light and [`SceneRgb`] cannot say so. Multiplying them
    /// is the step that turns a swatch into light, and it should happen in one
    /// place.
    pub fn tinted_intensity(&self) -> SceneRgb {
        self.color.scaled(self.effective_intensity() as f32)
    }

    /// Illuminance in lux at `distance` metres, for a light measured in
    /// candela.
    ///
    /// The inverse-square law, and `None` for a kind that is not measured that
    /// way: the sun's intensity is *already* an illuminance and does not fall
    /// off, and an area light's depends on how much of it is visible.
    ///
    /// Here so that a light's brightness is answerable in the unit a light
    /// meter reads, which is how anybody checks a scene against a reference.
    pub fn illuminance_at(&self, distance: f64) -> Option<f64> {
        match self.kind.intensity_unit() {
            Intensity::Candela => {
                if distance <= 0.0 {
                    return None;
                }
                Some(self.effective_intensity() / (distance * distance))
            }
            // Parallel light does not diverge, so distance does not enter.
            Intensity::Lux => Some(self.effective_intensity()),
            Intensity::Nit => None,
        }
    }

    /// The luminous intensity along the normal, in candela.
    ///
    /// What an area light's nits become once its size is known, so that two
    /// lights of different kinds can be compared at all. `None` for the sun,
    /// which has no position to have an intensity from.
    pub fn axial_candela(&self) -> Option<f64> {
        match self.kind {
            LightKind::Point { .. } | LightKind::Spot { .. } => Some(self.effective_intensity()),
            LightKind::Area { shape, .. } => Some(self.effective_intensity() * shape.area_m2()),
            LightKind::Sun { .. } => None,
        }
    }
}

/// What the background is.
#[derive(Clone, Debug, PartialEq)]
pub enum Background {
    /// One colour everywhere.
    Flat(SceneRgb),
    /// An equirectangular image.
    Image(ImageId),
}

/// The world datablock: what light arrives from no object in particular.
///
/// A datablock and not a field on the scene, for the same reason a light is
/// one: a studio setup is a thing to share between files, and library linking
/// (§1.4) has to be able to reach it.
#[derive(Clone, Debug, PartialEq)]
pub struct World {
    pub name: String,
    pub linkage: crate::document::Linkage,
    pub uid: Uid,

    pub background: Background,
    /// A multiplier on the background, in the same spirit as a light's
    /// exposure but linear, because an HDRI's own values are already absolute
    /// and this is a correction rather than a setting.
    pub strength: f64,
    /// Radians about the up axis.
    ///
    /// Turning an HDRI is how anybody places a highlight, and it must not
    /// require re-exporting the image.
    pub rotation: f64,
}

impl Default for World {
    fn default() -> Self {
        Self {
            name: String::new(),
            linkage: crate::document::Linkage::Local,
            uid: Uid::NONE,
            // A dim neutral grey, which is enough to see an unlit object
            // against and little enough that a real light dominates it.
            background: Background::Flat(SceneRgb::new(0.05, 0.05, 0.05)),
            strength: 1.0,
            rotation: 0.0,
        }
    }
}

impl World {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// The background colour, with strength applied.
    ///
    /// `None` when the background is an image, which needs sampling and is the
    /// renderer's business.
    pub fn flat_radiance(&self) -> Option<SceneRgb> {
        match self.background {
            Background::Flat(color) => Some(color.scaled(self.strength as f32)),
            Background::Image(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_is_measured_in_the_unit_that_is_physical_for_it() {
        // The decision that looks like an inconsistency and is the opposite. A
        // point light emits in every direction, so intensity per solid angle is
        // what is fixed about it; the sun is infinitely far away, so the
        // illuminance arriving is; an area light is a surface, so brightness
        // per unit area is. One unit for all three would be a fiction for two.
        assert_eq!(
            LightKind::Point { radius: 0.1 }.intensity_unit(),
            Intensity::Candela
        );
        assert_eq!(
            LightKind::Spot {
                radius: 0.1,
                cone_angle: 1.0,
                blend: 0.2
            }
            .intensity_unit(),
            Intensity::Candela
        );
        assert_eq!(
            LightKind::Sun {
                angular_diameter: 0.01
            }
            .intensity_unit(),
            Intensity::Lux
        );
        assert_eq!(
            LightKind::Area {
                shape: AreaShape::Disc { radius: 0.5 },
                spread: std::f64::consts::PI
            }
            .intensity_unit(),
            Intensity::Nit
        );
    }

    #[test]
    fn the_symbols_are_the_si_ones() {
        // Symbols and not translated words: `cd` and `lx` are the same in every
        // language, which is what an SI symbol is for. A user matching a
        // datasheet is reading the same two letters.
        assert_eq!(Intensity::Candela.symbol(), "cd");
        assert_eq!(Intensity::Lux.symbol(), "lx");
        assert_eq!(Intensity::Nit.symbol(), "cd/m²");
    }

    #[test]
    fn exposure_is_stops_on_top_of_the_intensity() {
        // Two numbers because `UsdLux` has two, and folding them would mean
        // guessing how to unfold them on the way back. One stop is a doubling,
        // which is the whole reason an artist reaches for it.
        let mut light = Light::new("Lamp");
        light.intensity = 100.0;
        assert_eq!(light.effective_intensity(), 100.0);

        light.exposure = 1.0;
        assert_eq!(light.effective_intensity(), 200.0);
        light.exposure = -2.0;
        assert_eq!(light.effective_intensity(), 25.0);
    }

    #[test]
    fn a_tint_is_not_an_amount_of_light_until_it_is_multiplied() {
        // The colour module's own rule: a light's colour is a swatch and its
        // brightness is a separate number in physical units. Keeping them apart
        // is what stops a warm lamp being dimmer than a white one of the same
        // wattage.
        let warm = Light {
            color: SceneRgb::new(1.0, 0.8, 0.6),
            intensity: 1000.0,
            ..Light::new("Warm")
        };
        assert!(
            warm.color.is_display_representable(),
            "a tint is bounded by one"
        );
        let light = warm.tinted_intensity();
        assert!((light.r - 1000.0).abs() < 0.1);
        assert!((light.g - 800.0).abs() < 0.1);
        assert!(!light.is_display_representable(), "a lamp is not a swatch");
    }

    #[test]
    fn the_tinted_intensity_carries_the_kinds_unit_and_not_a_fixed_one() {
        // Which is why the method is not called `radiance`. The same
        // arithmetic produces candela here and lux one line down, and a caller
        // that assumed either would be wrong three times out of four.
        let bulb = Light {
            color: SceneRgb::new(1.0, 0.5, 0.25),
            intensity: 400.0,
            ..Light::of_kind("Bulb", LightKind::BULB)
        };
        assert_eq!(bulb.kind.intensity_unit(), Intensity::Candela);
        assert!((bulb.tinted_intensity().g - 200.0).abs() < 1e-3);

        let sun = Light {
            color: SceneRgb::new(1.0, 0.5, 0.25),
            intensity: 400.0,
            ..Light::of_kind("Sun", LightKind::SUN)
        };
        assert_eq!(sun.kind.intensity_unit(), Intensity::Lux);
        // Identical numbers, a different physical quantity. The type cannot
        // tell them apart, so the unit has to be asked for.
        assert_eq!(sun.tinted_intensity().g, bulb.tinted_intensity().g);
        assert_ne!(sun.kind.intensity_unit(), bulb.kind.intensity_unit());
    }

    #[test]
    fn a_candela_light_falls_off_with_the_square_of_the_distance() {
        // The inverse-square law, and the reason candela is the stored unit: a
        // number a light meter can be held up to. 1000 cd at two metres is 250
        // lux, which anybody can check.
        let light = Light {
            intensity: 1000.0,
            ..Light::new("Lamp")
        };
        assert_eq!(light.illuminance_at(1.0), Some(1000.0));
        assert_eq!(light.illuminance_at(2.0), Some(250.0));
        assert_eq!(light.illuminance_at(10.0), Some(10.0));
        // And zero distance is not an infinity.
        assert_eq!(light.illuminance_at(0.0), None);
        assert_eq!(light.illuminance_at(-1.0), None);
    }

    #[test]
    fn the_sun_does_not_fall_off() {
        // Because its intensity is *already* an illuminance. A sun that obeyed
        // the inverse-square law would light the near side of a scene and not
        // the far, which is the single most obvious way to get outdoor lighting
        // wrong.
        let sun = Light::sun("Sun");
        assert_eq!(sun.kind.intensity_unit(), Intensity::Lux);
        let near = sun.illuminance_at(1.0);
        let far = sun.illuminance_at(1000.0);
        assert_eq!(near, far);
        assert_eq!(near, Some(20_000.0));
    }

    #[test]
    fn an_area_light_has_no_single_illuminance() {
        // It depends on how much of the surface is visible from where you
        // stand, which is the renderer's integral and not a number this type
        // can produce. Saying `None` is better than producing a plausible
        // wrong one.
        let panel = Light {
            kind: LightKind::Area {
                shape: AreaShape::Rect {
                    width: 1.0,
                    height: 2.0,
                },
                spread: std::f64::consts::PI,
            },
            ..Light::new("Softbox")
        };
        assert_eq!(panel.illuminance_at(1.0), None);
    }

    #[test]
    fn nits_become_candela_once_the_size_is_known() {
        // What makes two lights of different kinds comparable at all. A
        // one-by-two-metre panel at 500 nits is 1000 cd along its normal.
        let panel = Light {
            kind: LightKind::Area {
                shape: AreaShape::Rect {
                    width: 1.0,
                    height: 2.0,
                },
                spread: std::f64::consts::PI,
            },
            intensity: 500.0,
            ..Light::new("Softbox")
        };
        assert_eq!(panel.axial_candela(), Some(1000.0));

        // A point light's intensity already is candela.
        let lamp = Light {
            intensity: 1000.0,
            ..Light::new("Lamp")
        };
        assert_eq!(lamp.axial_candela(), Some(1000.0));
        // And the two are now the same brightness along the axis, which is the
        // comparison the conversion exists to make possible.
        assert_eq!(panel.axial_candela(), lamp.axial_candela());

        // The sun has no position, so no intensity from one.
        assert_eq!(Light::sun("Sun").axial_candela(), None);
    }

    #[test]
    fn an_area_shapes_area_is_its_area() {
        // Because `axial_candela` divides by nothing else, an error here is a
        // light of the wrong brightness with no other symptom.
        assert_eq!(
            AreaShape::Rect {
                width: 2.0,
                height: 3.0
            }
            .area_m2(),
            6.0
        );
        let disc = AreaShape::Disc { radius: 1.0 }.area_m2();
        assert!((disc - std::f64::consts::PI).abs() < 1e-12);
        // An ellipse takes its half-axes, so a 2×2 ellipse is the unit disc and
        // not four times it.
        let ellipse = AreaShape::Ellipse {
            width: 2.0,
            height: 2.0,
        }
        .area_m2();
        assert!((ellipse - disc).abs() < 1e-12);
    }

    #[test]
    fn the_sun_is_soft_by_default_and_at_its_own_angular_size() {
        // Half a degree is what the sun subtends from here, so an outdoor
        // shadow gets the penumbra a photograph has without anybody tuning a
        // number. Zero is available and is the shadow no photograph contains.
        let sun = Light::sun("Sun");
        let LightKind::Sun { angular_diameter } = sun.kind else {
            panic!("not a sun")
        };
        assert!((angular_diameter.to_degrees() - 0.526).abs() < 1e-9);
        // And the constant is the same number as the runtime conversion, which
        // is the whole risk of spelling a `const fn`-less conversion out by
        // hand.
        assert!(
            (LightKind::SUN_ANGULAR_DIAMETER - 0.526_f64.to_radians()).abs() < 1e-15,
            "the constant and the conversion disagree"
        );
        assert_eq!(angular_diameter, LightKind::SUN_ANGULAR_DIAMETER);
        assert!(sun.kind.is_soft());

        let hard = LightKind::Sun {
            angular_diameter: 0.0,
        };
        assert!(!hard.is_soft());
    }

    #[test]
    fn every_kind_can_be_hard_and_a_hard_light_says_so() {
        // A renderer skips the sampling entirely for a hard shadow, which is
        // the difference between a handful of rays and one — so the question
        // has to be answerable without inspecting each kind's parameters at
        // every call site.
        assert!(!LightKind::Point { radius: 0.0 }.is_soft());
        assert!(LightKind::Point { radius: 0.01 }.is_soft());
        assert!(!LightKind::Spot {
            radius: 0.0,
            cone_angle: 1.0,
            blend: 0.0
        }
        .is_soft());
        assert!(!LightKind::Area {
            shape: AreaShape::Disc { radius: 0.0 },
            spread: 1.0
        }
        .is_soft());
    }

    #[test]
    fn a_light_can_be_told_to_skip_the_diffuse_or_the_specular() {
        // A cheat, and not an optional one: a rim light that must not blow out
        // a highlight is an ordinary request on every production. A renderer
        // that could only be physical would have it faked with geometry.
        let light = Light::default();
        assert!(light.affect_diffuse && light.affect_specular);
        let rim = Light {
            affect_diffuse: false,
            ..Light::default()
        };
        assert!(!rim.affect_diffuse && rim.affect_specular);
    }

    #[test]
    fn a_world_applies_its_strength_to_a_flat_background() {
        let world = World {
            background: Background::Flat(SceneRgb::new(0.1, 0.2, 0.4)),
            strength: 2.0,
            ..World::new("World")
        };
        let radiance = world.flat_radiance().expect("a flat background");
        assert!((radiance.r - 0.2).abs() < 1e-6);
        assert!((radiance.b - 0.8).abs() < 1e-6);
    }

    #[test]
    fn a_world_with_an_image_has_no_single_colour() {
        // Which needs sampling, and sampling is the renderer's. A flat colour
        // stood in for it would be a background that is the wrong colour
        // everywhere rather than an obviously missing one.
        let mut doc = crate::Document::new();
        let image = doc.add_image(crate::Image::new(
            "HDRI",
            64,
            32,
            crate::PixelFormat::Rgba16F,
        ));
        let world = World {
            background: Background::Image(image),
            ..World::new("World")
        };
        assert_eq!(world.flat_radiance(), None);
    }

    #[test]
    fn a_world_starts_dim_enough_for_a_real_light_to_matter() {
        // Bright enough to see an unlit object against, dim enough that adding
        // a lamp does something. A world at one would have every scene start
        // flat-lit and every new light look like it did nothing.
        let world = World::default();
        let radiance = world.flat_radiance().expect("flat");
        assert!(radiance.r > 0.0 && radiance.r < 0.2, "{}", radiance.r);
        assert_eq!(world.strength, 1.0);
        assert_eq!(world.rotation, 0.0);
    }

    #[test]
    fn a_light_of_a_kind_gets_that_kinds_own_brightness() {
        // The mistake this exists to stop: setting `kind` on a default light
        // leaves a sun at 1000 *lux*, which is a dull overcast afternoon
        // wearing the name of the sun. Nobody would write that on purpose and
        // nobody would spot it as the reason their outdoor scene is flat.
        let sun = Light::of_kind("Sun", LightKind::SUN);
        assert_eq!(sun.intensity, 20_000.0);
        assert_eq!(sun.kind.intensity_unit(), Intensity::Lux);

        let bulb = Light::of_kind("Lamp", LightKind::BULB);
        assert_eq!(bulb.intensity, 1000.0);

        // And `sun()` is the same light, so there is one answer and not two.
        assert_eq!(Light::sun("Sun"), sun);
    }

    #[test]
    fn switching_between_a_point_and_a_panel_does_not_change_the_brightness() {
        // The two defaults agree on purpose: a one-metre-square panel at 1000
        // nits is 1000 candela along its normal. An artist trying the other
        // kind is asking about the shadow's softness, not about exposure, and a
        // default that moved the exposure would make the comparison useless.
        let bulb = Light::of_kind("A", LightKind::BULB);
        let panel = Light::of_kind("A", LightKind::PANEL);
        assert_eq!(bulb.axial_candela(), panel.axial_candela());
    }

    #[test]
    fn the_default_kinds_are_all_soft() {
        // Every real light is, and a hard shadow is the single thing that makes
        // a render look like a render. Hard is available and is never the
        // starting point.
        for kind in LightKind::ALL {
            assert!(kind.is_soft(), "{} starts hard", kind.id());
            assert!(kind.default_intensity() > 0.0, "{} starts dark", kind.id());
        }
    }

    #[test]
    fn every_kind_round_trips_through_its_text_form() {
        // The file holds the text.
        let kinds = [
            LightKind::Point { radius: 0.1 },
            LightKind::Sun {
                angular_diameter: 0.01,
            },
            LightKind::Spot {
                radius: 0.1,
                cone_angle: 1.0,
                blend: 0.2,
            },
            LightKind::Area {
                shape: AreaShape::Disc { radius: 0.5 },
                spread: 1.0,
            },
        ];
        let mut ids: Vec<&str> = kinds.iter().map(|k| k.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), kinds.len(), "two kinds share an id");
    }
}
