//! The document's lights, in the form the renderer takes.
//!
//! `kreate-render` cannot see a document — it depends on `kreate-math` and
//! `kreate-mesh` and nothing else of ours, the same as `kreate-ui`, because a
//! presentation crate consumes a snapshot and never reaches for the scene. So
//! the join lives here, in the app, which is the only place that can see both.
//!
//! That turns out to be the better arrangement rather than a concession. The
//! unit reasoning stays in `kreate-scene` beside the units — `axial_candela`
//! knows how each kind's brightness becomes an intensity from a point,
//! including an ellipse's area — and the renderer receives one unit and one
//! falloff rule. The alternative was four branches of unit arithmetic in the
//! shader, per light, per pixel.
//!
//! In `main.rs` this would be unguarded by construction, which is why it is
//! here: every decision below is a function of the document and a test can make
//! one.

use kreate_render::lights::{Falloff, PlacedLight};
use kreate_scene::{Document, LightKind};

/// Every light in the scene, placed and converted.
///
/// Order follows [`Document::lit_by`], which is arena slot order. Stable across
/// frames for an unchanged scene, which is what stops a light's index — and so
/// its shadow map — from moving under it.
pub fn placed_lights(doc: &Document) -> Vec<PlacedLight> {
    doc.lit_by()
        .map(|(_, light, world)| place(light, &world))
        .collect()
}

/// One light, placed by the transform of the object carrying it.
fn place(light: &kreate_scene::Light, world: &kreate_math::DAffine3) -> PlacedLight {
    // A light points down its object's -Z: glTF's convention and Blender's.
    // Transformed by the linear part, and left un-normalised because the packer
    // normalises — a non-uniform scale means this is not unit length, and doing
    // it in one place is better than doing it in two.
    let direction = (world.matrix3 * kreate_math::DVec3::NEG_Z).as_vec3();

    // Which unit the renderer gets, and how much.
    //
    // `axial_candela` is the one place that knows how each kind's brightness
    // becomes an intensity measured from a point, and it answers `None` for
    // exactly one kind: the sun, which has no position to have an intensity
    // from. So the two arms below are "measured from a point" and "parallel",
    // which is precisely the distinction the renderer's `Falloff` draws — and a
    // kind added later that has no axial intensity lands in the arm that does
    // not divide by a distance, which is the safe side.
    let (falloff, scale) = match light.axial_candela() {
        Some(candela) => (Falloff::InverseSquare, candela),
        None => (Falloff::Parallel, light.effective_intensity()),
    };

    // The tint applied to whichever scalar the falloff needs.
    //
    // `Light::tinted_intensity` is the tint times the amount in the *kind's*
    // own unit, and for every kind but an area light that is the same number as
    // this. An area light's nits become candela through its area, which is what
    // `axial_candela` has already accounted for, so the multiplication happens
    // here rather than there.
    let amount = light.color.scaled(scale as f32).to_array();

    PlacedLight {
        falloff,
        position: world.translation,
        direction,
        amount,
        radius: source_radius(light.kind),
        angular_radius: match light.kind {
            // Radius, not diameter. The datablock stores the diameter because
            // that is what an almanac quotes; the shader wants the half-angle
            // because that is what it compares a dot product against.
            LightKind::Sun { angular_diameter } => (angular_diameter * 0.5) as f32,
            _ => 0.0,
        },
        cone: match light.kind {
            LightKind::Spot {
                cone_angle, blend, ..
            } => Some((cone_angle as f32, blend as f32)),
            _ => None,
        },
        cast_shadow: light.cast_shadow,
        affect_diffuse: light.affect_diffuse,
        affect_specular: light.affect_specular,
    }
}

/// How big the source is, in metres, for a soft shadow and a highlight.
///
/// An area light arrives as the disc of the same area, because the viewport
/// treats it as a point at its axial candela and a disc is the shape a soft
/// shadow is sampled from. A two-by-one panel and a one-point-six-metre disc
/// cast nearly the same penumbra, which is what this number is for; the exact
/// shape matters to the renderer's integral and not to a raster pass.
fn source_radius(kind: LightKind) -> f32 {
    match kind {
        LightKind::Point { radius } | LightKind::Spot { radius, .. } => radius as f32,
        // The sun's size is an angle, not a length. It travels in
        // `angular_radius`, and a length here would be meaningless.
        LightKind::Sun { .. } => 0.0,
        LightKind::Area { shape, .. } => (shape.area_m2() / std::f64::consts::PI).sqrt() as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::{AreaShape, Light};

    /// A document with one light of `kind`, at the origin.
    fn one(kind: LightKind) -> (Document, PlacedLight) {
        let mut doc = Document::new();
        let id = doc.add_light(Light::of_kind("Lamp", kind));
        doc.add_object(Object::new("Lamp", ObjectData::Light(id)));
        let placed = placed_lights(&doc);
        assert_eq!(placed.len(), 1);
        (doc, placed[0])
    }

    #[test]
    fn a_point_light_arrives_in_candela_and_falls_off() {
        // Its stored unit is already what the renderer wants, so the number
        // must cross unchanged. A conversion here would be a rounding step for
        // nothing.
        let (_, placed) = one(LightKind::BULB);
        assert_eq!(placed.falloff, Falloff::InverseSquare);
        assert_eq!(placed.amount, [1000.0; 3]);
        assert_eq!(placed.radius, 0.05);
        assert_eq!(placed.cone, None);
    }

    #[test]
    fn a_sun_arrives_in_lux_and_does_not_fall_off() {
        // The one kind with no position to have an intensity from, which is
        // exactly the kind that must not be divided by a distance. A sun on the
        // inverse-square path would light the near side of a scene and not the
        // far.
        let (_, placed) = one(LightKind::SUN);
        assert_eq!(placed.falloff, Falloff::Parallel);
        assert_eq!(placed.amount, [20_000.0; 3]);
        // Its size is an angle, and a length would be meaningless.
        assert_eq!(placed.radius, 0.0);
        let half = (LightKind::SUN_ANGULAR_DIAMETER * 0.5) as f32;
        assert!((placed.angular_radius - half).abs() < 1e-9);
    }

    #[test]
    fn an_area_lights_nits_become_candela_through_its_area() {
        // The conversion that would otherwise have to happen per pixel, and the
        // reason it happens here: it needs the area of an ellipse.
        //
        // A one-by-two-metre panel at 500 nits is 1000 candela along its
        // normal, which is the same brightness as a 1000 cd bulb — so switching
        // kinds asks about the shadow rather than about exposure.
        let mut doc = Document::new();
        let mut panel = Light::of_kind(
            "Softbox",
            LightKind::Area {
                shape: AreaShape::Rect {
                    width: 1.0,
                    height: 2.0,
                },
                spread: std::f64::consts::PI,
            },
        );
        panel.intensity = 500.0;
        let id = doc.add_light(panel);
        doc.add_object(Object::new("Softbox", ObjectData::Light(id)));

        let placed = placed_lights(&doc)[0];
        assert_eq!(placed.falloff, Falloff::InverseSquare);
        assert!((placed.amount[0] - 1000.0).abs() < 1e-3, "{:?}", placed.amount);

        // And the source arrives as the disc of the same area, which is what a
        // soft shadow is sampled from.
        let expected = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
        assert!((placed.radius - expected).abs() < 1e-6, "{}", placed.radius);
    }

    #[test]
    fn exposure_reaches_the_renderer_for_every_kind() {
        // Because the renderer has no second exposure to apply, and the two
        // exposures in the product must not both be applied to the same light.
        //
        // Every kind, and a falsification sweep is why. This test used to use a
        // point light only, which goes through `axial_candela` — and that
        // already folds the exposure in. Dropping the exposure from the *sun's*
        // arm survived, because the sun's amount comes from
        // `effective_intensity` and nothing here went down that path.
        for kind in LightKind::ALL {
            let mut doc = Document::new();
            let mut lamp = Light::of_kind("Lamp", kind);
            lamp.intensity = 100.0;
            let id = doc.add_light(lamp);
            doc.add_object(Object::new("Lamp", ObjectData::Light(id)));
            let flat = placed_lights(&doc)[0].amount[0];

            let mut doc = Document::new();
            let mut lamp = Light::of_kind("Lamp", kind);
            lamp.intensity = 100.0;
            lamp.exposure = 2.0;
            let id = doc.add_light(lamp);
            doc.add_object(Object::new("Lamp", ObjectData::Light(id)));
            let raised = placed_lights(&doc)[0].amount[0];

            // Two stops is four times the light, whatever the unit.
            assert!(
                (raised - flat * 4.0).abs() < flat * 1e-4,
                "{}: {flat} then {raised}",
                kind.id()
            );
        }
    }

    #[test]
    fn the_tint_is_applied_and_is_not_an_amount_of_light_on_its_own() {
        let mut doc = Document::new();
        let mut warm = Light::of_kind("Warm", LightKind::BULB);
        warm.color = kreate_math::SceneRgb::new(1.0, 0.5, 0.25);
        warm.intensity = 400.0;
        let id = doc.add_light(warm);
        doc.add_object(Object::new("Warm", ObjectData::Light(id)));

        let placed = placed_lights(&doc)[0];
        assert!((placed.amount[0] - 400.0).abs() < 1e-3);
        assert!((placed.amount[1] - 200.0).abs() < 1e-3);
        assert!((placed.amount[2] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn a_light_points_down_its_objects_negative_z() {
        // glTF's convention and Blender's. An object with no rotation points
        // straight down, which is what a new lamp should do.
        let (_, placed) = one(LightKind::BULB);
        assert!((placed.direction - glam::Vec3::NEG_Z).length() < 1e-6);

        // Turned a quarter turn about X, it points along -Y.
        let mut doc = Document::new();
        let id = doc.add_light(Light::of_kind("Lamp", LightKind::SUN));
        let mut object = Object::new("Lamp", ObjectData::Light(id));
        object.transform.rotation =
            glam::DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2);
        doc.add_object(object);

        let placed = placed_lights(&doc)[0];
        assert!(
            (placed.direction - glam::Vec3::NEG_Y).length() < 1e-6,
            "{:?}",
            placed.direction
        );
    }

    #[test]
    fn a_light_is_placed_by_its_world_transform_and_not_its_local_one() {
        // A lamp parented to a rig moves with the rig. Reading the local
        // transform would leave every rigged light at the origin, and the
        // symptom is a scene that lights from the wrong place with nothing
        // obviously wrong in the outliner.
        let mut doc = Document::new();
        let id = doc.add_light(Light::of_kind("Lamp", LightKind::BULB));
        let rig = doc.add_object(Object::new("Rig", ObjectData::Empty));
        let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(id)));
        doc.set_parent(lamp, Some(rig)).expect("parent");
        doc.object_mut(rig).expect("rig").transform.translation =
            kreate_math::DVec3::new(0.0, 0.0, 7.0);

        assert_eq!(placed_lights(&doc)[0].position.z, 7.0);
    }

    #[test]
    fn a_spots_cone_travels_as_the_full_angle_and_the_blend() {
        // The renderer turns them into two cosines, which is its business. What
        // matters here is that both numbers arrive: a cone with no blend and a
        // blend with no cone are both wrong in ways that look deliberate.
        let (_, placed) = one(LightKind::LAMP);
        let (angle, blend) = placed.cone.expect("a spot has a cone");
        assert!((angle - 45f32.to_radians()).abs() < 1e-6, "{angle}");
        assert!((blend - 0.15).abs() < 1e-6, "{blend}");
    }

    #[test]
    fn only_a_spot_gets_a_cone() {
        // A cone on a point light would darken most of the scene, and a cone
        // on a sun would darken all of it.
        for kind in [LightKind::BULB, LightKind::SUN, LightKind::PANEL] {
            let (_, placed) = one(kind);
            assert_eq!(placed.cone, None, "{} got a cone", kind.id());
        }
    }

    #[test]
    fn the_switches_cross_unchanged() {
        // The lighting cheats. A rim light that must not blow out a highlight
        // is an ordinary request, and one dropped here would be faked with
        // geometry instead.
        let mut doc = Document::new();
        let mut rim = Light::of_kind("Rim", LightKind::PANEL);
        rim.cast_shadow = false;
        rim.affect_diffuse = false;
        let id = doc.add_light(rim);
        doc.add_object(Object::new("Rim", ObjectData::Light(id)));

        let placed = placed_lights(&doc)[0];
        assert!(!placed.cast_shadow);
        assert!(!placed.affect_diffuse);
        assert!(placed.affect_specular);
    }

    #[test]
    fn a_hidden_light_does_not_reach_the_renderer() {
        // `lit_by` skips invisible objects, and this is the test that says the
        // app relies on that rather than filtering again itself. Two filters
        // are two places to forget one.
        let mut doc = Document::new();
        let id = doc.add_light(Light::of_kind("Lamp", LightKind::BULB));
        let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(id)));
        assert_eq!(placed_lights(&doc).len(), 1);

        doc.object_mut(lamp).expect("lamp").visible = false;
        assert!(placed_lights(&doc).is_empty());
    }

    #[test]
    fn two_objects_sharing_one_bulb_are_two_lights() {
        // The datablock is shared; the placements are not. A join that
        // deduplicated by datablock would light the scene from one of the two
        // positions and drop the other.
        let mut doc = Document::new();
        let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
        let mut left = Object::new("Left", ObjectData::Light(bulb));
        left.transform.translation = kreate_math::DVec3::new(-3.0, 0.0, 0.0);
        doc.add_object(left);
        let mut right = Object::new("Right", ObjectData::Light(bulb));
        right.transform.translation = kreate_math::DVec3::new(3.0, 0.0, 0.0);
        doc.add_object(right);

        let placed = placed_lights(&doc);
        assert_eq!(placed.len(), 2);
        let xs: Vec<f64> = placed.iter().map(|l| l.position.x).collect();
        assert!(xs.contains(&-3.0) && xs.contains(&3.0), "{xs:?}");
        // And they are the same light otherwise, because they are one bulb.
        assert_eq!(placed[0].amount, placed[1].amount);
    }

    #[test]
    fn a_scene_with_no_lights_is_an_empty_list_and_not_a_default_one() {
        // A renderer handed a stand-in light would light every unlit scene, and
        // the user would have no way to turn it off.
        let mut doc = Document::new();
        assert!(placed_lights(&doc).is_empty());
        doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));
        assert!(placed_lights(&doc).is_empty());
    }
}
