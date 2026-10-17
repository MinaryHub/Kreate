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
/// The material preview's own light: what Blender's Material Preview lights
/// a material with, so it can be judged before the scene has a lamp.
pub struct Preview {
    /// The flat sky behind the studio, for what reads a flat world.
    pub world: [f32; 3],
    /// The studio's key lamp, for its shadow.
    pub lights: Vec<PlacedLight>,
    /// Stops: the light is the application's, so its exposure is too.
    pub exposure: f32,
    /// How far the studio is turned about up, in radians: see
    /// [`kreate_render::studio`].
    pub yaw: f32,
}

/// How bright the studio's key lamp is, beside the softbox it stands for:
/// the softbox lights through the environment and casts no shadow, so the
/// lamp is the share of its light that does.
const STUDIO_KEY_LUX: f32 = 10.0;

/// The exposure the studio is shown at.
pub const STUDIO_EXPOSURE: f32 = -0.6;

/// The studio, turned by `yaw` about up.
fn studio(yaw: f32) -> Preview {
    let turn = glam::Quat::from_rotation_z(yaw);
    let from = turn * kreate_render::studio::key_direction();
    Preview {
        world: [0.5, 0.5, 0.52],
        lights: vec![PlacedLight {
            falloff: Falloff::Parallel,
            direction: -from.normalize(),
            amount: [STUDIO_KEY_LUX; 3],
            angular_radius: 0.1,
            cast_shadow: true,
            ..Default::default()
        }],
        exposure: STUDIO_EXPOSURE,
        yaw,
    }
}

/// The preview light for a view turned by `view`: the studio turns with it
/// about up, as Blender's preview light does, so the key is always over the
/// viewer's left shoulder and the side being looked at is lit.
pub fn material_preview(view: glam::DQuat) -> Preview {
    let ahead = view.as_quat() * -glam::Vec3::Z;
    // The studio is laid out for a viewer looking along +Y.
    let yaw = if ahead.truncate().length_squared() > 1e-8 {
        (-ahead.x).atan2(ahead.y)
    } else {
        0.0
    };
    studio(yaw)
}

/// The studio the rendered view uses when the scene brings no light of its
/// own: fixed in the world, so turning round a part shows its other side
/// in shadow, as a real room would.
pub fn rendered_studio() -> Preview {
    // Turned to the default view, which looks across the diagonal: laid out
    // for a view along +Y, the key stood almost behind that viewer and every
    // shadow fell out of sight behind its part.
    studio(DEFAULT_VIEW_YAW)
}

/// The default view's turn about up from +Y, as `material_preview` measures
/// it: the new document's camera looks along about (0.5, 0.74).
const DEFAULT_VIEW_YAW: f32 = -0.6;

/// Whether the rendered view lights the scene with the studio: when it has
/// no lamp and its sky is not a picture. A scene lit only by a flat grey
/// sky is a scene with no light in it, and showing it so was a flat grey
/// picture; a lamp, or a sky image, is the user choosing their own.
pub fn uses_studio(doc: &Document) -> bool {
    doc.lit_by().next().is_none() && crate::environment::background_image(doc).is_none()
}

pub fn placed_lights(doc: &Document) -> Vec<PlacedLight> {
    doc.lit_by()
        .map(|(_, light, world)| place(light, &world))
        .collect()
}

/// The world's radiance, or black.
///
/// Black for a document with no world, and for one whose background is an
/// image: sampling a cube map is the IBL step and needs the archive's
/// `images/` layout first. Black rather than a stand-in colour, because a
/// scene lit only by its lamps is a real thing to want and an invented ambient
/// is one the user cannot turn off.
pub fn world_radiance(doc: &Document) -> [f32; 3] {
    doc.active_world()
        .and_then(|id| doc.world(id))
        .and_then(|world| world.flat_radiance())
        .map(|c| c.to_array())
        .unwrap_or([0.0; 3])
}

/// The exposure the shot is taken at, in stops.
///
/// The bridge between §2.2's photometric lights and §2.1's display transform.
/// A sun is twenty thousand lux and AgX expects scene-linear values where 0.18
/// is mid grey, so without this a physically correct scene saturates every
/// pixel — which is what the rendered viewport's first tests measured.
///
/// From the *scene* camera and not the viewport's orbit camera. The exposure
/// is a property of the shot: looking around should not change how bright the
/// room is, any more than turning your head does.
///
/// The first camera in the document, and a default one when there is none.
/// Not an error: a scene being modelled usually has no camera yet, and it
/// still has to be lookable-at.
pub fn shot_exposure(doc: &Document, rate: f64, subject: kreate_math::DVec3) -> f64 {
    shot(doc, rate, subject).exposure
}

/// How the picture is being exposed, and where the number came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    /// Stops, for the display transform.
    pub exposure: f64,
    /// Whether the meter produced it, rather than the camera's dials.
    ///
    /// Both answers come from here together, because they are one decision: a
    /// caller that asked for the exposure and then asked separately whether it
    /// was metered would have two chances to get a different answer, and the
    /// header would end up explaining a number it was not showing.
    pub metered: bool,
}

/// The exposure for this document's picture, and how it was arrived at.
pub fn shot(doc: &Document, rate: f64, subject: kreate_math::DVec3) -> Shot {
    let reading = metered_illuminance(doc, subject);
    // A document with no camera object still has a picture to expose, and the
    // defaults are the ones a new document gets.
    let fallback;
    let camera = match doc.cameras().next() {
        Some((_, camera)) => camera,
        None => {
            fallback = kreate_scene::Camera::new("");
            &fallback
        }
    };
    Shot {
        exposure: camera.metered_exposure(reading, rate),
        // The same condition `Camera::metered_exposure` applies, and the reason
        // it is spelled twice rather than returned from there: that function is
        // about a number, and this is about what to tell the user.
        metered: camera.auto_exposure && reading.is_some_and(|lux| lux > 0.0),
    }
}

/// How far below the scene's own light a lamp has to fall before it is dropped.
///
/// A thousandth, which is about ten stops under whatever the shot is exposed
/// for. Near mid grey a thousandth of the scene's light is under one code
/// value, so a lamp culled at this threshold cannot move a pixel there; in the
/// shadows the local level is lower and the same fraction is a larger share of
/// it, which is the error and is stated rather than hidden.
///
/// It is a *quality* cutoff and not the exact one. The exact one — the distance
/// past which a lamp cannot change a pixel at all — is
/// `kreate_render::lights::cull_range` against the display's black point, and
/// it comes out in the hundreds of metres for any lamp bright enough to matter.
/// Correct, and correct at a scale no viewport reaches: with it, `light_cost`'s
/// two hundred and fifty-six lamps are all in every tile and nothing is culled.
///
/// The honest summary is that culling needs a threshold somebody chose, and
/// this is the choice.
const CULL_FRACTION: f64 = 1.0 / 1000.0;

/// The lights of the scene, each with the distance past which it is not worth
/// computing.
///
/// The range comes from the same meter the exposure does, which is what makes
/// it scene-relative rather than absolute: a lamp is worth computing while it
/// is within a thousandth of what else is lighting the place. One reading, two
/// uses, and no second opinion about how bright the scene is.
///
/// `subject` is what the picture is about — the same point the exposure is
/// metered at.
pub fn culled_lights(doc: &Document, subject: kreate_math::DVec3) -> Vec<PlacedLight> {
    let mut lights = placed_lights(doc);
    let Some(reading) = metered_illuminance(doc, subject) else {
        // Nothing lighting anything. There is no scene level to be a thousandth
        // of, and a range worked out from zero is not a number — so nothing is
        // culled, which costs a loop over a light list that is about to produce
        // a black picture anyway.
        return lights;
    };
    let cutoff = reading * CULL_FRACTION;
    for light in &mut lights {
        if light.range.is_some() {
            // Already decided — by an artist, when the datablock grows the
            // field, or by a caller that knows something this does not.
            continue;
        }
        let brightest = light.amount.iter().copied().fold(0.0f32, f32::max) as f64;
        // Inverse square: the distance at which this lamp's own illuminance
        // falls to the cutoff.
        light.range = Some((brightest / cutoff.max(f64::MIN_POSITIVE)).sqrt() as f32);
    }
    lights
}

/// What a light meter at `subject` would read, in lux.
///
/// The sum of every light reaching the point, plus the world. A flat number and
/// not a direction, because that is what an incident meter reads and what the
/// grey-card calibration is stated against — the alternative is a normal, and
/// the subject has not been chosen yet when the exposure is.
///
/// `None` when nothing reaches it. An unlit scene has no exposure that helps,
/// and pretending otherwise would send the dial to infinity the moment the last
/// lamp was switched off.
///
/// # Why at a point rather than over the picture
///
/// Averaging the frame is the other way to meter and it is the one that makes a
/// viewport flicker: an object moving through shot changes the average, so the
/// walls change brightness as the camera turns. A spot reading at what the user
/// is looking at is steady, and it is what a photographer does.
pub fn metered_illuminance(doc: &Document, subject: kreate_math::DVec3) -> Option<f64> {
    let mut total = 0.0f64;
    for placed in placed_lights(doc) {
        let packed = kreate_render::lights::GpuLight::pack(&placed, subject);
        // At the subject itself, which is the origin of the eye-relative space
        // the packing just put the light into.
        let lux = packed.illuminance_at(kreate_math::glam::Vec3::ZERO);
        total += luminous(lux);
    }
    // The world arrives as a radiance in every direction; what a meter facing
    // any of them reads from a full hemisphere is π times it.
    let world = world_radiance(doc);
    total += std::f64::consts::PI * luminous(world);
    (total > 0.0).then_some(total)
}

/// One number from three channels, weighted the way brightness is seen.
///
/// Rec.709 luminance. A meter reads light, not colour, and summing the channels
/// instead would make a blue lamp meter as bright as a green one of the same
/// power — which is not what either of them looks like.
fn luminous(rgb: [f32; 3]) -> f64 {
    0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64
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
        // No end. Inverse square does not have one, and the distance past
        // which a lamp cannot change a pixel depends on the display it is being
        // looked at through — which the document does not know and this
        // function does not have. The renderer fills it in for the pass that
        // culls against it; when the datablock grows an artist-set distance, as
        // Blender's has, it arrives here instead.
        range: None,
        // The path tracer draws a panel for it instead. See `area_panels`.
        traced_as_geometry: matches!(light.kind, LightKind::Area { .. }),
    }
}

/// An area light's panel, for the path tracer: its outline in the light's own
/// XY plane, facing -Z the way the light shines, placed by `transform`, and
/// glowing with `radiance` — nits, the unit an area light's intensity is in.
pub struct AreaPanel {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<kreate_math::Vec3>,
    pub indices: Vec<u32>,
    pub transform: kreate_math::DAffine3,
    pub radiance: [f32; 3],
}

/// Every visible area light's panel.
///
/// What the path tracer lights an area light with: the surface it is, glowing
/// from its front, aimed at as any glowing surface is — rather than the sphere
/// the rasterised view stands in for it, which shines every way at once.
pub fn area_panels(doc: &Document) -> Vec<AreaPanel> {
    use kreate_scene::AreaShape;
    doc.lit_by()
        .filter_map(|(_, light, world)| {
            let LightKind::Area { shape, .. } = light.kind else {
                return None;
            };
            let (positions, indices) = match shape {
                AreaShape::Rect { width, height } => {
                    let (x, y) = ((width * 0.5) as f32, (height * 0.5) as f32);
                    (
                        vec![[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]],
                        vec![0, 2, 1, 0, 3, 2],
                    )
                }
                AreaShape::Disc { radius } => ellipse(radius as f32, radius as f32),
                AreaShape::Ellipse { width, height } => {
                    ellipse((width * 0.5) as f32, (height * 0.5) as f32)
                }
            };
            let normals = vec![kreate_math::Vec3::NEG_Z; positions.len()];
            Some(AreaPanel {
                positions,
                normals,
                indices,
                transform: world,
                radiance: light.tinted_intensity().to_array(),
            })
        })
        .collect()
}

/// An ellipse of half-widths `a` and `b` as a fan of 32 triangles.
fn ellipse(a: f32, b: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    const SIDES: u32 = 32;
    let mut positions = vec![[0.0; 3]];
    let mut indices = Vec::new();
    for i in 0..SIDES {
        let t = i as f32 / SIDES as f32 * std::f32::consts::TAU;
        positions.push([a * t.cos(), b * t.sin(), 0.0]);
        indices.extend([0, 1 + i, 1 + (i + 1) % SIDES]);
    }
    (positions, indices)
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

    #[test]
    fn an_area_light_is_a_panel_to_the_path_tracer_and_not_a_lamp() {
        use kreate_scene::AreaShape;
        let (doc, placed) = one(LightKind::Area {
            shape: AreaShape::Rect {
                width: 2.0,
                height: 1.0,
            },
            spread: std::f64::consts::PI,
        });
        assert!(
            placed.traced_as_geometry,
            "the lamp sampling would count it twice"
        );
        let panels = area_panels(&doc);
        assert_eq!(panels.len(), 1);
        let p = &panels[0];
        // Two by one, in its own plane, facing down its -Z.
        let (xs, ys): (Vec<f32>, Vec<f32>) = p.positions.iter().map(|v| (v[0], v[1])).unzip();
        assert_eq!(xs.iter().cloned().fold(f32::MIN, f32::max), 1.0);
        assert_eq!(ys.iter().cloned().fold(f32::MIN, f32::max), 0.5);
        assert!(p.normals.iter().all(|n| *n == kreate_math::Vec3::NEG_Z));
        assert!(p.radiance[0] > 0.0);
        // A point light is still a lamp.
        let (_, point) = one(LightKind::Point { radius: 0.1 });
        assert!(!point.traced_as_geometry);
    }
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
    fn a_scene_with_no_world_is_lit_only_by_its_lamps() {
        // Black, and not an invented ambient. A stand-in the user cannot turn
        // off is worse than a dark scene they can light.
        let doc = Document::new();
        assert_eq!(world_radiance(&doc), [0.0; 3]);
    }

    #[test]
    fn the_active_world_lights_the_scene_and_its_strength_is_applied() {
        let mut doc = Document::new();
        let id = doc.add_world(kreate_scene::World {
            background: kreate_scene::Background::Flat(kreate_math::SceneRgb::new(0.1, 0.2, 0.4)),
            strength: 2.0,
            ..kreate_scene::World::new("Studio")
        });
        // Present but not active: a document can hold a day setup and a night
        // one, and only the active one lights anything.
        assert_eq!(world_radiance(&doc), [0.0; 3]);

        doc.set_active_world(Some(id));
        let lit = world_radiance(&doc);
        assert!((lit[0] - 0.2).abs() < 1e-6, "{lit:?}");
        assert!((lit[2] - 0.8).abs() < 1e-6, "{lit:?}");
    }

    #[test]
    fn an_image_background_lights_nothing_yet() {
        // Sampling one is the IBL step, and it needs the archive's `images/`
        // layout first. Black is the honest answer until then — a flat colour
        // standing in for an HDRI would be a background that is the wrong
        // colour everywhere rather than an obviously missing one.
        let mut doc = Document::new();
        let image = doc.add_image(kreate_scene::Image::new(
            "sky",
            64,
            32,
            kreate_scene::PixelFormat::Rgba16F,
        ));
        let id = doc.add_world(kreate_scene::World {
            background: kreate_scene::Background::Image(image),
            ..kreate_scene::World::new("Sky")
        });
        doc.set_active_world(Some(id));
        assert_eq!(world_radiance(&doc), [0.0; 3]);
    }

    #[test]
    fn a_scene_with_no_camera_still_has_an_exposure() {
        // A scene being modelled usually has no camera yet, and it still has
        // to be lookable-at. Zero stops would be the wrong answer — that is
        // the default the photometric lights saturate at.
        let doc = Document::new();
        let exposure = shot_exposure(&doc, 24.0, kreate_math::DVec3::ZERO);
        assert!(exposure.is_finite());
        assert!(exposure < -5.0, "a default scene exposed at {exposure}");
    }

    #[test]
    fn the_scene_camera_sets_the_exposure() {
        // A property of the shot. Turning the aperture down two stops has to
        // darken the picture by two stops.
        let mut doc = Document::new();
        let id = doc.add_camera(kreate_scene::Camera {
            f_stop: Some(2.8),
            ..kreate_scene::Camera::new("Shot")
        });
        doc.add_object(Object::new("Shot", ObjectData::Camera(id)));
        let open = shot_exposure(&doc, 24.0, kreate_math::DVec3::ZERO);

        doc.camera_mut(id).expect("camera").f_stop = Some(5.6);
        let stopped_down = shot_exposure(&doc, 24.0, kreate_math::DVec3::ZERO);
        assert!(
            (open - stopped_down - 2.0).abs() < 1e-6,
            "{open} then {stopped_down}"
        );
    }

    #[test]
    fn the_rate_reaches_the_exposure() {
        // The shutter is stored as a fraction of a frame, so the same camera
        // is a different exposure at a different rate — which is correct, and
        // is why the rate has to be passed rather than assumed.
        let mut doc = Document::new();
        let id = doc.add_camera(kreate_scene::Camera::new("Shot"));
        doc.add_object(Object::new("Shot", ObjectData::Camera(id)));
        let slow = shot_exposure(&doc, 24.0, kreate_math::DVec3::ZERO);
        let fast = shot_exposure(&doc, 48.0, kreate_math::DVec3::ZERO);
        assert!((slow - fast - 1.0).abs() < 1e-6, "{slow} then {fast}");
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
        assert!(
            (placed.amount[0] - 1000.0).abs() < 1e-3,
            "{:?}",
            placed.amount
        );

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
        object.transform.rotation = glam::DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2);
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
