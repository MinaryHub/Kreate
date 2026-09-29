//! A point, its material, and every light in the scene.
//!
//! The two halves of §2.2 and §2.3 meet here: `lights` says how much light
//! arrives from each source, `brdf` says how much of it leaves toward the
//! viewer, and this multiplies and sums them. It is the innermost loop of the
//! rendered viewport and, later, of a path tracer's direct-lighting estimate.
//!
//! # Why this is its own thing and not a line in the fragment shader
//!
//! Because the sum has decisions in it that are worth testing on their own,
//! and none of them is visible in a picture:
//!
//! - A light can be told to skip the diffuse or the specular response (§2.2's
//!   lighting cheats). That is per light, so it lives in the loop.
//! - Lights add. Not the largest, not an average: the arithmetic of light is
//!   addition, and a renderer that got that wrong would look plausible under
//!   one lamp.
//! - The result is scene-linear and unbounded. Nothing is clamped here — the
//!   display transform is the one place a picture gets a range, and clamping
//!   early is exactly the mistake §2.1 records twice.
//!
//! # The composition
//!
//! There is no `#import` in this build, so the shader is assembled by
//! concatenating `lights`' and `brdf`'s library text with this module's own.
//! The two libraries declare their buffers in group 2 — a storage pointer
//! cannot be a WGSL function parameter, so a library that needs a buffer must
//! declare it — and leave groups 0 and 1 to the pass.

use crate::brdf::{self, Surface};
use crate::lights::GpuLight;

/// Everything the shading of one point needs, besides the lights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Eye-relative, in metres, like the packed lights.
    pub position: glam::Vec3,
    /// Unit, and pointing out of the surface.
    pub normal: glam::Vec3,
    /// Unit, and pointing at the viewer.
    ///
    /// For a camera at the origin — which eye-relative space puts it at — this
    /// is just `-normalize(position)`, and [`Point::facing_the_eye`] builds it
    /// that way. Kept as a field because a path tracer's viewer is a bounce
    /// direction and not a camera.
    pub view: glam::Vec3,
    pub surface: Surface,
}

impl Point {
    /// A point seen by the camera, which eye-relative space puts at the origin.
    pub fn facing_the_eye(position: glam::Vec3, normal: glam::Vec3, surface: Surface) -> Self {
        Self {
            position,
            normal,
            // Away from the point, toward the origin. A point *at* the origin
            // has no direction to be seen from, and the normal is the least
            // wrong answer — it makes the surface face the viewer, which is
            // what a degenerate camera distance should look like.
            view: (-position).try_normalize().unwrap_or(normal),
            surface,
        }
    }
}

/// What a uniform environment contributes to a point.
///
/// Exact for a flat world, which is worth saying because an ambient term
/// usually is not. Under an environment the same in every direction the
/// prefiltered radiance is that colour whatever the roughness, so the
/// split-sum specular is the DFG table straight out; and the diffuse half
/// integrates to `albedo × radiance`, coupled by the share the direct path
/// uses. A white surface under `L` returns `L`, and a furnace test says so.
///
/// An *image* background goes through the same function with a lookup in place
/// of the constant — see [`ambient_from`] and [`crate::environment`].
pub fn ambient(surface: &Surface, normal: glam::Vec3, view: glam::Vec3, radiance: [f32; 3]) -> [f32; 3] {
    let nov = normal.dot(view).clamp(1e-4, 1.0);
    // A uniform environment irradiates every direction with `πL` and
    // prefilters to `L` at every roughness. Spelled that way rather than
    // simplified, because it is the same call an *image* environment makes and
    // a second formula here is the drift §5's risk 2 is about.
    let irradiance = [
        radiance[0] * std::f32::consts::PI,
        radiance[1] * std::f32::consts::PI,
        radiance[2] * std::f32::consts::PI,
    ];
    ambient_from(surface, nov, irradiance, radiance)
}

/// The environment's contribution, given what it sends and what it reflects.
///
/// The one formula, with the two sources as arguments. A flat world supplies
/// `πL` and `L`; an image supplies [`crate::environment`]'s coefficients and a
/// cube lookup. Written this way so that "the image path and the flat path
/// agree" is a property of the *arguments* rather than a claim about two
/// bodies of arithmetic staying in step.
pub fn ambient_from(
    surface: &Surface,
    nov: f32,
    irradiance: [f32; 3],
    prefiltered: [f32; 3],
) -> [f32; 3] {
    let roughness = surface.roughness.clamp(0.0, 1.0);
    let share = brdf::specular_share(surface.f0(), nov, roughness);
    let albedo = surface.diffuse_albedo();
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        // Irradiance over `π` is the radiance a Lambertian surface reflects.
        out[c] = prefiltered[c] * share[c]
            + albedo[c] * (1.0 - share[c]) * irradiance[c] / std::f32::consts::PI;
    }
    out
}

/// The light leaving `point` toward its viewer, in the working space.
///
/// Scene-linear and unbounded: a surface under a sun is in the thousands, and
/// the display transform is what turns that into a picture. Clamping here would
/// be the bug §2.1 records — twice.
pub fn shade(point: &Point, lights: &[GpuLight], world: [f32; 3]) -> [f32; 3] {
    // The world first, because it is the light that is always there.
    let mut total = ambient(&point.surface, point.normal, point.view, world);
    for light in lights {
        let to_light = direction_to(light, point.position);
        let illuminance = light.illuminance_at(point.position);
        let contribution = brdf::shade(
            &point.surface,
            point.normal,
            point.view,
            to_light,
            illuminance,
            light.flags & crate::lights::FLAG_DIFFUSE != 0,
            light.flags & crate::lights::FLAG_SPECULAR != 0,
        );
        for c in 0..3 {
            total[c] += contribution[c];
        }
    }
    total
}

/// Which way the light is, from `at`.
///
/// A parallel light has only a direction, and it is the direction it *points*,
/// so the way to it is the other way. Getting this backwards lights the far
/// side of everything, which reads as a normal problem rather than a sign.
pub fn direction_to(light: &GpuLight, at: glam::Vec3) -> glam::Vec3 {
    let axis = glam::Vec3::from_slice(&light.direction[..3]);
    if light.falloff == PARALLEL {
        return -axis;
    }
    let offset = glam::Vec3::from_slice(&light.position[..3]) - at;
    // A point exactly on the light has no direction to be lit from. The
    // illuminance is zero there too, so any unit vector does; the axis is one
    // that exists.
    offset.try_normalize().unwrap_or(-axis)
}

/// The falloff id for a parallel light. See `lights::Falloff`.
const PARALLEL: u32 = 1;

/// This module's own shader text, probe and all.
fn own_source() -> &'static str {
    include_str!("shaders/shading.wgsl")
}

const PROBE_MARKER: &str = "// --- the probe pass";

/// The composed shader a drawing pass includes: three libraries, no entry
/// points.
///
/// Assembled rather than written out, because a copy of either library is the
/// drift the parity tests exist to prevent. The order matters only in that WGSL
/// wants a declaration before its use, and the libraries are self-contained.
///
/// A pass appends its own vertex and fragment entry points to this and binds
/// group 2: the DFG table at 0, the lights at 1, the shading globals at 2.
pub fn shader_source() -> String {
    let own = match own_source().split_once(PROBE_MARKER) {
        Some((library, _)) => library,
        None => panic!("the probe marker is gone from shading.wgsl"),
    };
    format!(
        "{}\n{}\n{}",
        crate::brdf::library_source(),
        crate::lights::library_source(),
        own,
    )
}

/// The rendered viewport's shader, **without a material**.
///
/// The library composition plus a vertex and a fragment stage. Nothing in
/// `rendered.wgsl` shades: every light and every lobe is `shade_point`, so
/// what the viewport runs is the function the device tests measure.
///
/// This does not compile on its own, and that is the point: the fragment stage
/// calls `material_sample`, which a material's own compiled code provides. A
/// version of it that worked alone would need a built-in surface, and a
/// built-in surface is the second implementation §3's fifth decision exists to
/// prevent. Use [`rendered_source_with`].
pub fn rendered_source() -> String {
    format!("{}\n{}", shader_source(), include_str!("shaders/rendered.wgsl"))
}

/// The same with one material's compiled code on the end, which is what a
/// pipeline is built from.
pub fn rendered_source_with(material: &str) -> String {
    format!("{}\n{}", rendered_source(), material)
}

/// The shader that draws the environment behind the scene.
///
/// The same library as everything else: it reads `shading`'s uniform and the
/// same cube, and a second declaration of either is the drift the parity tests
/// exist to prevent.
pub fn background_source() -> String {
    format!(
        "{}
{}",
        shader_source(),
        include_str!("shaders/background.wgsl")
    )
}

/// The same, with this module's probe entry point on the end.
///
/// For the device test, which needs something to dispatch. Kept apart from
/// [`shader_source`] because the probe declares buffers in group 0 that a
/// drawing pass wants for its camera.
pub fn probe_source() -> String {
    format!(
        "{}\n{}\n{}",
        crate::brdf::library_source(),
        crate::lights::library_source(),
        own_source(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene lit only by its lamps.
    ///
    /// Spelled out at every call rather than defaulted, because "no world" is
    /// a state a scene is deliberately in and not an omission — and a default
    /// would let a test forget which it was measuring.
    const NO_WORLD: [f32; 3] = [0.0; 3];
    use crate::lights::{Falloff, PlacedLight, FLAG_DIFFUSE, FLAG_SHADOW, FLAG_SPECULAR};
    use kreate_math::SceneRgb;

    fn bulb(position: glam::Vec3, candela: f32) -> GpuLight {
        GpuLight::pack(
            &PlacedLight {
                position: position.as_dvec3(),
                amount: [candela; 3],
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        )
    }

    fn sun(direction: glam::Vec3, lux: f32) -> GpuLight {
        GpuLight::pack(
            &PlacedLight {
                falloff: Falloff::Parallel,
                direction,
                amount: [lux; 3],
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        )
    }

    fn grey_point(normal: glam::Vec3) -> Point {
        Point::facing_the_eye(
            glam::Vec3::new(0.0, 0.0, 0.0),
            normal,
            Surface {
                base_color: SceneRgb::new(0.8, 0.8, 0.8),
                metallic: 0.0,
                roughness: 0.5,
            },
        )
    }

    #[test]
    fn a_white_surface_under_a_uniform_environment_returns_it() {
        // The furnace test, at the scene level. If the light arriving from
        // every direction is `L` and the surface reflects everything, the
        // light leaving is `L` — whatever the material, whatever the angle.
        //
        // This is the strongest statement available about an ambient term, and
        // most renderers cannot make it: an ambient that is `albedo × L` plus a
        // specular term reflects more than arrives, and one that ignores the
        // specular reflects less. The share is what makes the two add to one.
        // Zero and one, which are the two materials that exist. See the test
        // below for why the values between them are not held to this.
        let environment = [3.0f32, 3.0, 3.0];
        for metallic in [0.0f32, 1.0] {
            for roughness in [0.05f32, 0.3, 0.7, 1.0] {
                let surface = Surface {
                    base_color: SceneRgb::WHITE,
                    metallic,
                    roughness,
                };
                for angle in [0.0f32, 0.7, 1.4] {
                    let normal = glam::Vec3::Z;
                    let view = glam::Vec3::new(angle.sin(), 0.0, angle.cos());
                    let out = ambient(&surface, normal, view, environment);
                    assert!(
                        (out[0] - environment[0]).abs() < environment[0] * 0.02,
                        "metallic {metallic} roughness {roughness} at {angle} \
                         returned {} of {}",
                        out[0],
                        environment[0]
                    );
                }
            }
        }
    }

    #[test]
    fn a_half_metal_loses_energy_and_that_is_the_parameterisation() {
        // Recorded rather than fixed, because it is not a defect in the
        // shading: metal-roughness blends the *parameters* and not the
        // results, so at metallic one half the reflectance has moved most of
        // the way to a metal's while the diffuse albedo has only halved, and
        // the two no longer add to one.
        //
        // Values between zero and one are not a material. They are what a
        // texture does while crossing from one to the other — which is the
        // only reason the parameter is continuous at all, and `Material::
        // metallic` says so.
        //
        // What is asserted is that nothing *gains* energy, which would show as
        // a surface that glows, and that the loss is where it is expected.
        let environment = [1.0f32; 3];
        let mut worst = 1.0f32;
        for metallic in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let surface = Surface {
                base_color: SceneRgb::WHITE,
                metallic,
                roughness: 0.3,
            };
            let out = ambient(&surface, glam::Vec3::Z, glam::Vec3::Z, environment)[0];
            assert!(out <= 1.02, "metallic {metallic} returned {out}");
            worst = worst.min(out);
        }
        // The dip is real and is in the middle, not at either end.
        assert!(worst < 0.9, "the blend lost nothing: {worst}");
        let ends = [0.0f32, 1.0].map(|metallic| {
            ambient(
                &Surface {
                    base_color: SceneRgb::WHITE,
                    metallic,
                    roughness: 0.3,
                },
                glam::Vec3::Z,
                glam::Vec3::Z,
                environment,
            )[0]
        });
        for end in ends {
            assert!(end > 0.97, "an end lost energy: {end}");
        }
    }

    #[test]
    fn a_darker_surface_returns_less_and_never_more() {
        // The other half: conservation is a bound, not an identity. A black
        // surface under an environment still shows its specular reflection,
        // and nothing shows more than arrived.
        let environment = [1.0f32; 3];
        for albedo in [0.0f32, 0.25, 0.5, 0.8, 1.0] {
            let surface = Surface {
                base_color: SceneRgb::new(albedo, albedo, albedo),
                metallic: 0.0,
                roughness: 0.4,
            };
            let out = ambient(&surface, glam::Vec3::Z, glam::Vec3::Z, environment)[0];
            assert!(out <= environment[0] * 1.02, "{albedo} returned {out}");
            assert!(out > 0.0, "even black has a reflection: {albedo}");
        }

        // And brighter albedo returns more, or the parameter means nothing.
        let dark = ambient(
            &Surface {
                base_color: SceneRgb::new(0.1, 0.1, 0.1),
                metallic: 0.0,
                roughness: 0.4,
            },
            glam::Vec3::Z,
            glam::Vec3::Z,
            environment,
        )[0];
        let light = ambient(
            &Surface {
                base_color: SceneRgb::new(0.9, 0.9, 0.9),
                metallic: 0.0,
                roughness: 0.4,
            },
            glam::Vec3::Z,
            glam::Vec3::Z,
            environment,
        )[0];
        assert!(light > dark * 3.0, "{dark} against {light}");
    }

    #[test]
    fn a_black_world_contributes_nothing() {
        // Which is what a scene with no world is, and it has to cost nothing
        // rather than a small grey.
        let point = grey_point(glam::Vec3::Z);
        assert_eq!(
            ambient(&point.surface, point.normal, point.view, NO_WORLD),
            [0.0; 3]
        );
        assert_eq!(shade(&point, &[], NO_WORLD), [0.0; 3]);
    }

    #[test]
    fn the_world_lights_a_scene_with_no_lamps_in_it() {
        // An environment is a light, and the commonest one. A renderer that
        // only summed the lamps would show a black object against a coloured
        // background, which reads as the object being broken.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let lit = shade(&point, &[], [0.5, 0.25, 0.125]);
        assert!(lit[0] > 0.0 && lit[1] > 0.0 && lit[2] > 0.0, "{lit:?}");
        // And it carries the world's colour rather than greying it.
        assert!(lit[0] > lit[1] && lit[1] > lit[2], "{lit:?}");
    }

    #[test]
    fn the_world_and_the_lamps_add() {
        // Two lights, and one of them is everywhere. Nothing about the sum is
        // special, which is the point — an ambient that replaced the direct
        // light, or that was multiplied into it, would look plausible in a
        // scene with one of the two.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let lamp = bulb(glam::Vec3::new(0.0, 0.0, 2.0), 400.0);

        let only_world = shade(&point, &[], [0.4; 3])[0];
        let only_lamp = shade(&point, &[lamp], NO_WORLD)[0];
        let both = shade(&point, &[lamp], [0.4; 3])[0];
        assert!(only_world > 0.0 && only_lamp > 0.0);
        assert!(
            (both - (only_world + only_lamp)).abs() < both * 1e-5,
            "{only_world} + {only_lamp} against {both}"
        );
    }

    #[test]
    fn lights_add() {
        // The arithmetic of light is addition. A renderer taking the largest,
        // or averaging, would look plausible under one lamp and wrong under
        // two — and "wrong under two" is every scene anybody makes.
        let mut point = grey_point(glam::Vec3::Z);
        point.position = glam::Vec3::ZERO;
        point.view = glam::Vec3::Z;

        let a = bulb(glam::Vec3::new(0.0, 0.0, 2.0), 400.0);
        let b = bulb(glam::Vec3::new(1.0, 0.0, 2.0), 400.0);

        let only_a = shade(&point, std::slice::from_ref(&a), NO_WORLD)[0];
        let only_b = shade(&point, std::slice::from_ref(&b), NO_WORLD)[0];
        let both = shade(&point, &[a, b], NO_WORLD)[0];

        assert!(only_a > 0.0 && only_b > 0.0);
        assert!((both - (only_a + only_b)).abs() < both * 1e-5, "{both}");
    }

    #[test]
    fn no_lights_is_black_and_not_a_default_grey() {
        // A surface lit by nothing is black. Anything else is an ambient term
        // nobody asked for, and it would make every light the user adds look
        // like it did less than it did.
        let point = grey_point(glam::Vec3::Z);
        assert_eq!(shade(&point, &[], NO_WORLD), [0.0; 3]);
    }

    #[test]
    fn a_light_facing_away_contributes_nothing() {
        // Below the horizon of the surface, not merely dim. A negative
        // contribution would darken the surface, which reads as a shadow bug.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let behind = bulb(glam::Vec3::new(0.0, 0.0, -2.0), 1000.0);
        assert_eq!(shade(&point, &[behind], NO_WORLD), [0.0; 3]);
    }

    #[test]
    fn the_way_to_a_parallel_light_is_against_the_way_it_points() {
        // A sun pointing down lights a surface facing up. The other sign
        // lights the far side of everything, which reads as a normal problem
        // rather than as a sign.
        let overhead = sun(glam::Vec3::NEG_Z, 20_000.0);
        let to_light = direction_to(&overhead, glam::Vec3::ZERO);
        assert!((to_light - glam::Vec3::Z).length() < 1e-6, "{to_light}");

        let mut up = grey_point(glam::Vec3::Z);
        up.view = glam::Vec3::Z;
        assert!(shade(&up, &[overhead], NO_WORLD)[0] > 0.0, "the ground is unlit");

        let mut down = grey_point(glam::Vec3::NEG_Z);
        down.view = glam::Vec3::NEG_Z;
        assert_eq!(shade(&down, &[overhead], NO_WORLD), [0.0; 3], "the underside is lit");
    }

    #[test]
    fn the_two_lobe_switches_are_read_per_light() {
        // §2.2's lighting cheats reach all the way here: a rim light that must
        // not blow out a highlight is an ordinary request, and it is a property
        // of the light rather than of the surface — so two lights on one point
        // can differ.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let at = glam::Vec3::new(0.5, 0.0, 2.0);

        let both = GpuLight::pack(
            &PlacedLight {
                position: at.as_dvec3(),
                amount: [500.0; 3],
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        let diffuse_only = GpuLight::pack(
            &PlacedLight {
                position: at.as_dvec3(),
                amount: [500.0; 3],
                affect_specular: false,
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        let specular_only = GpuLight::pack(
            &PlacedLight {
                position: at.as_dvec3(),
                amount: [500.0; 3],
                affect_diffuse: false,
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );

        assert_eq!(both.flags, FLAG_SHADOW | FLAG_DIFFUSE | FLAG_SPECULAR);
        let all = shade(&point, &[both], NO_WORLD)[0];
        let d = shade(&point, &[diffuse_only], NO_WORLD)[0];
        let s = shade(&point, &[specular_only], NO_WORLD)[0];
        assert!(d > 0.0 && s > 0.0);
        assert!((d + s - all).abs() < all * 1e-5, "{d} + {s} against {all}");
    }

    #[test]
    fn the_result_is_scene_linear_and_is_not_clamped() {
        // A surface under a sun is in the thousands. Clamping here would be the
        // bug §2.1 records twice — a picture gets its range from the display
        // transform and nowhere else.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let noon = sun(glam::Vec3::NEG_Z, 100_000.0);
        let lit = shade(&point, &[noon], NO_WORLD)[0];
        assert!(lit > 100.0, "the sun was clamped away: {lit}");
    }

    #[test]
    fn brightness_is_linear_in_the_lights_intensity() {
        // Which is what "scene-linear" means at this seam, and what makes the
        // exposure controls behave. A tone curve applied here would break it.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let one = shade(&point, &[bulb(glam::Vec3::new(0.0, 0.0, 2.0), 100.0)], NO_WORLD)[0];
        let ten = shade(&point, &[bulb(glam::Vec3::new(0.0, 0.0, 2.0), 1000.0)], NO_WORLD)[0];
        assert!((ten - one * 10.0).abs() < one * 1e-4, "{one} then {ten}");
    }

    #[test]
    fn a_point_at_the_eye_does_not_produce_a_nan() {
        // Reachable: a fragment exactly at the near plane's origin in
        // eye-relative space. The view direction is undefined there, and a NaN
        // in a fragment spreads through the blend.
        let point = Point::facing_the_eye(
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            Surface::default(),
        );
        assert!(point.view.is_finite());
        let lit = shade(&point, &[bulb(glam::Vec3::new(0.0, 0.0, 2.0), 500.0)], NO_WORLD);
        assert!(lit.iter().all(|c| c.is_finite()), "{lit:?}");
    }

    #[test]
    fn the_composed_shader_has_no_probe_in_it() {
        // The libraries' probe passes declare bindings a forward pass does not
        // have. Composing them in would make the pass demand buffers nobody
        // binds, and the failure arrives as a pipeline validation error a long
        // way from the edit.
        let source = shader_source();
        assert!(!source.contains("probe_illuminance"));
        assert!(!source.contains("probe_brdf"));
        assert!(!source.contains("struct Probe"));

        // And it does have the two libraries in it, or the strip took too much.
        assert!(source.contains("fn illuminance_at"));
        assert!(source.contains("fn brdf_evaluate"));
        assert!(source.contains("fn shade_point"));
    }

    #[test]
    fn the_composed_shader_declares_each_buffer_once() {
        // Two copies of a binding is a compile error, and the composition is
        // exactly the operation that could produce one.
        let source = shader_source();
        assert_eq!(source.matches("var<storage, read> lights:").count(), 1);
        assert_eq!(source.matches("var<storage, read> dfg:").count(), 1);
    }

    #[test]
    fn both_compositions_parse() {
        // The whole point of composing rather than duplicating, and the one
        // thing that could break without any single file being wrong.
        //
        // The library form has no entry point, which is a valid module: it is
        // what a drawing pass appends its own entry points to.
        naga::front::wgsl::parse_str(&shader_source()).expect("the library composition parses");
        naga::front::wgsl::parse_str(&probe_source()).expect("the probe composition parses");
    }

    #[test]
    fn the_drawing_composition_parses_and_shades_through_the_library() {
        // The viewport's shader is the same three libraries with a vertex and
        // a fragment stage on the end. If it ever grew a second copy of the
        // shading, the device tests would still pass and the picture would
        // still be plausible — which is exactly the drift §5 warns about.
        // With a material on the end, because that is what a pipeline is:
        // the fragment stage calls `material_sample` and a material's own code
        // is where that comes from. Parsing the library alone would be parsing
        // something no device is ever given.
        let source = rendered_source_with(&a_material());
        naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        assert!(source.contains("fn vs_rendered"));
        assert!(source.contains("fn fs_rendered"));
        // One `shade_point`, and the fragment stage calls it rather than
        // reimplementing it.
        assert_eq!(source.matches("fn shade_point").count(), 1);
        assert!(source.contains("shade_point(surface, in.world, normal, view, vec2<u32>(in.clip.xy))"));
        // And no probe: those declare buffers a drawing pass does not bind.
        assert!(!source.contains("fn probe_shading"));
        // One material function, from the material, and the fragment calls it.
        assert_eq!(source.matches("fn material_sample").count(), 1);
        assert!(source.contains("material_sample(in.uv, uv_dx, uv_dy)"));
    }

    /// A plain grey material, compiled.
    ///
    /// Through the real compiler rather than written out here: a hand-written
    /// stand-in would let this test pass while the thing a pipeline is actually
    /// given did not compile.
    fn a_material() -> String {
        let graph = kreate_scene::shader::ShaderGraph::standard(
            "test",
            kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            0.0,
            0.5,
        );
        kreate_shader::compile(&graph)
            .expect("a standard surface compiles")
            .wgsl
    }

    #[test]
    fn the_probe_composition_is_the_library_plus_an_entry_point() {
        // So that what the device test measures is what a drawing pass would
        // run, rather than a second assembly of the same parts.
        let library = shader_source();
        let probe = probe_source();
        assert!(
            probe.starts_with(library.trim_end()),
            "the probe is not the library with something appended"
        );
        assert!(probe.contains("fn probe_shading"));
        assert!(!library.contains("fn probe_shading"));
    }
}
