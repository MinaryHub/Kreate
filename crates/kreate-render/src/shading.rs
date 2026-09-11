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

/// The light leaving `point` toward its viewer, in the working space.
///
/// Scene-linear and unbounded: a surface under a sun is in the thousands, and
/// the display transform is what turns that into a picture. Clamping here would
/// be the bug §2.1 records — twice.
pub fn shade(point: &Point, lights: &[GpuLight]) -> [f32; 3] {
    let mut total = [0.0f32; 3];
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

/// The shader the rendered viewport draws with.
///
/// The library composition plus a vertex and a fragment stage. Nothing in
/// `rendered.wgsl` shades: every light and every lobe is `shade_point`, so
/// what the viewport runs is the function the device tests measure.
pub fn rendered_source() -> String {
    format!("{}\n{}", shader_source(), include_str!("shaders/rendered.wgsl"))
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
    fn lights_add() {
        // The arithmetic of light is addition. A renderer taking the largest,
        // or averaging, would look plausible under one lamp and wrong under
        // two — and "wrong under two" is every scene anybody makes.
        let mut point = grey_point(glam::Vec3::Z);
        point.position = glam::Vec3::ZERO;
        point.view = glam::Vec3::Z;

        let a = bulb(glam::Vec3::new(0.0, 0.0, 2.0), 400.0);
        let b = bulb(glam::Vec3::new(1.0, 0.0, 2.0), 400.0);

        let only_a = shade(&point, std::slice::from_ref(&a))[0];
        let only_b = shade(&point, std::slice::from_ref(&b))[0];
        let both = shade(&point, &[a, b])[0];

        assert!(only_a > 0.0 && only_b > 0.0);
        assert!((both - (only_a + only_b)).abs() < both * 1e-5, "{both}");
    }

    #[test]
    fn no_lights_is_black_and_not_a_default_grey() {
        // A surface lit by nothing is black. Anything else is an ambient term
        // nobody asked for, and it would make every light the user adds look
        // like it did less than it did.
        let point = grey_point(glam::Vec3::Z);
        assert_eq!(shade(&point, &[]), [0.0; 3]);
    }

    #[test]
    fn a_light_facing_away_contributes_nothing() {
        // Below the horizon of the surface, not merely dim. A negative
        // contribution would darken the surface, which reads as a shadow bug.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let behind = bulb(glam::Vec3::new(0.0, 0.0, -2.0), 1000.0);
        assert_eq!(shade(&point, &[behind]), [0.0; 3]);
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
        assert!(shade(&up, &[overhead])[0] > 0.0, "the ground is unlit");

        let mut down = grey_point(glam::Vec3::NEG_Z);
        down.view = glam::Vec3::NEG_Z;
        assert_eq!(shade(&down, &[overhead]), [0.0; 3], "the underside is lit");
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
        let all = shade(&point, &[both])[0];
        let d = shade(&point, &[diffuse_only])[0];
        let s = shade(&point, &[specular_only])[0];
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
        let lit = shade(&point, &[noon])[0];
        assert!(lit > 100.0, "the sun was clamped away: {lit}");
    }

    #[test]
    fn brightness_is_linear_in_the_lights_intensity() {
        // Which is what "scene-linear" means at this seam, and what makes the
        // exposure controls behave. A tone curve applied here would break it.
        let mut point = grey_point(glam::Vec3::Z);
        point.view = glam::Vec3::Z;
        let one = shade(&point, &[bulb(glam::Vec3::new(0.0, 0.0, 2.0), 100.0)])[0];
        let ten = shade(&point, &[bulb(glam::Vec3::new(0.0, 0.0, 2.0), 1000.0)])[0];
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
        let lit = shade(&point, &[bulb(glam::Vec3::new(0.0, 0.0, 2.0), 500.0)]);
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
        let source = rendered_source();
        naga::front::wgsl::parse_str(&source).expect("the drawing composition parses");
        assert!(source.contains("fn vs_rendered"));
        assert!(source.contains("fn fs_rendered"));
        // One `shade_point`, and the fragment stage calls it rather than
        // reimplementing it.
        assert_eq!(source.matches("fn shade_point").count(), 1);
        assert!(source.contains("shade_point(surface, in.world, normal, view)"));
        // And no probe: those declare buffers a drawing pass does not bind.
        assert!(!source.contains("fn probe_shading"));
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
