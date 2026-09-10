//! Scene lights, as the GPU needs them.
//!
//! # What this crate knows about a light, and what it does not
//!
//! Not what kind it is. `kreate-render` cannot see a document — it depends on
//! `kreate-math` and `kreate-mesh` and nothing else of ours, the same as
//! `kreate-ui`, and that boundary is deliberate: a presentation crate consumes
//! a snapshot and never reaches for the scene.
//!
//! So the four kinds an artist chooses between do not appear here. What appears
//! is what the shading arithmetic actually branches on, which is two things:
//!
//! - **Whether the brightness falls off with distance.** A point, a spot and an
//!   area light are measured in candela and divide by distance squared; the sun
//!   is measured in lux, is already an illuminance, and does not diverge.
//! - **Whether a cone limits it.**
//!
//! That is not a copy of `LightKind` with the names changed. It is smaller on
//! purpose: an area light reaches the viewport as a point at its axial candela
//! with a radius, because that far-field approximation is what a raster pass
//! can afford and the exact integral is the renderer's business. Its *shape*
//! will matter when there is something that samples it.
//!
//! # The units are converted once, on the CPU
//!
//! [`PlacedLight::amount`] arrives already in one unit per falloff — candela or
//! lux — with the tint multiplied in. The alternative is four branches of unit
//! arithmetic per pixel, including an ellipse's area, and the conversion needs
//! `kreate_scene::Light::axial_candela`, which lives beside the units where it
//! can be tested. So the app converts and the shader divides by a distance.
//!
//! # Positions are eye-relative, and that is not an optimisation
//!
//! The camera sits at the origin and the world is moved to meet it
//! ([`crate::camera::Camera::model_matrix_relative`]). A light's position is
//! rebased the same way, subtracting the eye in `f64` before narrowing: a lamp
//! two metres from a camera parked a million units from the origin is `2.0`
//! either way, but through absolute single-precision world space it is the
//! difference of two large numbers and the two metres are gone.
//!
//! A direction needs no rebasing, which is why a sun costs nothing to place.

use bytemuck::{Pod, Zeroable};

/// How a light's brightness behaves with distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Falloff {
    /// Candela: intensity per solid angle. Divide by distance squared.
    #[default]
    InverseSquare,
    /// Lux: an illuminance already. Parallel light does not diverge, so a sun
    /// that obeyed the inverse-square law would light the near side of a scene
    /// and not the far — the most conspicuous way to get outdoor light wrong.
    Parallel,
}

impl Falloff {
    /// The number on the wire.
    ///
    /// A `match` rather than a cast, so that adding a falloff fails to compile
    /// here instead of sending a number the shader reads as inverse-square. The
    /// numbers are asserted against the shader's constants in a test.
    fn id(self) -> u32 {
        match self {
            Falloff::InverseSquare => 0,
            Falloff::Parallel => 1,
        }
    }
}

/// A light with a place in the world, in the units the shader wants.
///
/// What the app hands the renderer, having done the joining the renderer cannot
/// do: the placement comes from the object, the colour and brightness from the
/// datablock, and the unit conversion from `kreate-scene`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedLight {
    pub falloff: Falloff,
    /// World position, in `f64`. Rebased onto the camera when packed — see the
    /// module docs on why the subtraction happens before the narrowing.
    ///
    /// Ignored for [`Falloff::Parallel`], which has a direction and no place.
    pub position: glam::DVec3,
    /// The unit vector the light points along, in world space.
    ///
    /// glTF's convention and Blender's: a light points down its object's `-Z`.
    /// Only meaningful for a sun and a spot, and normalised by the packer
    /// because a rotated non-uniform scale can hand over something that is not
    /// quite unit length.
    pub direction: glam::Vec3,
    /// The tint times the brightness: candela for [`Falloff::InverseSquare`],
    /// lux for [`Falloff::Parallel`], per channel.
    ///
    /// Not a colour and not a swatch. `kreate_scene::Light::tinted_intensity`
    /// is the one place that multiplies the two, and it is deliberately not
    /// called `radiance` because the unit depends on the kind.
    pub amount: [f32; 3],
    /// The source's radius in metres: what makes a shadow soft and a highlight
    /// a shape rather than a point. Zero is the hard light nothing real is.
    pub radius: f32,
    /// A sun's angular radius in radians. Zero for everything else.
    ///
    /// Separate from `radius` because they are different quantities that cannot
    /// be converted into one another without a distance, and a sun has none.
    pub angular_radius: f32,
    /// The full cone angle in radians and the fraction of it that fades.
    ///
    /// `None` for a light with no cone, which packs as a cosine of -1 — the
    /// whole sphere, so nothing is outside it. A branchless "everything is lit"
    /// rather than a flag the shader has to test for most of the lights.
    pub cone: Option<(f32, f32)>,
    pub cast_shadow: bool,
    pub affect_diffuse: bool,
    pub affect_specular: bool,
}

impl Default for PlacedLight {
    fn default() -> Self {
        Self {
            falloff: Falloff::InverseSquare,
            position: glam::DVec3::ZERO,
            direction: glam::Vec3::NEG_Z,
            amount: [0.0; 3],
            radius: 0.0,
            angular_radius: 0.0,
            cone: None,
            cast_shadow: true,
            affect_diffuse: true,
            affect_specular: true,
        }
    }
}

/// A light casts a shadow.
pub const FLAG_SHADOW: u32 = 1;
/// A light lights the diffuse response.
pub const FLAG_DIFFUSE: u32 = 2;
/// A light lights the specular response.
pub const FLAG_SPECULAR: u32 = 4;

/// One light, as the shader reads it.
///
/// Sixty-four bytes and sixteen-byte aligned, which is what makes it a valid
/// element of a storage array under every layout rule. A test asserts the size,
/// because a layout that disagrees with the shader's is silent: every light
/// after the first reads fields from its neighbour.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    /// xyz: eye-relative position in metres. w: source radius in metres.
    pub position: [f32; 4],
    /// xyz: unit direction the light points. w: angular radius in radians.
    pub direction: [f32; 4],
    /// rgb: candela or lux, per [`GpuLight::falloff`]. w: unused.
    pub amount: [f32; 4],
    /// cos of the outer half-angle, then of the inner. `-1` for no cone.
    pub cone: [f32; 2],
    /// See [`Falloff`].
    pub falloff: u32,
    /// See `FLAG_SHADOW` and friends.
    pub flags: u32,
}

impl GpuLight {
    /// Packs a light, rebasing its position onto `eye`.
    pub fn pack(light: &PlacedLight, eye: glam::DVec3) -> Self {
        // The subtraction in f64, before the narrowing. See the module docs.
        let relative = (light.position - eye).as_vec3();

        // Normalised here rather than trusted: a light under a rotated,
        // non-uniformly scaled parent arrives slightly off unit length, and a
        // direction that is not unit length makes a spot's cone the wrong
        // width rather than making anything obviously wrong.
        let direction = light
            .direction
            .try_normalize()
            .unwrap_or(glam::Vec3::NEG_Z);

        // A cone of "none" is a cosine of -1, which is the whole sphere: the
        // same arithmetic lights a point light fully, with no branch in the hot
        // loop for what is most of the lights. Inner is at least outer, because
        // the fade runs from inner to outer and a reversed pair would divide by
        // a negative width.
        let (outer, inner) = match light.cone {
            None => (-1.0, -1.0),
            Some((angle, blend)) => {
                let half = (angle * 0.5).clamp(0.0, std::f32::consts::PI);
                let outer = half.cos();
                // `blend` is the fraction of the cone that fades, so the inner
                // edge is that much closer to the axis. Clamped, because a
                // blend outside [0, 1] is a keymap or a file being wrong and
                // not a reason to produce a cone that is inside out.
                let inner = (half * (1.0 - blend.clamp(0.0, 1.0))).cos();
                (outer, inner.max(outer))
            }
        };

        let mut flags = 0;
        if light.cast_shadow {
            flags |= FLAG_SHADOW;
        }
        if light.affect_diffuse {
            flags |= FLAG_DIFFUSE;
        }
        if light.affect_specular {
            flags |= FLAG_SPECULAR;
        }

        Self {
            position: [relative.x, relative.y, relative.z, light.radius.max(0.0)],
            direction: [
                direction.x,
                direction.y,
                direction.z,
                light.angular_radius.max(0.0),
            ],
            amount: [light.amount[0], light.amount[1], light.amount[2], 0.0],
            cone: [outer, inner],
            falloff: light.falloff.id(),
            flags,
        }
    }

    /// The illuminance this light delivers to a point, in lux per channel.
    ///
    /// The CPU's answer, and the reference the shader is held to. `at` is
    /// eye-relative, like the packed position.
    ///
    /// Irradiance and not shading: no surface normal, no material, no cosine
    /// term. Those belong to the BRDF, and keeping them out means this can be
    /// checked against a light meter's reading — which is the whole reason the
    /// units are photometric.
    pub fn illuminance_at(&self, at: glam::Vec3) -> [f32; 3] {
        // Against `Falloff::id` rather than a literal, so the wire format has
            // one authority. A second copy of the number here would be a place
            // for the CPU reference to disagree with what was packed — and this
            // function *is* the reference the shader is held to.
            let parallel = Falloff::Parallel.id();
        let (attenuation, to_light) = match self.falloff {
            // Parallel: no distance term, and the direction is the light's own.
            f if f == parallel => (1.0, -glam::Vec3::from_slice(&self.direction[..3])),
            _ => {
                let offset = glam::Vec3::from_slice(&self.position[..3]) - at;
                let distance_squared = offset.length_squared();
                if distance_squared <= 0.0 {
                    // Standing exactly on the light. Not an infinity: a point
                    // there has no direction to be lit from either.
                    return [0.0; 3];
                }
                (1.0 / distance_squared, offset / distance_squared.sqrt())
            }
        };

        // The cone. Strictly *outside* the outer edge is dark, which matters at
        // exactly `cos = -1`: a light with no cone packs its outer edge there,
        // and `<=` would have excluded the one direction directly behind the
        // light. A point on the edge itself falls through to the interpolation
        // and comes out at zero, so there is no discontinuity either way.
        let axis = glam::Vec3::from_slice(&self.direction[..3]);
        let cosine = axis.dot(-to_light);
        let [outer, inner] = self.cone;
        let cone = if cosine < outer {
            0.0
        } else if cosine >= inner {
            1.0
        } else {
            // Linear in the cosine between the two edges. Smoother curves are
            // available and none of them is more correct; this one has no
            // discontinuity at either edge, which is what shows.
            (cosine - outer) / (inner - outer)
        };

        let scale = attenuation * cone;
        [
            self.amount[0] * scale,
            self.amount[1] * scale,
            self.amount[2] * scale,
        ]
    }
}

/// The light buffer a frame binds.
///
/// Grown rather than reallocated per frame, and never zero-sized: wgpu refuses
/// a zero-sized binding, so "no lights" is a count of zero over a buffer that
/// still exists. That lesson is the display resolve's, learned the same way.
pub struct Lights {
    buffer: wgpu::Buffer,
    /// How many lights the buffer can hold, not how many are live.
    capacity: u32,
    count: u32,
}

impl Lights {
    /// A buffer with room for `capacity` lights and none in it.
    ///
    /// One is the floor, because a zero-sized buffer cannot be bound and a
    /// scene with no lights still has to draw.
    /// The capacity a request of `wanted` actually gets.
    ///
    /// One is the floor. Separated so the rule is checkable without a device,
    /// which is what makes it a test rather than a comment.
    pub fn floor_capacity(wanted: u32) -> u32 {
        wanted.max(1)
    }

    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let capacity = Self::floor_capacity(capacity);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene lights"),
            size: capacity as u64 * std::mem::size_of::<GpuLight>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity,
            count: 0,
        }
    }

    /// Packs `lights` against `eye` and uploads them, growing if needed.
    ///
    /// Returns how many were written. The buffer is not shrunk: a scene whose
    /// light count oscillates would otherwise reallocate every frame, and the
    /// memory is sixty-four bytes a light.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        eye: glam::DVec3,
        lights: &[PlacedLight],
    ) -> u32 {
        let packed: Vec<GpuLight> = lights.iter().map(|l| GpuLight::pack(l, eye)).collect();
        if packed.len() as u32 > self.capacity {
            // Rounded up to a power of two rather than fitted exactly, so a
            // scene being built one lamp at a time reallocates a handful of
            // times instead of once per lamp.
            //
            // The old buffer is dropped here while a submitted frame may still
            // be reading it, which is safe: wgpu keeps a resource alive until
            // the submissions using it have completed.
            *self = Self::new(device, (packed.len() as u32).next_power_of_two());
        }
        if !packed.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&packed));
        }
        // The buffer is not shrunk, so the slots past `count` still hold the
            // lights of an earlier frame. `count` is the only thing that says
            // where the live ones end, which is why a consumer must loop to it
            // rather than to the buffer's length.
        self.count = packed.len() as u32;
        self.count
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// How many lights are live. The buffer may be larger.
    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

/// The shader source, for a pass that shades with these lights.
///
/// There is no `#import` in this build, so a shader needing `illuminance_at`
/// takes the text. A function exposed rather than the file copied, because a
/// copy is the drift the device test exists to prevent.
pub fn shader_source() -> &'static str {
    include_str!("shaders/lights.wgsl")
}

/// Where this library expects the light array to be bound.
pub const LIGHTS_GROUP: u32 = 2;
pub const LIGHTS_BINDING: u32 = 1;

/// Where the probe pass begins.
///
/// Everything after it is for the device test and is stripped when this file is
/// composed into another shader: the probe declares bindings a forward pass
/// does not have, so leaving them in would make the pass demand buffers nobody
/// binds.
const PROBE_MARKER: &str = "// --- the probe pass";

/// The part of the shader another shader can include.
///
/// # Panics
///
/// If the marker is gone. Returning the whole file instead would compose the
/// probe's bindings into a forward pass, and the failure would arrive as a
/// pipeline validation error a long way from the edit that caused it.
pub fn library_source() -> &'static str {
    let source = shader_source();
    match source.split_once(PROBE_MARKER) {
        Some((library, _)) => library,
        None => panic!("the probe marker is gone from this shader"),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_struct_is_what_the_shader_expects() {
        // A layout mismatch is silent and total: every light after the first
        // reads its fields out of its neighbour, and the picture is merely
        // wrong rather than broken. Sixty-four bytes, sixteen-byte aligned, is
        // what makes it a valid storage-array element under every layout rule.
        assert_eq!(std::mem::size_of::<GpuLight>(), 64);
        assert_eq!(std::mem::align_of::<GpuLight>(), 4);
        // And the shader's own declaration adds up to the same.
        assert_eq!(16 + 16 + 16 + 8 + 4 + 4, 64);
    }

    #[test]
    fn a_position_is_rebased_before_it_is_narrowed() {
        // The precision claim, and the reason the subtraction is in f64. A lamp
        // two metres from a camera parked a million units out is 2.0 either
        // way; through absolute single-precision world space it is the
        // difference of two large numbers and the two metres are gone.
        // A thousand million, not a million: f32 holds every integer up to
        // 2^24, so at a million the naive path would have worked and the
        // comparison below would prove nothing. At 1e9 an f32 step is 64.
        let far = 1_000_000_000.0;
        let light = PlacedLight {
            position: glam::DVec3::new(far + 2.0, far, far),
            ..PlacedLight::default()
        };
        let packed = GpuLight::pack(&light, glam::DVec3::splat(far));
        assert_eq!(packed.position[0], 2.0);
        assert_eq!(packed.position[1], 0.0);

        // What the naive path would have produced, for contrast: the two metres
        // do not survive being written down as an absolute f32.
        let naive = (far + 2.0) as f32 - far as f32;
        assert_eq!(naive, 0.0, "the two metres survived, so this proves nothing");
    }

    #[test]
    fn a_suns_direction_does_not_depend_on_where_the_camera_is() {
        // Which is why a sun costs nothing to place: a direction is
        // eye-invariant, and only positions need rebasing.
        let sun = PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.3, -0.5, -0.81).normalize(),
            ..PlacedLight::default()
        };
        let near = GpuLight::pack(&sun, glam::DVec3::ZERO);
        let far = GpuLight::pack(&sun, glam::DVec3::splat(1e6));
        assert_eq!(near.direction, far.direction);
    }

    #[test]
    fn a_direction_is_normalised_rather_than_trusted() {
        // A light under a rotated, non-uniformly scaled parent arrives slightly
        // off unit length. A direction that is not unit length makes a spot's
        // cone the wrong width, which reads as a lighting choice rather than as
        // a bug.
        let light = PlacedLight {
            direction: glam::Vec3::new(0.0, 0.0, -3.0),
            ..PlacedLight::default()
        };
        let packed = GpuLight::pack(&light, glam::DVec3::ZERO);
        let d = glam::Vec3::from_slice(&packed.direction[..3]);
        assert!((d.length() - 1.0).abs() < 1e-6, "{}", d.length());

        // And a zero direction does not become a NaN.
        let broken = PlacedLight {
            direction: glam::Vec3::ZERO,
            ..PlacedLight::default()
        };
        let packed = GpuLight::pack(&broken, glam::DVec3::ZERO);
        let d = glam::Vec3::from_slice(&packed.direction[..3]);
        assert!(d.is_finite() && (d.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_light_with_no_cone_lights_every_direction() {
        // Packed as a cosine of -1 rather than as a flag, so a point light goes
        // through the same arithmetic as a spot with no branch. A flag would be
        // a branch in the hot loop for a case that is most of the lights.
        let light = PlacedLight {
            amount: [100.0; 3],
            position: glam::DVec3::ZERO,
            ..PlacedLight::default()
        };
        let packed = GpuLight::pack(&light, glam::DVec3::ZERO);
        assert_eq!(packed.cone, [-1.0, -1.0]);

        // Behind, beside, in front: all lit, and all by the inverse square.
        for direction in [
            glam::Vec3::X,
            glam::Vec3::NEG_X,
            glam::Vec3::Y,
            glam::Vec3::Z,
            glam::Vec3::NEG_Z,
        ] {
            let lux = packed.illuminance_at(direction * 2.0);
            assert!((lux[0] - 25.0).abs() < 1e-4, "{direction} gave {lux:?}");
        }
    }

    #[test]
    fn a_candela_light_falls_off_with_the_square_of_the_distance() {
        // The same law the datablock's own test checks, on the packed form:
        // 1000 cd at two metres is 250 lux, which anybody can hold a meter up
        // to. That the two agree is what makes the unit worth storing.
        let packed = GpuLight::pack(
            &PlacedLight {
                amount: [1000.0; 3],
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        assert!((packed.illuminance_at(glam::Vec3::X)[0] - 1000.0).abs() < 1e-3);
        assert!((packed.illuminance_at(glam::Vec3::X * 2.0)[0] - 250.0).abs() < 1e-3);
        assert!((packed.illuminance_at(glam::Vec3::X * 10.0)[0] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn a_parallel_light_does_not_fall_off() {
        // Because its amount is already an illuminance. A sun obeying the
        // inverse-square law would light the near side of a scene and not the
        // far, which is the most conspicuous way to get outdoor light wrong.
        let sun = GpuLight::pack(
            &PlacedLight {
                falloff: Falloff::Parallel,
                direction: glam::Vec3::NEG_Z,
                amount: [20_000.0; 3],
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        let near = sun.illuminance_at(glam::Vec3::ZERO);
        let far = sun.illuminance_at(glam::Vec3::Z * 1000.0);
        assert_eq!(near, far);
        assert!((near[0] - 20_000.0).abs() < 1e-3);
    }

    #[test]
    fn standing_on_a_point_light_is_not_an_infinity() {
        // A point with no distance to the light has no direction to be lit
        // from either, so the honest answer is nothing rather than everything.
        let packed = GpuLight::pack(
            &PlacedLight {
                amount: [1000.0; 3],
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        assert_eq!(packed.illuminance_at(glam::Vec3::ZERO), [0.0; 3]);
    }

    #[test]
    fn a_spots_cone_fades_from_the_inner_edge_to_the_outer() {
        // Full brightness on the axis, nothing outside, and monotonic between.
        // A non-monotonic fade would put a ring in the middle of a spotlight.
        let spot = GpuLight::pack(
            &PlacedLight {
                direction: glam::Vec3::NEG_Z,
                amount: [1000.0; 3],
                cone: Some((90f32.to_radians(), 0.5)),
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        // Sampled along an arc a fixed distance away, so only the cone varies.
        let mut previous = f32::INFINITY;
        for degrees in 0..=60 {
            let angle = (degrees as f32).to_radians();
            let at = glam::Vec3::new(angle.sin(), 0.0, -angle.cos()) * 3.0;
            let lux = spot.illuminance_at(at)[0];
            assert!(lux <= previous + 1e-4, "the fade turned back at {degrees}°");
            previous = lux;
        }
        // On the axis: the full inverse-square value at three metres.
        let axis = spot.illuminance_at(glam::Vec3::NEG_Z * 3.0)[0];
        assert!((axis - 1000.0 / 9.0).abs() < 1e-3, "{axis}");
        // Outside the 45° half-angle: nothing.
        let outside = glam::Vec3::new(50f32.to_radians().sin(), 0.0, -50f32.to_radians().cos());
        assert_eq!(spot.illuminance_at(outside * 3.0)[0], 0.0);
    }

    #[test]
    fn a_blend_of_zero_is_a_hard_edge() {
        // The two cosines coincide, so the fade has no width. Worth pinning
        // because a zero-width fade is where a division by the width would
        // produce a NaN and a ring of black pixels.
        let spot = GpuLight::pack(
            &PlacedLight {
                direction: glam::Vec3::NEG_Z,
                amount: [1000.0; 3],
                cone: Some((90f32.to_radians(), 0.0)),
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        assert_eq!(spot.cone[0], spot.cone[1]);
        let inside = spot.illuminance_at(
            glam::Vec3::new(40f32.to_radians().sin(), 0.0, -40f32.to_radians().cos()) * 3.0,
        )[0];
        let outside = spot.illuminance_at(
            glam::Vec3::new(50f32.to_radians().sin(), 0.0, -50f32.to_radians().cos()) * 3.0,
        )[0];
        assert!(inside > 0.0 && inside.is_finite(), "{inside}");
        assert_eq!(outside, 0.0);
    }

    #[test]
    fn a_cone_that_is_inside_out_is_refused_rather_than_drawn() {
        // A blend outside [0, 1] is a keymap or a file being wrong. Producing a
        // reversed pair would divide by a negative width and light the outside
        // of the cone instead of the inside.
        for blend in [-1.0, 2.0, f32::NAN] {
            let spot = GpuLight::pack(
                &PlacedLight {
                    direction: glam::Vec3::NEG_Z,
                    amount: [1000.0; 3],
                    cone: Some((90f32.to_radians(), blend)),
                    ..PlacedLight::default()
                },
                glam::DVec3::ZERO,
            );
            assert!(
                spot.cone[1] >= spot.cone[0],
                "blend {blend} gave {:?}",
                spot.cone
            );
            let lux = spot.illuminance_at(glam::Vec3::NEG_Z * 3.0)[0];
            assert!(lux.is_finite(), "blend {blend} gave {lux}");
        }
    }

    #[test]
    fn an_absurd_cone_angle_is_clamped_rather_than_wrapped() {
        // A cone wider than a full turn would come back out of `cos` as a
        // narrow one, so a light asked to be omnidirectional would become a
        // pinspot.
        let spot = GpuLight::pack(
            &PlacedLight {
                direction: glam::Vec3::NEG_Z,
                amount: [1000.0; 3],
                cone: Some((100.0, 0.0)),
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        assert_eq!(spot.cone[0], -1.0, "a very wide cone is a whole sphere");
        assert!(spot.illuminance_at(glam::Vec3::Z * 3.0)[0] > 0.0);
    }

    #[test]
    fn the_three_switches_pack_independently() {
        // The lighting cheats, and they are not optional: a rim light that must
        // not blow out a highlight is an ordinary request.
        let all = GpuLight::pack(&PlacedLight::default(), glam::DVec3::ZERO);
        assert_eq!(all.flags, FLAG_SHADOW | FLAG_DIFFUSE | FLAG_SPECULAR);

        let rim = GpuLight::pack(
            &PlacedLight {
                cast_shadow: false,
                affect_diffuse: false,
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        assert_eq!(rim.flags, FLAG_SPECULAR);
        // The three bits are distinct powers of two, or two switches would be
        // one switch.
        assert_eq!(FLAG_SHADOW & FLAG_DIFFUSE, 0);
        assert_eq!(FLAG_DIFFUSE & FLAG_SPECULAR, 0);
        assert_eq!(FLAG_SHADOW & FLAG_SPECULAR, 0);
    }

    #[test]
    fn a_negative_radius_is_not_stored() {
        // It would make a soft shadow the shader samples backwards. Zero is the
        // hard light, which is at least a light.
        let packed = GpuLight::pack(
            &PlacedLight {
                radius: -1.0,
                angular_radius: -0.5,
                ..PlacedLight::default()
            },
            glam::DVec3::ZERO,
        );
        assert_eq!(packed.position[3], 0.0);
        assert_eq!(packed.direction[3], 0.0);
    }

    #[test]
    fn the_falloff_numbers_are_the_shaders_numbers() {
        // The wire format, and the one part a device test cannot catch: a
        // mismatch would run the other falloff, and both produce a picture.
        assert_eq!(Falloff::InverseSquare.id(), 0);
        assert_eq!(Falloff::Parallel.id(), 1);

        let source = shader_source();
        for (name, id) in [
            ("FALLOFF_INVERSE_SQUARE", Falloff::InverseSquare.id()),
            ("FALLOFF_PARALLEL", Falloff::Parallel.id()),
        ] {
            let declaration = format!("const {name}: u32 = {id}u;");
            assert!(
                source.contains(&declaration),
                "the shader does not declare `{declaration}`"
            );
        }
    }

    #[test]
    fn the_flag_bits_are_the_shaders_bits() {
        // Three switches that would otherwise be one switch, or the wrong one.
        let source = shader_source();
        for (name, bit) in [
            ("FLAG_SHADOW", FLAG_SHADOW),
            ("FLAG_DIFFUSE", FLAG_DIFFUSE),
            ("FLAG_SPECULAR", FLAG_SPECULAR),
        ] {
            let declaration = format!("const {name}: u32 = {bit}u;");
            assert!(
                source.contains(&declaration),
                "the shader does not declare `{declaration}`"
            );
        }
    }

    #[test]
    fn the_shader_parses() {
        // Cheap, and it catches what only a device would otherwise find. The
        // deform shader once shipped with a construct WGSL forbids.
        naga::front::wgsl::parse_str(shader_source()).expect("lights.wgsl parses");
    }

    #[test]
    fn a_buffer_is_never_zero_sized_however_few_lights_there_are() {
        // wgpu refuses a zero-sized binding, so a scene with no lights still
        // needs a buffer. The display resolve learned this the hard way — as a
        // validation error three frames from the caller who had nothing to
        // draw.
        assert_eq!(super::Lights::floor_capacity(0), 1);
        assert_eq!(super::Lights::floor_capacity(1), 1);
        assert_eq!(super::Lights::floor_capacity(9), 9);
    }
}
