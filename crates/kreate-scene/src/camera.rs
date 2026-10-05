//! The camera as a physical instrument.
//!
//! §3's seventh "decide now": focal length, sensor size, f-stop, focus distance
//! and shutter, before there is a render setting to add them to. The cost of
//! deferring is not the fields — it is that every camera path and every exporter
//! is written against whatever *is* stored, and today that would be a field of
//! view.
//!
//! # Focal length and sensor size are stored; the field of view is derived
//!
//! They carry the same information — `fov = 2·atan(sensor / 2·focal)` — so one
//! of them is authoritative and the other is a view of it. Storing the angle is
//! the natural first choice and the wrong one:
//!
//! - Every real camera, every lens and every glTF and USD file speaks focal
//!   length and sensor size. An exporter that had only an angle would have to
//!   invent a sensor to divide by, and two exporters would invent different
//!   ones.
//! - Depth of field needs an aperture *diameter*, which is `focal / f_stop`. An
//!   angle cannot produce one.
//! - A user who matches a real photograph types `35mm`, and a tool that made
//!   them convert to degrees would be answering a question nobody asked.
//!
//! The other direction is a formula anyone can check, which is what makes this
//! the right way round rather than merely a choice.
//!
//! # Millimetres, and why that is not the colour mistake again
//!
//! `kreate-math` says the base unit is the metre, and a focal length is stored
//! in millimetres anyway. That looks like the sort of unit confusion the colour
//! and time types exist to prevent, and it is not, for a specific reason: the
//! only thing the two millimetre quantities are ever used for is their *ratio*,
//! where the unit cancels. Nothing in this module converts, so nothing can
//! convert wrongly.
//!
//! The one place a length in metres is needed is the aperture diameter, for a
//! circle of confusion, and that conversion belongs to the renderer that wants
//! it — with the division by a thousand written where somebody can see it.
//!
//! # What is not here
//!
//! Nothing looks through one of these yet. The viewport camera is a *view* — an
//! orbit pivot and a distance — and making it able to adopt a scene camera's
//! framing is interface work with its own design (navigation has to lock, or
//! the camera has to move with it). What could not wait is which of focal
//! length and field of view is the stored one, because the answer propagates
//! into every consumer the moment there is more than one.

use kreate_reflect::Uid;

/// Which sensor dimension the render fills.
///
/// The difference between a 36mm sensor at 50mm framing a 16:9 image and the
/// same numbers framing a square one, and getting it wrong makes an imported
/// camera frame differently from the tool that exported it — visible, and
/// attributable to nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SensorFit {
    /// The sensor's width applies to whichever image dimension is larger.
    ///
    /// What every other tool defaults to, so a round trip through one does not
    /// change the framing.
    #[default]
    Auto,
    /// The sensor's width maps to the image width.
    Horizontal,
    /// The sensor's height maps to the image height.
    Vertical,
}

impl SensorFit {
    pub fn id(self) -> &'static str {
        match self {
            SensorFit::Auto => "auto",
            SensorFit::Horizontal => "horizontal",
            SensorFit::Vertical => "vertical",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "auto" => SensorFit::Auto,
            "horizontal" => SensorFit::Horizontal,
            "vertical" => SensorFit::Vertical,
            _ => return None,
        })
    }
}

/// How a camera turns a scene into an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Perspective,
    /// No convergence. `ortho_scale` is the width of the view in metres.
    Orthographic,
}

/// A camera datablock.
///
/// Physical parameters, in the units the instruments they describe are labelled
/// in. See the module docs for why the focal length is stored and the field of
/// view is not.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub name: String,
    pub linkage: crate::document::Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,

    pub projection: Projection,
    /// Focal length in millimetres. What a lens is labelled with.
    pub focal_length_mm: f64,
    /// Sensor width in millimetres. 36 is a full 35mm frame.
    pub sensor_width_mm: f64,
    /// Sensor height in millimetres. 24 completes a full 35mm frame.
    pub sensor_height_mm: f64,
    pub sensor_fit: SensorFit,
    /// The width of an orthographic view, in metres.
    ///
    /// Ignored by a perspective camera, and kept rather than reset when the
    /// projection changes: switching to orthographic to check a silhouette and
    /// back is a thing a modeller does dozens of times, and it must not lose
    /// the framing.
    pub ortho_scale: f64,

    /// Near and far clipping, in metres.
    pub clip_near: f64,
    pub clip_far: f64,

    /// Whether the exposure is metered off the scene rather than dialled in.
    ///
    /// On by default, and it has to be. The lights are photometric and the
    /// exposure is photographic, so the two have independent defaults that must
    /// somehow agree — and no single pair agrees for both an indoor lamp and
    /// the sun. The shipped defaults were a bright-overcast exposure and a
    /// thousand-candela bulb, eleven stops apart: adding a lamp from the menu
    /// produced a black screen, and adding a sun produced a picture. Three of
    /// the four kinds were unusable.
    ///
    /// This is what a real camera does about the same problem, and what its
    /// light meter is for. `exposure_compensation` still applies on top, so the
    /// ±EV dial means what it means on a camera: a deliberate departure from
    /// what the meter said.
    ///
    /// Turning it off hands the picture back to `f_stop`, `shutter` and `iso` —
    /// which is what a shot being matched to a plate needs, and why those
    /// fields are not merely decorative.
    pub auto_exposure: bool,
    /// The f-number: focal length divided by the aperture diameter.
    ///
    /// `None` is a pinhole — everything in focus, no depth of field, and no
    /// cost. Distinguished from a very large f-number because "off" and
    /// "f/1000" want different code paths: one skips the sampling entirely.
    pub f_stop: Option<f64>,
    /// How far away the plane of focus is, in metres.
    pub focus_distance: f64,

    /// How long the shutter is open, as a fraction of one frame's interval.
    ///
    /// A fraction and not a duration, because that is what the number means to
    /// the person setting it: half a frame is a 180° shutter whatever the frame
    /// rate, and it stays a 180° shutter when the rate changes. A duration in
    /// seconds would have to be re-entered every time the scene's rate moved,
    /// which is exactly the trap the time types avoid by storing the frame.
    ///
    /// Zero is an instant — no motion blur, and no cost.
    pub shutter: f64,

    /// Sensitivity, as an ISO number. 100 is a slow film and a base sensor.
    ///
    /// The third leg of the exposure triangle, and the one that had no
    /// representation until the renderer needed a picture: the aperture and
    /// the shutter were here for depth of field and motion blur, which are
    /// what a modeller sets them for, and neither of those needs to know how
    /// sensitive the sensor is.
    pub iso: f64,
    /// The ±EV dial, in stops.
    ///
    /// Every camera has one, and it is the control a person actually reaches
    /// for: the triangle sets the look — how much is in focus, how much is
    /// blurred — and this sets the brightness without disturbing either.
    pub exposure_compensation: f64,
}

/// The aperture the exposure assumes when depth of field is off.
///
/// [`Camera::f_stop`] of `None` means "do not simulate depth of field", which
/// is a rendering choice and not a claim that no light enters. Something has
/// to stand in, and f/5.6 is the middle of every lens.
pub const DEFAULT_EXPOSURE_F_STOP: f64 = 5.6;

/// The shutter the exposure assumes when motion blur is off.
///
/// Same argument: zero shutter means no blur, not a closed camera.
pub const DEFAULT_EXPOSURE_SHUTTER_SECONDS: f64 = 1.0 / 60.0;

impl Default for Camera {
    /// A 50mm lens on a full 35mm frame, wide open at nothing.
    ///
    /// 50mm because it is the one focal length that is a default rather than a
    /// choice, and 36×24 because every sensor size anyone quotes is quoted
    /// relative to it.
    fn default() -> Self {
        Self {
            name: String::new(),
            linkage: crate::document::Linkage::Local,
            uid: Uid::NONE,
            projection: Projection::Perspective,
            focal_length_mm: 50.0,
            sensor_width_mm: 36.0,
            sensor_height_mm: 24.0,
            sensor_fit: SensorFit::Auto,
            ortho_scale: 8.0,
            clip_near: 0.1,
            clip_far: 1000.0,
            f_stop: None,
            focus_distance: 10.0,
            shutter: 0.5,
            iso: 100.0,
            auto_exposure: true,
            exposure_compensation: 0.0,
        }
    }
}

impl Camera {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// The sensor dimension the image's given axis fills, in millimetres.
    ///
    /// `aspect` is the image's width divided by its height.
    fn sensor_along(&self, aspect: f64, horizontal: bool) -> f64 {
        let fits_horizontally = match self.sensor_fit {
            SensorFit::Horizontal => true,
            SensorFit::Vertical => false,
            // The width applies to the longer axis. A square image is neither
            // longer, and taking it as horizontal keeps the answer continuous
            // as the aspect crosses one — a discontinuity there would make a
            // render at 1000x1000 frame differently from one at 1000x1001.
            SensorFit::Auto => aspect >= 1.0,
        };
        match (fits_horizontally, horizontal) {
            (true, true) => self.sensor_width_mm,
            (true, false) => self.sensor_width_mm / aspect,
            // Auto fitting vertically still uses the *width* value, because
            // "sensor size" in that mode means the size of the longer axis and
            // the longer axis is the vertical one. Only an explicit vertical
            // fit reads the height.
            (false, false) => match self.sensor_fit {
                SensorFit::Vertical => self.sensor_height_mm,
                _ => self.sensor_width_mm,
            },
            (false, true) => {
                let vertical = match self.sensor_fit {
                    SensorFit::Vertical => self.sensor_height_mm,
                    _ => self.sensor_width_mm,
                };
                vertical * aspect
            }
        }
    }

    /// Vertical field of view in radians, for an image of this aspect.
    ///
    /// Derived, never stored. `aspect` is width over height.
    pub fn fov_y(&self, aspect: f64) -> f64 {
        let sensor = self.sensor_along(aspect, false);
        2.0 * (sensor / (2.0 * self.focal_length_mm)).atan()
    }

    /// Horizontal field of view in radians, for an image of this aspect.
    pub fn fov_x(&self, aspect: f64) -> f64 {
        let sensor = self.sensor_along(aspect, true);
        2.0 * (sensor / (2.0 * self.focal_length_mm)).atan()
    }

    /// The focal length that would give this vertical field of view.
    ///
    /// For a caller that has an angle and wants to store a camera — an importer
    /// reading a format that only carries one, or a user dragging a
    /// field-of-view slider. The sensor stays as it is, which is what makes the
    /// answer well defined.
    pub fn focal_length_for_fov_y(&self, fov_y: f64, aspect: f64) -> f64 {
        let sensor = self.sensor_along(aspect, false);
        sensor / (2.0 * (fov_y * 0.5).tan())
    }

    /// The aperture's diameter in metres, or `None` for a pinhole.
    ///
    /// The one place a millimetre becomes a metre, written where it can be
    /// seen. A renderer wants this for a circle of confusion, in the same units
    /// as the scene.
    pub fn aperture_diameter_m(&self) -> Option<f64> {
        let f_stop = self.f_stop?;
        if f_stop <= 0.0 {
            return None;
        }
        Some(self.focal_length_mm / f_stop / 1000.0)
    }

    /// How long the shutter is open, in seconds, at a given frame rate.
    ///
    /// `None` when it is shut — an instant, and no motion blur to integrate.
    pub fn shutter_seconds(&self, rate: f64) -> Option<f64> {
        if self.shutter <= 0.0 || rate <= 0.0 {
            return None;
        }
        Some(self.shutter / rate)
    }

    /// The exposure value at ISO 100: `log2(N²/t)`, corrected for sensitivity.
    ///
    /// The standard photographic quantity, and the bridge §2.2's photometric
    /// lights were missing. A sun is twenty thousand lux and the display
    /// transform expects scene-linear values where 0.18 is mid grey; without
    /// this, a physically correct scene saturates every pixel at the default
    /// exposure, which is exactly what the rendered viewport's first tests
    /// measured.
    ///
    /// `rate` is the scene's, because the shutter is stored as a fraction of a
    /// frame (see [`Camera::shutter`]) and an exposure needs a duration.
    pub fn ev100(&self, rate: f64) -> f64 {
        let n = self
            .f_stop
            .filter(|f| *f > 0.0)
            .unwrap_or(DEFAULT_EXPOSURE_F_STOP);
        let t = self
            .shutter_seconds(rate)
            .filter(|t| *t > 0.0)
            .unwrap_or(DEFAULT_EXPOSURE_SHUTTER_SECONDS);
        let iso = self.iso.max(1.0);
        (n * n / t).log2() - (iso / 100.0).log2()
    }

    /// What a scene-linear luminance is multiplied by to become a display
    /// value.
    ///
    /// The Saturation Based Sensitivity form, `1 / (1.2·2^EV)`. The 1.2 is the
    /// constant that puts an 18% grey card at 0.18 when the exposure is
    /// correct — which is the whole calibration, and why sunny sixteen lands
    /// where a photographer expects.
    pub fn exposure_scale(&self, rate: f64) -> f64 {
        let ev = self.ev100(rate) - self.exposure_compensation;
        1.0 / (1.2 * 2f64.powf(ev))
    }

    /// The same, in stops, which is what the display transform takes.
    ///
    /// A separate accessor rather than leaving the logarithm to the caller,
    /// because the two forms are easy to confuse and one of them silently
    /// produces a picture.
    pub fn display_exposure(&self, rate: f64) -> f64 {
        self.exposure_scale(rate).log2()
    }

    /// The exposure to use, given what a light meter read at the subject.
    ///
    /// `metered` is the illuminance in lux arriving where the picture is about,
    /// and `None` is a meter that found nothing — an unlit scene, where there
    /// is no exposure that helps and the dialled-in one is as good an answer as
    /// any.
    ///
    /// The calibration is the grey card: a surface of 18% albedo under
    /// `E` lux returns `0.18 · E / π`, and the exposure that puts that at 0.18
    /// is `π / E`. No 1.2 here, unlike [`Camera::exposure_scale`] — that
    /// constant belongs to the photographic form, where it relates a scene
    /// luminance to a sensor's saturation. This form is a direct statement
    /// about a grey card and needs nothing on top of it.
    ///
    /// `exposure_compensation` still applies, because it is what a photographer
    /// means by it: a deliberate departure from what the meter said.
    pub fn metered_exposure(&self, metered: Option<f64>, rate: f64) -> f64 {
        match metered.filter(|e| *e > 0.0) {
            Some(lux) if self.auto_exposure => {
                (std::f64::consts::PI / lux).log2() + self.exposure_compensation
            }
            _ => self.display_exposure(rate),
        }
    }

    /// Whether this camera needs depth of field sampled at all.
    pub fn has_depth_of_field(&self) -> bool {
        self.aperture_diameter_m().is_some_and(|d| d > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Degrees, for a comparison a person can check against a lens chart.
    fn deg(radians: f64) -> f64 {
        radians.to_degrees()
    }

    #[test]
    fn a_fifty_millimetre_lens_on_a_full_frame_is_forty_degrees_across() {
        // The number every photographer knows, and the reason it is the test:
        // `2·atan(36 / 100)` is 39.6°, and anybody can check that against a
        // lens chart without reading this code.
        let c = Camera::default();
        assert!(
            (deg(c.fov_x(16.0 / 9.0)) - 39.597).abs() < 0.01,
            "got {}",
            deg(c.fov_x(16.0 / 9.0))
        );
    }

    #[test]
    fn eighteen_millimetres_on_a_full_frame_is_a_right_angle() {
        // `2·atan(36 / 36)` is exactly 90°, which makes this the one point on
        // the curve with no rounding to argue about.
        let c = Camera {
            focal_length_mm: 18.0,
            ..Camera::default()
        };
        assert!((deg(c.fov_x(16.0 / 9.0)) - 90.0).abs() < 1e-9);
    }

    #[test]
    fn the_vertical_angle_follows_the_aspect_under_a_horizontal_fit() {
        // The sensor's width fills the image width, so a taller image sees
        // *more* vertically. A camera that ignored the aspect would frame a
        // square render like a widescreen one.
        let c = Camera {
            sensor_fit: SensorFit::Horizontal,
            ..Camera::default()
        };
        let wide = c.fov_y(16.0 / 9.0);
        let square = c.fov_y(1.0);
        assert!(square > wide, "{} should exceed {}", deg(square), deg(wide));
        // And the horizontal angle does not move, because that is what a
        // horizontal fit means.
        assert!((c.fov_x(16.0 / 9.0) - c.fov_x(1.0)).abs() < 1e-12);
    }

    #[test]
    fn a_vertical_fit_pins_the_vertical_angle_instead() {
        let c = Camera {
            sensor_fit: SensorFit::Vertical,
            ..Camera::default()
        };
        assert!((c.fov_y(16.0 / 9.0) - c.fov_y(1.0)).abs() < 1e-12);
        // 24mm of sensor at 50mm: `2·atan(24 / 100)` is 27.0°.
        assert!(
            (deg(c.fov_y(2.0)) - 27.0).abs() < 0.05,
            "{}",
            deg(c.fov_y(2.0))
        );
    }

    #[test]
    fn an_automatic_fit_follows_the_longer_axis() {
        // What every other tool defaults to. A landscape image fits the sensor
        // width across; a portrait image fits it *down*, which is the half that
        // is easy to get wrong and shows as a portrait render framed too
        // tightly.
        let auto = Camera::default();
        let horizontal = Camera {
            sensor_fit: SensorFit::Horizontal,
            ..Camera::default()
        };
        let landscape = 16.0 / 9.0;
        assert!((auto.fov_x(landscape) - horizontal.fov_x(landscape)).abs() < 1e-12);

        // Portrait: the width value now governs the vertical angle, and it is
        // the *width* and not the height even under an automatic fit.
        let portrait = 9.0 / 16.0;
        assert!(
            (deg(auto.fov_y(portrait)) - deg(horizontal.fov_x(landscape))).abs() < 1e-9,
            "a portrait image should see the same angle down that a landscape \
             one sees across: {} against {}",
            deg(auto.fov_y(portrait)),
            deg(horizontal.fov_x(landscape))
        );
        assert!(
            (auto.fov_y(portrait) - auto.fov_y(1.0 / portrait)).abs() > 0.1,
            "the automatic fit is not responding to the aspect at all"
        );
    }

    #[test]
    fn the_automatic_fit_does_not_jump_at_a_square_image() {
        // A discontinuity here would make a render at 1000x1000 frame
        // differently from one at 1000x1001, which is the kind of thing nobody
        // reproduces.
        let c = Camera::default();
        let just_under = c.fov_y(0.999_999);
        let square = c.fov_y(1.0);
        let just_over = c.fov_y(1.000_001);
        assert!((square - just_under).abs() < 1e-5);
        assert!((square - just_over).abs() < 1e-5);
    }

    #[test]
    fn the_focal_length_and_the_angle_are_the_same_information() {
        // What makes storing the focal length a choice about *which* is
        // authoritative rather than a loss. An importer with only an angle can
        // land on a focal length, and it comes back.
        let c = Camera::default();
        for aspect in [16.0 / 9.0, 1.0, 9.0 / 16.0, 2.39] {
            for degrees in [10.0, 27.0, 45.0, 90.0, 120.0] {
                let wanted = degrees * std::f64::consts::PI / 180.0;
                let focal = c.focal_length_for_fov_y(wanted, aspect);
                let round_tripped = Camera {
                    focal_length_mm: focal,
                    ..c.clone()
                }
                .fov_y(aspect);
                assert!(
                    (round_tripped - wanted).abs() < 1e-12,
                    "{degrees}° at aspect {aspect} came back as {}",
                    deg(round_tripped)
                );
            }
        }
    }

    #[test]
    fn a_pinhole_has_no_aperture_and_no_depth_of_field() {
        // "Off" and "f/1000" want different code paths — one skips the
        // sampling entirely — so they are different values and not the same
        // value with a large number in it.
        let c = Camera::default();
        assert_eq!(c.f_stop, None);
        assert_eq!(c.aperture_diameter_m(), None);
        assert!(!c.has_depth_of_field());

        let stopped_down = Camera {
            f_stop: Some(1000.0),
            ..Camera::default()
        };
        assert!(stopped_down.has_depth_of_field());
    }

    #[test]
    fn the_aperture_is_the_focal_length_over_the_f_number_in_metres() {
        // A 50mm lens at f/2 has a 25mm aperture, which is 0.025 of a scene
        // unit. The one conversion in this module, and the test that says which
        // way round it goes.
        let c = Camera {
            f_stop: Some(2.0),
            ..Camera::default()
        };
        let d = c.aperture_diameter_m().expect("an aperture");
        assert!((d - 0.025).abs() < 1e-12, "{d}");
    }

    #[test]
    fn a_nonsense_f_number_is_a_pinhole_rather_than_an_infinity() {
        // A zero or negative f-stop from a hand-edited file or a script would
        // otherwise divide into an infinite aperture, and every ray would come
        // out `NaN` with the cause a long way from the symptom.
        for bad in [0.0, -2.0] {
            let c = Camera {
                f_stop: Some(bad),
                ..Camera::default()
            };
            assert_eq!(c.aperture_diameter_m(), None, "f/{bad}");
            assert!(!c.has_depth_of_field());
        }
    }

    #[test]
    fn the_shutter_is_a_fraction_of_a_frame_and_not_a_duration() {
        // Which is what makes it survive a change of frame rate. A 180°
        // shutter is half a frame at any rate, and a duration in seconds would
        // have to be re-entered every time the scene's rate moved — the trap
        // the time types avoid by storing the frame.
        let c = Camera::default();
        assert_eq!(c.shutter, 0.5);
        let at_24 = c.shutter_seconds(24.0).expect("open");
        let at_48 = c.shutter_seconds(48.0).expect("open");
        assert!((at_24 - 1.0 / 48.0).abs() < 1e-12, "{at_24}");
        assert!((at_48 - 1.0 / 96.0).abs() < 1e-12, "{at_48}");
        // Still half a frame in both cases, which is the invariant.
        assert!((at_24 * 24.0 - at_48 * 48.0).abs() < 1e-12);
    }

    #[test]
    fn a_shut_shutter_is_an_instant() {
        let c = Camera {
            shutter: 0.0,
            ..Camera::default()
        };
        assert_eq!(c.shutter_seconds(24.0), None);
        // And a nonsense rate cannot produce an infinite exposure.
        assert_eq!(Camera::default().shutter_seconds(0.0), None);
    }

    #[test]
    fn the_orthographic_framing_survives_a_trip_through_perspective() {
        // Switching to orthographic to check a silhouette and back is
        // something a modeller does dozens of times an hour, and it must not
        // lose the framing.
        let mut c = Camera {
            ortho_scale: 3.5,
            projection: Projection::Orthographic,
            ..Camera::default()
        };
        c.projection = Projection::Perspective;
        c.projection = Projection::Orthographic;
        assert_eq!(c.ortho_scale, 3.5);
    }

    #[test]
    fn a_sensor_fit_round_trips_through_its_text_form() {
        for fit in [SensorFit::Auto, SensorFit::Horizontal, SensorFit::Vertical] {
            assert_eq!(SensorFit::from_id(fit.id()), Some(fit));
        }
        assert_eq!(SensorFit::from_id("diagonal"), None);
    }
}
