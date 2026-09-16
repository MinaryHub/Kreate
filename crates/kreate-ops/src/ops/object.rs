//! Object-mode operators.

use kreate_math::{DAffine3, DQuat, DVec3};
use kreate_mesh::edit;
use kreate_scene::document::{MeshId, ObjectId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

/// The primitive shapes `object.add_primitive` can build.
///
/// Indices are part of the scripting and keymap surface, so entries are only
/// ever appended — reordering them would silently change what a saved keymap or
/// an old script does.
pub const PRIMITIVES: &[&str] = &[
    "plane",
    "cube",
    "grid",
    "uv_sphere",
    "ico_sphere",
    "cylinder",
    "cone",
    "torus",
];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddPrimitiveParams {
    /// Which shape to add.
    #[param(variants(
        "plane",
        "cube",
        "grid",
        "uv_sphere",
        "ico_sphere",
        "cylinder",
        "cone",
        "torus"
    ))]
    pub shape: u32,
    /// Overall size, in metres.
    #[param(min = 0.0001, soft_max = 100.0, unit = "length")]
    pub size: f64,
    /// Segments around the main axis, for the rounded shapes.
    #[param(min = 3, max = 512, unit = "count")]
    pub segments: u32,
    /// Rings from pole to pole, or minor segments for a torus.
    #[param(min = 2, max = 512, unit = "count")]
    pub rings: u32,
    /// Where to place the new object.
    pub location: [f64; 3],
}

impl Default for AddPrimitiveParams {
    fn default() -> Self {
        Self {
            shape: 1, // cube
            size: 2.0,
            segments: 32,
            rings: 16,
            location: [0.0; 3],
        }
    }
}

/// Adds a primitive mesh object.
pub struct AddPrimitive;

impl Operator for AddPrimitive {
    type Params = AddPrimitiveParams;

    fn id() -> &'static str {
        "object.add_primitive"
    }
    fn label() -> &'static str {
        "Add Primitive"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::primitives as p;
        let size = params.size as f32;
        let seg = params.segments;
        let rings = params.rings;
        let name = PRIMITIVES
            .get(params.shape as usize)
            .copied()
            .unwrap_or("cube");

        let mesh = match name {
            "plane" => p::plane(size),
            "grid" => p::grid(size, seg.min(256), rings.min(256)),
            "uv_sphere" => p::uv_sphere(size * 0.5, seg, rings),
            "ico_sphere" => p::ico_sphere(size * 0.5, (rings as f32).log2().round() as u32),
            "cylinder" => p::cylinder(size * 0.5, size, seg),
            "cone" => p::cone(size * 0.5, size, seg),
            "torus" => p::torus(size * 0.5, size * 0.15, seg, rings),
            _ => p::cube(size),
        };

        // Validated here rather than trusted: a primitive generator that emits
        // invalid topology would seed corruption that only surfaces three
        // operators later, with no way to attribute it.
        let violations = mesh.validate();
        if !violations.is_empty() {
            return Err(OpError::Failed(format!(
                "generated {name} is invalid: {violations:?}"
            )));
        }

        let d = doc.document_mut();
        let id = d.add_mesh_object(capitalise(name), mesh);
        d.object_mut(id).expect("just added").transform.translation =
            DVec3::from_array(params.location);
        // A newly added object becomes the selection, which is what every
        // subsequent operator acts on.
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddCameraParams {
    /// Focal length in millimetres. What a lens is labelled with.
    ///
    /// The stored quantity, not the field of view — see
    /// `kreate_scene::camera` for why that is the way round it is.
    #[param(min = 1.0, soft_min = 8.0, soft_max = 300.0)]
    pub focal_length_mm: f64,
    /// Sensor width in millimetres. 36 is a full 35mm frame.
    #[param(min = 0.1, soft_max = 100.0)]
    pub sensor_width_mm: f64,
    /// The f-number. Zero is a pinhole: everything in focus, and nothing to
    /// sample.
    #[param(min = 0.0, soft_min = 0.0, soft_max = 32.0)]
    pub f_stop: f64,
    /// How far away the plane of focus is, in metres.
    #[param(min = 0.0, soft_max = 100.0, unit = "length")]
    pub focus_distance: f64,
    /// Where to place the new camera.
    pub location: [f64; 3],
}

impl Default for AddCameraParams {
    fn default() -> Self {
        Self {
            focal_length_mm: 50.0,
            sensor_width_mm: 36.0,
            f_stop: 0.0,
            focus_distance: 10.0,
            location: [0.0, -8.0, 2.0],
        }
    }
}

/// Adds a camera object and the datablock behind it.
///
/// Two things, and the distinction is the point: the object is *where* the
/// camera is and the datablock is *what* it is. Two objects can share one lens.
pub struct AddCamera;

impl Operator for AddCamera {
    type Params = AddCameraParams;

    fn id() -> &'static str {
        "object.add_camera"
    }
    fn label() -> &'static str {
        "Add Camera"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let camera = kreate_scene::Camera {
            focal_length_mm: params.focal_length_mm,
            sensor_width_mm: params.sensor_width_mm,
            // A zero f-number is a pinhole and not an infinite aperture. The
            // parameter is a plain float because a slider cannot be `None`, and
            // this is where the two forms meet.
            f_stop: (params.f_stop > 0.0).then_some(params.f_stop),
            focus_distance: params.focus_distance,
            ..kreate_scene::Camera::new("Camera")
        };

        let d = doc.document_mut();
        let datablock = d.add_camera(camera);
        let mut object = kreate_scene::document::Object::new(
            "Camera",
            kreate_scene::document::ObjectData::Camera(datablock),
        );
        object.transform.translation = DVec3::from_array(params.location);
        let id = d.add_object(object);
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddLightParams {
    /// What kind of light.
    ///
    /// Indices are part of the scripting and keymap surface, so entries are
    /// only ever appended.
    #[param(variants("point", "sun", "spot", "area"))]
    pub kind: u32,
    /// The light's tint, as the colour picker holds it: sRGB, bounded by one.
    ///
    /// Not an amount of light — that is `intensity`, in physical units. The
    /// operator decodes it into the working space on the way in, which is the
    /// one place a colour conversion belongs.
    #[param(color)]
    pub color: [f64; 3],
    /// How much light, in the unit the chosen kind uses — candela for a point
    /// or a spot, lux for a sun, nits for an area light.
    ///
    /// **Zero means the kind's own default**, because a light of zero intensity
    /// is a light that does nothing and nobody adds one deliberately. The
    /// alternative is a single number in the parameter's default, and there is
    /// no single number: 1000 is a bright room in candela and a dull overcast
    /// afternoon in lux.
    #[param(min = 0.0, soft_max = 100_000.0)]
    pub intensity: f64,
    /// Stops on top of `intensity`.
    #[param(soft_min = -10.0, soft_max = 10.0)]
    pub exposure: f64,
    /// Where to put it.
    pub location: [f64; 3],
}

impl Default for AddLightParams {
    fn default() -> Self {
        Self {
            kind: 0,
            color: [1.0, 1.0, 1.0],
            // The kind's own default. See the field.
            intensity: 0.0,
            exposure: 0.0,
            location: [0.0, 0.0, 4.0],
        }
    }
}


#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetLightParams {
    /// What kind of light.
    ///
    /// Indices are part of the scripting and keymap surface, so entries are
    /// only ever appended. The same order as `AddLightParams::kind`, because
    /// two orders for one enum is a script that adds a sun and edits a spot.
    #[param(variants("point", "sun", "spot", "area"))]
    pub kind: u32,
    /// The light's tint, as the colour picker holds it: sRGB, bounded by one.
    ///
    /// Not an amount of light. See `AddLightParams::color`.
    #[param(color)]
    pub color: [f64; 3],
    /// How much light, in the unit the chosen kind uses \u2014 candela for a point
    /// or a spot, lux for a sun, nits for an area light.
    ///
    /// Unlike the add operator, zero here means zero. Adding a light of no
    /// brightness is a thing nobody does on purpose; turning one down to
    /// nothing is a thing people do all the time, and a panel that snapped it
    /// back to a default would be unusable.
    #[param(min = 0.0, soft_max = 100_000.0)]
    pub intensity: f64,
    /// Stops on top of `intensity`.
    ///
    /// The dial an artist actually reaches for, and the one `UsdLux` keeps
    /// separately \u2014 see \u00a77.1 on why the two are not collapsed into one number.
    #[param(soft_min = -10.0, soft_max = 10.0)]
    pub exposure: f64,
    /// How big the source is, in metres: what makes a shadow soft.
    ///
    /// A point's or a spot's. Ignored by the other two kinds, which have their
    /// own shapes \u2014 carried in the same parameter block rather than split into
    /// an operator per kind, because changing the kind is one of the edits and
    /// an operator that could not do it would leave the panel switching between
    /// operators mid-gesture.
    #[param(min = 0.0, soft_max = 2.0, unit = "length")]
    pub radius: f64,
    /// How wide a sun appears, in radians. The sun's own is 0.526\u00b0.
    #[param(min = 0.0, soft_max = 0.2, unit = "angle")]
    pub angular_diameter: f64,
    /// A spot's full cone angle, in radians.
    ///
    /// The soft maximum is a hair under a half turn: a cone of exactly pi is a
    /// hemisphere, which is a point light with extra arithmetic, and the dial
    /// should stop where the shape stops meaning anything.
    #[param(min = 0.0, soft_max = 3.1, unit = "angle")]
    pub cone_angle: f64,
    /// How much of that cone fades, from a hard edge at zero.
    #[param(min = 0.0, max = 1.0)]
    pub blend: f64,
    /// Whether the light casts a shadow.
    pub cast_shadow: bool,
    /// Whether it lights the diffuse response. \u00a72.2's lighting cheat.
    pub affect_diffuse: bool,
    /// And the specular one.
    pub affect_specular: bool,
}

impl Default for SetLightParams {
    fn default() -> Self {
        Self {
            kind: 0,
            color: [1.0, 1.0, 1.0],
            intensity: 1000.0,
            exposure: 0.0,
            radius: 0.1,
            angular_diameter: kreate_scene::LightKind::SUN_ANGULAR_DIAMETER,
            cone_angle: std::f64::consts::FRAC_PI_4,
            blend: 0.15,
            cast_shadow: true,
            affect_diffuse: true,
            affect_specular: true,
        }
    }
}

/// Sets the active object's light exactly.
///
/// Absolute, for the reason `SetTransform` is: these are numbers a person
/// types into a panel, and expressing a typed number as a delta means the
/// interface computing `wanted - current` and sending the difference.
///
/// # The datablock, not the object
///
/// Where the lamp *is* belongs to the object and goes through `SetTransform`;
/// a sun's direction and a spot's aim are its rotation. What kind of lamp it is
/// and how bright belong to the datablock, which a hundred linked copies share
/// \u2014 \u00a72.2's argument, and the reason editing one of them changes all of them.
/// That is the intent, and it is the whole reason the datablock is not a field
/// on the object.
pub struct SetLight;

impl Operator for SetLight {
    type Params = SetLightParams;

    fn id() -> &'static str {
        "light.set_properties"
    }
    fn label() -> &'static str {
        "Set Light"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        // Only when there is a light to set. Without this the panel would offer
        // itself for a mesh, and the operator would have to decide what "set the
        // light of a cube" means.
        active_light(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = active_light(doc.document()) else {
            // Polled, but a keymap or a script can call anything at any time.
            return Ok(OpResult::Cancelled);
        };

        // Clamped rather than refused, as the add operator does: a stale script
        // naming a kind this build does not have asked for a light, and the
        // worst honest answer is the first kind rather than nothing.
        let index = (params.kind as usize).min(kreate_scene::LightKind::ALL.len() - 1);
        let kind = match kreate_scene::LightKind::ALL[index] {
            kreate_scene::LightKind::Point { .. } => kreate_scene::LightKind::Point {
                radius: params.radius.max(0.0),
            },
            kreate_scene::LightKind::Sun { .. } => kreate_scene::LightKind::Sun {
                angular_diameter: params.angular_diameter.max(0.0),
            },
            kreate_scene::LightKind::Spot { .. } => kreate_scene::LightKind::Spot {
                radius: params.radius.max(0.0),
                cone_angle: params.cone_angle.max(0.0),
                blend: params.blend.clamp(0.0, 1.0),
            },
            kreate_scene::LightKind::Area { shape, spread } => {
                // The one kind whose own shape this operator does not carry.
                // An area light's rectangle or disc is two more numbers and a
                // second enum, and a panel for it wants its own row; until
                // there is one, changing to an area light keeps whatever shape
                // it had rather than flattening it to a default.
                kreate_scene::LightKind::Area { shape, spread }
            }
        };

        let d = doc.document_mut();
        let Some(light) = d.light_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        light.kind = kind;
        light.intensity = params.intensity.max(0.0);
        light.exposure = params.exposure;
        light.color = kreate_math::Srgb::new(
            params.color[0] as f32,
            params.color[1] as f32,
            params.color[2] as f32,
        )
        .as_reflectance();
        light.cast_shadow = params.cast_shadow;
        light.affect_diffuse = params.affect_diffuse;
        light.affect_specular = params.affect_specular;
        Ok(OpResult::Finished)
    }
}

/// The light datablock the active object points at, if it points at one.
fn active_light(doc: &Document) -> Option<kreate_scene::LightId> {
    let object = doc.object(doc.selection.active?)?;
    match object.data {
        kreate_scene::document::ObjectData::Light(id) => Some(id),
        _ => None,
    }
}

/// The parameters that would leave a light exactly as it is.
///
/// The panel's other half: an absolute operator needs the current values to
/// start from, and working them out at the call site is where a panel and an
/// operator drift apart about what a field means. One function, so the round
/// trip through `SetLight` is an identity and a test can say so.
pub fn light_params(light: &kreate_scene::Light) -> SetLightParams {
    let mut p = SetLightParams {
        kind: kreate_scene::LightKind::ALL
            .iter()
            .position(|k| std::mem::discriminant(k) == std::mem::discriminant(&light.kind))
            .unwrap_or(0) as u32,
        color: {
            let c = light.color.as_swatch();
            [c.r as f64, c.g as f64, c.b as f64]
        },
        intensity: light.intensity,
        exposure: light.exposure,
        cast_shadow: light.cast_shadow,
        affect_diffuse: light.affect_diffuse,
        affect_specular: light.affect_specular,
        ..SetLightParams::default()
    };
    // The kind's own numbers, and only the ones this kind has: a spot's cone
    // read off a sun would be the default showing up as though it were the
    // light's, which is the panel telling a small lie every time the kind
    // changes.
    match light.kind {
        kreate_scene::LightKind::Point { radius } => p.radius = radius,
        kreate_scene::LightKind::Sun { angular_diameter } => {
            p.angular_diameter = angular_diameter
        }
        kreate_scene::LightKind::Spot {
            radius,
            cone_angle,
            blend,
        } => {
            p.radius = radius;
            p.cone_angle = cone_angle;
            p.blend = blend;
        }
        kreate_scene::LightKind::Area { .. } => {}
    }
    p
}


/// Whether a parameter means anything for a light of this kind.
///
/// **Why the panel needs this and the operator does not.** Every kind's
/// numbers ride in one parameter block, deliberately: changing the kind is one
/// of the edits, and an operator that could not do it would leave the panel
/// switching between operators mid-gesture. `light_params` already declines to
/// read another kind's numbers off a light that has none — the comment there
/// calls showing them "a small lie" — but the *fields* were drawn regardless,
/// so a point light's panel offered a cone angle and an angular diameter that
/// did nothing at all.
///
/// Here rather than in the shell, for the reason `light_params` is here: two
/// places deciding what a kind has is how they come to disagree.
pub fn light_param_applies(kind: u32, param: &str) -> bool {
    // Indices are the ones `LightKind::ALL` is in, which is the order the
    // enum's variants are listed and the order a script sends.
    let point = kind == 0;
    let sun = kind == 1;
    let spot = kind == 2;
    match param {
        // The size of the source, which is what makes a shadow soft. An area
        // light has its own shape instead.
        "radius" => point || spot,
        "angular_diameter" => sun,
        "cone_angle" | "blend" => spot,
        // Everything else is every light's: what kind it is, its colour, how
        // much of it there is, and what it touches.
        _ => true,
    }
}

// --- the world -------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetWorldParams {
    /// The background's colour, as the picker holds it: sRGB, bounded by one.
    ///
    /// A reflectance-shaped number for something that is an *emission*, which
    /// is the compromise a colour picker forces: a swatch cannot express a
    /// hundred nits. `strength` is the amount, and the two multiply.
    #[param(color)]
    pub color: [f64; 3],
    /// How much light the background gives, linear.
    ///
    /// Zero is a scene lit only by its lamps, which is a real thing to want and
    /// is what a document with no world means \u2014 so it is reachable from the
    /// panel rather than only by deleting the world.
    #[param(min = 0.0, soft_max = 10.0)]
    pub strength: f64,
    /// Radians about the up axis.
    #[param(unit = "angle")]
    pub rotation: f64,
}

impl Default for SetWorldParams {
    fn default() -> Self {
        Self {
            color: [0.05, 0.05, 0.06],
            strength: 1.0,
            rotation: 0.0,
        }
    }
}

/// Sets the active world exactly.
///
/// The *active* one, which is the document's pointer and not a selection: a
/// world is not an object and cannot be clicked on. A file with a day setup and
/// a night one has two, and switching between them is `Document::set_active_world`
/// rather than an edit to either.
pub struct SetWorld;

impl Operator for SetWorld {
    type Params = SetWorldParams;

    fn id() -> &'static str {
        "world.set_properties"
    }
    fn label() -> &'static str {
        "Set World"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.active_world().is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().active_world() else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document_mut();
        let Some(world) = d.world_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        // An image background keeps its image: the panel offers a colour, and
        // a colour typed into a panel is not a reason to throw away an HDRI
        // somebody loaded. Changing *which* it is belongs to the control that
        // picks the image.
        if let kreate_scene::Background::Flat(_) = world.background {
            world.background = kreate_scene::Background::Flat(
                kreate_math::Srgb::new(
                    params.color[0] as f32,
                    params.color[1] as f32,
                    params.color[2] as f32,
                )
                .as_reflectance(),
            );
        }
        world.strength = params.strength.max(0.0);
        world.rotation = params.rotation;
        Ok(OpResult::Finished)
    }
}

/// The parameters that would leave the active world exactly as it is.
pub fn world_params(world: &kreate_scene::World) -> SetWorldParams {
    SetWorldParams {
        color: match world.background {
            kreate_scene::Background::Flat(c) => {
                let s = c.as_swatch();
                [s.r as f64, s.g as f64, s.b as f64]
            }
            // An image has no one colour. White, so that a panel showing it does
            // not claim the picture is tinted.
            kreate_scene::Background::Image(_) => [1.0, 1.0, 1.0],
        },
        strength: world.strength,
        rotation: world.rotation,
    }
}


#[derive(Clone, Debug, PartialEq, Default, OpParams)]
pub struct SetWorldImageParams {
    /// The image datablock's identity, in decimal. Empty goes back to a colour.
    ///
    /// **An identity and not an index**, for the reason §7.9's archive stores
    /// one: a position says where the image sits today and a `Uid` says which
    /// image it is. An undo step holding a position would point at whatever
    /// moved into that slot.
    ///
    /// Text rather than a number because a `Uid` is sixty-four bits and the
    /// property system's integer is signed — a round trip through it would turn
    /// half the possible identities negative, and the half it turns is the half
    /// a long session reaches.
    pub uid: String,
}

/// Points the active world's background at an image, or back at its colour.
///
/// **The operator does not load anything.** `kreate-ops` cannot read a file and
/// should not: an operator that touched the filesystem could not be replayed,
/// and the undo step for "a sky arrived" would have to hold the sky. So the
/// application opens the file, adds the datablock, and this makes the *link* —
/// which is the part worth undoing.
///
/// Undoing it therefore leaves the image in the document with nothing pointing
/// at it. That is what Blender does with an unused image and it is the right
/// answer: the alternative is an undo that deletes a datablock another world,
/// or a material, may have started using in the meantime.
pub struct SetWorldImage;

impl Operator for SetWorldImage {
    type Params = SetWorldImageParams;

    fn id() -> &'static str {
        "world.set_image"
    }
    fn label() -> &'static str {
        "Set World Image"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.active_world().is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().active_world() else {
            return Ok(OpResult::Cancelled);
        };
        // Resolved before the world is borrowed mutably, and resolved against
        // the *document* rather than trusted: a uid naming an image that is not
        // here is a parameter from a script or a stale redo, and the honest
        // answer to it is to refuse rather than to make a background that
        // points at nothing.
        let image = match params.uid.trim() {
            "" => None,
            text => {
                let Ok(raw) = text.parse::<u64>() else {
                    return Ok(OpResult::Cancelled);
                };
                match doc.document().image_by_uid(kreate_scene::Uid::from_raw(raw)) {
                    Some(image) => Some(image),
                    None => return Ok(OpResult::Cancelled),
                }
            }
        };
        let d = doc.document_mut();
        let Some(world) = d.world_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        world.background = match image {
            Some(image) => kreate_scene::Background::Image(image),
            // Back to a colour, and to *the* colour a world with no image has.
            // Not black: a background nobody chose should read as a background
            // and not as a failure, which is the same judgement the archive's
            // fallback makes.
            None => kreate_scene::Background::Flat(kreate_math::SceneRgb::new(0.05, 0.05, 0.06)),
        };
        Ok(OpResult::Finished)
    }
}

/// The parameters that would leave the world's background as it is.
pub fn world_image_params(world: &kreate_scene::World, doc: &Document) -> SetWorldImageParams {
    SetWorldImageParams {
        uid: match world.background {
            kreate_scene::Background::Image(id) => doc
                .image_uid(id)
                .map(|uid| uid.raw().to_string())
                .unwrap_or_default(),
            kreate_scene::Background::Flat(_) => String::new(),
        },
    }
}

// --- a material ------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetMaterialParams {
    /// The base colour: a reflectance, bounded by one.
    ///
    /// For a dielectric this is the diffuse albedo and for a metal the specular
    /// colour. `metallic` says which, which is the whole trick of the
    /// metal-roughness parameterisation.
    #[param(color)]
    pub color: [f64; 3],
    #[param(min = 0.0, max = 1.0)]
    pub metallic: f64,
    #[param(min = 0.0, max = 1.0)]
    pub roughness: f64,
    #[param(min = 0.0, max = 1.0)]
    pub alpha: f64,
}

impl Default for SetMaterialParams {
    fn default() -> Self {
        Self {
            color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            alpha: 1.0,
        }
    }
}

/// Sets the active object's first material slot.
///
/// The first slot, because that is the one the raster pass draws \u2014 see \u00a77.5 on
/// why per-face slots wait for the draws to be split. A panel that offered the
/// others would be editing materials nothing shows.
pub struct SetMaterial;

impl Operator for SetMaterial {
    type Params = SetMaterialParams;

    fn id() -> &'static str {
        "material.set_properties"
    }
    fn label() -> &'static str {
        "Set Material"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_material(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = active_material(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document_mut();
        let Some(material) = d.material_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        // Linear on both sides: glTF's `baseColorFactor` is linear too, and a
        // material's base colour is a reflectance rather than a swatch \u2014 so
        // unlike a light's tint there is no transfer function in the way.
        material.base_color = kreate_math::Srgb::new(
            params.color[0] as f32,
            params.color[1] as f32,
            params.color[2] as f32,
        )
        .as_reflectance();
        material.metallic = params.metallic.clamp(0.0, 1.0) as f32;
        material.roughness = params.roughness.clamp(0.0, 1.0) as f32;
        material.alpha = params.alpha.clamp(0.0, 1.0) as f32;
        Ok(OpResult::Finished)
    }
}

/// The material the active object draws with, if it has one.
pub fn active_material(doc: &Document) -> Option<kreate_scene::MaterialId> {
    let object = doc.selection.active?;
    // Through `material_at`, which is the one place the override rule lives:
    // slots belong to the mesh and overrides to the object, and a panel that
    // resolved them itself would be a second opinion about which wins.
    doc.material_id_at(object, 0)
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetMaterialImageParams {
    /// The image datablock's identity, in decimal. Empty takes the texture off.
    ///
    /// Text and an identity for the two reasons [`SetWorldImageParams::uid`]
    /// gives: a position says where the image sits today, and a `Uid` does not
    /// survive the property system's signed integer.
    pub uid: String,
}

/// Points the active object's material at a base colour texture, or takes it off.
///
/// The same division of labour as [`SetWorldImage`], and for the same reason:
/// the application opens the file and adds the datablock, and this makes the
/// *link*. An operator that read a file could not be replayed.
///
/// **The texture tints rather than replaces.** glTF's base colour is
/// `baseColorTexture * baseColorFactor`, and
/// [`ShaderGraph::for_material`](kreate_scene::shader::ShaderGraph::for_material)
/// builds exactly that — so the colour in the panel goes on meaning something
/// once a texture is assigned, which is what a person adjusting it expects.
pub struct SetMaterialImage;

impl Operator for SetMaterialImage {
    type Params = SetMaterialImageParams;

    fn id() -> &'static str {
        "material.set_image"
    }
    fn label() -> &'static str {
        "Set Material Image"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_material(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = active_material(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        // Resolved against the document before anything is borrowed mutably,
        // and refused rather than trusted: a uid naming an image that is not
        // here is a parameter from a script or a stale redo, and a material
        // pointing at nothing is worse than one that declined to change.
        let image = match params.uid.trim() {
            "" => None,
            text => {
                let Ok(raw) = text.parse::<u64>() else {
                    return Ok(OpResult::Cancelled);
                };
                match doc.document().image_by_uid(kreate_scene::Uid::from_raw(raw)) {
                    Some(image) => Some(image),
                    None => return Ok(OpResult::Cancelled),
                }
            }
        };
        let d = doc.document_mut();
        let Some(material) = d.material_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        material.base_color_image = image;
        Ok(OpResult::Finished)
    }
}

/// Gives the active object a material, so that there is one to edit.
///
/// **Why this has to exist at all.** A mesh starts with no material and an
/// empty slot list, which is the honest state — §1.4 is deliberate that a face
/// index means nothing until a slot fills it. But nothing else creates one, so
/// every material control in the application was unreachable: the Material tab
/// is offered only when there is a material, and there was no way to get the
/// first one. The colour, the metallic and roughness dials and the base colour
/// texture were all finished and none of them was a feature anybody had.
///
/// **On the mesh's slot, not the object's override.** The slot is what a face
/// index points at and what a linked copy shares, so this is "the material of
/// this chair" rather than "the material of this one copy of it". The override
/// stays what it is for — making one copy differ.
#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct NewMaterialParams {}

pub struct NewMaterial;

impl Operator for NewMaterial {
    type Params = NewMaterialParams;

    fn id() -> &'static str {
        "material.new"
    }
    fn label() -> &'static str {
        "New Material"
    }
    /// Only a mesh, and only one that has no material yet.
    ///
    /// **Why not "make another one", which is what Blender's button does.**
    /// Blender's Material tab carries a slot *list*, so a second material is
    /// visible the moment it exists and the first is one click away. There is
    /// no list here yet, and the panel shows this button only while the slot
    /// is empty — so a second material would be created, assigned, and leave
    /// the first with nothing pointing at it and no way back to it. Refusing
    /// is the honest answer until there is a list to pick from.
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_mesh(doc).is_some() && active_material(doc).is_none()
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        // Checked again rather than trusted: `poll` gates the button, and a
        // script or a replayed step reaches `execute` without it.
        if active_material(doc.document()).is_some() {
            return Ok(OpResult::Cancelled);
        }
        // Blender's defaults, and not an accident: a mid grey at 0.5 roughness
        // is the surface a modeller can read form from. Pure white blows out
        // under any light worth having and a mirror shows the room instead of
        // the model.
        let d = doc.document_mut();
        let material = d.add_material(kreate_scene::Material {
            name: "Material".into(),
            ..Default::default()
        });
        d.set_mesh_slot(mesh, 0, Some(material));
        Ok(OpResult::Finished)
    }
}

/// The mesh the active object draws, if it draws one.
pub fn active_mesh(doc: &Document) -> Option<kreate_scene::MeshId> {
    match doc.object(doc.selection.active?)?.data {
        kreate_scene::document::ObjectData::Mesh(mesh) => Some(mesh),
        _ => None,
    }
}

/// The parameters that would leave a material's texture exactly as it is.
pub fn material_image_params(
    doc: &Document,
    material: &kreate_scene::Material,
) -> SetMaterialImageParams {
    SetMaterialImageParams {
        uid: material
            .base_color_image
            .and_then(|id| doc.image_uid(id))
            .map(|uid| uid.raw().to_string())
            .unwrap_or_default(),
    }
}

/// The parameters that would leave a material exactly as it is.
pub fn material_params(material: &kreate_scene::Material) -> SetMaterialParams {
    let c = material.base_color.as_swatch();
    SetMaterialParams {
        color: [c.r as f64, c.g as f64, c.b as f64],
        metallic: material.metallic as f64,
        roughness: material.roughness as f64,
        alpha: material.alpha as f64,
    }
}



// --- a camera --------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetCameraParams {
    /// Focal length in millimetres. What a lens is labelled with.
    #[param(min = 1.0, soft_max = 300.0)]
    pub focal_length_mm: f64,
    /// Sensor width in millimetres. 36 is a full 35mm frame.
    #[param(min = 1.0, soft_max = 100.0)]
    pub sensor_width_mm: f64,
    /// Whether the exposure is metered off the scene.
    ///
    /// On by default and the reason a lamp added from the menu is visible at
    /// all: the lights are photometric and the dials below are photographic, so
    /// no fixed pair of defaults suits both a bulb and the sun.
    pub auto_exposure: bool,
    /// The f-number. Zero is a pinhole: everything in focus, and no cost.
    #[param(min = 0.0, soft_max = 22.0)]
    pub f_stop: f64,
    /// How far away the plane of focus is, in metres.
    #[param(min = 0.0, soft_max = 100.0, unit = "length")]
    pub focus_distance: f64,
    /// Sensitivity. Doubling it is one stop.
    #[param(min = 1.0, soft_max = 12_800.0)]
    pub iso: f64,
    /// The \u00b1EV dial, in stops.
    ///
    /// Applies whether or not the meter is on, which is what it means on a
    /// camera: a deliberate departure from what the meter said.
    #[param(soft_min = -5.0, soft_max = 5.0)]
    pub exposure_compensation: f64,
}

impl Default for SetCameraParams {
    fn default() -> Self {
        let c = kreate_scene::Camera::new("");
        Self {
            focal_length_mm: c.focal_length_mm,
            sensor_width_mm: c.sensor_width_mm,
            auto_exposure: c.auto_exposure,
            f_stop: c.f_stop.unwrap_or(0.0),
            focus_distance: c.focus_distance,
            iso: c.iso,
            exposure_compensation: c.exposure_compensation,
        }
    }
}

/// Sets the active object's camera exactly.
pub struct SetCamera;

impl Operator for SetCamera {
    type Params = SetCameraParams;

    fn id() -> &'static str {
        "camera.set_properties"
    }
    fn label() -> &'static str {
        "Set Camera"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_camera(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = active_camera(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document_mut();
        let Some(camera) = d.camera_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        camera.focal_length_mm = params.focal_length_mm.max(1.0);
        camera.sensor_width_mm = params.sensor_width_mm.max(1.0);
        camera.auto_exposure = params.auto_exposure;
        // Zero is a pinhole and `None` is how that is spelled, because "off"
        // and "f/1000" want different code paths: one skips the depth of field
        // sampling entirely rather than sampling a lens that is nearly a point.
        camera.f_stop = (params.f_stop > 0.0).then_some(params.f_stop);
        camera.focus_distance = params.focus_distance.max(0.0);
        camera.iso = params.iso.max(1.0);
        camera.exposure_compensation = params.exposure_compensation;
        Ok(OpResult::Finished)
    }
}

/// The camera datablock the active object points at, if it points at one.
pub fn active_camera(doc: &Document) -> Option<kreate_scene::document::CameraId> {
    let object = doc.object(doc.selection.active?)?;
    match object.data {
        kreate_scene::document::ObjectData::Camera(id) => Some(id),
        _ => None,
    }
}

/// The parameters that would leave a camera exactly as it is.
pub fn camera_params(camera: &kreate_scene::Camera) -> SetCameraParams {
    SetCameraParams {
        focal_length_mm: camera.focal_length_mm,
        sensor_width_mm: camera.sensor_width_mm,
        auto_exposure: camera.auto_exposure,
        f_stop: camera.f_stop.unwrap_or(0.0),
        focus_distance: camera.focus_distance,
        iso: camera.iso,
        exposure_compensation: camera.exposure_compensation,
    }
}


/// Adds a light object and the datablock behind it.
///
/// Two things, as with a camera, and §2.2 is explicit about why: a hundred
/// linked copies of a lamp share one bulb. The object is *where* the light is
/// — a sun's direction and a spot's aim are its rotation — and the datablock is
/// what kind of light it is and how bright.
pub struct AddLight;

impl Operator for AddLight {
    type Params = AddLightParams;

    fn id() -> &'static str {
        "object.add_light"
    }
    fn label() -> &'static str {
        "Add Light"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // A stale keymap or an old script can name a kind this build does not
        // have. Clamping rather than refusing, for the same reason the property
        // map clamps an out-of-range enum: the user asked for a light and the
        // worst honest answer is the first kind, not nothing.
        let kind = kreate_scene::LightKind::ALL
            [(params.kind as usize).min(kreate_scene::LightKind::ALL.len() - 1)];

        let name = capitalise(kind.id());
        let mut light = kreate_scene::Light::of_kind(name.clone(), kind);
        if params.intensity > 0.0 {
            light.intensity = params.intensity;
        }
        light.exposure = params.exposure;
        // The one conversion: a picker's swatch read as a proportion of light.
        // `as_reflectance` says in its own name that it is not an amount of
        // light, which is exactly the distinction a light's two fields make.
        light.color = kreate_math::Srgb::new(
            params.color[0] as f32,
            params.color[1] as f32,
            params.color[2] as f32,
        )
        .as_reflectance();

        let d = doc.document_mut();
        let datablock = d.add_light(light);
        let mut object = kreate_scene::document::Object::new(
            name,
            kreate_scene::document::ObjectData::Light(datablock),
        );
        object.transform.translation = DVec3::from_array(params.location);
        let id = d.add_object(object);
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteParams {}

/// Deletes the selected objects.
pub struct Delete;

impl Operator for Delete {
    type Params = DeleteParams;

    fn id() -> &'static str {
        "object.delete"
    }
    fn label() -> &'static str {
        "Delete Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in targets {
            d.remove_object(id);
        }
        d.selection.clear();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DuplicateParams {
    /// Share the original's geometry instead of copying it.
    ///
    /// A linked duplicate costs one transform; editing either copy changes both.
    pub linked: bool,
}

/// Duplicates the selected objects.
pub struct Duplicate;

impl Operator for Duplicate {
    type Params = DuplicateParams;

    fn id() -> &'static str {
        "object.duplicate"
    }
    fn label() -> &'static str {
        "Duplicate Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        let mut copies = Vec::with_capacity(targets.len());
        for id in targets {
            let copy = if params.linked {
                d.duplicate_linked(id)
            } else {
                d.duplicate_full(id)
            };
            if let Some(c) = copy {
                copies.push(c);
            }
        }
        if copies.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        // The copies become the selection, so a duplicate-then-move macro moves
        // the new objects rather than the originals.
        d.selection.clear();
        for c in copies {
            d.selection.add(c);
        }
        Ok(OpResult::Finished)
    }
}

/// Which point a rotation or scale happens about.
pub const PIVOTS: &[&str] = &["median", "individual", "active", "world_origin"];

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateParams {
    /// Offset applied to every selected object, in metres.
    #[param(unit = "length")]
    pub translation: [f64; 3],
}

/// Moves the selected objects.
pub struct Translate;

impl Operator for Translate {
    type Params = TranslateParams;

    fn id() -> &'static str {
        "object.translate"
    }
    fn label() -> &'static str {
        "Move Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let offset = DVec3::from_array(params.translation);
        let d = doc.document_mut();
        for id in targets {
            if let Some(o) = d.object_mut(id) {
                o.transform.translation += offset;
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetTransformParams {
    /// Where the object goes, in its parent's space.
    #[param(unit = "length")]
    pub location: [f64; 3],
    /// How it is turned, as XYZ euler angles.
    ///
    /// Euler rather than the quaternion the object stores, because these are
    /// numbers a person types. The conversion is not one-to-one — several euler
    /// triples name the same rotation — so reading a rotation back can give a
    /// different triple from the one that was written. That is visible only when
    /// a rotation is *read*, it never changes where the object points, and the
    /// fix is a rotation mode on the object itself. The plan wants one for
    /// animation regardless.
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl Default for SetTransformParams {
    fn default() -> Self {
        Self {
            location: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0; 3],
        }
    }
}

/// Places the active object exactly.
///
/// # Why this exists beside translate, rotate and scale
///
/// Those three are *relative*: they take a delta, which is what a drag and a
/// keyboard gesture produce, and what makes them composable and drift-free
/// (§2.2). Typing a coordinate is the other thing, and expressing it as a delta
/// means the interface computing `wanted - current` and sending the difference —
/// which is a subtraction the user did not ask for, in a place where a rounding
/// error becomes a nudge.
///
/// The active object alone, not the selection. Typing a location means "put
/// *this* here"; applying it to everything selected would stack them all on one
/// point, which nobody has ever wanted.
pub struct SetTransform;

impl Operator for SetTransform {
    type Params = SetTransformParams;

    fn id() -> &'static str {
        "object.set_transform"
    }
    fn label() -> &'static str {
        "Set Transform"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // Unreachable through the dispatcher, which asks `poll` first. Here
        // for a caller that reaches `execute` directly, and stated in the test
        // as uncovered rather than left looking guarded.
        let Some(active) = doc.document().selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let wanted = kreate_math::DTransform {
            translation: DVec3::from_array(params.location),
            rotation: kreate_math::glam::DQuat::from_euler(
                kreate_math::glam::EulerRot::XYZ,
                params.rotation[0],
                params.rotation[1],
                params.rotation[2],
            )
            .normalize(),
            scale: DVec3::from_array(params.scale),
        };
        let d = doc.document_mut();
        let Some(object) = d.object_mut(active) else {
            return Ok(OpResult::Cancelled);
        };
        // Nothing to record if nothing moved. An operator that reports success
        // for a no-op puts an entry in the history that undoes to the same
        // scene — and this one runs on every keystroke in a coordinate field,
        // where the value is usually the one already there.
        if object.transform == wanted {
            return Ok(OpResult::Cancelled);
        }
        object.transform = wanted;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RotateParams {
    /// Angle in radians. The UI shows and parses degrees.
    #[param(unit = "angle")]
    pub angle: f64,
    /// Axis to rotate about.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    /// What the rotation happens about.
    #[param(variants("median", "individual", "active", "world_origin"))]
    pub pivot: u32,
}

/// Rotates the selected objects.
pub struct Rotate;

impl Operator for Rotate {
    type Params = RotateParams;

    fn id() -> &'static str {
        "object.rotate"
    }
    fn label() -> &'static str {
        "Rotate Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let axis = match params.axis {
            0 => DVec3::X,
            1 => DVec3::Y,
            _ => DVec3::Z,
        };
        let rotation = DQuat::from_axis_angle(axis, params.angle);
        let pivot = pivot_point(doc.document(), &targets, params.pivot);

        let d = doc.document_mut();
        for id in targets {
            // `individual` rotates each object about its own origin, so there is
            // no orbiting component.
            let about = if params.pivot == 1 {
                d.object(id).map(|o| o.transform.translation)
            } else {
                pivot
            };
            let Some(about) = about else { continue };
            if let Some(o) = d.object_mut(id) {
                let offset = o.transform.translation - about;
                o.transform.translation = about + rotation * offset;
                o.transform.rotation = (rotation * o.transform.rotation).normalize();
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ScaleParams {
    /// Scale factor per axis.
    pub scale: [f64; 3],
    /// What the scaling happens about.
    #[param(variants("median", "individual", "active", "world_origin"))]
    pub pivot: u32,
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            pivot: 0,
        }
    }
}

/// Scales the selected objects.
pub struct Scale;

impl Operator for Scale {
    type Params = ScaleParams;

    fn id() -> &'static str {
        "object.scale"
    }
    fn label() -> &'static str {
        "Scale Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let factor = DVec3::from_array(params.scale);
        let pivot = pivot_point(doc.document(), &targets, params.pivot);

        let d = doc.document_mut();
        for id in targets {
            let about = if params.pivot == 1 {
                d.object(id).map(|o| o.transform.translation)
            } else {
                pivot
            };
            let Some(about) = about else { continue };
            if let Some(o) = d.object_mut(id) {
                let offset = o.transform.translation - about;
                o.transform.translation = about + offset * factor;
                o.transform.scale *= factor;
            }
        }
        Ok(OpResult::Finished)
    }
}

/// Resolves a pivot mode to a point, or `None` for `individual`, which has a
/// different point per object.
fn pivot_point(doc: &Document, targets: &[ObjectId], mode: u32) -> Option<DVec3> {
    match mode {
        // median
        0 => {
            let mut sum = DVec3::ZERO;
            let mut count = 0.0;
            for id in targets {
                if let Some(o) = doc.object(*id) {
                    sum += o.transform.translation;
                    count += 1.0;
                }
            }
            (count > 0.0).then(|| sum / count)
        }
        1 => None,
        // active
        2 => doc
            .selection
            .active
            .and_then(|a| doc.object(a))
            .map(|o| o.transform.translation),
        _ => Some(DVec3::ZERO),
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllParams {
    /// What to do to the selection.
    #[param(variants("select", "deselect", "toggle", "invert"))]
    pub action: u32,
}

/// Selects, deselects, toggles or inverts the object selection.
pub struct SelectAll;

impl Operator for SelectAll {
    type Params = SelectAllParams;

    fn id() -> &'static str {
        "object.select_all"
    }
    fn label() -> &'static str {
        "Select All Objects"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let selectable: Vec<ObjectId> = doc
            .document()
            .objects()
            .filter(|(_, o)| o.selectable)
            .map(|(id, _)| id)
            .collect();
        let currently = doc.document().selection.objects.clone();

        // `toggle` means "select all unless everything already is", which is what
        // a single key doing double duty has to mean.
        let action = match params.action {
            2 => {
                if currently.len() == selectable.len() && !selectable.is_empty() {
                    1
                } else {
                    0
                }
            }
            other => other,
        };

        let d = doc.document_mut();
        match action {
            0 => {
                d.selection.clear();
                for id in selectable {
                    d.selection.add(id);
                }
            }
            1 => d.selection.clear(),
            _ => {
                d.selection.clear();
                for id in selectable.into_iter().filter(|id| !currently.contains(id)) {
                    d.selection.add(id);
                }
            }
        }
        // Selection changes are undoable: users expect Ctrl+Z to restore a
        // selection, and losing a careful one is more annoying than losing an edit.
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParentParams {
    /// Clear the parent instead of setting it to the active object.
    pub clear: bool,
}

/// Parents the selected objects to the active one.
pub struct Parent;

impl Operator for Parent {
    type Params = ParentParams;

    fn id() -> &'static str {
        "object.parent"
    }
    fn label() -> &'static str {
        "Parent Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let selection = doc.document().selection.clone();
        let parent = if params.clear { None } else { selection.active };
        if !params.clear && parent.is_none() {
            return Err(OpError::Failed("no active object to parent to".into()));
        }

        let d = doc.document_mut();
        let mut changed = false;
        let mut refused = 0;
        for id in selection.objects.iter().copied() {
            if Some(id) == parent {
                continue;
            }
            match d.set_parent(id, parent) {
                Ok(()) => changed = true,
                // A cycle is refused per object rather than failing the whole
                // operation: parenting a mixed selection where one member is an
                // ancestor should do the sensible part.
                Err(_) => refused += 1,
            }
        }
        if !changed {
            return if refused > 0 {
                Err(OpError::Failed(format!(
                    "{refused} object(s) would have created a cycle"
                )))
            } else {
                Ok(OpResult::Cancelled)
            };
        }
        Ok(OpResult::Finished)
    }
}

/// Registers every object-mode operator.
#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct JoinParams {}

/// Joins the selected objects' meshes into the active object's.
///
/// The active object survives and keeps its transform; the others are removed.
/// Each arrival is brought through the active object's inverse world transform,
/// so nothing moves on screen — which is the whole point of a join, and the
/// part that is easy to get wrong.
pub struct Join;

impl Operator for Join {
    type Params = JoinParams;

    fn id() -> &'static str {
        "object.join"
    }
    fn label() -> &'static str {
        "Join Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        // Two mesh objects at least, one of them active. A join of one is a
        // no-op and a join with nothing active has no target.
        let Some(active) = doc.selection.active else {
            return false;
        };
        if doc.object(active).and_then(|o| o.mesh()).is_none() {
            return false;
        }
        doc.selection
            .objects
            .iter()
            .filter(|id| doc.object(**id).and_then(|o| o.mesh()).is_some())
            .count()
            >= 2
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let selection = doc.document().selection.clone();
        let Some(active) = selection.active else {
            return Err(OpError::Failed("no active object to join into".into()));
        };
        let Some(target_mesh) = doc.document().object(active).and_then(|o| o.mesh()) else {
            return Err(OpError::Failed("the active object has no mesh".into()));
        };

        // Into the target's space. Read before anything is mutated, because a
        // world transform depends on the hierarchy and the hierarchy is about
        // to lose objects.
        let into_target = doc.document().world_transform(active).inverse();
        let sources: Vec<(ObjectId, MeshId, DAffine3)> = selection
            .objects
            .iter()
            .copied()
            .filter(|id| *id != active)
            .filter_map(|id| {
                let mesh = doc.document().object(id).and_then(|o| o.mesh())?;
                // A linked datablock is read-only, but *reading* it into a join
                // is fine: what the join writes is the target.
                Some((id, mesh, into_target * doc.document().world_transform(id)))
            })
            .collect();
        if sources.is_empty() {
            return Ok(OpResult::Cancelled);
        }

        let (joined, report) = {
            let d = doc.document();
            let Some(target) = d.mesh(target_mesh) else {
                return Err(OpError::Failed("the active mesh is missing".into()));
            };
            let mut parts = Vec::with_capacity(sources.len() + 1);
            parts.push(edit::join::Part {
                mesh: target,
                transform: DAffine3::IDENTITY,
            });
            for (_, mesh, transform) in &sources {
                if let Some(m) = d.mesh(*mesh) {
                    parts.push(edit::join::Part {
                        mesh: m,
                        transform: *transform,
                    });
                }
            }
            edit::join::join(&parts)
        };

        let d = doc.document_mut();
        if let Some(slot) = d.mesh_mut(target_mesh) {
            *slot = joined;
        }
        for (id, _, _) in &sources {
            d.remove_object(*id);
        }
        d.selection.objects.retain(|id| *id == active);
        d.selection.active = Some(active);

        if !report.conflicted.is_empty() {
            // Said out loud rather than dropped in silence: these are layers
            // the user authored, and §5.6's rule about never losing data
            // quietly applies most where the data is their own.
            //
            // The log is not good enough, honestly. This belongs in the report
            // panel the plan puts beside the importer, and an operator has no
            // channel to carry a warning to the interface yet — adding one is a
            // change to `OpResult` and to every caller, which is not a thing to
            // do in passing. Returning an error is not the answer either: the
            // join succeeded, and an error would roll it back.
            tracing::warn!(
                layers = %report.conflicted.join(", "),
                "joined objects disagreed on these attribute layers; \
                 they were left out"
            );
        }
        Ok(OpResult::Finished)
    }
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddPrimitive>();
    registry.register::<AddCamera>();
    registry.register::<AddLight>();
    registry.register::<SetLight>();
    registry.register::<SetWorld>();
    registry.register::<SetWorldImage>();
    registry.register::<NewMaterial>();
    registry.register::<SetMaterialImage>();
    registry.register::<SetMaterial>();
    registry.register::<SetCamera>();
    registry.register::<Delete>();
    registry.register::<Duplicate>();
    registry.register::<Translate>();
    registry.register::<SetTransform>();
    registry.register::<Rotate>();
    registry.register::<Scale>();
    registry.register::<SelectAll>();
    registry.register::<Parent>();
    registry.register::<Join>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::OperatorRegistry;
    use crate::{PropertyMap, PropertyValue};

    fn setup() -> (OperatorRegistry, History, Document, Context) {
        let mut r = OperatorRegistry::new();
        register(&mut r);
        (r, History::new(), Document::new(), Context::default())
    }

    /// Two cubes, both selected, the first active — a join's usual setup.
    fn two_cubes() -> (
        OperatorRegistry,
        History,
        Document,
        Context,
        ObjectId,
        ObjectId,
    ) {
        let (r, h, mut doc, ctx) = setup();
        let a = doc.add_mesh_object("a", kreate_mesh::primitives::cube(2.0));
        let b = doc.add_mesh_object("b", kreate_mesh::primitives::cube(2.0));
        doc.selection.add(a);
        doc.selection.add(b);
        doc.selection.active = Some(a);
        (r, h, doc, ctx, a, b)
    }

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> OpResult {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(r, d, c, id, &p).expect(id)
    }

    #[test]
    fn every_primitive_can_be_added_and_is_valid() {
        // The generators are validated inside the operator, so this also proves
        // no shape seeds invalid topology into a document.
        let (r, mut h, mut d, c) = setup();
        for (i, name) in PRIMITIVES.iter().enumerate() {
            let result = run(
                &r,
                &mut h,
                &mut d,
                &c,
                "object.add_primitive",
                &[("shape", PropertyValue::Enum(i as u32))],
            );
            assert_eq!(result, OpResult::Finished, "{name}");
        }
        assert_eq!(d.object_count(), PRIMITIVES.len());
    }

    /// One cube, selected and active: what the N panel edits.
    fn one_cube() -> (OperatorRegistry, History, Document, Context, ObjectId) {
        let (r, h, mut doc, ctx) = setup();
        let cube = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));
        doc.selection.add(cube);
        doc.selection.active = Some(cube);
        (r, h, doc, ctx, cube)
    }

    fn placed(
        location: [f64; 3],
        rotation: [f64; 3],
        scale: [f64; 3],
    ) -> Vec<(&'static str, PropertyValue)> {
        vec![
            ("location", PropertyValue::Vec3(location)),
            ("rotation", PropertyValue::Vec3(rotation)),
            ("scale", PropertyValue::Vec3(scale)),
        ]
    }

    #[test]
    fn setting_a_transform_places_the_object_exactly() {
        // The point of it: typing a coordinate puts the object there, rather
        // than moving it by the difference. Every component distinct and none of
        // them a default, because a version that wrote one value into all three,
        // or ignored an axis, would pass a symmetric fixture.
        let (r, mut h, mut doc, ctx, cube) = one_cube();
        let result = run(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            "object.set_transform",
            &placed(
                [1.5, -2.5, 7.0],
                [0.0, std::f64::consts::FRAC_PI_2, 0.0],
                [2.0, 3.0, 4.0],
            ),
        );
        assert_eq!(result, OpResult::Finished);

        let t = doc.object(cube).expect("still there").transform;
        assert_eq!(t.translation, DVec3::new(1.5, -2.5, 7.0));
        assert_eq!(t.scale, DVec3::new(2.0, 3.0, 4.0));

        // A quarter turn about Y takes +X to -Z, which pins the axis and the
        // direction. The wrong axis, or the wrong sign, sends it elsewhere.
        let turned = t.rotation * DVec3::X;
        assert!(
            (turned - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-9,
            "+X went to {turned:?}"
        );
    }

    #[test]
    fn setting_the_transform_it_already_has_records_nothing() {
        // This runs on every keystroke in a coordinate field, where the value is
        // usually the one already there. An operator that reported success would
        // fill the history with entries that undo to the same scene, and the
        // user's real last action would be buried under them.
        let (r, mut h, mut doc, ctx, cube) = one_cube();
        let fields = placed([1.0, 2.0, 3.0], [0.0; 3], [1.0; 3]);
        run(&r, &mut h, &mut doc, &ctx, "object.set_transform", &fields);
        let placed_at = doc.object(cube).expect("there").transform;

        let mut second = History::new();
        let again = run(
            &r,
            &mut second,
            &mut doc,
            &ctx,
            "object.set_transform",
            &fields,
        );
        assert_eq!(again, OpResult::Cancelled);
        assert!(!second.can_undo(), "a no-op reached the history");
        assert_eq!(
            doc.object(cube).expect("there").transform,
            placed_at,
            "the object moved anyway"
        );
    }

    #[test]
    fn setting_a_transform_touches_the_active_object_and_no_other() {
        // Typing a location means "put *this* here". Applying it to everything
        // selected would stack them all on one point.
        //
        // **The active object is deliberately not the first one selected.**
        // `two_cubes` makes it the first, and with that fixture a version
        // reading `selection.objects.first()` instead of `selection.active`
        // passes every line below — a sweep proved it. The two only differ when
        // they disagree, so the fixture has to make them disagree.
        let (r, mut h, mut doc, ctx, a, b) = two_cubes();
        doc.selection.active = Some(b);
        assert_eq!(
            doc.selection.objects.first().copied(),
            Some(a),
            "the fixture must not have the active object first"
        );
        let before = doc.object(a).expect("there").transform;

        run(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            "object.set_transform",
            &placed([5.0, 6.0, 7.0], [0.0; 3], [1.0; 3]),
        );
        assert_eq!(
            doc.object(b).expect("there").transform.translation,
            DVec3::new(5.0, 6.0, 7.0),
            "the active object was not the one placed"
        );
        assert_eq!(
            doc.object(a).expect("there").transform,
            before,
            "the other selected object moved too"
        );
    }

    #[test]
    fn setting_a_transform_needs_something_to_set() {
        // With nothing active there is nothing to place, and the *dispatcher*
        // is what refuses: it asks `poll` first and returns `PollFailed`
        // without running the operator at all.
        //
        // Which makes the operator's own re-check unreachable from here, and
        // the first version of this test asserted `Cancelled` and failed. The
        // check stays as defence for a caller that reaches `execute` directly —
        // a script, or a test — and it is stated here that nothing exercises it,
        // rather than left looking covered.
        let (r, mut h, mut doc, ctx) = setup();
        assert!(!r
            .get("object.set_transform")
            .expect("registered")
            .poll(&ctx, &doc));

        let mut p = PropertyMap::new();
        for (k, v) in placed([1.0, 0.0, 0.0], [0.0; 3], [1.0; 3]) {
            p.set(k, v);
        }
        let error = h
            .dispatch(&r, &mut doc, &ctx, "object.set_transform", &p)
            .expect_err("the dispatcher must refuse it");
        assert!(
            matches!(error, crate::operator::OpError::PollFailed(_)),
            "{error:?}"
        );
        assert!(!h.can_undo(), "a refused operator reached the history");
    }

    #[test]
    fn a_transform_that_was_set_can_be_undone() {
        // The coarse undo path, which is right here: an object transform is nine
        // numbers, and the snapshot shares every array in the document.
        let (r, mut h, mut doc, ctx, cube) = one_cube();
        let before = doc.object(cube).expect("there").transform;

        run(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            "object.set_transform",
            &placed([9.0, 8.0, 7.0], [0.5, 0.0, 0.0], [2.0; 3]),
        );
        assert_ne!(doc.object(cube).expect("there").transform, before);

        assert!(h.undo(&mut doc));
        assert_eq!(
            doc.object(cube).expect("there").transform,
            before,
            "undo did not put it back"
        );
    }

    #[test]
    fn adding_a_primitive_selects_it() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        assert_eq!(d.selection.objects.len(), 1);
        assert!(d.selection.active.is_some());
    }

    #[test]
    fn a_primitive_is_placed_at_the_requested_location() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([1.0, 2.0, 3.0]))],
        );
        let id = d.selection.active.unwrap();
        assert_eq!(
            d.object(id).unwrap().transform.translation,
            DVec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn an_out_of_range_shape_index_falls_back_rather_than_failing() {
        // A saved keymap or script may name a shape that no longer exists.
        let (r, mut h, mut d, c) = setup();
        let result = run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("shape", PropertyValue::Enum(999))],
        );
        assert_eq!(result, OpResult::Finished);
        assert_eq!(d.object_count(), 1);
    }

    #[test]
    fn adding_then_undoing_leaves_an_empty_document() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        h.undo(&mut d);
        assert_eq!(d.object_count(), 0);
        assert!(d.selection.is_empty(), "the selection must be restored too");
    }

    #[test]
    fn deleting_removes_the_selection_and_undo_brings_it_back() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(&r, &mut h, &mut d, &c, "object.delete", &[]);
        assert_eq!(d.object_count(), 0);

        h.undo(&mut d);
        assert_eq!(d.object_count(), 1);
        assert_eq!(
            d.selection.objects.len(),
            1,
            "the selection comes back as well"
        );
    }

    #[test]
    fn a_linked_duplicate_shares_geometry_and_a_full_one_does_not() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(true))],
        );
        assert_eq!(d.object_count(), 2);
        assert_eq!(d.mesh_count(), 1, "linked must not copy geometry");

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(false))],
        );
        assert_eq!(d.mesh_count(), 2);
    }

    #[test]
    fn duplicating_selects_the_copies() {
        // So that a duplicate-then-move macro moves the new objects.
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let original = d.selection.active.unwrap();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(true))],
        );
        assert_eq!(d.selection.objects.len(), 1);
        assert_ne!(d.selection.active, Some(original));
    }

    #[test]
    fn translate_moves_every_selected_object() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(true))],
        );
        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.translate",
            &[("translation", PropertyValue::Vec3([0.0, 0.0, 5.0]))],
        );
        for (_, o) in d.objects() {
            assert_eq!(o.transform.translation.z, 5.0);
        }
    }

    #[test]
    fn rotating_about_the_median_orbits_the_objects() {
        let (r, mut h, mut d, c) = setup();
        // Two objects at x = 0 and x = 2, median x = 1.
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([2.0, 0.0, 0.0]))],
        );
        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.rotate",
            &[
                ("angle", PropertyValue::Float(std::f64::consts::PI)),
                ("axis", PropertyValue::Enum(2)),
                ("pivot", PropertyValue::Enum(0)),
            ],
        );
        // A half turn about x = 1 swaps them.
        let xs: Vec<f64> = d
            .objects()
            .map(|(_, o)| (o.transform.translation.x * 1e6).round() / 1e6)
            .collect();
        assert!(xs.contains(&2.0) && xs.contains(&0.0), "got {xs:?}");
    }

    #[test]
    fn rotating_about_individual_origins_does_not_move_anything() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([5.0, 0.0, 0.0]))],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.rotate",
            &[
                ("angle", PropertyValue::Float(1.0)),
                ("pivot", PropertyValue::Enum(1)),
            ],
        );
        let id = d.selection.active.unwrap();
        let o = d.object(id).unwrap();
        assert_eq!(o.transform.translation, DVec3::new(5.0, 0.0, 0.0));
        assert_ne!(
            o.transform.rotation,
            DQuat::IDENTITY,
            "it should still have turned"
        );
    }

    #[test]
    fn scaling_about_the_world_origin_moves_objects_outward() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([2.0, 0.0, 0.0]))],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.scale",
            &[
                ("scale", PropertyValue::Vec3([2.0, 2.0, 2.0])),
                ("pivot", PropertyValue::Enum(3)),
            ],
        );
        let id = d.selection.active.unwrap();
        assert_eq!(d.object(id).unwrap().transform.translation.x, 4.0);
        assert_eq!(d.object(id).unwrap().transform.scale.x, 2.0);
    }

    #[test]
    fn select_all_toggles_between_all_and_none() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);

        // One is selected after the second add; toggle should select all.
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select_all",
            &[("action", PropertyValue::Enum(2))],
        );
        assert_eq!(d.selection.objects.len(), 2);

        // Everything is selected, so toggle now deselects.
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select_all",
            &[("action", PropertyValue::Enum(2))],
        );
        assert!(d.selection.is_empty());
    }

    #[test]
    fn select_all_can_invert() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let first = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let second = d.selection.active.unwrap();

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select_all",
            &[("action", PropertyValue::Enum(3))],
        );
        assert!(d.selection.contains(first));
        assert!(!d.selection.contains(second));
    }

    #[test]
    fn a_selection_change_is_undoable() {
        // Users expect Ctrl+Z to restore a selection; losing a careful one is
        // more annoying than losing an edit.
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let before = d.selection.clone();

        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        assert_ne!(d.selection, before);

        h.undo(&mut d);
        assert_eq!(d.selection, before);
    }

    #[test]
    fn parenting_refuses_a_cycle_per_object_and_does_the_rest() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let a = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let b = d.selection.active.unwrap();
        d.set_parent(b, Some(a)).unwrap();

        // Select both with `a` active: `b` is fine, `a` itself is skipped.
        d.selection.clear();
        d.selection.add(b);
        d.selection.add(a);
        let result = run(&r, &mut h, &mut d, &c, "object.parent", &[]);
        assert_eq!(result, OpResult::Finished);
        assert_eq!(d.object(b).unwrap().parent, Some(a));
    }

    #[test]
    fn parenting_can_clear() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let a = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let b = d.selection.active.unwrap();
        d.set_parent(b, Some(a)).unwrap();

        d.selection.clear();
        d.selection.add(b);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.parent",
            &[("clear", PropertyValue::Bool(true))],
        );
        assert_eq!(d.object(b).unwrap().parent, None);
    }

    #[test]
    fn operators_needing_a_selection_are_polled_out_when_there_is_none() {
        let (r, _h, d, c) = setup();
        for id in [
            "object.delete",
            "object.duplicate",
            "object.translate",
            "object.parent",
        ] {
            let info = r.get(id).expect(id);
            assert!(!info.poll(&c, &d), "{id} should be unavailable");
        }
        // Adding and select-all always work.
        assert!(r.get("object.add_primitive").unwrap().poll(&c, &d));
        assert!(r.get("object.select_all").unwrap().poll(&c, &d));
    }

    #[test]
    fn a_session_of_operators_reads_back_as_a_script() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("shape", PropertyValue::Enum(5))],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.translate",
            &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
        );
        let script = h.to_script();
        assert!(
            script.contains("kreate.ops.object.add_primitive("),
            "{script}"
        );
        assert!(
            script.contains("kreate.ops.object.translate(translation=(0.0, 0.0, 1.0))"),
            "{script}"
        );
    }

    #[test]
    fn adjusting_the_last_add_changes_the_shape_in_place() {
        // The F9 panel on a real operator: change the primitive after adding it.
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("shape", PropertyValue::Enum(1))],
        );
        assert_eq!(d.object_count(), 1);
        let cube_faces = d
            .mesh(
                d.object(d.selection.active.unwrap())
                    .unwrap()
                    .mesh()
                    .unwrap(),
            )
            .unwrap()
            .num_faces();

        let mut p = PropertyMap::new();
        p.set("shape", PropertyValue::Enum(5)); // cylinder
        p.set("segments", PropertyValue::Int(8));
        h.adjust_last(&r, &mut d, &c, &p).unwrap();

        assert_eq!(d.object_count(), 1, "adjusting must replace, not add");
        let faces = d
            .mesh(
                d.object(d.selection.active.unwrap())
                    .unwrap()
                    .mesh()
                    .unwrap(),
            )
            .unwrap()
            .num_faces();
        assert_ne!(faces, cube_faces);
        assert_eq!(faces, 10, "8 sides plus two caps");
    }

    #[test]
    fn joining_moves_the_geometry_into_the_active_object_and_removes_the_rest() {
        let (r, mut h, mut doc, ctx, a, b) = two_cubes();
        if let Some(o) = doc.object_mut(b) {
            o.transform.translation = DVec3::new(5.0, 0.0, 0.0);
        }
        h.dispatch(&r, &mut doc, &ctx, "object.join", &PropertyMap::new())
            .expect("join ran");

        assert_eq!(
            doc.objects().count(),
            1,
            "the source object was not removed"
        );
        assert_eq!(doc.selection.active, Some(a), "the active object changed");
        let mesh = doc.object(a).and_then(|o| o.mesh()).expect("still a mesh");
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 12, "twelve faces");
    }

    #[test]
    fn a_join_does_not_move_the_geometry_on_screen() {
        // The whole point, and the part that is easy to get wrong: each arrival
        // has to come through the *target's* inverse world transform, or the
        // joined geometry snaps somewhere other than where it was drawn.
        let (r, mut h, mut doc, ctx, a, b) = two_cubes();
        // Both objects moved, so neither transform is the identity and getting
        // the composition backwards is visible.
        if let Some(o) = doc.object_mut(a) {
            o.transform.translation = DVec3::new(0.0, 0.0, 10.0);
        }
        if let Some(o) = doc.object_mut(b) {
            o.transform.translation = DVec3::new(5.0, 0.0, 10.0);
        }

        let before: Vec<DVec3> = {
            let world = doc.world_transform(b);
            let mesh = doc.object(b).and_then(|o| o.mesh()).unwrap();
            doc.mesh(mesh)
                .unwrap()
                .positions()
                .iter()
                .map(|p| world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64)))
                .collect()
        };

        h.dispatch(&r, &mut doc, &ctx, "object.join", &PropertyMap::new())
            .expect("join ran");

        let world = doc.world_transform(a);
        let mesh = doc.object(a).and_then(|o| o.mesh()).unwrap();
        let after: Vec<DVec3> = doc
            .mesh(mesh)
            .unwrap()
            .positions()
            .iter()
            .skip(8)
            .map(|p| world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64)))
            .collect();

        assert_eq!(after.len(), before.len());
        for (was, now) in before.iter().zip(&after) {
            assert!(
                (*was - *now).length() < 1e-9,
                "a vertex moved from {was:?} to {now:?}"
            );
        }
    }

    #[test]
    fn a_join_needs_two_mesh_objects_and_an_active_one() {
        let (_, _, mut doc, ctx) = setup();
        let a = doc.add_mesh_object("a", kreate_mesh::primitives::cube(2.0));

        assert!(!Join::poll(&ctx, &doc), "nothing selected");

        doc.selection.add(a);
        doc.selection.active = Some(a);
        assert!(!Join::poll(&ctx, &doc), "one object is a no-op, not a join");

        let empty = doc.add_object(kreate_scene::document::Object::new(
            "empty",
            kreate_scene::document::ObjectData::Empty,
        ));
        doc.selection.add(empty);
        assert!(!Join::poll(&ctx, &doc), "an empty counted as a mesh");

        let b = doc.add_mesh_object("b", kreate_mesh::primitives::cube(2.0));
        doc.selection.add(b);
        assert!(Join::poll(&ctx, &doc));
    }

    #[test]
    fn undo_puts_the_joined_objects_back() {
        let (r, mut h, mut doc, ctx, a, _) = two_cubes();
        h.dispatch(&r, &mut doc, &ctx, "object.join", &PropertyMap::new())
            .expect("join ran");
        assert_eq!(doc.objects().count(), 1);

        assert!(h.undo(&mut doc), "nothing to undo");
        assert_eq!(
            doc.objects().count(),
            2,
            "the removed object did not return"
        );
        let mesh = doc.object(a).and_then(|o| o.mesh()).unwrap();
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6, "the target grew");
    }
}
