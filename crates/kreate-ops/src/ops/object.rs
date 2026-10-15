//! Object-mode operators.

use kreate_math::{DAffine3, DQuat, DVec3};
use kreate_mesh::edit;
use kreate_scene::document::{MeshId, ObjectId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::property::PropertyMap;
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
    /// Overall size, in metres: a plane's, grid's or cube's edge; a sphere's
    /// diameter; a cylinder's or cone's diameter and also its height; a
    /// torus's ring diameter through the tube's centre, with a tube 0.3 of it
    /// thick. Centred on `location`. `object.set_dimensions` sizes it exactly
    /// afterwards.
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
    /// How it is turned, as Euler XYZ: square to the work plane when the
    /// Cursor orientation is chosen.
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
    /// Lifted so its lowest point is at `location` rather than its middle:
    /// a cube added at the cursor on the floor stands on the floor instead of
    /// half through it. Off by default, as Blender and a script expect; the
    /// Add menu turns it on.
    pub on_floor: bool,
}

impl Default for AddPrimitiveParams {
    fn default() -> Self {
        Self {
            shape: 1, // cube
            size: 2.0,
            segments: 32,
            rings: 16,
            location: [0.0; 3],
            rotation: [0.0; 3],
            on_floor: false,
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

        let angles = DVec3::from_array(params.rotation);
        let turn = if angles.is_finite() {
            kreate_scene::animation::from_euler(angles)
        } else {
            kreate_math::DQuat::IDENTITY
        };
        // How far below its middle the shape reaches, turned as it will be.
        let below = if params.on_floor {
            mesh.positions()
                .iter()
                .map(|p| (turn * p.as_dvec3()).z)
                .fold(0.0f64, f64::min)
        } else {
            0.0
        };
        let d = doc.document_mut();
        let id = d.add_mesh_object(capitalise(name), mesh);
        let o = d.object_mut(id).expect("just added");
        o.transform.translation = DVec3::from_array(params.location) - DVec3::Z * below;
        o.transform.rotation = turn;
        // A newly added object becomes the selection, which is what every
        // subsequent operator acts on.
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddGearParams {
    /// What kind: a spur or helical gear; an internal (ring) gear; a rack; a
    /// straight bevel gear; a worm and its wheel.
    #[param(variants("spur", "internal", "rack", "bevel", "worm"))]
    pub kind: u32,
    /// The module, in millimetres, from ISO 54's first series: what sets
    /// the size of the teeth, and what two gears that mesh share.
    #[param(variants(
        "0.5", "0.6", "0.8", "1", "1.25", "1.5", "2", "2.5", "3", "4", "5", "6", "8", "10"
    ))]
    pub module: u32,
    /// How many teeth.
    #[param(min = 6, max = 400, unit = "count")]
    pub teeth: u32,
    /// The face width, along the axis.
    #[param(min = 0.0001, soft_max = 0.2, unit = "length")]
    pub width: f64,
    /// The bore's diameter; zero for none.
    #[param(min = 0.0, soft_max = 0.1, unit = "length")]
    pub bore: f64,
    /// The basic rack's pressure angle, in degrees: 20 is ISO 53's, and
    /// what nearly every gear made now has.
    #[param(variants("20", "14.5", "25"))]
    pub pressure_angle: u32,
    /// The helix angle; zero for a spur gear. The mate is wound the other
    /// way, as a pair on parallel shafts must be.
    #[param(min = -0.79, max = 0.79, unit = "angle")]
    pub helix: f64,
    /// The profile shift, in modules: positive for fewer than 17 teeth,
    /// whose roots a cutter would otherwise undercut.
    #[param(min = -1.0, max = 1.0)]
    pub shift: f64,
    /// Play between the teeth, taken off each tooth's thickness on the
    /// pitch circle: a tenth of a millimetre suits a printed pair.
    #[param(min = 0.0, soft_max = 0.001, unit = "length")]
    pub backlash: f64,
    /// Teeth of a second gear made to run with this one, set at the centre
    /// distance and turned to mesh: beside a spur gear, inside a ring gear,
    /// on a rack, at the shaft angle to a bevel gear; zero for none. A bevel
    /// gear with none is a mitre gear, made for a mate of its own size.
    #[param(min = 0, max = 400, unit = "count")]
    pub mate_teeth: u32,
    /// A ring gear's wall round its teeth; a rack's back below its roots.
    #[param(min = 0.0, soft_max = 0.05, unit = "length")]
    pub rim: f64,
    /// A worm's starts: how many threads wind round it.
    #[param(min = 1, max = 6, unit = "count")]
    pub starts: u32,
    /// A worm's diameter factor q: its pitch diameter over the module.
    #[param(min = 4.0, max = 25.0)]
    pub diameter_factor: f64,
    /// The angle between a bevel gear's shaft and its mate's: a right
    /// angle for most, less or more for a drive turned off square.
    #[param(min = 0.35, max = 2.8, unit = "angle")]
    pub shaft_angle: f64,
    /// Where the gear's axis stands, on its bottom face.
    pub location: [f64; 3],
}

impl Default for AddGearParams {
    fn default() -> Self {
        Self {
            module: 6,
            teeth: 20,
            width: 0.01,
            bore: 0.005,
            pressure_angle: 0,
            helix: 0.0,
            shift: 0.0,
            backlash: 0.0001,
            mate_teeth: 0,
            rim: 0.005,
            starts: 1,
            diameter_factor: 10.0,
            shaft_angle: std::f64::consts::FRAC_PI_2,
            kind: 0,
            location: [0.0; 3],
        }
    }
}

impl AddGearParams {
    /// The gear these describe, in the mesh crate's terms.
    pub fn gear(&self) -> kreate_mesh::gear::GearParams {
        use kreate_mesh::gear::ISO_MODULES;
        let module = ISO_MODULES[(self.module as usize).min(ISO_MODULES.len() - 1)];
        let pressure = match self.pressure_angle {
            1 => 14.5f64,
            2 => 25.0,
            _ => 20.0,
        };
        kreate_mesh::gear::GearParams {
            module: module * 0.001,
            teeth: self.teeth.clamp(6, 400),
            width: self.width.max(0.0001),
            bore: self.bore.max(0.0),
            pressure_angle: pressure.to_radians(),
            helix: self.helix.clamp(-0.79, 0.79),
            shift: self.shift.clamp(-1.0, 1.0),
            backlash: self.backlash.max(0.0),
        }
    }
}

/// An involute gear to the standard basic rack (ISO 53) in a standard
/// module (ISO 54) — with, if asked, a second one set beside it to mesh.
/// Named for what it is ("Gear m2 z20"), so the numbers to order or cut it
/// by are on it.
pub struct AddGear;

impl Operator for AddGear {
    type Params = AddGearParams;
    fn id() -> &'static str {
        "object.add_gear"
    }
    fn label() -> &'static str {
        "Add Gear"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::gear;
        let g = params.gear();
        let m_mm = g.module * 1000.0;
        // i18n-exempt: parts' designations, numbers and letters
        let label = |what: &str, teeth: u32| format!("{what} m{m_mm} z{teeth}");
        let mate = params.mate_teeth.min(400);
        let at = DVec3::from_array(params.location);
        // What to make: each part's name, mesh, and where it goes.
        // Asked for more than the shape can be: said, with the most it can,
        // rather than quietly made smaller — a value that changes nothing is
        // a value the user cannot tell was ignored.
        let mm = |v: f64| (v * 10000.0).round() / 10.0;
        let too_big = |what: &str, asked: f64, most: f64| -> Result<(), OpError> {
            if asked > most + 1e-12 {
                return Err(OpError::Failed(format!(
                    "the {what} can be at most {} mm here; {} mm was asked for",
                    mm(most),
                    mm(asked)
                )));
            }
            Ok(())
        };
        match params.kind {
            1 | 2 => {}
            3 => {
                let partner = if mate >= 6 { mate } else { g.teeth };
                let shaft = params.shaft_angle;
                // Each cone must open less than flat: past it the gear is a
                // crown or an internal bevel, which these are not.
                let d1 = gear::bevel_cone_angle_at(g.teeth, partner, shaft);
                let limit = 85f64.to_radians();
                if !(d1 > 0.05 && d1 < limit && shaft - d1 > 0.05 && shaft - d1 < limit) {
                    return Err(OpError::Failed(format!(
                        "at a shaft angle of {:.0}° these teeth make a cone {:.0}° and {:.0}° from its axis: each must be under 85°; bring the shaft angle nearer a right angle or change the teeth",
                        shaft.to_degrees(),
                        d1.to_degrees(),
                        (shaft - d1).to_degrees()
                    )));
                }
                let face = gear::bevel_face_limit_at(g.teeth, partner, shaft, g.module);
                if g.width > face + 1e-12 {
                    return Err(OpError::Failed(format!(
                        "the face width of this bevel gear can be at most {} mm, half its                          cone distance (a third, {} mm, is usual); {} mm was asked for",
                        mm(face),
                        mm(face * 2.0 / 3.0),
                        mm(g.width)
                    )));
                }
                too_big(
                    "bore",
                    g.bore,
                    gear::bevel_bore_limit_at(&g, partner, shaft),
                )?;
                if mate >= 6 {
                    let other = gear::GearParams { teeth: mate, ..g };
                    too_big(
                        "bore",
                        g.bore,
                        gear::bevel_bore_limit_at(&other, g.teeth, shaft),
                    )?;
                }
            }
            4 => {
                let w = gear::WormParams {
                    module: g.module,
                    starts: params.starts.clamp(1, 6),
                    diameter_factor: params.diameter_factor.clamp(4.0, 25.0),
                    length: 0.0,
                    pressure_angle: g.pressure_angle,
                    bore: g.bore,
                    backlash: g.backlash,
                };
                too_big("worm's bore", g.bore, gear::worm_bore_limit(&w))?;
                let wheel = gear::worm_wheel_params(&w, g.teeth, g.width, g.bore);
                too_big("wheel's bore", g.bore, gear::bore_limit(&wheel))?;
            }
            _ => {
                too_big("bore", g.bore, gear::bore_limit(&g))?;
                if mate >= 6 {
                    let other = gear::GearParams { teeth: mate, ..g };
                    too_big("mate's bore", g.bore, gear::bore_limit(&other))?;
                }
            }
        }
        let mut parts: Vec<(String, kreate_mesh::MeshData, DQuat, DVec3)> = Vec::new();
        match params.kind {
            // A ring gear, and a pinion running inside it.
            1 => {
                parts.push((
                    label("Ring gear", g.teeth),
                    gear::internal_gear(&g, params.rim, 0.0),
                    DQuat::IDENTITY,
                    at,
                ));
                if mate >= 6 {
                    if g.teeth < mate + 10 {
                        return Err(OpError::Failed(format!(
                            "a pinion of {mate} teeth inside a ring of {} would foul it: \
                             keep the ring at least ten teeth larger",
                            g.teeth
                        )));
                    }
                    let pinion = gear::GearParams { teeth: mate, ..g };
                    too_big("pinion's bore", g.bore, gear::bore_limit(&pinion))?;
                    parts.push((
                        label("Gear", mate),
                        gear::gear(&pinion, 0.0),
                        DQuat::IDENTITY,
                        at + DVec3::X * gear::internal_centre_distance(&g, &pinion),
                    ));
                }
            }
            // A rack along X, and a pinion standing on it.
            2 => {
                parts.push((
                    // i18n-exempt: a part's designation
                    format!("Rack m{m_mm} {}", g.teeth),
                    gear::rack(&g, params.rim),
                    DQuat::IDENTITY,
                    at,
                ));
                if mate >= 6 {
                    // Its teeth lean as the rack's do where they meet: the
                    // same helix.
                    let pinion = gear::GearParams { teeth: mate, ..g };
                    too_big("pinion's bore", g.bore, gear::bore_limit(&pinion))?;
                    let r = gear::gear_info(&pinion).pitch * 0.5;
                    parts.push((
                        label("Gear", mate),
                        gear::gear(&pinion, gear::rack_mate_turn(mate)),
                        DQuat::IDENTITY,
                        at + DVec3::Y * r,
                    ));
                }
            }
            // A straight bevel gear, and its mate at the shaft angle.
            3 => {
                let partner = if mate >= 6 { mate } else { g.teeth };
                let shaft = params.shaft_angle;
                parts.push((
                    label("Bevel gear", g.teeth),
                    gear::bevel_gear_at(&g, partner, shaft, 0.0),
                    DQuat::IDENTITY,
                    at,
                ));
                if mate >= 6 {
                    let (turn, offset, own) =
                        gear::bevel_mate_place_at(g.teeth, mate, shaft, g.module);
                    parts.push((
                        label("Bevel gear", mate),
                        gear::bevel_gear_at(
                            &gear::GearParams { teeth: mate, ..g },
                            g.teeth,
                            shaft,
                            own,
                        ),
                        turn,
                        at + offset,
                    ));
                }
            }
            // A worm along X, and the wheel of `teeth` it drives.
            4 => {
                let w = gear::WormParams {
                    module: g.module,
                    starts: params.starts.clamp(1, 6),
                    diameter_factor: params.diameter_factor.clamp(4.0, 25.0),
                    length: (11.0 + 0.06 * g.teeth as f64) * g.module,
                    pressure_angle: g.pressure_angle,
                    bore: g.bore,
                    backlash: g.backlash,
                };
                let length = w.length;
                // Its own Z along the world's X, its groove (its X) facing +Y.
                let along_x =
                    DQuat::from_mat3(&kreate_math::DMat3::from_cols(DVec3::Y, DVec3::Z, DVec3::X));
                parts.push((
                    // i18n-exempt: a part's designation
                    format!("Worm m{m_mm} n{}", w.starts),
                    gear::worm(&w),
                    along_x,
                    at,
                ));
                let wheel = gear::worm_wheel_params(&w, g.teeth, g.width, g.bore);
                let info = gear::gear_info(&wheel);
                let centre = (w.pitch_diameter() + info.pitch) * 0.5;
                // A tooth facing the worm at the wheel's middle height.
                let twist = wheel.width * wheel.helix.tan() / (info.pitch * 0.5);
                let turn = -std::f64::consts::FRAC_PI_2 - twist * 0.5;
                parts.push((
                    label("Worm wheel", g.teeth),
                    gear::gear(&wheel, turn),
                    DQuat::IDENTITY,
                    at + DVec3::new(length * 0.5, centre, -wheel.width * 0.5),
                ));
            }
            // A spur or helical gear, and its mate beside it.
            _ => {
                parts.push((
                    label("Gear", g.teeth),
                    gear::gear(&g, 0.0),
                    DQuat::IDENTITY,
                    at,
                ));
                if mate >= 6 {
                    let other = gear::GearParams {
                        teeth: mate,
                        helix: -g.helix,
                        ..g
                    };
                    parts.push((
                        label("Gear", mate),
                        gear::gear(&other, gear::mate_turn(mate)),
                        DQuat::IDENTITY,
                        at + DVec3::X * gear::centre_distance(&g, &other),
                    ));
                }
            }
        }
        for (name, mesh, _, _) in &parts {
            let violations = mesh.validate();
            if !violations.is_empty() {
                return Err(OpError::Failed(format!(
                    "{name} is invalid: {violations:?}"
                )));
            }
        }
        let d = doc.document_mut();
        d.selection.clear();
        let mut first = None;
        for (name, mesh, turn, place) in parts {
            let id = d.add_mesh_object(name, mesh);
            let o = d.object_mut(id).expect("just added");
            o.transform.translation = place;
            o.transform.rotation = turn;
            d.selection.add(id);
            first.get_or_insert(id);
        }
        d.selection.active = first;
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

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddEmptyParams {
    /// Where the empty goes, in metres.
    #[param(unit = "length")]
    pub location: [f64; 3],
}

impl Default for AddEmptyParams {
    fn default() -> Self {
        Self { location: [0.0; 3] }
    }
}

/// Adds an empty: an object with a place and nothing else.
///
/// Blender's Add ▸ Empty. What a parent for a rig is, and a target for a
/// constraint, and a marker for where something will go — a transform that
/// other things are relative to. It has no datablock, so unlike a camera or a
/// lamp there is only the one thing to add.
pub struct AddEmpty;

impl Operator for AddEmpty {
    type Params = AddEmptyParams;

    fn id() -> &'static str {
        "object.add_empty"
    }
    fn label() -> &'static str {
        "Add Empty"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document_mut();
        let mut object =
            kreate_scene::document::Object::new("Empty", kreate_scene::document::ObjectData::Empty);
        object.transform.translation = DVec3::from_array(params.location);
        let id = d.add_object(object);
        // What was just added is what the user is about to move.
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
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
    /// How much light, in the unit the chosen kind uses — candela for a point
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
    /// separately — see §7.1 on why the two are not collapsed into one number.
    #[param(soft_min = -10.0, soft_max = 10.0)]
    pub exposure: f64,
    /// How big the source is, in metres: what makes a shadow soft.
    ///
    /// A point's or a spot's. Ignored by the other two kinds, which have their
    /// own shapes — carried in the same parameter block rather than split into
    /// an operator per kind, because changing the kind is one of the edits and
    /// an operator that could not do it would leave the panel switching between
    /// operators mid-gesture.
    #[param(min = 0.0, soft_max = 2.0, unit = "length")]
    pub radius: f64,
    /// How wide a sun appears, in radians. The sun's own is 0.526°.
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
    /// An area light's outline.
    #[param(variants("rectangle", "disc", "ellipse"))]
    pub shape: u32,
    /// An area light's width in metres — a disc's diameter.
    #[param(min = 0.0, soft_max = 10.0, unit = "length")]
    pub width: f64,
    /// An area light's height in metres. A disc has only a width.
    #[param(min = 0.0, soft_max = 10.0, unit = "length")]
    pub height: f64,
    /// How far off its face an area light reaches, in radians: a half turn
    /// is a full hemisphere, less narrows it towards a beam.
    #[param(min = 0.0, soft_max = 3.1, unit = "angle")]
    pub spread: f64,
    /// Whether the light casts a shadow.
    pub cast_shadow: bool,
    /// Whether it lights the diffuse response. §2.2's lighting cheat.
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
            shape: 0,
            width: 1.0,
            height: 1.0,
            spread: std::f64::consts::PI,
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
/// — §2.2's argument, and the reason editing one of them changes all of them.
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

    fn current(_ctx: &Context, doc: &Document, _given: &PropertyMap) -> Option<Self::Params> {
        doc.light(active_light(doc)?).map(light_params)
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
            kreate_scene::LightKind::Area { .. } => {
                let (w, h) = (params.width.max(0.0), params.height.max(0.0));
                kreate_scene::LightKind::Area {
                    shape: match params.shape {
                        1 => kreate_scene::AreaShape::Disc { radius: w * 0.5 },
                        2 => kreate_scene::AreaShape::Ellipse {
                            width: w,
                            height: h,
                        },
                        _ => kreate_scene::AreaShape::Rect {
                            width: w,
                            height: h,
                        },
                    },
                    spread: params.spread.clamp(0.0, std::f64::consts::PI),
                }
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
        kreate_scene::LightKind::Sun { angular_diameter } => p.angular_diameter = angular_diameter,
        kreate_scene::LightKind::Spot {
            radius,
            cone_angle,
            blend,
        } => {
            p.radius = radius;
            p.cone_angle = cone_angle;
            p.blend = blend;
        }
        kreate_scene::LightKind::Area { shape, spread } => {
            p.spread = spread;
            match shape {
                kreate_scene::AreaShape::Rect { width, height } => {
                    (p.shape, p.width, p.height) = (0, width, height);
                }
                kreate_scene::AreaShape::Disc { radius } => {
                    (p.shape, p.width, p.height) = (1, radius * 2.0, radius * 2.0);
                }
                kreate_scene::AreaShape::Ellipse { width, height } => {
                    (p.shape, p.width, p.height) = (2, width, height);
                }
            }
        }
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
    let area = kind == 3;
    match param {
        // An area light's outline. A disc's height is its width, but the
        // field stays: hiding it as the shape changes would move the rows
        // under the pointer.
        "shape" | "width" | "height" | "spread" => area,
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
    /// is what a document with no world means — so it is reachable from the
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

    fn current(_ctx: &Context, doc: &Document, _given: &PropertyMap) -> Option<Self::Params> {
        doc.world(doc.active_world()?).map(world_params)
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
                match doc
                    .document()
                    .image_by_uid(kreate_scene::Uid::from_raw(raw))
                {
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
    /// Which of the mesh's material slots to edit.
    ///
    /// **A parameter rather than document state.** Blender keeps the active
    /// slot on the object, which makes it undoable and saved; here the panel
    /// knows which row is selected and says so. That keeps the document to
    /// what a *file* has to hold, and it makes the operator sayable from a
    /// script: "set the material of slot 2", rather than "select slot 2, then
    /// set the material".
    #[param(min = 0.0)]
    pub slot: u32,
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
    /// How far the normal map tilts the surface: zero is flat, one is the map
    /// as painted. Does nothing without a normal map.
    #[param(min = 0.0, soft_max = 2.0)]
    pub normal_strength: f64,
    /// The colour the surface glows with.
    #[param(color)]
    pub emission: [f64; 3],
    /// How bright the glow is, in nits (cd/m²). Zero is none.
    #[param(min = 0.0, soft_max = 10000.0)]
    pub emission_strength: f64,
    /// How much of the light passes through: zero is solid, one is glass.
    #[param(min = 0.0, max = 1.0)]
    pub transmission: f64,
    /// How sharply what passes through bends: 1.5 is glass, 1.33 water.
    #[param(min = 1.0, soft_max = 3.0)]
    pub ior: f64,
    /// Cut the surface away where its opacity — alpha, times the base colour
    /// image's — is below this: a leaf on a card. Zero is never.
    #[param(min = 0.0, max = 1.0)]
    pub alpha_clip: f64,
}

impl Default for SetMaterialParams {
    fn default() -> Self {
        Self {
            slot: 0,
            color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            alpha: 1.0,
            normal_strength: 1.0,
            emission: [1.0, 1.0, 1.0],
            emission_strength: 0.0,
            transmission: 0.0,
            ior: 1.5,
            alpha_clip: 0.0,
        }
    }
}

/// Sets the material in one of the active object's slots.
///
/// Any slot, now that the rendered pass draws one call per slot. It used to be
/// the first and only the first, because that was the only one that reached
/// the screen: a panel offering the others would have been editing materials
/// nothing showed.
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
        active_slot_count(doc) > 0
    }

    fn current(_ctx: &Context, doc: &Document, given: &PropertyMap) -> Option<Self::Params> {
        let slot = match given.get("slot") {
            Some(kreate_reflect::PropertyValue::Int(i)) => u32::try_from(*i).ok()?,
            _ => 0,
        };
        let material = doc.material(doc.material_id_at(doc.selection.active?, slot as usize)?)?;
        Some(material_params(material, slot))
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = material_in_slot(doc.document(), params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document_mut();
        let Some(material) = d.material_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        // Linear on both sides: glTF's `baseColorFactor` is linear too, and a
        // material's base colour is a reflectance rather than a swatch — so
        // unlike a light's tint there is no transfer function in the way.
        material.base_color = kreate_math::Srgb::new(
            params.color[0] as f32,
            params.color[1] as f32,
            params.color[2] as f32,
        )
        .as_reflectance();
        material.metallic = params.metallic.clamp(0.0, 1.0) as f32;
        material.roughness = params.roughness.clamp(0.0, 1.0) as f32;
        material.normal_strength = params.normal_strength.max(0.0) as f32;
        material.alpha = params.alpha.clamp(0.0, 1.0) as f32;
        material.emission = kreate_math::Srgb::new(
            params.emission[0] as f32,
            params.emission[1] as f32,
            params.emission[2] as f32,
        )
        .as_reflectance();
        material.emission_strength = params.emission_strength.max(0.0) as f32;
        material.transmission = params.transmission.clamp(0.0, 1.0) as f32;
        material.ior = params.ior.max(1.0) as f32;
        material.alpha_clip = (params.alpha_clip > 0.0).then(|| params.alpha_clip.min(1.0) as f32);
        let (glass, bend) = (material.transmission, material.ior);
        let (opacity, cut) = (material.alpha, material.alpha_clip.unwrap_or(0.0));
        // The glow has no socket, so a material with its own graph takes it
        // from here too: the panel's dial is the one place it is set.
        let emitted = material.emitted();
        if let Some(graph) = material.graph.as_mut() {
            if let Some(o) = graph.output() {
                if let kreate_scene::shader::NodeKind::Surface {
                    emission,
                    transmission,
                    ior,
                    alpha,
                    clip,
                    ..
                } = &mut graph.nodes[o].kind
                {
                    *emission = emitted;
                    *transmission = glass;
                    *ior = bend;
                    *alpha = opacity;
                    *clip = cut;
                }
            }
        }
        Ok(OpResult::Finished)
    }
}

/// The material the active object draws with in its first slot, if it has one.
pub fn active_material(doc: &Document) -> Option<kreate_scene::MaterialId> {
    material_in_slot(doc, 0)
}

/// The material the active object draws with in `slot`.
///
/// Through `material_id_at`, which is the one place the override rule lives:
/// slots belong to the mesh and overrides to the object, and a panel that
/// resolved them itself would be a second opinion about which chair is
/// upholstered.
pub fn material_in_slot(doc: &Document, slot: u32) -> Option<kreate_scene::MaterialId> {
    doc.material_id_at(doc.selection.active?, slot as usize)
}

/// How many material slots the active object's mesh has.
///
/// Zero for anything that is not a mesh, and for a mesh nobody has assigned a
/// material to, which is a state and not an error.
pub fn active_slot_count(doc: &Document) -> usize {
    match active_mesh(doc) {
        Some(mesh) => doc.mesh_slots(mesh).len(),
        None => 0,
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetMaterialImageParams {
    /// Which of the mesh's material slots the texture goes on.
    ///
    /// See [`SetMaterialParams::slot`] on why the slot travels as a parameter.
    #[param(min = 0.0)]
    pub slot: u32,
    /// Which of the material's textures: the order of
    /// [`MaterialMap::ALL`](kreate_scene::MaterialMap::ALL).
    ///
    /// Base colour first and so the default, which keeps every script written
    /// before there were other maps meaning what it meant.
    #[param(variants("base_color", "metallic_roughness", "normal", "emission"))]
    pub map: u32,
    /// The image datablock's identity, in decimal. Empty takes the texture off.
    ///
    /// Text and an identity for the two reasons [`SetWorldImageParams::uid`]
    /// gives: a position says where the image sits today, and a `Uid` does not
    /// survive the property system's signed integer.
    pub uid: String,
}

/// Points one of the active object's material's textures at an image, or takes
/// it off.
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
        active_slot_count(doc) > 0
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = material_in_slot(doc.document(), params.slot) else {
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
                match doc
                    .document()
                    .image_by_uid(kreate_scene::Uid::from_raw(raw))
                {
                    Some(image) => Some(image),
                    None => return Ok(OpResult::Cancelled),
                }
            }
        };
        let d = doc.document_mut();
        let Some(material) = d.material_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        let map = kreate_scene::MaterialMap::ALL
            .get(params.map as usize)
            .copied()
            .unwrap_or(kreate_scene::MaterialMap::BaseColor);
        *material.map_mut(map) = image;
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
pub struct NewMaterialParams {
    /// Which slot to fill. See [`SetMaterialParams::slot`].
    #[param(min = 0.0)]
    pub slot: u32,
}

pub struct NewMaterial;

impl Operator for NewMaterial {
    type Params = NewMaterialParams;

    fn id() -> &'static str {
        "material.new"
    }
    fn label() -> &'static str {
        "New Material"
    }
    /// Any mesh. Whether the *slot* is already filled is decided in `execute`,
    /// because `poll` cannot see the parameter.
    ///
    /// This used to refuse whenever the object had a material at all, on the
    /// grounds that a second one would be created, assigned, and left with
    /// nothing pointing at it — true while the panel had no slot list and the
    /// renderer drew one material per object. Both of those have changed.
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_mesh(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        // A slot that already holds a material is left alone rather than
        // overwritten. The button is only offered on an empty slot, so
        // reaching here means a script or a replayed step — and silently
        // replacing somebody's material is the worse of the two answers.
        if material_in_slot(doc.document(), params.slot).is_some() {
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
        d.set_mesh_slot(mesh, params.slot as usize, Some(material));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddMaterialSlotParams {}

/// Appends an empty material slot to the active object's mesh.
///
/// **Empty, not filled.** Blender's `+` adds a slot and leaves it without a
/// material, and the separate New button fills it — two steps, because they
/// are two decisions: how many parts this model has, and what each part is
/// made of. A `+` that also made a material would give somebody who wanted
/// three slots three materials they then had to delete.
pub struct AddMaterialSlot;

impl Operator for AddMaterialSlot {
    type Params = AddMaterialSlotParams;

    fn id() -> &'static str {
        "material.add_slot"
    }
    fn label() -> &'static str {
        "Add Material Slot"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_mesh(doc).is_some()
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let at = doc.document().mesh_slots(mesh).len();
        doc.document_mut().set_mesh_slot(mesh, at, None);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RemoveMaterialSlotParams {
    /// Which slot to take away. See [`SetMaterialParams::slot`].
    #[param(min = 0.0)]
    pub slot: u32,
}

/// Takes a material slot off the active object's mesh.
///
/// **The faces that pointed at it are not deleted**, and neither is the
/// material datablock. A face's `material_index` past the end of the slot list
/// resolves to the first slot, the same place an unassigned face lands — so
/// removing a slot changes what those faces are made of and never how many
/// there are. Deleting geometry from a material panel would be a surprise
/// nobody could undo by re-adding the slot.
///
/// The indices of later slots do shift down, which is the part a face's stored
/// index notices. Blender renumbers the faces to match; that needs an edit to
/// the mesh attribute and is left for when the panel can show which faces are
/// on which slot, because doing it silently is how a model comes back wrong.
pub struct RemoveMaterialSlot;

impl Operator for RemoveMaterialSlot {
    type Params = RemoveMaterialSlotParams;

    fn id() -> &'static str {
        "material.remove_slot"
    }
    fn label() -> &'static str {
        "Remove Material Slot"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_slot_count(doc) > 0
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let mut slots = doc.document().mesh_slots(mesh).to_vec();
        if params.slot as usize >= slots.len() {
            return Ok(OpResult::Cancelled);
        }
        slots.remove(params.slot as usize);
        doc.document_mut().set_mesh_slots(mesh, slots);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MaterialSlotParams {
    /// Which slot. See [`SetMaterialParams::slot`].
    #[param(min = 0.0)]
    pub slot: u32,
}

/// Blender's Assign: in edit mode, the selected faces take this slot.
pub struct AssignMaterial;

impl Operator for AssignMaterial {
    type Params = MaterialSlotParams;

    fn id() -> &'static str {
        "material.assign"
    }
    fn label() -> &'static str {
        "Assign Material"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        ctx.mode.edits_elements() && active_slot_count(doc) > 0
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::attribute::{names, AttrType, Domain};
        let slot = params.slot as i32;
        let changed: usize = crate::ops::mesh::each_bmesh(doc, |bm| {
            let faces = bm.selected_faces();
            if faces.is_empty() {
                return Ok(0);
            }
            let len = bm.slot_count(Domain::Face);
            bm.attributes_mut()
                .ensure(names::MATERIAL_INDEX, Domain::Face, AttrType::I32, len);
            let Some(layer) = bm.attributes_mut().make_mut_i32(names::MATERIAL_INDEX) else {
                return Ok(0);
            };
            let mut n = 0;
            for f in faces {
                if let Some(at) = layer.get_mut(kreate_mesh::arena::ArenaIndex::index(f) as usize) {
                    if *at != slot {
                        *at = slot;
                        n += 1;
                    }
                }
            }
            Ok(n)
        })?
        .into_iter()
        .sum();
        Ok(if changed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

/// Blender's Select: in edit mode, the faces on this slot join the selection.
pub struct SelectByMaterial;

impl Operator for SelectByMaterial {
    type Params = MaterialSlotParams;

    fn id() -> &'static str {
        "material.select"
    }
    fn label() -> &'static str {
        "Select by Material"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        ctx.mode.edits_elements() && active_slot_count(doc) > 0
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::attribute::names;
        let slot = params.slot as i32;
        let found: usize = crate::ops::mesh::each_bmesh(doc, |bm| {
            // A face with no index is on the first slot, as the renderer
            // draws it.
            let on: Vec<_> = bm
                .face_ids()
                .filter(|f| {
                    let index = bm
                        .attributes()
                        .get_i32(names::MATERIAL_INDEX)
                        .and_then(|l| {
                            l.get(kreate_mesh::arena::ArenaIndex::index(*f) as usize)
                                .copied()
                        })
                        .unwrap_or(0);
                    index == slot
                })
                .collect();
            for f in &on {
                bm.set_face_selected(*f, true);
            }
            if !on.is_empty() {
                bm.flush_selection(kreate_mesh::bmesh::SelectMode::Face);
            }
            Ok(on.len())
        })?
        .into_iter()
        .sum();
        Ok(if found == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LinkMaterialParams {
    /// Which slot. See [`SetMaterialParams::slot`].
    #[param(min = 0.0)]
    pub slot: u32,
    /// The material's identity, in decimal: an identity rather than a name
    /// or a position, so a script naming it survives both changing.
    pub uid: String,
}

/// Blender's browse button: a slot takes a material that already exists, so
/// two parts — or two models — can share one.
pub struct LinkMaterial;

impl Operator for LinkMaterial {
    type Params = LinkMaterialParams;

    fn id() -> &'static str {
        "material.link"
    }
    fn label() -> &'static str {
        "Link Material"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_mesh(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let uid = params.uid.trim();
        let material = uid
            .parse::<u64>()
            .ok()
            .and_then(|raw| {
                doc.document()
                    .material_by_uid(kreate_scene::Uid::from_raw(raw))
            })
            .ok_or_else(|| {
                OpError::Failed(format!("no material has uid {uid:?}; get_scene lists them"))
            })?;
        // The next slot as well as an existing one: a freshly added object
        // has none, and making the slot first is a step a script should not
        // have to know about.
        let slot = params.slot as usize;
        let slots = doc.document().mesh_slots(mesh);
        if slot > slots.len() {
            return Err(OpError::Failed(format!(
                "the object has {} slot(s); slot {slot} would leave a gap",
                slots.len()
            )));
        }
        if slots.get(slot) == Some(&Some(material)) {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().set_mesh_slot(mesh, slot, Some(material));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetVisibleParams {
    /// The object's identity, in decimal. See [`SetWorldImageParams::uid`] on
    /// why an identity and not a position.
    pub uid: String,
    pub visible: bool,
}

/// Shows or hides one object, by identity.
///
/// **Not the selection.** The outliner's eye acts on its own row whether or
/// not that row is selected — a user hides the thing in the way, which is by
/// definition not the thing they are working on. So the object is named, and
/// named by the uid a file and a script agree on rather than by an arena slot
/// that is reused the moment something is deleted.
///
/// Setting what is already so is a cancel, not an edit: pressing undo after
/// clicking the eye of a visible object should bring back the last *change*,
/// not replay a no-op.
pub struct SetVisible;

impl Operator for SetVisible {
    type Params = SetVisibleParams;

    fn id() -> &'static str {
        "object.set_visible"
    }
    fn label() -> &'static str {
        "Set Visibility"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.object_count() > 0
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Ok(raw) = params.uid.trim().parse::<u64>() else {
            return Ok(OpResult::Cancelled);
        };
        let Some(id) = doc
            .document()
            .object_by_uid(kreate_scene::Uid::from_raw(raw))
        else {
            return Ok(OpResult::Cancelled);
        };
        if doc.document().object(id).map(|o| o.visible) == Some(params.visible) {
            return Ok(OpResult::Cancelled);
        }
        let Some(object) = doc.document_mut().object_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        object.visible = params.visible;
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

/// The parameters that would leave one of a material's textures exactly as it
/// is.
pub fn material_image_params(
    doc: &Document,
    material: &kreate_scene::Material,
    slot: u32,
    map: kreate_scene::MaterialMap,
) -> SetMaterialImageParams {
    SetMaterialImageParams {
        slot,
        map: kreate_scene::MaterialMap::ALL
            .iter()
            .position(|m| *m == map)
            .unwrap_or(0) as u32,
        uid: material
            .map(map)
            .and_then(|id| doc.image_uid(id))
            .map(|uid| uid.raw().to_string())
            .unwrap_or_default(),
    }
}

/// The parameters that would leave the material in `slot` exactly as it is.
pub fn material_params(material: &kreate_scene::Material, slot: u32) -> SetMaterialParams {
    let c = material.base_color.as_swatch();
    SetMaterialParams {
        slot,
        color: [c.r as f64, c.g as f64, c.b as f64],
        metallic: material.metallic as f64,
        roughness: material.roughness as f64,
        alpha: material.alpha as f64,
        normal_strength: material.normal_strength as f64,
        emission: {
            let e = material.emission.as_swatch();
            [e.r as f64, e.g as f64, e.b as f64]
        },
        emission_strength: material.emission_strength as f64,
        transmission: material.transmission as f64,
        ior: material.ior as f64,
        alpha_clip: material.alpha_clip.unwrap_or(0.0) as f64,
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
    /// The ±EV dial, in stops.
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

    fn current(_ctx: &Context, doc: &Document, _given: &PropertyMap) -> Option<Self::Params> {
        doc.camera(active_camera(doc)?).map(camera_params)
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

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PasteParams {
    /// The copied objects, as the text `kreate_io::clipboard::encode` makes.
    ///
    /// **The bytes ride in the parameters** rather than the operator reading
    /// a clipboard, because an operator has to be replayable: a paste in the
    /// history, redone a week later with the copied objects long deleted, must
    /// produce what it produced the first time. Nothing but its own bytes can
    /// promise that.
    ///
    /// Hidden from the adjust panel: it is megabytes for an imported model,
    /// and nothing a person would type.
    #[param(hidden)]
    pub data: String,
}

/// Adds the copied objects to the document — Ctrl+V.
///
/// Copy is not an operator. It changes nothing in the document; it turns the
/// selection into a fragment and the application keeps the text. This is the
/// half that edits, and it is the half that can be undone.
///
/// The paste becomes the selection, and only it: the next thing a person does
/// with something they pasted is move it.
pub struct Paste;

impl Operator for Paste {
    type Params = PasteParams;

    fn id() -> &'static str {
        "object.paste"
    }
    fn label() -> &'static str {
        "Paste Objects"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // Text that is not a fragment — whatever another application left on
        // the clipboard — is declined, not an error: "that was not ours" is the
        // normal case for a paste.
        let Some(fragment) = kreate_io::clipboard::decode(&params.data) else {
            return Ok(OpResult::Cancelled);
        };
        if fragment.object_count() == 0 {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        let pasted = d.merge_from(&fragment);
        if pasted.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        d.selection.clear();
        for id in &pasted {
            d.selection.add(*id);
        }
        d.selection.active = pasted.last().copied();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ImportParams {
    /// The scene file: glTF, FBX, OBJ, STL, PLY, 3MF, OFF or a `.kreate`.
    pub path: String,
    /// How far the whole model is moved from where its file puts it. The
    /// application sets it so a model lands beside what is already there
    /// rather than inside it — every asset is made at the origin, and a
    /// bottle imported after a barrel otherwise disappears into it.
    pub offset: [f64; 3],
    /// How tall the model is made, in metres, scaled evenly about the
    /// origin before it is moved; 0 keeps the file's size. Assets come in
    /// every scale a maker chose.
    #[param(unit = "length", min = 0.0)]
    pub height: f64,
}

/// Adds a scene file's contents beside what is there: File ▸ Import, a
/// dropped file, an asset from the browser.
///
/// **The file's path rides in the parameters, not its contents.** A paste
/// carries its bytes because the clipboard will not be there to replay it
/// from; a file usually will, and carrying an imported model in the history
/// — fifty megabytes for a clock — made every line of it that heavy. The
/// application reads the file ahead on a thread of its own
/// (`kreate_io::scene_file::prepare`), so adding it does not stop the
/// interface; a replay reads it where it is.
pub struct Import;

impl Operator for Import {
    type Params = ImportParams;

    fn id() -> &'static str {
        "object.import"
    }
    fn label() -> &'static str {
        "Import"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let path = std::path::Path::new(params.path.trim());
        let fragment = kreate_io::scene_file::take(path)
            .map_err(|why| OpError::Failed(format!("cannot import {}: {why}", path.display())))?;
        if fragment.object_count() == 0 {
            return Ok(OpResult::Cancelled);
        }
        // Scaled to the height asked for, measured as the file stands.
        let factor = {
            let bounds = fragment.world_bounds();
            let tall = (bounds.max.z - bounds.min.z) as f64;
            if params.height > 0.0 && !bounds.is_empty() && tall > 1e-9 {
                params.height / tall
            } else {
                1.0
            }
        };
        let d = doc.document_mut();
        let added = d.merge_from(&fragment);
        if added.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        // The roots only: a child rides on its parent.
        let offset = DVec3::from_array(params.offset);
        if offset != DVec3::ZERO || factor != 1.0 {
            for id in &added {
                if let Some(object) = d.object_mut(*id).filter(|o| o.parent.is_none()) {
                    object.transform.translation = object.transform.translation * factor + offset;
                    object.transform.scale *= factor;
                }
            }
        }
        d.selection.clear();
        for id in &added {
            d.selection.add(*id);
        }
        d.selection.active = added.last().copied();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LinkLibraryParams {
    /// The `.kreate` to link from, as the document should name it.
    pub path: String,
}

/// Links every mesh object of a `.kreate` library: read-only here, and read
/// again from the library each time the document opens — so fixing a prop
/// in its library fixes it in every scene that links it.
pub struct LinkLibrary;

impl Operator for LinkLibrary {
    type Params = LinkLibraryParams;

    fn id() -> &'static str {
        "object.link_library"
    }
    fn label() -> &'static str {
        "Link Library"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document_mut();
        let linked = kreate_io::library::link_all(d, std::path::Path::new("."), params.path.trim())
            .map_err(|e| OpError::Failed(e.to_string()))?;
        if linked.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        d.selection.clear();
        for id in &linked {
            d.selection.add(*id);
        }
        d.selection.active = linked.last().copied();
        Ok(OpResult::Finished)
    }
}

/// Which point a rotation or scale happens about.
pub const PIVOTS: &[&str] = &["median", "individual", "active", "world_origin", "cursor"];

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateParams {
    /// Offset applied to every selected object, in metres.
    #[param(unit = "length")]
    pub translation: [f64; 3],
    /// Key what this changed, at `key_frame`: auto-keying, filled in by the
    /// application when its record button is on. Hidden, because it is the
    /// application's state and not a choice about this operation — but a
    /// parameter, so the move and its key are one step to undo and a replay
    /// keys as the original did.
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
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
        // The move is in the world: each object takes it in its parent's
        // space, so a child of a turned or scaled parent goes as far and the
        // same way as the rest; one under a selected parent goes with it.
        for id in movable(d, &targets) {
            let local = d.above(id).inverse().transform_vector3(offset);
            if let Some(o) = d.object_mut(id) {
                o.transform.translation += local;
            }
        }
        if params.autokey {
            crate::ops::anim::key_objects(d, &targets, params.key_frame, &["location"]);
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
    /// Key what this changed, at `key_frame`: auto-keying, filled in by the
    /// application when its record button is on. Hidden, because it is the
    /// application's state and not a choice about this operation — but a
    /// parameter, so the move and its key are one step to undo and a replay
    /// keys as the original did.
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
}

impl Default for SetTransformParams {
    fn default() -> Self {
        Self {
            location: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0; 3],
            autokey: false,
            key_frame: 1.0,
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

    fn current(_ctx: &Context, doc: &Document, _given: &PropertyMap) -> Option<Self::Params> {
        let t = &doc.object(doc.selection.active?)?.transform;
        let r = kreate_scene::animation::to_euler(t.rotation);
        Some(SetTransformParams {
            location: t.translation.to_array(),
            rotation: r.to_array(),
            scale: t.scale.to_array(),
            ..SetTransformParams::default()
        })
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
            // Blender's XYZ Euler — X first, then Y, then Z — the convention
            // the rotation curves are keyed in, so the numbers in the sidebar
            // and the values on a curve are the same numbers.
            rotation: kreate_scene::animation::from_euler(DVec3::from_array(params.rotation))
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
        if params.autokey {
            crate::ops::anim::key_objects(
                d,
                &[active],
                params.key_frame,
                crate::ops::anim::KEYING_SETS[0],
            );
        }
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
    #[param(variants("median", "individual", "active", "world_origin", "cursor"))]
    pub pivot: u32,
    /// Key what this changed, at `key_frame`: auto-keying, filled in by the
    /// application when its record button is on. Hidden, because it is the
    /// application's state and not a choice about this operation — but a
    /// parameter, so the move and its key are one step to undo and a replay
    /// keys as the original did.
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
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
        for id in movable(d, &targets) {
            let at = d.world_transform(id).translation;
            // `individual` rotates each object about its own origin, so there is
            // no orbiting component.
            let about = if params.pivot == 1 { Some(at) } else { pivot };
            let Some(about) = about else { continue };
            let above = d.above(id);
            let (_, turn, _) = above.to_scale_rotation_translation();
            let to = above
                .inverse()
                .transform_point3(about + rotation * (at - about));
            if let Some(o) = d.object_mut(id) {
                o.transform.translation = to;
                // The world turn, seen from the parent's frame.
                o.transform.rotation =
                    (turn.inverse() * rotation * turn * o.transform.rotation).normalize();
            }
        }
        if params.autokey {
            // Location too: turning about any pivot but an object's own origin
            // moves it, and a key of the rotation alone would snap it back.
            crate::ops::anim::key_objects(
                d,
                &targets,
                params.key_frame,
                &["location", "rotation_euler"],
            );
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ScaleParams {
    /// Scale factor per axis.
    pub scale: [f64; 3],
    /// What the scaling happens about.
    #[param(variants("median", "individual", "active", "world_origin", "cursor"))]
    pub pivot: u32,
    /// Key what this changed, at `key_frame`: auto-keying, filled in by the
    /// application when its record button is on. Hidden, because it is the
    /// application's state and not a choice about this operation — but a
    /// parameter, so the move and its key are one step to undo and a replay
    /// keys as the original did.
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            pivot: 0,
            autokey: false,
            key_frame: 1.0,
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
        for id in movable(d, &targets) {
            let at = d.world_transform(id).translation;
            let about = if params.pivot == 1 { Some(at) } else { pivot };
            let Some(about) = about else { continue };
            let to = d
                .above(id)
                .inverse()
                .transform_point3(about + (at - about) * factor);
            if let Some(o) = d.object_mut(id) {
                o.transform.translation = to;
                o.transform.scale *= factor;
            }
        }
        if params.autokey {
            // Location too, for the reason rotate gives.
            crate::ops::anim::key_objects(d, &targets, params.key_frame, &["location", "scale"]);
        }
        Ok(OpResult::Finished)
    }
}

/// The selected objects a transform moves itself: those with no selected
/// object above them, which carry them along — moved twice otherwise.
fn movable(doc: &Document, targets: &[ObjectId]) -> Vec<ObjectId> {
    targets
        .iter()
        .copied()
        .filter(|id| !targets.iter().any(|a| a != id && doc.is_ancestor(*a, *id)))
        .collect()
}

/// Resolves a pivot mode to a point in the world, or `None` for
/// `individual`, which has a different point per object.
fn pivot_point(doc: &Document, targets: &[ObjectId], mode: u32) -> Option<DVec3> {
    match mode {
        // median
        0 => {
            let mut sum = DVec3::ZERO;
            let mut count = 0.0;
            for id in targets {
                if doc.object(*id).is_some() {
                    sum += doc.world_transform(*id).translation;
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
            .filter(|a| doc.object(*a).is_some())
            .map(|a| doc.world_transform(a).translation),
        // The 3D cursor, which is in the world; exact for objects without a
        // parent, as the median is.
        4 => Some(doc.cursor),
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
        let previous_active = doc.document().selection.active;

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
        // Each `add` made its object active, which left whichever came last in
        // the document active. Blender keeps the active object through Select
        // All, and things are built on that: select the cutters, the target
        // active, A, Ctrl+minus. So the one that was active stays so, as long
        // as it is still selected.
        if let Some(previous) = previous_active.filter(|a| d.selection.contains(*a)) {
            d.selection.active = Some(previous);
        }
        // Selection changes are undoable: users expect Ctrl+Z to restore a
        // selection, and losing a careful one is more annoying than losing an edit.
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectParams {
    /// The object's name, exactly as the outliner shows it.
    pub name: String,
    /// Add to the selection rather than replace it.
    pub extend: bool,
}

/// Selects an object by its name, and makes it the active one.
///
/// What a script has instead of a click. Every object operator acts on the
/// selection, so without this a script that lost its selection — to a click
/// in the viewport, or an undo to before the object it wanted was made — had
/// no way back to it short of making the object again.
pub struct Select;

impl Operator for Select {
    type Params = SelectParams;

    fn id() -> &'static str {
        "object.select"
    }
    fn label() -> &'static str {
        "Select Object"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc
            .document()
            .objects()
            .find(|(_, o)| o.name == params.name)
            .map(|(id, _)| id)
        else {
            return Err(OpError::Failed(format!("no object called {}", params.name)));
        };
        let d = doc.document_mut();
        if !params.extend {
            d.selection.clear();
        }
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParentOneParams {
    /// The child's identity, in decimal.
    pub child: String,
    /// The parent's, or empty to clear.
    pub parent: String,
}

/// One object parented to another, by identity — the outliner's drag and
/// drop, which names both and leaves the selection alone. The child stays
/// where it is.
pub struct ParentOne;

impl Operator for ParentOne {
    type Params = ParentOneParams;

    fn id() -> &'static str {
        "object.parent_one"
    }
    fn label() -> &'static str {
        "Parent"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let find = |text: &str| {
            text.trim().parse::<u64>().ok().and_then(|raw| {
                doc.document()
                    .object_by_uid(kreate_scene::Uid::from_raw(raw))
            })
        };
        let Some(child) = find(&params.child) else {
            return Ok(OpResult::Cancelled);
        };
        let parent = if params.parent.trim().is_empty() {
            None
        } else {
            match find(&params.parent) {
                Some(p) => Some(p),
                None => return Ok(OpResult::Cancelled),
            }
        };
        if parent == Some(child) || doc.document().object(child).map(|o| o.parent) == Some(parent) {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut()
            .set_parent(child, parent)
            .map_err(|_| OpError::Failed("that would make an object its own ancestor".into()))?;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct HideObjectsParams {
    /// Hide what is *not* selected instead — Blender's Shift+H.
    pub unselected: bool,
}

/// Blender's H in object mode: the selected objects disappear, and leave the
/// selection — a hidden object that stays selected is moved by the next G.
pub struct HideObjects;

impl Operator for HideObjects {
    type Params = HideObjectsParams;

    fn id() -> &'static str {
        "object.hide"
    }
    fn label() -> &'static str {
        "Hide"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let targets: Vec<ObjectId> = d
            .objects()
            .filter(|(id, o)| o.visible && d.selection.objects.contains(id) != params.unselected)
            .map(|(id, _)| id)
            .collect();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in &targets {
            if let Some(o) = d.object_mut(*id) {
                o.visible = false;
            }
            d.selection.remove(*id);
        }
        Ok(OpResult::Finished)
    }
}

/// Blender's Alt+H in object mode: every hidden object back, and selected, so
/// what came back is what the next action acts on.
pub struct RevealObjects;

impl Operator for RevealObjects {
    type Params = crate::ops::mesh::EmptyParams;

    fn id() -> &'static str {
        "object.reveal"
    }
    fn label() -> &'static str {
        "Reveal Hidden"
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let hidden: Vec<ObjectId> = doc
            .document()
            .objects()
            .filter(|(_, o)| !o.visible)
            .map(|(id, _)| id)
            .collect();
        if hidden.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in &hidden {
            if let Some(o) = d.object_mut(*id) {
                o.visible = true;
            }
            d.selection.add(*id);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RenameParams {
    pub name: String,
}

/// Renames the active object — F2, or a double-click in the outliner.
///
/// Numbered (`.001`) when another object has the name, as Blender does: a
/// name is how a person and a script find an object, and two of the same is
/// one neither can find. Animation, drivers and links hold the object by
/// identity, so nothing that points at it is broken by the new name.
pub struct Rename;

impl Operator for Rename {
    type Params = RenameParams;

    fn id() -> &'static str {
        "object.rename"
    }
    fn label() -> &'static str {
        "Rename"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(id) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let base = params.name.trim();
        if base.is_empty() || d.object(id).is_some_and(|o| o.name == base) {
            return Ok(OpResult::Cancelled);
        }
        let taken = |name: &str| d.objects().any(|(other, o)| other != id && o.name == name);
        let name = if taken(base) {
            (1..)
                .map(|i| format!("{base}.{i:03}"))
                .find(|n| !taken(n))
                .expect("names run out after integers")
        } else {
            base.to_string()
        };
        if let Some(object) = doc.document_mut().object_mut(id) {
            object.name = name;
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParentParams {
    /// Clear the parent instead of setting it to the active object.
    pub clear: bool,
    /// Hang them from the active armature's active bone rather than from the
    /// armature: Blender's Ctrl+P ▸ Bone.
    pub bone: bool,
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

        // The bone, for a parent to a bone: the armature's active one.
        let bone = match (params.bone && !params.clear, parent) {
            (true, Some(p)) => {
                let d = doc.document();
                let name = d
                    .object(p)
                    .and_then(|o| o.pose.active.clone())
                    .filter(|b| d.bone_frame(p, b).is_some());
                match name {
                    Some(b) => Some(b),
                    None => {
                        return Err(OpError::Failed(
                            "make an armature active, with a bone active in it".into(),
                        ))
                    }
                }
            }
            _ => None,
        };
        let d = doc.document_mut();
        let mut changed = false;
        let mut refused = 0;
        for id in selection.objects.iter().copied() {
            if Some(id) == parent {
                continue;
            }
            let done = match (&bone, parent) {
                (Some(b), Some(p)) => d.set_parent_bone(id, p, b),
                _ => d.set_parent(id, parent),
            };
            match done {
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

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddLatheParams {
    /// The silhouette as radius, height pairs: r0, z0, r1, z1, … in metres,
    /// from one end to the other. A point at radius zero closes that end; an
    /// end off the axis is left open, like a cup's mouth. Come back down the
    /// inside to give a wall thickness.
    pub points: Vec<f64>,
    /// Segments around the axis.
    #[param(min = 3, max = 512, unit = "count")]
    pub segments: u32,
    /// Curve steps between neighbouring profile points: 1 joins them with
    /// straight lines, more with a smooth curve through every point — a
    /// rounded belly rather than a polygon's.
    #[param(min = 1, max = 64, unit = "count")]
    pub resolution: u32,
    /// Where the axis stands; the profile's heights are measured from here.
    pub location: [f64; 3],
}

impl Default for AddLatheParams {
    fn default() -> Self {
        Self {
            points: vec![0.0, 0.0, 0.4, 0.0, 0.5, 0.6, 0.25, 1.0, 0.0, 1.0],
            segments: 32,
            resolution: 1,
            location: [0.0; 3],
        }
    }
}

/// Adds a mesh made by turning a profile about the vertical axis: a vase, a
/// cup, a bottle, a lampshade — the silhouette itself rather than a sphere
/// squashed towards it.
pub struct AddLathe;

impl Operator for AddLathe {
    type Params = AddLatheParams;

    fn id() -> &'static str {
        "object.add_lathe"
    }
    fn label() -> &'static str {
        "Add Lathe"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if params.points.len() < 4 || params.points.len() % 2 != 0 {
            return Err(OpError::Failed(
                "points wants radius, height pairs: at least two points, an even count of numbers"
                    .into(),
            ));
        }
        let given: Vec<glam::Vec3> = params
            .points
            .chunks(2)
            .map(|p| glam::Vec3::new(p[0] as f32, 0.0, p[1] as f32))
            .collect();
        let profile: Vec<(f32, f32)> =
            kreate_mesh::primitives::smooth_path(&given, params.resolution, false)
                .into_iter()
                .map(|p| (p.x.max(0.0), p.z))
                .collect();
        let mesh = kreate_mesh::primitives::lathe(&profile, params.segments);
        let violations = mesh.validate();
        if !violations.is_empty() {
            return Err(OpError::Failed(format!(
                "the profile makes an invalid mesh: {violations:?}"
            )));
        }
        let d = doc.document_mut();
        let id = d.add_mesh_object("Lathe", mesh);
        d.object_mut(id).expect("just added").transform.translation =
            DVec3::from_array(params.location);
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddSweepParams {
    /// The points the path passes through: x0, y0, z0, x1, y1, z1, … in
    /// metres from `location`. At least two.
    pub path: Vec<f64>,
    /// A round profile's radius, in metres. Ignored when `profile` is given.
    #[param(min = 0.0001, soft_max = 1.0, unit = "length")]
    pub radius: f64,
    /// A round profile's segments.
    #[param(min = 3, max = 256, unit = "count")]
    pub segments: u32,
    /// A profile of its own instead of a circle: x0, y0, x1, y1, … across the
    /// path, a closed outline — a moulding, a rail, a frame's section.
    pub profile: Vec<f64>,
    /// Curve steps between neighbouring path points; 1 joins them straight.
    #[param(min = 1, max = 64, unit = "count")]
    pub resolution: u32,
    /// The path is a loop: its end joins its start.
    pub closed: bool,
    /// The profile's size at the start of the path…
    #[param(min = 0.0, soft_max = 4.0)]
    pub scale_start: f64,
    /// …and at its end: a spout that narrows.
    #[param(min = 0.0, soft_max = 4.0)]
    pub scale_end: f64,
    /// Close the ends of an open path.
    pub caps: bool,
    pub location: [f64; 3],
}

impl Default for AddSweepParams {
    fn default() -> Self {
        Self {
            path: vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.5],
            radius: 0.05,
            segments: 16,
            profile: Vec::new(),
            resolution: 8,
            closed: false,
            scale_start: 1.0,
            scale_end: 1.0,
            caps: true,
            location: [0.0; 3],
        }
    }
}

/// Adds a mesh made by carrying a profile along a path: a pipe, a handle, a
/// kettle's spout, a cable, a picture frame, a moulding.
pub struct AddSweep;

impl Operator for AddSweep {
    type Params = AddSweepParams;

    fn id() -> &'static str {
        "object.add_sweep"
    }
    fn label() -> &'static str {
        "Add Sweep"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if params.path.len() < 6 || params.path.len() % 3 != 0 {
            return Err(OpError::Failed(
                "path wants x, y, z triples: at least two points".into(),
            ));
        }
        if !params.profile.is_empty() && (params.profile.len() < 6 || params.profile.len() % 2 != 0)
        {
            return Err(OpError::Failed(
                "profile wants x, y pairs: at least three points".into(),
            ));
        }
        let path: Vec<glam::Vec3> = params
            .path
            .chunks(3)
            .map(|p| glam::Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
            .collect();
        let profile: Vec<glam::Vec2> = if params.profile.is_empty() {
            kreate_mesh::primitives::circle_profile(params.radius as f32, params.segments)
        } else {
            params
                .profile
                .chunks(2)
                .map(|p| glam::Vec2::new(p[0] as f32, p[1] as f32))
                .collect()
        };
        let mesh = kreate_mesh::primitives::sweep(&kreate_mesh::primitives::Sweep {
            path: &path,
            profile: &profile,
            resolution: params.resolution,
            closed: params.closed,
            scale: (params.scale_start as f32, params.scale_end as f32),
            caps: params.caps,
        });
        if mesh.num_faces() == 0 {
            return Err(OpError::Failed(
                "the path is too short to sweep along".into(),
            ));
        }
        let violations = mesh.validate();
        if !violations.is_empty() {
            return Err(OpError::Failed(format!(
                "the sweep makes an invalid mesh: {violations:?}"
            )));
        }
        let d = doc.document_mut();
        let id = d.add_mesh_object("Sweep", mesh);
        d.object_mut(id).expect("just added").transform.translation =
            DVec3::from_array(params.location);
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PurgeParams {}

/// Removes meshes, materials and images that nothing uses — Blender's Purge.
///
/// What an experiment leaves behind: every `material.new` that was tried and
/// replaced is still in the file, and still in a list a script reads to find
/// one to reuse.
pub struct Purge;

impl Operator for Purge {
    type Params = PurgeParams;

    fn id() -> &'static str {
        "data.purge"
    }
    fn label() -> &'static str {
        "Purge Unused Data"
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let purged = doc.document_mut().purge_unused();
        Ok(if purged == kreate_scene::document::Purged::default() {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

/// An object's box in world space, or its origin as a point when it has no
/// mesh: a lamp or an empty is aligned by where it is.
fn bounds_or_point(doc: &Document, id: ObjectId) -> Option<(DVec3, DVec3)> {
    doc.object_world_bounds(id).or_else(|| {
        doc.object(id)?;
        let p = doc.world_transform(id).translation;
        Some((p, p))
    })
}

/// Moves an object by a distance given in world space, whatever its parent.
fn move_in_world(doc: &mut Document, id: ObjectId, delta: DVec3) {
    let local = doc.above(id).inverse().transform_vector3(delta);
    if let Some(object) = doc.object_mut(id) {
        object.transform.translation += local;
    }
}

fn object_named(doc: &Document, name: &str) -> Option<ObjectId> {
    doc.objects()
        .find(|(_, o)| o.name == name)
        .map(|(id, _)| id)
}

/// Where along one axis of a box `side` is: its low face, its middle, its
/// high face.
fn side_of(bounds: (DVec3, DVec3), axis: usize, side: u32) -> f64 {
    match side {
        0 => bounds.0[axis],
        2 => bounds.1[axis],
        _ => (bounds.0[axis] + bounds.1[axis]) * 0.5,
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AlignParams {
    /// The object to line up with, by name. Empty: the 3D cursor.
    pub target: String,
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    /// Which side of the active object is moved into line.
    #[param(variants("min", "center", "max"))]
    pub side: u32,
    /// Which side of the target it is lined up with.
    #[param(variants("min", "center", "max"))]
    pub to: u32,
    /// Then this much further along the axis, in metres.
    #[param(unit = "length")]
    pub offset: f64,
}

impl Default for AlignParams {
    fn default() -> Self {
        Self {
            target: String::new(),
            axis: 2,
            side: 0,
            to: 2,
            offset: 0.0,
        }
    }
}

/// Moves the active object along one axis until one side of its box lines up
/// with a side of another object's box, by their world-space bounds.
///
/// The defaults stand it on top of the target: its bottom (min Z) to the
/// target's top (max Z). `side="center", to="center"` centres it; `side="max",
/// to="min"` puts it against the near face.
pub struct Align;

impl Operator for Align {
    type Params = AlignParams;

    fn id() -> &'static str {
        "object.align"
    }
    fn label() -> &'static str {
        "Align to Object"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(active) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let target = match params.target.trim() {
            "" => (d.cursor, d.cursor),
            name => {
                let id = object_named(d, name)
                    .ok_or_else(|| OpError::Failed(format!("no object called {name:?}")))?;
                if id == active {
                    return Err(OpError::Failed(
                        "an object cannot be aligned to itself".into(),
                    ));
                }
                bounds_or_point(d, id)
                    .ok_or_else(|| OpError::Failed(format!("{name:?} has no position")))?
            }
        };
        let Some(own) = bounds_or_point(d, active) else {
            return Ok(OpResult::Cancelled);
        };
        let axis = (params.axis as usize).min(2);
        let distance =
            side_of(target, axis, params.to) + params.offset - side_of(own, axis, params.side);
        if distance.abs() < 1e-12 {
            return Ok(OpResult::Cancelled);
        }
        let mut delta = DVec3::ZERO;
        delta[axis] = distance;
        move_in_world(doc.document_mut(), active, delta);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PlaceOnParams {
    /// The object to stand it on, by name.
    pub target: String,
    /// Also centre it on the target in X and Y.
    pub center: bool,
}

/// Stands the active object on top of another: its lowest point on the
/// target's highest, by their world-space bounds — a vase on a table. With
/// `center`, in the middle of it as well.
pub struct PlaceOn;

impl Operator for PlaceOn {
    type Params = PlaceOnParams;

    fn id() -> &'static str {
        "object.place_on"
    }
    fn label() -> &'static str {
        "Place On"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(active) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let name = params.target.trim();
        let target = object_named(d, name)
            .ok_or_else(|| OpError::Failed(format!("no object called {name:?}")))?;
        if target == active {
            return Err(OpError::Failed(
                "an object cannot be placed on itself".into(),
            ));
        }
        let (Some(under), Some(own)) = (bounds_or_point(d, target), bounds_or_point(d, active))
        else {
            return Ok(OpResult::Cancelled);
        };
        let mut delta = DVec3::new(0.0, 0.0, under.1.z - own.0.z);
        if params.center {
            for axis in 0..2 {
                delta[axis] = side_of(under, axis, 1) - side_of(own, axis, 1);
            }
        }
        if delta.length() < 1e-12 {
            return Ok(OpResult::Cancelled);
        }
        move_in_world(doc.document_mut(), active, delta);
        Ok(OpResult::Finished)
    }
}

// --- for scripts: sizes, shading, names --------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetDimensionsParams {
    /// The size wanted along the object's own X, Y and Z, in metres. Zero
    /// leaves that axis as it is.
    #[param(min = 0.0, unit = "length")]
    pub dimensions: [f64; 3],
}

impl Default for SetDimensionsParams {
    fn default() -> Self {
        Self {
            dimensions: [0.0; 3],
        }
    }
}

/// Scales the active mesh object so its bounding box is the size asked for.
///
/// Blender's Dimensions field: the mesh's extent along each of its own axes,
/// times the scale. A script knows the size it wants — a table top 1.6 m by
/// 0.9 m — and working the scale out from a primitive's size means knowing how
/// each primitive was built. An axis the mesh has no extent along (a plane's
/// Z) cannot be sized and is left alone.
pub struct SetDimensions;

impl Operator for SetDimensions {
    type Params = SetDimensionsParams;

    fn id() -> &'static str {
        "object.set_dimensions"
    }
    fn label() -> &'static str {
        "Set Dimensions"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|id| doc.object(id))
            .is_some_and(|o| o.mesh().is_some())
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(active) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let Some(mesh) = d
            .object(active)
            .and_then(|o| o.mesh())
            .and_then(|m| d.mesh(m))
        else {
            return Ok(OpResult::Cancelled);
        };
        let extent = mesh.bounds();
        let extent = (extent.max - extent.min).as_dvec3();
        let Some(object) = doc.document_mut().object_mut(active) else {
            return Ok(OpResult::Cancelled);
        };
        let mut scale = object.transform.scale;
        for axis in 0..3 {
            if params.dimensions[axis] > 0.0 && extent[axis] > 1e-9 {
                scale[axis] = params.dimensions[axis] / extent[axis];
            }
        }
        if scale == object.transform.scale {
            return Ok(OpResult::Cancelled);
        }
        object.transform.scale = scale;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ShadeObjectParams {
    /// Smooth, or flat.
    pub smooth: bool,
    /// With smooth: keep edges sharper than this angle sharp — Blender's Auto
    /// Smooth. Zero smooths every edge.
    #[param(min = 0.0, max = 3.15, unit = "angle")]
    pub angle: f64,
}

impl Default for ShadeObjectParams {
    fn default() -> Self {
        Self {
            smooth: true,
            angle: 0.0,
        }
    }
}

/// Shades the selected mesh objects smooth or flat, every face — Blender's
/// object-mode Shade Smooth and Shade Auto Smooth.
///
/// `mesh.shade` does the same to the selected faces and needs edit mode;
/// this is the whole object, which is what is usually meant.
pub struct ShadeObjects;

impl Operator for ShadeObjects {
    type Params = ShadeObjectParams;

    fn id() -> &'static str {
        "object.shade_smooth"
    }
    fn label() -> &'static str {
        "Shade Smooth"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !selected_mesh_ids(doc).is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mut changed = 0;
        for mesh in selected_mesh_ids(doc.document()) {
            changed += crate::ops::mesh::with_bmesh_id(doc, mesh, |bm| {
                let faces: Vec<_> = bm.face_ids().collect();
                let mut n = edit::set_shade_smooth(bm, &faces, params.smooth);
                if params.smooth && params.angle > 0.0 {
                    n += edit::auto_smooth(bm, params.angle as f32);
                }
                Ok(n)
            })?;
        }
        Ok(if changed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

/// The meshes of the selected objects, each once: two linked copies share one.
fn selected_mesh_ids(doc: &Document) -> Vec<MeshId> {
    let mut out = Vec::new();
    for id in doc.selection.objects.iter() {
        if let Some(mesh) = doc.object(*id).and_then(|o| o.mesh()) {
            if !out.contains(&mesh) {
                out.push(mesh);
            }
        }
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RenameMaterialParams {
    /// Which of the active object's slots holds the material.
    #[param(min = 0.0)]
    pub slot: u32,
    pub name: String,
}

/// Renames the material in one of the active object's slots.
///
/// Numbered (`.001`) when another material has the name, as objects are: a
/// script finds a material to reuse by its name.
pub struct RenameMaterial;

impl Operator for RenameMaterial {
    type Params = RenameMaterialParams;

    fn id() -> &'static str {
        "material.rename"
    }
    fn label() -> &'static str {
        "Rename Material"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .is_some_and(|id| doc.material_id_at(id, 0).is_some())
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(active) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let Some(id) = d.material_id_at(active, params.slot as usize) else {
            return Ok(OpResult::Cancelled);
        };
        let base = params.name.trim();
        if base.is_empty() || d.material(id).is_some_and(|m| m.name == base) {
            return Ok(OpResult::Cancelled);
        }
        let taken = |name: &str| {
            d.materials()
                .any(|(other, m)| other != id && m.name == name)
        };
        let name = if taken(base) {
            (1..)
                .map(|i| format!("{base}.{i:03}"))
                .find(|n| !taken(n))
                .expect("names run out after integers")
        } else {
            base.to_string()
        };
        if let Some(material) = doc.document_mut().material_mut(id) {
            material.name = name;
        }
        Ok(OpResult::Finished)
    }
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddPrimitive>();
    registry.register::<AddGear>();
    registry.register::<AddCamera>();
    registry.register::<AddLight>();
    registry.register::<SetLight>();
    registry.register::<SetWorld>();
    registry.register::<SetWorldImage>();
    registry.register::<NewMaterial>();
    registry.register::<AddMaterialSlot>();
    registry.register::<SetVisible>();
    registry.register::<Paste>();
    registry.register::<Import>();
    registry.register::<LinkLibrary>();
    registry.register::<AddEmpty>();
    registry.register::<RemoveMaterialSlot>();
    registry.register::<AssignMaterial>();
    registry.register::<SelectByMaterial>();
    registry.register::<LinkMaterial>();
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
    registry.register::<Select>();
    registry.register::<Parent>();
    registry.register::<Rename>();
    registry.register::<ParentOne>();
    registry.register::<HideObjects>();
    registry.register::<RevealObjects>();
    registry.register::<Join>();
    registry.register::<SetDimensions>();
    registry.register::<ShadeObjects>();
    registry.register::<RenameMaterial>();
    registry.register::<AddLathe>();
    registry.register::<AddSweep>();
    registry.register::<Purge>();
    registry.register::<Align>();
    registry.register::<PlaceOn>();
}

#[cfg(test)]
mod tests {

    /// Several objects moved, turned or scaled together go together, whatever
    /// their parents: a child of a turned, scaled parent as far and the same
    /// way as the rest, and one under a selected parent once, with it.
    #[test]
    fn a_selection_moves_as_one_whatever_its_parents() {
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut doc = Document::new();
        let mesh = || kreate_mesh::primitives::cube(1.0);
        let parent = doc.add_mesh_object("Parent", mesh());
        let child = doc.add_mesh_object("Child", mesh());
        let loose = doc.add_mesh_object("Loose", mesh());
        doc.set_parent(child, Some(parent)).unwrap();
        {
            let p = doc.object_mut(parent).unwrap();
            p.transform.rotation = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
            p.transform.scale = DVec3::splat(2.0);
        }
        doc.object_mut(child).unwrap().transform.translation = DVec3::new(1.0, 0.0, 0.0);
        let world = |doc: &Document, id| doc.world_transform(id).translation;
        let mut h = History::new();
        let run = |doc: &mut Document,
                   h: &mut History,
                   id: &str,
                   pairs: &[(&'static str, PropertyValue)]| {
            let mut p = PropertyMap::new();
            for (k, v) in pairs {
                p.set(k, v.clone());
            }
            h.dispatch(&r, doc, &Context::default(), id, &p).unwrap();
        };
        let close = |a: DVec3, b: DVec3| assert!((a - b).length() < 1e-9, "{a} against {b}");
        // The child and a loose object: both a metre up x.
        doc.selection.add(child);
        doc.selection.add(loose);
        let (c0, l0) = (world(&doc, child), world(&doc, loose));
        run(
            &mut doc,
            &mut h,
            "object.translate",
            &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
        );
        close(world(&doc, child) - c0, DVec3::X);
        close(world(&doc, loose) - l0, DVec3::X);
        // A quarter turn about their median: both orbit it alike.
        let (c0, l0) = (world(&doc, child), world(&doc, loose));
        let mid = (c0 + l0) * 0.5;
        run(
            &mut doc,
            &mut h,
            "object.rotate",
            &[
                ("angle", PropertyValue::Float(std::f64::consts::FRAC_PI_2)),
                ("axis", PropertyValue::Enum(2)),
            ],
        );
        let turn = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        close(world(&doc, child), mid + turn * (c0 - mid));
        close(world(&doc, loose), mid + turn * (l0 - mid));
        // Parent and child together: the child goes once, with its parent.
        doc.selection.clear();
        doc.selection.add(parent);
        doc.selection.add(child);
        let c0 = world(&doc, child);
        run(
            &mut doc,
            &mut h,
            "object.translate",
            &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
        );
        close(world(&doc, child) - c0, DVec3::Z);
    }

    /// An area light's outline and spread are set from the panel and read
    /// back the same, for each outline.
    #[test]
    fn an_area_light_takes_its_shape_and_spread() {
        use kreate_scene::document::{Object, ObjectData};
        for (shape, w, h) in [(0u32, 2.0, 0.5), (1, 1.5, 1.5), (2, 3.0, 1.0)] {
            let mut r = OperatorRegistry::new();
            crate::ops::register_builtin(&mut r);
            let mut doc = Document::new();
            let light = doc.add_light(kreate_scene::Light::of_kind(
                "Panel",
                kreate_scene::LightKind::PANEL,
            ));
            let o = doc.add_object(Object::new("Panel", ObjectData::Light(light)));
            doc.selection.add(o);
            let wanted = SetLightParams {
                kind: 3,
                shape,
                width: w,
                height: h,
                spread: 1.2,
                ..light_params(doc.light(light).unwrap())
            };
            let mut h_ = History::new();
            h_.dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "light.set_properties",
                &kreate_reflect::OpParams::to_map(&wanted),
            )
            .unwrap();
            let back = light_params(doc.light(light).unwrap());
            assert_eq!(
                (back.shape, back.width, back.height, back.spread),
                (shape, w, h, 1.2)
            );
        }
    }

    /// Blender's Assign, Select and browse: a face takes a slot, the faces on a
    /// slot are found again, and a slot takes a material that exists.
    #[test]
    fn faces_are_assigned_to_slots_and_found_by_them() {
        use kreate_mesh::attribute::names;
        use kreate_scene::document::{Object, ObjectData};
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let o = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc.selection.add(o);
        let edit = Context {
            mode: crate::operator::Mode::EditMesh,
        };
        let mut h = History::new();
        let slot = |n: i64| {
            let mut p = crate::PropertyMap::new();
            p.set("slot", crate::PropertyValue::Int(n));
            p
        };
        h.dispatch(
            &r,
            &mut doc,
            &edit,
            "material.add_slot",
            &crate::PropertyMap::new(),
        )
        .unwrap();
        h.dispatch(
            &r,
            &mut doc,
            &edit,
            "material.add_slot",
            &crate::PropertyMap::new(),
        )
        .unwrap();
        // One face selected.
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            bm.select_none();
            let f = bm.face_ids().next().unwrap();
            bm.set_face_selected(f, true);
            bm.flush_selection(kreate_mesh::bmesh::SelectMode::Face);
            *m = bm.to_mesh_data();
        }
        h.dispatch(&r, &mut doc, &edit, "material.assign", &slot(1))
            .unwrap();
        let index = doc
            .mesh(mesh)
            .unwrap()
            .attributes()
            .get_i32(names::MATERIAL_INDEX)
            .unwrap()
            .to_vec();
        assert_eq!(index.iter().filter(|i| **i == 1).count(), 1, "{index:?}");

        kreate_mesh::selection::select_none(doc.mesh_mut(mesh).unwrap());
        h.dispatch(&r, &mut doc, &edit, "material.select", &slot(1))
            .unwrap();
        let picked = kreate_mesh::selection::selected(
            doc.mesh(mesh).unwrap(),
            kreate_mesh::attribute::Domain::Face,
        );
        assert_eq!(picked.len(), 1);
        assert_eq!(index[picked[0] as usize], 1);

        let red = doc.add_material(kreate_scene::document::Material {
            name: "Red".into(),
            ..Default::default()
        });
        let uid = doc.material(red).unwrap().uid.raw().to_string();
        let mut p = slot(0);
        p.set("uid", crate::PropertyValue::Text(uid));
        h.dispatch(&r, &mut doc, &Context::default(), "material.link", &p)
            .unwrap();
        assert_eq!(doc.mesh_slots(mesh)[0], Some(red));
    }

    /// H hides the selection and takes it out of the selection; Shift+H hides
    /// the rest; Alt+H brings everything back, selected.
    #[test]
    fn hide_and_reveal_objects() {
        use kreate_scene::document::{Object, ObjectData};
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut doc = Document::new();
        let a = doc.add_object(Object::new("A", ObjectData::Empty));
        let b = doc.add_object(Object::new("B", ObjectData::Empty));
        doc.selection.add(a);
        let mut h = History::new();
        let ctx = Context::default();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "object.hide",
            &crate::PropertyMap::new(),
        )
        .unwrap();
        assert!(!doc.object(a).unwrap().visible && doc.object(b).unwrap().visible);
        assert!(!doc.selection.objects.contains(&a));
        let mut p = crate::PropertyMap::new();
        p.set("unselected", crate::PropertyValue::Bool(true));
        h.dispatch(&r, &mut doc, &ctx, "object.hide", &p).unwrap();
        assert!(!doc.object(b).unwrap().visible);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "object.reveal",
            &crate::PropertyMap::new(),
        )
        .unwrap();
        assert!(doc.object(a).unwrap().visible && doc.object(b).unwrap().visible);
        assert!(doc.selection.objects.contains(&a) && doc.selection.objects.contains(&b));
    }

    /// The outliner's drop: one object parented to another by identity, the
    /// selection untouched; a loop refused.
    #[test]
    fn parent_one_by_identity() {
        use kreate_scene::document::{Object, ObjectData};
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut doc = Document::new();
        let a = doc.add_object(Object::new("A", ObjectData::Empty));
        let b = doc.add_object(Object::new("B", ObjectData::Empty));
        let uid = |doc: &Document, id| {
            crate::PropertyValue::Text(doc.object_uid(id).unwrap().raw().to_string())
        };
        let mut h = History::new();
        let mut p = crate::PropertyMap::new();
        p.set("child", uid(&doc, b));
        p.set("parent", uid(&doc, a));
        h.dispatch(&r, &mut doc, &Context::default(), "object.parent_one", &p)
            .unwrap();
        assert_eq!(doc.object(b).unwrap().parent, Some(a));
        assert!(doc.selection.objects.is_empty());
        let mut back = crate::PropertyMap::new();
        back.set("child", uid(&doc, a));
        back.set("parent", uid(&doc, b));
        assert!(h
            .dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "object.parent_one",
                &back
            )
            .is_err());
    }

    #[test]
    fn rename_names_the_active_object_and_numbers_a_taken_name() {
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut doc = Document::new();
        use kreate_scene::document::{Object, ObjectData};
        let _a = doc.add_object(Object::new("Lamp", ObjectData::Empty));
        let b = doc.add_object(Object::new("Cube", ObjectData::Empty));
        doc.selection.add(b);
        let mut h = History::new();
        let mut p = crate::PropertyMap::new();
        p.set("name", crate::PropertyValue::Text("Lamp".into()));
        h.dispatch(&r, &mut doc, &Context::default(), "object.rename", &p)
            .unwrap();
        assert_eq!(doc.object(b).unwrap().name, "Lamp.001");
        p.set("name", crate::PropertyValue::Text("  Box ".into()));
        h.dispatch(&r, &mut doc, &Context::default(), "object.rename", &p)
            .unwrap();
        assert_eq!(doc.object(b).unwrap().name, "Box");
        assert!(h.undo(&mut doc));
        assert_eq!(doc.object(b).unwrap().name, "Lamp.001");
    }

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
            matches!(error, crate::operator::OpError::PollFailed(..)),
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

    /// On the floor: a 2 m cube's bottom at the location, not its middle;
    /// turned on its edge, its lowest corner there.
    #[test]
    fn a_primitive_on_the_floor_stands_on_its_location() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[
                ("location", PropertyValue::Vec3([1.0, 2.0, 0.5])),
                ("on_floor", PropertyValue::Bool(true)),
            ],
        );
        let id = d.selection.active.unwrap();
        let t = d.object(id).unwrap().transform.translation;
        assert!((t - DVec3::new(1.0, 2.0, 1.5)).length() < 1e-9, "{t:?}");
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[
                ("on_floor", PropertyValue::Bool(true)),
                (
                    "rotation",
                    PropertyValue::Vec3([std::f64::consts::FRAC_PI_4, 0.0, 0.0]),
                ),
            ],
        );
        let id = d.selection.active.unwrap();
        let t = d.object(id).unwrap().transform.translation;
        assert!((t.z - 2f64.sqrt()).abs() < 1e-6, "on its edge: {t:?}");
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
    fn an_object_is_selected_by_name() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.rename",
            &[("name", PropertyValue::Text("Boiler".into()))],
        );
        let boiler = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let other = d.selection.active.unwrap();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select",
            &[("name", PropertyValue::Text("Boiler".into()))],
        );
        assert_eq!(d.selection.objects, vec![boiler]);
        assert_eq!(d.selection.active, Some(boiler));
        // Extending keeps what was there, and the named one becomes active.
        d.selection.clear();
        d.selection.add(other);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select",
            &[
                ("name", PropertyValue::Text("Boiler".into())),
                ("extend", PropertyValue::Bool(true)),
            ],
        );
        assert_eq!(d.selection.objects.len(), 2);
        assert_eq!(d.selection.active, Some(boiler));
        // A name nobody has is an error that says so, and changes nothing.
        let mut p = PropertyMap::new();
        p.set("name", PropertyValue::Text("Nobody".into()));
        assert!(h.dispatch(&r, &mut d, &c, "object.select", &p).is_err());
        assert_eq!(d.selection.objects.len(), 2);
    }

    #[test]
    fn select_all_keeps_the_active_object() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let first = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        // The first made active again; the other comes later in the document.
        d.selection.clear();
        d.selection.add(first);
        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        assert_eq!(d.selection.objects.len(), 2);
        assert_eq!(d.selection.active, Some(first));
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

#[cfg(test)]
mod gear_tests {
    use crate::history::History;
    use crate::operator::{Context, Mode};
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::Document;

    #[test]
    fn a_gear_and_its_mate_are_added_meshed_in_one_step() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut p = PropertyMap::new();
        // Module 2 (the seventh of ISO 54's series), 20 teeth and a mate of 30.
        p.set("module", PropertyValue::Enum(6));
        p.set("teeth", PropertyValue::Int(20));
        p.set("mate_teeth", PropertyValue::Int(30));
        p.set("location", PropertyValue::Vec3([0.1, 0.0, 0.0]));
        h.dispatch(
            &r,
            &mut doc,
            &Context { mode: Mode::Object },
            "object.add_gear",
            &p,
        )
        .unwrap();
        assert_eq!(h.entries().len(), 1, "one step");
        let find = |name: &str| {
            doc.objects()
                .find(|(_, o)| o.name == name)
                .map(|(id, _)| id)
                .unwrap_or_else(|| panic!("no {name}"))
        };
        let (a, b) = (find("Gear m2 z20"), find("Gear m2 z30"));
        // The centre distance: half of 40 + 60 mm.
        let gap = doc.world_transform(b).translation - doc.world_transform(a).translation;
        assert!((gap.x - 0.050).abs() < 1e-12, "{gap:?}");
        assert_eq!(doc.selection.active, Some(a));
        assert_eq!(doc.selection.objects.len(), 2);
        // The outside diameter is what the blank is turned to: 44 mm.
        let mesh = doc.mesh(doc.object(a).unwrap().mesh().unwrap()).unwrap();
        let size = mesh.bounds().size();
        assert!((size.x - 0.044).abs() < 1e-5, "{size:?}");
        assert!((size.z - 0.010).abs() < 1e-6);
        h.undo(&mut doc);
        assert_eq!(doc.object_count(), 0);
    }

    fn add(kind: u32, pairs: &[(&'static str, PropertyValue)]) -> Document {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut p = PropertyMap::new();
        p.set("kind", PropertyValue::Enum(kind));
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(
            &r,
            &mut doc,
            &Context { mode: Mode::Object },
            "object.add_gear",
            &p,
        )
        .unwrap();
        doc
    }

    fn names(doc: &Document) -> Vec<String> {
        let mut n: Vec<String> = doc.objects().map(|(_, o)| o.name.clone()).collect();
        n.sort();
        n
    }

    #[test]
    fn every_kind_makes_its_parts_named_for_what_they_are() {
        let mate = [("mate_teeth", PropertyValue::Int(20))];
        let ring = add(1, &[("teeth", PropertyValue::Int(60)), mate[0].clone()]);
        assert_eq!(names(&ring), vec!["Gear m2 z20", "Ring gear m2 z60"]);
        let rack = add(2, &[("teeth", PropertyValue::Int(12)), mate[0].clone()]);
        assert_eq!(names(&rack), vec!["Gear m2 z20", "Rack m2 12"]);
        let bevel = add(
            3,
            &[
                ("teeth", PropertyValue::Int(20)),
                ("mate_teeth", PropertyValue::Int(30)),
            ],
        );
        assert_eq!(
            names(&bevel),
            vec!["Bevel gear m2 z20", "Bevel gear m2 z30"]
        );
        let mitre = add(3, &[]);
        assert_eq!(names(&mitre), vec!["Bevel gear m2 z20"]);
        let worm = add(
            4,
            &[
                ("teeth", PropertyValue::Int(30)),
                ("starts", PropertyValue::Int(2)),
            ],
        );
        assert_eq!(names(&worm), vec!["Worm m2 n2", "Worm wheel m2 z30"]);
        // The wheel at the centre distance: (20 + 60) / 2 mm from the worm.
        let wheel = worm
            .objects()
            .find(|(_, o)| o.name.starts_with("Worm wheel"))
            .unwrap()
            .0;
        let y = worm.world_transform(wheel).translation.y;
        assert!((y - 0.040).abs() < 1e-9, "{y}");
    }

    /// Bevel gears on shafts at 60°: two gears, the mate's axis 60° from
    /// the first's; and a shaft angle that would make a crown, refused.
    #[test]
    fn bevel_gears_can_meet_at_an_angle_other_than_square() {
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut p = PropertyMap::new();
        p.set("kind", PropertyValue::Enum(3));
        p.set("mate_teeth", PropertyValue::Int(30));
        p.set("width", PropertyValue::Float(0.004));
        p.set("bore", PropertyValue::Float(0.0));
        p.set("shaft_angle", PropertyValue::Float(60f64.to_radians()));
        h.dispatch(&r, &mut doc, &Context::default(), "object.add_gear", &p)
            .unwrap();
        let gears: Vec<_> = doc.objects().map(|(id, _)| id).collect();
        assert_eq!(gears.len(), 2);
        let axes: Vec<kreate_math::DVec3> = gears
            .iter()
            .map(|id| {
                doc.world_transform(*id)
                    .transform_vector3(kreate_math::DVec3::Z)
                    .normalize()
            })
            .collect();
        assert!((axes[0].dot(axes[1]).acos().to_degrees() - 60.0).abs() < 1e-6);
        let mut crown = p.clone();
        crown.set("teeth", PropertyValue::Int(60));
        crown.set("mate_teeth", PropertyValue::Int(12));
        crown.set("shaft_angle", PropertyValue::Float(150f64.to_radians()));
        assert!(h
            .dispatch(&r, &mut doc, &Context::default(), "object.add_gear", &crown)
            .is_err());
    }

    #[test]
    fn a_bevel_face_too_wide_is_refused_with_the_most_it_can_be() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut p = PropertyMap::new();
        p.set("kind", PropertyValue::Enum(3));
        p.set("teeth", PropertyValue::Int(20));
        p.set("mate_teeth", PropertyValue::Int(30));
        // m 2, z 20 against 30: a cone distance of 36.06 mm, so at most 18.0.
        p.set("width", PropertyValue::Float(0.018));
        h.dispatch(
            &r,
            &mut doc,
            &Context { mode: Mode::Object },
            "object.add_gear",
            &p,
        )
        .expect("just within");
        p.set("width", PropertyValue::Float(0.025));
        let refused = h
            .dispatch(
                &r,
                &mut doc,
                &Context { mode: Mode::Object },
                "object.add_gear",
                &p,
            )
            .unwrap_err()
            .to_string();
        assert!(refused.contains("18 mm"), "{refused}");
        // And a wider face, within the limit, is a different shape.
        let small = kreate_mesh::gear::bevel_gear(
            &kreate_mesh::gear::GearParams {
                width: 0.006,
                ..Default::default()
            },
            30,
            0.0,
        );
        let wide = kreate_mesh::gear::bevel_gear(
            &kreate_mesh::gear::GearParams {
                width: 0.016,
                ..Default::default()
            },
            30,
            0.0,
        );
        assert!(wide.bounds().size().z > small.bounds().size().z + 0.005);
    }

    #[test]
    fn a_bore_wider_than_the_root_is_refused() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut p = PropertyMap::new();
        // m 2, 20 teeth: a 35 mm root, so a bore up to 31.5 mm.
        p.set("bore", PropertyValue::Float(0.040));
        let refused = h
            .dispatch(
                &r,
                &mut doc,
                &Context { mode: Mode::Object },
                "object.add_gear",
                &p,
            )
            .unwrap_err()
            .to_string();
        assert!(refused.contains("31.5 mm"), "{refused}");
        assert_eq!(doc.object_count(), 0);
    }

    #[test]
    fn a_ring_too_small_for_its_pinion_is_refused() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut p = PropertyMap::new();
        p.set("kind", PropertyValue::Enum(1));
        p.set("teeth", PropertyValue::Int(25));
        p.set("mate_teeth", PropertyValue::Int(20));
        assert!(h
            .dispatch(
                &r,
                &mut doc,
                &Context { mode: Mode::Object },
                "object.add_gear",
                &p
            )
            .is_err());
        assert_eq!(doc.object_count(), 0);
    }
}
