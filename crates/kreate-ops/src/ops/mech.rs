//! Machine design: holes cut to the sizes a screw wants, and the standard
//! parts a machine is put together with. Every one an operator and one undo
//! step; the parts are added at the 3D cursor, turned as it is, so a cursor
//! put on a face (Shift+S, Cursor to Selected) sets where and which way.

use kreate_math::{DQuat, DVec3};
use kreate_mesh::mech::{self, HoleKind, METRIC};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::modifier::{BooleanModifier, ModifierInstance, ModifierKind};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Hole>();
    registry.register::<AddFastener>();
    registry.register::<CheckInterference>();
    registry.register::<AddBearing>();
    registry.register::<AddShaft>();
    registry.register::<AddPulley>();
    registry.register::<AddSprocket>();
    registry.register::<AddSpring>();
    registry.register::<GearLink>();
    registry.register::<Shell>();
    registry.register::<Draft>();
    registry.register::<SweepAlong>();
    registry.register::<Loft>();
    registry.register::<AddSheetMetal>();
}

/// Where a part or a cutter goes: the cursor, or a place given.
fn placement(
    doc: &Document,
    at_cursor: bool,
    location: [f64; 3],
    rotation: [f64; 3],
) -> (DVec3, DQuat) {
    if at_cursor {
        (doc.cursor, doc.cursor_rotation)
    } else {
        let angles = DVec3::from_array(rotation);
        let turn = if angles.is_finite() {
            kreate_scene::animation::from_euler(angles)
        } else {
            DQuat::IDENTITY
        };
        (DVec3::from_array(location), turn)
    }
}

/// The object to cut: named, or the active one; a mesh either way.
fn target(doc: &Document, name: &str) -> Result<ObjectId, OpError> {
    let id = if name.trim().is_empty() {
        doc.selection.active
    } else {
        doc.objects()
            .find(|(_, o)| o.name == name)
            .map(|(id, _)| id)
    };
    id.filter(|id| doc.object(*id).and_then(|o| o.mesh()).is_some())
        .ok_or_else(|| OpError::Failed("pick the part to cut the hole in".into()))
}

/// A hidden object holding `mesh`, placed, that `into` loses as a boolean:
/// what makes a cut that can be moved or taken off later.
fn cut_with(
    doc: &mut DocMut<'_>,
    into: ObjectId,
    name: String,
    mesh: kreate_mesh::MeshData,
    at: DVec3,
    turn: DQuat,
) {
    let d = doc.document_mut();
    let mesh = d.add_mesh(mesh);
    let mut cutter = Object::new(name, ObjectData::Mesh(mesh));
    cutter.visible = false;
    cutter.transform.translation = at;
    cutter.transform.rotation = turn;
    let cutter = d.add_object(cutter);
    let operand = d.object_uid(cutter).expect("just added");
    d.object_mut(into)
        .expect("the target")
        .modifiers
        .push(ModifierInstance::new(ModifierKind::Boolean(
            BooleanModifier {
                operand,
                op: kreate_mesh::modifiers::BooleanOp::Difference,
            },
        )));
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct HoleParams {
    /// A plain clearance hole, one counterbored for a socket head cap
    /// screw, one countersunk for a countersunk screw, or a tapped hole.
    #[param(variants("simple", "counterbore", "countersink", "tapped"))]
    pub kind: u32,
    /// The screw it is for, ISO metric.
    #[param(variants(
        "M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10", "M12", "M14", "M16", "M20", "M24"
    ))]
    pub size: u32,
    /// A plain hole's diameter, if not its size's clearance; zero for that.
    #[param(min = 0.0, soft_max = 0.05, unit = "length")]
    pub diameter: f64,
    /// How deep, below the surface, for a blind hole.
    #[param(min = 0.0, soft_max = 0.1, unit = "length")]
    pub depth: f64,
    /// Right through the part.
    pub through: bool,
    /// A tapped hole's thread cut as well as its drill: for a part to be
    /// printed. A part to be tapped wants only the drill.
    pub thread: bool,
    /// At the 3D cursor, going down its Z; otherwise at `location`.
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
    /// The part to cut, by name; empty for the active object.
    pub target: String,
}

impl Default for HoleParams {
    fn default() -> Self {
        Self {
            kind: 0,
            size: 5,
            diameter: 0.0,
            depth: 0.01,
            through: true,
            thread: false,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
            target: String::new(),
        }
    }
}

/// A hole for an ISO metric screw, cut into the active part at the 3D
/// cursor and down its Z: a clearance hole, counterbored or countersunk for
/// the head, or a tapped hole's drill — and its thread, if asked. The
/// cutter stays as a hidden object and the part as a boolean against it,
/// so the hole can be moved or taken off later.
pub struct Hole;

impl Operator for Hole {
    type Params = HoleParams;
    fn id() -> &'static str {
        "object.hole"
    }
    fn label() -> &'static str {
        "Hole"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let into = target(d, &params.target)?;
        let size = METRIC[(params.size as usize).min(METRIC.len() - 1)];
        let kind = match params.kind {
            1 => HoleKind::Counterbore,
            2 => HoleKind::Countersink,
            3 => HoleKind::Tapped,
            _ => HoleKind::Simple,
        };
        let (at, turn) = placement(d, params.at_cursor, params.location, params.rotation);
        // Through: deeper than the part is across, wherever on it it starts.
        let depth = if params.through {
            let o = d.object(into).expect("checked");
            let b = o
                .mesh()
                .and_then(|m| d.mesh(m))
                .map(|m| m.bounds().size().as_dvec3().length())
                .unwrap_or(1.0);
            let scale = o.transform.scale.abs().max_element();
            (b * scale * 2.0).max(params.depth)
        } else {
            params.depth
        };
        let cutter = mech::hole_cutter(
            kind,
            &size,
            (params.diameter > 0.0).then_some(params.diameter),
            depth,
            params.through,
        );
        let what = ["", " counterbore", " countersink", " tapped"][params.kind.min(3) as usize];
        // i18n-exempt: a feature's designation
        let name = format!("Hole {}{what}", size.name);
        cut_with(doc, into, name.clone(), cutter, at, turn);
        if kind == HoleKind::Tapped && params.thread {
            // The thread, the tap drill's depth, cut downward: an external
            // thread turned over, with a tenth of a millimetre of play.
            let mut thread =
                mech::internal_thread_cutter(size.d * 0.001, size.pitch * 0.001, depth, 0.0001);
            for p in thread.positions_mut() {
                // Half a turn about X: down −Z, still the right hand.
                p.y = -p.y;
                p.z = -p.z;
            }
            // i18n-exempt: a feature's designation
            cut_with(doc, into, format!("{name} thread"), thread, at, turn);
        }
        Ok(OpResult::Finished)
    }
}

/// Adds a finished part at the cursor, selected and active.
fn add_part(
    doc: &mut DocMut<'_>,
    name: String,
    mesh: kreate_mesh::MeshData,
    at: DVec3,
    turn: DQuat,
) -> ObjectId {
    let d = doc.document_mut();
    let id = d.add_mesh_object(name, mesh);
    let o = d.object_mut(id).expect("just added");
    o.transform.translation = at;
    o.transform.rotation = turn;
    d.selection.clear();
    d.selection.add(id);
    d.selection.active = Some(id);
    id
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddFastenerParams {
    /// A socket head cap screw (ISO 4762), a hex bolt (ISO 4017), a nut
    /// (ISO 4032) or a washer (ISO 7089).
    #[param(variants("cap_screw", "hex_bolt", "nut", "washer"))]
    pub kind: u32,
    /// Its ISO metric size.
    #[param(variants(
        "M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10", "M12", "M14", "M16", "M20", "M24"
    ))]
    pub size: u32,
    /// A screw's length under its head.
    #[param(min = 0.001, soft_max = 0.2, unit = "length")]
    pub length: f64,
    /// The thread modelled to ISO 68-1, for printing or close looks; plain
    /// at its major diameter otherwise, which is lighter.
    pub thread: bool,
    /// At the 3D cursor, sitting on it; otherwise at `location`.
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddFastenerParams {
    fn default() -> Self {
        Self {
            kind: 0,
            size: 5,
            length: 0.02,
            thread: true,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// An ISO screw, bolt, nut or washer, set on the cursor as it would sit on
/// a surface — a screw's head on it and its shank going in — and named by
/// its designation, which is what it is ordered by.
pub struct AddFastener;

impl Operator for AddFastener {
    type Params = AddFastenerParams;
    fn id() -> &'static str {
        "object.add_fastener"
    }
    fn label() -> &'static str {
        "Add Fastener"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::mech::FastenerKind as K;
        let size = (params.size as usize).min(METRIC.len() - 1);
        let kind = match params.kind {
            1 => K::HexBolt,
            2 => K::Nut,
            3 => K::Washer,
            _ => K::CapScrew,
        };
        let name = METRIC[size].name;
        let l = (params.length * 1000.0 * 10.0).round() / 10.0;
        // i18n-exempt: designations, as parts are ordered by
        let label = match kind {
            K::CapScrew => format!("ISO 4762 {name}x{l}"),
            K::HexBolt => format!("ISO 4017 {name}x{l}"),
            K::Nut => format!("ISO 4032 {name}"),
            K::Washer => format!("ISO 7089 {name}"),
        };
        let mesh = mech::fastener(kind, size, params.length, params.thread);
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        add_part(doc, label, mesh, at, turn);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct CheckInterferenceParams {}

/// The name the interference check's results go under, to be found and
/// replaced on the next check.
pub const INTERFERENCE: &str = "Interference";

/// Where parts run into each other: the selected parts — or, with fewer
/// than two selected, every part shown — taken two at a time, and wherever
/// two overlap, the overlap itself made a red object named for the two and
/// how much of each other they hold. Put in the Interference collection,
/// whose last check's results it replaces. A check that finds nothing says
/// so as its answer and leaves nothing behind.
pub struct CheckInterference;

impl Operator for CheckInterference {
    type Params = CheckInterferenceParams;
    fn id() -> &'static str {
        "object.check_interference"
    }
    fn label() -> &'static str {
        "Check Interference"
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::modifiers::boolean::boolean;
        use kreate_mesh::modifiers::BooleanOp;
        use kreate_scene::modifier::{EvalCtx, ModifierCache};
        let d = doc.document();
        let old = d.find_collection(INTERFERENCE);
        let is_result = |id: ObjectId| old.is_some() && d.collection_of(id) == old;
        let part = |id: &ObjectId| {
            d.object(*id).is_some_and(|o| {
                o.mesh().is_some() && o.sketch_lines.is_empty() && o.sketch.is_empty()
            }) && d.is_shown(*id)
                && !is_result(*id)
        };
        let mut parts: Vec<ObjectId> = d
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| part(id))
            .collect();
        if parts.len() < 2 {
            parts = d
                .objects()
                .map(|(id, _)| id)
                .filter(|id| part(id))
                .collect();
        }
        if parts.len() < 2 {
            return Err(OpError::Failed(
                "there are not two parts to check against each other".into(),
            ));
        }
        // Each part as drawn, in the world.
        let mut cache = ModifierCache::new();
        let world: Vec<(
            ObjectId,
            kreate_mesh::MeshData,
            kreate_math::Vec3,
            kreate_math::Vec3,
        )> = parts
            .iter()
            .filter_map(|id| {
                let mesh = cache.evaluated(d, *id, &EvalCtx::still())?;
                let to = d.world_transform(*id);
                let mut m = (*mesh).clone();
                for p in m.positions_mut() {
                    *p = to.transform_point3(p.as_dvec3()).as_vec3();
                }
                let b = m.bounds();
                Some((*id, m, b.min, b.max))
            })
            .collect();
        let mut found: Vec<(String, kreate_mesh::MeshData, f64)> = Vec::new();
        for i in 0..world.len() {
            for j in i + 1..world.len() {
                let (a, b) = (&world[i], &world[j]);
                // Boxes apart, parts apart.
                if a.2.cmpgt(b.3).any() || b.2.cmpgt(a.3).any() {
                    continue;
                }
                let overlap = boolean(&a.1, &b.1, BooleanOp::Intersect);
                if overlap.num_faces() == 0 {
                    continue;
                }
                let v = kreate_mesh::measure::mass_properties(&overlap, |p| p)
                    .volume
                    .abs();
                // Less than a cubic tenth of a millimetre is touching.
                if v < 1e-13 {
                    continue;
                }
                let name = |id: ObjectId| d.object(id).map(|o| o.name.clone()).unwrap_or_default();
                // i18n-exempt: two part names and a volume
                let label = format!("{} x {} ({:.3} mm3)", name(a.0), name(b.0), v * 1e9);
                found.push((label, overlap, v));
            }
        }
        // The last check's results go, found or not: what is left must be
        // what is true now.
        let d = doc.document_mut();
        if let Some(c) = d.find_collection(INTERFERENCE) {
            let stale: Vec<ObjectId> = d
                .objects()
                .map(|(id, _)| id)
                .filter(|id| d.collection_of(*id) == Some(c))
                .collect();
            for id in stale {
                d.remove_object(id);
            }
        }
        if found.is_empty() {
            return Err(OpError::Failed(format!(
                "no interference: the {} parts are clear of each other",
                parts.len()
            )));
        }
        let collection = match d.find_collection(INTERFERENCE) {
            Some(c) => c,
            None => d.add_collection(kreate_scene::document::Collection {
                name: INTERFERENCE.into(),
                objects: Vec::new(),
                hide: false,
            }),
        };
        let red = d.add_material(kreate_scene::document::Material {
            name: INTERFERENCE.into(),
            base_color: kreate_math::SceneRgb::new(0.9, 0.05, 0.03),
            ..Default::default()
        });
        for (label, mesh, _) in found {
            let mesh_id = d.add_mesh(mesh);
            d.set_mesh_slots(mesh_id, vec![Some(red)]);
            let id = d.add_object(Object::new(label, ObjectData::Mesh(mesh_id)));
            d.move_to_collection(id, Some(collection));
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddBearingParams {
    /// Its designation (ISO 15): miniature, 60, 62 and 63 series.
    #[param(variants(
        "625", "626", "608", "6000", "6001", "6002", "6003", "6004", "6005", "6006", "6007",
        "6008", "6009", "6010", "6200", "6201", "6202", "6203", "6204", "6205", "6206", "6207",
        "6208", "6209", "6210", "6300", "6301"
    ))]
    pub size: u32,
    /// At the 3D cursor, standing on it; otherwise at `location`.
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddBearingParams {
    fn default() -> Self {
        Self {
            size: 18,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// A deep groove ball bearing (ISO 15), named by its designation and its
/// size — what a shaft and a housing are made to.
pub struct AddBearing;

impl Operator for AddBearing {
    type Params = AddBearingParams;
    fn id() -> &'static str {
        "object.add_bearing"
    }
    fn label() -> &'static str {
        "Add Bearing"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let b = mech::BEARINGS[(params.size as usize).min(mech::BEARINGS.len() - 1)];
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        // i18n-exempt: a designation and its size
        let name = format!("Bearing {} ({}x{}x{})", b.name, b.bore, b.outside, b.width);
        add_part(doc, name, mech::bearing(&b), at, turn);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddShaftParams {
    #[param(min = 0.003, soft_max = 0.065, unit = "length")]
    pub diameter: f64,
    #[param(min = 0.005, soft_max = 0.5, unit = "length")]
    pub length: f64,
    /// A keyway for a parallel key (DIN 6885 A) of the shaft's size, half
    /// way up.
    pub keyway: bool,
    /// The key's length, and the keyway's.
    #[param(min = 0.004, soft_max = 0.1, unit = "length")]
    pub key_length: f64,
    /// The key itself, standing in the keyway, as a part of its own.
    pub with_key: bool,
    /// A retaining ring groove (DIN 471) this far in from each end; zero
    /// for none.
    #[param(min = 0.0, soft_max = 0.05, unit = "length")]
    pub groove_at: f64,
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddShaftParams {
    fn default() -> Self {
        Self {
            diameter: 0.02,
            length: 0.1,
            keyway: true,
            key_length: 0.025,
            with_key: true,
            groove_at: 0.005,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// A round shaft standing up the cursor's Z, with a keyway milled for its
/// size's parallel key and retaining ring grooves turned near its ends.
pub struct AddShaft;

impl Operator for AddShaft {
    type Params = AddShaftParams;
    fn id() -> &'static str {
        "object.add_shaft"
    }
    fn label() -> &'static str {
        "Add Shaft"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let (d, l) = (params.diameter, params.length);
        let key = mech::key_for(d * 1000.0);
        if params.keyway && key.is_none() {
            return Err(OpError::Failed(format!(
                "DIN 6885 has keys for shafts over 6 and up to 65 mm; this one is {} mm",
                (d * 10000.0).round() / 10.0
            )));
        }
        if params.keyway && params.key_length > l {
            return Err(OpError::Failed("the key is longer than the shaft".into()));
        }
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        // i18n-exempt: a part's size
        let name = format!(
            "Shaft {}x{}",
            (d * 10000.0).round() / 10.0,
            (l * 10000.0).round() / 10.0
        );
        let id = add_part(
            doc,
            name.clone(),
            mech::shaft(d, l, params.groove_at),
            at,
            turn,
        );
        if let (true, Some(k)) = (params.keyway, key) {
            let cutter = mech::keyway_cutter(d, &k, params.key_length, l * 0.5);
            // i18n-exempt: a feature's size
            cut_with(
                doc,
                id,
                format!("{name} keyway {}x{}", k.width, k.height),
                cutter,
                at,
                turn,
            );
            if params.with_key {
                let part = mech::key(d, &k, params.key_length, l * 0.5);
                // i18n-exempt: a designation
                let key_name = format!(
                    "Key DIN 6885 A {}x{}x{}",
                    k.width,
                    k.height,
                    (params.key_length * 1000.0).round()
                );
                let d = doc.document_mut();
                let kid = d.add_mesh_object(key_name, part);
                let o = d.object_mut(kid).expect("just added");
                o.transform.translation = at;
                o.transform.rotation = turn;
            }
            let d = doc.document_mut();
            d.selection.clear();
            d.selection.add(id);
            d.selection.active = Some(id);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddPulleyParams {
    /// The belt it runs: GT2, GT3 or HTD 5M.
    #[param(variants("gt2", "gt3", "htd5m"))]
    pub belt: u32,
    #[param(min = 8, max = 150, unit = "count")]
    pub teeth: u32,
    /// The toothed width, the belt's and a little.
    #[param(min = 0.001, soft_max = 0.05, unit = "length")]
    pub width: f64,
    #[param(min = 0.0, soft_max = 0.03, unit = "length")]
    pub bore: f64,
    /// A flange each side, to keep the belt on.
    pub flanges: bool,
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddPulleyParams {
    fn default() -> Self {
        Self {
            belt: 0,
            teeth: 20,
            width: 0.007,
            bore: 0.005,
            flanges: true,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// A timing pulley for a GT2, GT3 or HTD 5M belt, named by its belt and
/// teeth.
pub struct AddPulley;

impl Operator for AddPulley {
    type Params = AddPulleyParams;
    fn id() -> &'static str {
        "object.add_pulley"
    }
    fn label() -> &'static str {
        "Add Timing Pulley"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let belt = mech::BELTS[(params.belt as usize).min(mech::BELTS.len() - 1)];
        let teeth = params.teeth.clamp(8, 150);
        let root =
            (teeth as f64 * belt.pitch / std::f64::consts::PI - 2.0 * belt.gap_depth) * 0.001;
        if params.bore >= root * 0.9 {
            return Err(OpError::Failed(format!(
                "the bore can be at most {} mm on this pulley",
                (root * 0.9 * 10000.0).round() / 10.0
            )));
        }
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        // i18n-exempt: a designation
        let name = format!("Pulley {} {}T", belt.name, teeth);
        add_part(
            doc,
            name,
            mech::pulley(&belt, teeth, params.width, params.bore, params.flanges),
            at,
            turn,
        );
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddSprocketParams {
    /// The roller chain (ISO 606, and two ANSI sizes).
    #[param(variants("04B", "05B", "06B", "08B", "10B", "ansi25", "ansi40"))]
    pub chain: u32,
    #[param(min = 6, max = 120, unit = "count")]
    pub teeth: u32,
    #[param(min = 0.0, soft_max = 0.05, unit = "length")]
    pub bore: f64,
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddSprocketParams {
    fn default() -> Self {
        Self {
            chain: 3,
            teeth: 17,
            bore: 0.01,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// A roller chain sprocket to ISO 606's tooth form, named by its chain and
/// teeth.
pub struct AddSprocket;

impl Operator for AddSprocket {
    type Params = AddSprocketParams;
    fn id() -> &'static str {
        "object.add_sprocket"
    }
    fn label() -> &'static str {
        "Add Sprocket"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let chain = mech::CHAINS[(params.chain as usize).min(mech::CHAINS.len() - 1)];
        let teeth = params.teeth.clamp(6, 120);
        let root = mech::sprocket_pitch_diameter(&chain, teeth) - chain.roller * 0.001;
        if params.bore >= root * 0.85 {
            return Err(OpError::Failed(format!(
                "the bore can be at most {} mm on this sprocket",
                (root * 0.85 * 10000.0).round() / 10.0
            )));
        }
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        // i18n-exempt: a designation
        let name = format!("Sprocket {} {}T", chain.name, teeth);
        add_part(
            doc,
            name,
            mech::sprocket(&chain, teeth, params.bore),
            at,
            turn,
        );
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddSpringParams {
    #[param(variants("compression", "extension", "torsion"))]
    pub kind: u32,
    /// The wire's diameter.
    #[param(min = 0.0001, soft_max = 0.01, unit = "length")]
    pub wire: f64,
    /// The coil's mean diameter: through the middle of the wire.
    #[param(min = 0.001, soft_max = 0.1, unit = "length")]
    pub diameter: f64,
    /// Total turns, the ends' included.
    #[param(min = 1.0, soft_max = 50.0)]
    pub coils: f64,
    /// A compression spring's free length; the others are close-wound.
    #[param(min = 0.001, soft_max = 0.3, unit = "length")]
    pub length: f64,
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddSpringParams {
    fn default() -> Self {
        Self {
            kind: 0,
            wire: 0.001,
            diameter: 0.01,
            coils: 8.0,
            length: 0.03,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// A coil spring: compression with closed, ground ends; extension with a
/// hook each end; torsion with a leg each end. Named by its wire and coil.
pub struct AddSpring;

impl Operator for AddSpring {
    type Params = AddSpringParams;
    fn id() -> &'static str {
        "object.add_spring"
    }
    fn label() -> &'static str {
        "Add Spring"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::mech::SpringKind as K;
        let kind = match params.kind {
            1 => K::Extension,
            2 => K::Torsion,
            _ => K::Compression,
        };
        if params.wire * 2.0 >= params.diameter {
            return Err(OpError::Failed("the coil is too tight for the wire: its diameter must be more than twice the wire's".into()));
        }
        if kind == K::Compression && params.length < params.wire * params.coils {
            return Err(OpError::Failed(format!(
                "{} coils of this wire need at least {} mm",
                params.coils,
                (params.wire * params.coils * 10000.0).round() / 10.0
            )));
        }
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        let what = ["Compression", "Extension", "Torsion"][params.kind.min(2) as usize];
        // i18n-exempt: a part's description by its sizes
        let name = format!(
            "{what} spring {}x{}",
            (params.wire * 10000.0).round() / 10.0,
            (params.diameter * 10000.0).round() / 10.0
        );
        let mesh = mech::spring(
            kind,
            params.wire,
            params.diameter,
            params.coils,
            params.length,
        );
        add_part(doc, name, mesh, at, turn);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct GearLinkParams {
    /// How many turns the driven gear makes to one of the active one's;
    /// zero to read it from their names' tooth counts ("Gear m2 z20").
    #[param(soft_min = -10.0, soft_max = 10.0)]
    pub ratio: f64,
}

/// A gear's module (mm) and teeth, read from its name as the gear tools
/// write it: "Gear m2 z20", "Ring gear m1.5 z60", "Rack m2 12".
pub fn gear_numbers(name: &str) -> Option<(f64, u32)> {
    let mut module = None;
    let mut teeth = None;
    for word in name.split_whitespace() {
        if let Some(m) = word.strip_prefix('m').and_then(|v| v.parse::<f64>().ok()) {
            module = Some(m);
        } else if let Some(z) = word.strip_prefix('z').and_then(|v| v.parse::<u32>().ok()) {
            teeth = Some(z);
        }
    }
    Some((module?, teeth.unwrap_or(0)))
}

/// The selected gears made to turn with the active one, by drivers — so
/// turning or animating it turns them, at the ratio their teeth give: the
/// other way round for an external pair, the same way for a pinion in a
/// ring gear; a rack slides along by the pinion's pitch circle. Where each
/// is now stays where it is.
pub struct GearLink;

impl Operator for GearLink {
    type Params = GearLinkParams;
    fn id() -> &'static str {
        "object.gear_link"
    }
    fn label() -> &'static str {
        "Link Gears"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_scene::driver::{Driver, DriverVariable};
        let d = doc.document();
        let lead = d
            .selection
            .active
            .ok_or_else(|| OpError::Failed("make the driving gear the active object".into()))?;
        let driven: Vec<ObjectId> = d
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| *id != lead)
            .collect();
        if driven.is_empty() {
            return Err(OpError::Failed(
                "select the gears to drive, then the driving one last".into(),
            ));
        }
        let name = |id: ObjectId| d.object(id).map(|o| o.name.clone()).unwrap_or_default();
        let lead_name = name(lead);
        let lead_uid = d.object_uid(lead).expect("an object");
        let lead_rot = d.object(lead).expect("an object").transform.rotation;
        let lead_angle = kreate_scene::animation::to_euler(lead_rot).z;
        let lead_x = d.object(lead).expect("an object").transform.translation.x;
        let is_rack = |n: &str| n.starts_with("Rack");
        let is_ring = |n: &str| n.starts_with("Ring gear");
        let mut links: Vec<(ObjectId, Driver)> = Vec::new();
        for id in driven {
            let other = name(id);
            let o = d.object(id).expect("an object");
            let here = kreate_scene::animation::to_euler(o.transform.rotation).z;
            let mut var = DriverVariable::new("a");
            var.target = lead_uid;
            let numbers = (gear_numbers(&lead_name), gear_numbers(&other));
            let driver = if is_rack(&other) {
                // The pinion turns, the rack slides: its pitch circle's arc.
                let (Some((m, z)), _) = numbers else {
                    return Err(OpError::Failed(format!(
                        "{lead_name} names no module and teeth"
                    )));
                };
                let r = m * 0.001 * z as f64 * 0.5;
                var.property = "rotation_euler".into();
                var.index = 2;
                let x0 = o.transform.translation.x;
                let mut dr = Driver::new("location", 0, format!("{x0} + {r} * (a - {lead_angle})"));
                dr.variables.push(var);
                dr
            } else if is_rack(&lead_name) {
                // The rack slides, the pinion turns.
                let (_, Some((m, z))) = numbers else {
                    return Err(OpError::Failed(format!(
                        "{other} names no module and teeth"
                    )));
                };
                let r = m * 0.001 * z as f64 * 0.5;
                var.property = "location".into();
                var.index = 0;
                let mut dr = Driver::new(
                    "rotation_euler",
                    2,
                    format!("{here} + (a - {lead_x}) / {r}"),
                );
                dr.variables.push(var);
                dr
            } else {
                let ratio = if params.ratio != 0.0 {
                    params.ratio
                } else {
                    match numbers {
                        (Some((_, za)), Some((_, zb))) if za > 0 && zb > 0 => {
                            // Inside a ring gear the two turn the same way.
                            let same = is_ring(&lead_name) || is_ring(&other);
                            let r = za as f64 / zb as f64;
                            if same {
                                r
                            } else {
                                -r
                            }
                        }
                        _ => {
                            return Err(OpError::Failed(format!(
                                "cannot tell the teeth of {lead_name} and {other}: give the ratio"
                            )))
                        }
                    }
                };
                var.property = "rotation_euler".into();
                var.index = 2;
                let mut dr = Driver::new(
                    "rotation_euler",
                    2,
                    format!("{here} + {ratio} * (a - {lead_angle})"),
                );
                dr.variables.push(var);
                dr
            };
            links.push((id, driver));
        }
        let d = doc.document_mut();
        for (id, driver) in links {
            let o = d.object_mut(id).expect("an object");
            // One link a channel: a gear linked again is relinked.
            o.drivers
                .retain(|x| !(x.property == driver.property && x.index == driver.index));
            o.drivers.push(driver);
        }
        Ok(OpResult::Finished)
    }
}

/// The selected faces of an edit mesh, by where they fall in its stored
/// order — the order `to_mesh_data` writes them in.
fn selected_face_indices(bm: &kreate_mesh::BMesh) -> Vec<u32> {
    bm.face_ids()
        .enumerate()
        .filter(|(_, f)| bm.is_face_selected(*f))
        .map(|(i, _)| i as u32)
        .collect()
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ShellParams {
    /// The wall's thickness, inside the part.
    #[param(min = 0.0001, soft_max = 0.02, unit = "length")]
    pub thickness: f64,
}

impl Default for ShellParams {
    fn default() -> Self {
        Self { thickness: 0.002 }
    }
}

/// The part hollowed: the selected faces opened, the rest given a wall
/// this thick on the inside — a box made a tray, a case made from a block.
pub struct Shell;

impl Operator for Shell {
    type Params = ShellParams;
    fn id() -> &'static str {
        "mesh.shell"
    }
    fn label() -> &'static str {
        "Shell"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        crate::ops::mesh::can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let done: usize = crate::ops::mesh::each_bmesh(doc, |bm| {
            let open = selected_face_indices(bm);
            if open.is_empty() {
                return Ok(0);
            }
            let shelled = mech::shell(&bm.to_mesh_data(), &open, params.thickness as f32);
            *bm = kreate_mesh::BMesh::from_mesh_data(&shelled);
            Ok(1)
        })?
        .into_iter()
        .sum();
        if done == 0 {
            return Err(OpError::Failed(
                "select the faces to open, then shell".into(),
            ));
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DraftParams {
    /// The draft angle: how far the walls lean in as they rise.
    #[param(min = 0.0, max = 0.7, unit = "angle")]
    pub angle: f64,
    /// Which way the part comes out of the mould.
    #[param(variants("z", "minus_z", "y", "minus_y", "x", "minus_x"))]
    pub pull: u32,
}

impl Default for DraftParams {
    fn default() -> Self {
        Self {
            angle: 1f64.to_radians(),
            pull: 0,
        }
    }
}

/// The selected faces given a draft for moulding: leaning in by the angle
/// as they rise along the pull, from the lowest of them, which stays.
pub struct Draft;

impl Operator for Draft {
    type Params = DraftParams;
    fn id() -> &'static str {
        "mesh.draft"
    }
    fn label() -> &'static str {
        "Draft"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        crate::ops::mesh::can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_math::Vec3;
        let pull =
            [Vec3::Z, -Vec3::Z, Vec3::Y, -Vec3::Y, Vec3::X, -Vec3::X][params.pull.min(5) as usize];
        let done: usize = crate::ops::mesh::each_bmesh(doc, |bm| {
            let faces = selected_face_indices(bm);
            if faces.is_empty() {
                return Ok(0);
            }
            let selected = bm.selected_faces();
            let mut m = bm.to_mesh_data();
            mech::draft(&mut m, &faces, pull, params.angle as f32);
            // Back with the same faces selected, to draft again or move on.
            let mut back = kreate_mesh::BMesh::from_mesh_data(&m);
            let ids: Vec<_> = back.face_ids().collect();
            for i in &faces {
                if let Some(f) = ids.get(*i as usize) {
                    back.set_face_selected(*f, true);
                }
            }
            back.flush_selection(kreate_mesh::bmesh::SelectMode::Face);
            let _ = selected;
            *bm = back;
            Ok(1)
        })?
        .into_iter()
        .sum();
        if done == 0 {
            return Err(OpError::Failed("select the walls to draft".into()));
        }
        Ok(OpResult::Finished)
    }
}

/// The largest region of a sketch, in its own plane: what a sweep carries
/// or a loft passes through. `None` for an object with no closed region.
fn sketch_outline(doc: &Document, id: ObjectId) -> Option<Vec<kreate_math::Vec2>> {
    kreate_scene::feature::sketch_regions(doc, id)
        .into_iter()
        .max_by(|a, b| {
            a.area()
                .abs()
                .partial_cmp(&b.area().abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.outline)
}

/// An object's lines as one path, in the world: a sketch's lines, or a
/// mesh's edges — the longest chain through them, and whether it closes.
fn object_path(doc: &Document, id: ObjectId) -> Option<(Vec<DVec3>, bool)> {
    let o = doc.object(id)?;
    let mesh = doc.mesh(o.mesh()?)?;
    let mut edges: Vec<[u32; 2]> = mesh.edges().to_vec();
    edges.extend(o.sketch_lines.iter().copied());
    let (points, closed) =
        kreate_mesh::modifiers::path_array::path_chain(mesh.positions(), &edges)?;
    let world = doc.world_transform(id);
    let points: Vec<DVec3> = points
        .iter()
        .map(|p| world.transform_point3(p.as_dvec3()))
        .collect();
    (points.len() >= 2).then_some((points, closed))
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SweepAlongParams {
    /// A round profile's radius, when no sketch is selected to give one.
    #[param(min = 0.0001, soft_max = 0.1, unit = "length")]
    pub radius: f64,
    /// A round profile's segments.
    #[param(min = 3, max = 128, unit = "count")]
    pub segments: u32,
    /// The profile's size at the path's end, against its start: a taper.
    #[param(min = 0.0, soft_max = 4.0)]
    pub scale_end: f64,
    /// Close the ends of an open path.
    pub caps: bool,
}

impl Default for SweepAlongParams {
    fn default() -> Self {
        Self {
            radius: 0.005,
            segments: 24,
            scale_end: 1.0,
            caps: true,
        }
    }
}

/// A profile carried along the active object's lines — a sketch's or a
/// mesh's edges: a pipe run, a handrail, a gasket in its groove. The profile
/// is another selected sketch's largest region, as drawn about that
/// sketch's origin, or else a circle.
pub struct SweepAlong;

impl Operator for SweepAlong {
    type Params = SweepAlongParams;
    fn id() -> &'static str {
        "object.sweep_along"
    }
    fn label() -> &'static str {
        "Sweep Along Path"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let path_id = d
            .selection
            .active
            .ok_or_else(|| OpError::Failed("make the path the active object".into()))?;
        let (path, closed) = object_path(d, path_id).ok_or_else(|| {
            OpError::Failed("the active object has no lines to sweep along".into())
        })?;
        let profile = match d
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| *id != path_id)
            .find_map(|id| sketch_outline(d, id))
        {
            Some(outline) => outline,
            None => kreate_mesh::primitives::circle_profile(params.radius as f32, params.segments),
        };
        let start = path[0];
        let local: Vec<kreate_math::Vec3> = path.iter().map(|p| (*p - start).as_vec3()).collect();
        let mesh = kreate_mesh::primitives::sweep(&kreate_mesh::primitives::Sweep {
            path: &local,
            profile: &profile,
            resolution: 1,
            closed,
            scale: (1.0, params.scale_end as f32),
            caps: params.caps,
        });
        if mesh.num_faces() == 0 || !mesh.validate().is_empty() {
            return Err(OpError::Failed(
                "the profile cannot be swept along this path".into(),
            ));
        }
        add_part(doc, "Sweep".into(), mesh, start, DQuat::IDENTITY);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LoftParams {}

/// A solid through the selected sketches' outlines, nearest to nearest:
/// a square base to a round top, a duct, a transition piece.
pub struct Loft;

impl Operator for Loft {
    type Params = LoftParams;
    fn id() -> &'static str {
        "object.loft"
    }
    fn label() -> &'static str {
        "Loft"
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let mut sections: Vec<Vec<DVec3>> = Vec::new();
        for id in d.selection.objects.iter().copied() {
            if let Some(outline) = sketch_outline(d, id) {
                let world = d.world_transform(id);
                sections.push(
                    outline
                        .iter()
                        .map(|p| world.transform_point3(DVec3::new(p.x as f64, p.y as f64, 0.0)))
                        .collect(),
                );
            }
        }
        if sections.len() < 2 {
            return Err(OpError::Failed(
                "select two or more sketches with closed outlines".into(),
            ));
        }
        // In order through space: from an end, each next the nearest left.
        let centre = |s: &[DVec3]| s.iter().copied().sum::<DVec3>() / s.len() as f64;
        let mid = sections.iter().map(|s| centre(s)).sum::<DVec3>() / sections.len() as f64;
        let first = (0..sections.len())
            .max_by(|a, b| {
                centre(&sections[*a])
                    .distance(mid)
                    .partial_cmp(&centre(&sections[*b]).distance(mid))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let mut order = vec![sections.swap_remove(first)];
        while !sections.is_empty() {
            let here = centre(order.last().expect("one"));
            let next = (0..sections.len())
                .min_by(|a, b| {
                    centre(&sections[*a])
                        .distance(here)
                        .partial_cmp(&centre(&sections[*b]).distance(here))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("one left");
            order.push(sections.swap_remove(next));
        }
        let start = centre(&order[0]);
        let local: Vec<Vec<kreate_math::Vec3>> = order
            .iter()
            .map(|s| s.iter().map(|p| (*p - start).as_vec3()).collect())
            .collect();
        let mesh = mech::loft(&local);
        if mesh.num_faces() == 0 || !mesh.validate().is_empty() {
            return Err(OpError::Failed("these outlines cannot be lofted".into()));
        }
        add_part(doc, "Loft".into(), mesh, start, DQuat::IDENTITY);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddSheetMetalParams {
    /// The base's flat between the bends, across.
    #[param(min = 0.001, soft_max = 1.0, unit = "length")]
    pub width: f64,
    /// The base's flat, along.
    #[param(min = 0.001, soft_max = 1.0, unit = "length")]
    pub length: f64,
    /// The sheet's gauge.
    #[param(min = 0.0001, soft_max = 0.01, unit = "length")]
    pub thickness: f64,
    /// The bends' inside radius.
    #[param(min = 0.0001, soft_max = 0.02, unit = "length")]
    pub bend_radius: f64,
    /// Where the neutral line lies through the thickness, from the inside.
    #[param(min = 0.0, max = 0.5)]
    pub k_factor: f64,
    /// The flanges' bend, from flat.
    #[param(min = 0.0, max = 3.0, unit = "angle")]
    pub angle: f64,
    /// The flanges' straight length past the bend.
    #[param(min = 0.0, soft_max = 0.2, unit = "length")]
    pub flange: f64,
    pub front: bool,
    pub right: bool,
    pub back: bool,
    pub left: bool,
    /// Laid out flat: the blank to cut before bending.
    pub flat: bool,
    pub at_cursor: bool,
    pub location: [f64; 3],
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

impl Default for AddSheetMetalParams {
    fn default() -> Self {
        Self {
            width: 0.1,
            length: 0.06,
            thickness: 0.0015,
            bend_radius: 0.0015,
            k_factor: 0.44,
            angle: std::f64::consts::FRAC_PI_2,
            flange: 0.02,
            front: true,
            right: true,
            back: true,
            left: true,
            flat: false,
            at_cursor: true,
            location: [0.0; 3],
            rotation: [0.0; 3],
        }
    }
}

/// A sheet metal part: a base plate with a flange bent up on any of its
/// sides, the corners relieved — a tray, a bracket, a cover. Flat lays it
/// out as the blank, each bend unrolled to its bend allowance, and names the
/// blank's size.
pub struct AddSheetMetal;

impl Operator for AddSheetMetal {
    type Params = AddSheetMetalParams;
    fn id() -> &'static str {
        "object.add_sheet_metal"
    }
    fn label() -> &'static str {
        "Add Sheet Metal"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let p = mech::SheetMetal {
            width: params.width as f32,
            length: params.length as f32,
            thickness: params.thickness as f32,
            radius: params.bend_radius as f32,
            k_factor: params.k_factor as f32,
            angle: params.angle as f32,
            height: params.flange as f32,
            flanges: [params.front, params.right, params.back, params.left],
            flat: params.flat,
        };
        let mesh = mech::sheet_metal(&p);
        if mesh.num_faces() == 0 || !mesh.validate().is_empty() {
            return Err(OpError::Failed(
                "these sizes make no sheet metal part".into(),
            ));
        }
        let (at, turn) = placement(
            doc.document(),
            params.at_cursor,
            params.location,
            params.rotation,
        );
        let mm = |v: f64| (v * 10000.0).round() / 10.0;
        let name = if params.flat {
            let (w, l) = p.flat_size();
            // i18n-exempt: a part's description by its sizes
            format!(
                "Sheet {} mm blank {}x{}",
                mm(params.thickness),
                mm(w as f64),
                mm(l as f64)
            )
        } else {
            // i18n-exempt: a part's description by its sizes
            format!(
                "Sheet {} mm {}x{}",
                mm(params.thickness),
                mm(params.width),
                mm(params.length)
            )
        };
        add_part(doc, name, mesh, at, turn);
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::modifier::{EvalCtx, ModifierCache};

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        doc: &mut Document,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(r, doc, &Context { mode: Mode::Object }, id, &p)
    }

    fn volume(doc: &Document, id: ObjectId) -> f64 {
        let mut cache = ModifierCache::new();
        let mesh = cache.evaluated(doc, id, &EvalCtx::still()).unwrap();
        kreate_mesh::measure::mass_properties(&mesh, |p| p).volume
    }

    /// A 40 × 40 × 10 mm plate, its top at z = 0.
    fn plate(doc: &mut Document) -> ObjectId {
        let mut m = kreate_mesh::primitives::cube(1.0);
        for p in m.positions_mut() {
            p.x *= 0.04;
            p.y *= 0.04;
            p.z = (p.z - 0.5) * 0.01;
        }
        let id = doc.add_mesh_object("Plate", m);
        doc.selection.add(id);
        doc.selection.active = Some(id);
        id
    }

    #[test]
    fn sheet_metal_is_added_folded_or_as_its_blank() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        run(&r, &mut h, &mut doc, "object.add_sheet_metal", &[]).unwrap();
        let folded = volume(&doc, doc.selection.active.unwrap());
        run(
            &r,
            &mut h,
            &mut doc,
            "object.add_sheet_metal",
            &[("flat", PropertyValue::Bool(true))],
        )
        .unwrap();
        let id = doc.selection.active.unwrap();
        let flat = volume(&doc, id);
        // K 0.44 puts the neutral line inside the middle: the blank a little
        // shorter than the metal the bend holds.
        assert!(
            flat < folded && flat > folded * 0.99,
            "{flat} against {folded}"
        );
        assert!(
            doc.object(id).unwrap().name.contains("blank"),
            "{}",
            doc.object(id).unwrap().name
        );
    }

    #[test]
    fn a_sketch_sweeps_along_a_path_and_two_sketches_loft() {
        use kreate_math::Vec3;
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        // A 10 mm square drawn about its sketch's origin.
        let square = |doc: &mut Document, z: f32, half: f32| {
            let pts: Vec<Vec3> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                .iter()
                .map(|(x, y)| Vec3::new(x * half, y * half, 0.0))
                .collect();
            let id = doc.add_mesh_object("Sketch", kreate_mesh::MeshData::from_faces(pts, &[]));
            let o = doc.object_mut(id).unwrap();
            o.sketch_lines = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
            o.transform.translation = kreate_math::DVec3::new(0.0, 0.0, z as f64);
            id
        };
        // A path 100 mm long up x, as a sketch's line.
        let path = doc.add_mesh_object(
            "Path",
            kreate_mesh::MeshData::from_faces(vec![Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)], &[]),
        );
        doc.object_mut(path).unwrap().sketch_lines = vec![[0, 1]];
        let profile = square(&mut doc, 0.0, 0.005);
        doc.selection.clear();
        doc.selection.add(profile);
        doc.selection.add(path);
        doc.selection.active = Some(path);
        run(&r, &mut h, &mut doc, "object.sweep_along", &[]).unwrap();
        let swept = doc.selection.active.unwrap();
        let v = volume(&doc, swept);
        assert!((v - 0.01 * 0.01 * 0.1).abs() < 1e-8, "{v}");
        // With no profile sketch, a round one.
        doc.selection.clear();
        doc.selection.add(path);
        doc.selection.active = Some(path);
        run(
            &r,
            &mut h,
            &mut doc,
            "object.sweep_along",
            &[("radius", PropertyValue::Float(0.01))],
        )
        .unwrap();
        let v = volume(&doc, doc.selection.active.unwrap());
        assert!(
            v > 0.9 * std::f64::consts::PI * 1e-4 * 0.1 && v < std::f64::consts::PI * 1e-4 * 0.1,
            "{v}"
        );
        // Two squares 20 mm apart, 20 and 10 mm wide: a frustum.
        let a = square(&mut doc, 0.0, 0.01);
        let b = square(&mut doc, 0.02, 0.005);
        doc.selection.clear();
        doc.selection.add(b);
        doc.selection.add(a);
        doc.selection.active = Some(a);
        run(&r, &mut h, &mut doc, "object.loft", &[]).unwrap();
        let v = volume(&doc, doc.selection.active.unwrap());
        let want = 0.02 / 3.0 * (4e-4 + 1e-4 + (4e-4f64 * 1e-4).sqrt());
        assert!((v - want).abs() < want * 1e-3, "{v} against {want}");
        // One sketch is not a loft.
        doc.selection.clear();
        doc.selection.add(a);
        assert!(run(&r, &mut h, &mut doc, "object.loft", &[]).is_err());
    }

    #[test]
    fn a_hole_takes_out_what_its_size_says() {
        let r = {
            let mut r = OperatorRegistry::new();
            crate::register_builtin(&mut r);
            r
        };
        let whole = 0.04 * 0.04 * 0.01;
        let prism = 48.0 / std::f64::consts::TAU * (std::f64::consts::TAU / 48.0).sin();
        let area = |d_mm: f64| std::f64::consts::PI * (d_mm * 0.0005).powi(2) * prism;
        for (kind, want) in [
            // Through, 6.6 mm across.
            (0u32, area(6.6) * 0.01),
            // 11 mm across 6.4 mm down, then 6.6.
            (1, area(11.0) * 0.0064 + area(6.6) * 0.0036),
            // The tap drill, 5 mm.
            (3, area(5.0) * 0.01),
        ] {
            let mut h = History::new();
            let mut doc = Document::new();
            let id = plate(&mut doc);
            run(
                &r,
                &mut h,
                &mut doc,
                "object.hole",
                &[("kind", PropertyValue::Enum(kind))],
            )
            .unwrap();
            assert_eq!(h.entries().len(), 1, "one step");
            let gone = whole - volume(&doc, id);
            assert!(
                (gone - want).abs() < want * 0.02,
                "kind {kind}: {gone} against {want}"
            );
        }
    }

    #[test]
    fn fasteners_are_named_by_their_designation() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        for kind in 0..4 {
            run(
                &r,
                &mut h,
                &mut doc,
                "object.add_fastener",
                &[
                    ("kind", PropertyValue::Enum(kind)),
                    ("thread", PropertyValue::Bool(false)),
                ],
            )
            .unwrap();
        }
        let mut names: Vec<String> = doc.objects().map(|(_, o)| o.name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "ISO 4017 M6x20",
                "ISO 4032 M6",
                "ISO 4762 M6x20",
                "ISO 7089 M6"
            ]
        );
    }

    #[test]
    fn interference_is_found_where_parts_overlap_and_nowhere_else() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        // Two 10 mm cubes overlapping by 2 mm, and a third well clear.
        let cube = || kreate_mesh::primitives::cube(0.01);
        let a = doc.add_mesh_object("A", cube());
        let b = doc.add_mesh_object("B", cube());
        let c = doc.add_mesh_object("C", cube());
        doc.object_mut(b).unwrap().transform.translation = DVec3::new(0.008, 0.0, 0.0);
        doc.object_mut(c).unwrap().transform.translation = DVec3::new(0.0, 0.05, 0.0);
        let _ = (a, c);
        run(&r, &mut h, &mut doc, "object.check_interference", &[]).unwrap();
        let found: Vec<String> = doc
            .objects()
            .filter(|(id, _)| doc.collection_of(*id) == doc.find_collection(INTERFERENCE))
            .map(|(_, o)| o.name.clone())
            .collect();
        assert_eq!(found.len(), 1, "{found:?}");
        // 2 × 10 × 10 mm: 200 mm³.
        assert!(found[0].starts_with("A x B (200.0"), "{}", found[0]);
        // Moved apart, checked again: the result goes, and it says so.
        doc.object_mut(b).unwrap().transform.translation = DVec3::new(0.02, 0.0, 0.0);
        assert!(run(&r, &mut h, &mut doc, "object.check_interference", &[]).is_err());
    }

    #[test]
    fn standard_parts_are_added_named_and_refused_when_they_cannot_be() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        for id in [
            "object.add_bearing",
            "object.add_shaft",
            "object.add_pulley",
            "object.add_sprocket",
            "object.add_spring",
        ] {
            run(&r, &mut h, &mut doc, id, &[]).unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let mut names: Vec<String> = doc
            .objects()
            .filter(|(_, o)| o.visible)
            .map(|(_, o)| o.name.clone())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "Bearing 6204 (20x47x14)",
                "Compression spring 1x10",
                "Key DIN 6885 A 6x6x25",
                "Pulley GT2 20T",
                "Shaft 20x100",
                "Sprocket 08B 17T",
            ]
        );
        // The keyway is cut: the shaft holds less than its plain turned
        // self.
        let shaft = doc
            .objects()
            .find(|(_, o)| o.name == "Shaft 20x100")
            .unwrap()
            .0;
        let plain = kreate_mesh::measure::mass_properties(
            &kreate_mesh::mech::shaft(0.02, 0.1, 0.005),
            |p| p,
        )
        .volume;
        assert!(volume(&doc, shaft) < plain - 1e-7);
        // A shaft too thin for any key, a spring too tight to wind.
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "object.add_shaft",
            &[("diameter", PropertyValue::Float(0.004))]
        )
        .is_err());
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "object.add_spring",
            &[("wire", PropertyValue::Float(0.006))]
        )
        .is_err());
    }

    #[test]
    fn linked_gears_turn_by_their_teeth_and_a_rack_slides_by_its_pinion() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let small = doc.add_mesh_object("Gear m2 z20", kreate_mesh::primitives::cube(0.01));
        let big = doc.add_mesh_object("Gear m2 z40", kreate_mesh::primitives::cube(0.01));
        let rack = doc.add_mesh_object("Rack m2 12", kreate_mesh::primitives::cube(0.01));
        doc.selection.add(big);
        doc.selection.add(rack);
        doc.selection.add(small);
        doc.selection.active = Some(small);
        run(&r, &mut h, &mut doc, "object.gear_link", &[]).unwrap();
        // The small gear a quarter turn: the big one an eighth back, the
        // rack along by a quarter of a 40 mm pitch circle.
        let quarter = std::f64::consts::FRAC_PI_2;
        doc.object_mut(small).unwrap().transform.rotation = DQuat::from_rotation_z(quarter);
        doc.drive();
        let angle =
            |id| kreate_scene::animation::to_euler(doc.object(id).unwrap().transform.rotation).z;
        assert!((angle(big) + quarter / 2.0).abs() < 1e-9, "{}", angle(big));
        let x = doc.object(rack).unwrap().transform.translation.x;
        assert!((x - 0.02 * quarter).abs() < 1e-9, "{x}");
    }

    #[test]
    fn a_tapped_hole_can_carry_its_thread() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let id = plate(&mut doc);
        run(
            &r,
            &mut h,
            &mut doc,
            "object.hole",
            &[
                ("kind", PropertyValue::Enum(3)),
                ("size", PropertyValue::Enum(2)),
                ("through", PropertyValue::Bool(false)),
                ("depth", PropertyValue::Float(0.006)),
                ("thread", PropertyValue::Bool(true)),
            ],
        )
        .unwrap();
        // The drill and the thread: two cuts.
        assert_eq!(doc.object(id).unwrap().modifiers.len(), 2);
        let v = volume(&doc, id);
        let drill_only = 0.04 * 0.04 * 0.01 - std::f64::consts::PI * 0.00125f64.powi(2) * 0.006;
        assert!(v < drill_only, "the thread took nothing: {v}");
    }
}
