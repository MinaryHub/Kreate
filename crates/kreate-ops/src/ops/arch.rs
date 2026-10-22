//! Architecture: walls, doors, windows, stairs, columns, railings, slabs and
//! roofs, each one operator whose sizes and details are its parameters.
//!
//! An element keeps the line that made it (`Object::generator`), so its
//! parameters stay open after it is made: the Properties editor's
//! Architecture panel reads them back, and `object.arch_set` makes the mesh
//! again from the edited line, in place — the object, its transform, its
//! materials and anything linked to its mesh stay as they are.
//!
//! The geometry is `kreate_mesh::arch`. The materials an element comes with
//! (a door's wood and brass, a window's glass) are found by name in the
//! document and made only the first time, so a house of twenty windows has
//! one glass, which is how it is edited.

use kreate_math::{DQuat, DVec3, SceneRgb};
use kreate_mesh::{arch, MeshData};
use kreate_reflect::{OpParams as _, PropertyMap};
use kreate_scene::document::{Material, MaterialId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddWall>();
    registry.register::<AddDoor>();
    registry.register::<AddWindow>();
    registry.register::<AddStairs>();
    registry.register::<AddColumn>();
    registry.register::<AddRailing>();
    registry.register::<AddSlab>();
    registry.register::<AddRoof>();
    registry.register::<ArchSet>();
}

/// The operators that make an architectural element, which `object.arch_set`
/// can make again.
pub const ELEMENTS: [&str; 8] = [
    "object.add_wall",
    "object.add_door",
    "object.add_window",
    "object.add_stairs",
    "object.add_column",
    "object.add_railing",
    "object.add_slab",
    "object.add_roof",
];

/// A material slot an element comes with: its name, and how a new one looks.
type Slot = (&'static str, fn(&mut Material));

/// What an element's parameters make: its name, its mesh, its slots.
struct Built {
    name: &'static str,
    mesh: MeshData,
    slots: Vec<Slot>,
}

/// A material by name: the document's, or a new one looking like `look`.
fn material(doc: &mut Document, name: &str, look: fn(&mut Material)) -> MaterialId {
    if let Some((id, _)) = doc.materials().find(|(_, m)| m.name == name) {
        return id;
    }
    let mut m = Material {
        name: name.into(),
        ..Default::default()
    };
    look(&mut m);
    doc.add_material(m)
}

fn painted(m: &mut Material) {
    m.base_color = SceneRgb::new(0.8, 0.79, 0.76);
    m.roughness = 0.6;
}
fn wood(m: &mut Material) {
    m.base_color = SceneRgb::new(0.24, 0.12, 0.05);
    m.roughness = 0.45;
}
fn metal(m: &mut Material) {
    m.base_color = SceneRgb::new(0.75, 0.62, 0.38);
    m.metallic = 1.0;
    m.roughness = 0.3;
}
fn glass(m: &mut Material) {
    m.base_color = SceneRgb::new(0.9, 0.95, 1.0);
    m.roughness = 0.02;
    m.transmission = 1.0;
    m.ior = 1.5;
}
fn concrete(m: &mut Material) {
    m.base_color = SceneRgb::new(0.42, 0.41, 0.39);
    m.roughness = 0.85;
}
fn stone(m: &mut Material) {
    m.base_color = SceneRgb::new(0.6, 0.57, 0.52);
    m.roughness = 0.7;
}
fn steel(m: &mut Material) {
    m.base_color = SceneRgb::new(0.12, 0.12, 0.13);
    m.metallic = 1.0;
    m.roughness = 0.4;
}
fn tiles(m: &mut Material) {
    m.base_color = SceneRgb::new(0.3, 0.08, 0.04);
    m.roughness = 0.65;
}

/// The line that makes the element again: its parameters, less where it
/// stands, which is the object's own.
fn generator_line(op_id: &str, map: &PropertyMap) -> String {
    let mut kept = PropertyMap::new();
    for (name, value) in map.iter() {
        if name != "location" && name != "rotation" {
            kept.set(name, value.clone());
        }
    }
    kept.to_script_call(op_id)
}

/// Puts `built` in as a new object at `location`, turned by `rotation`,
/// keeping the line that made it; it is selected.
fn place(
    doc: &mut DocMut<'_>,
    built: Built,
    generator: String,
    location: [f64; 3],
    rotation: [f64; 3],
) -> Result<OpResult, OpError> {
    let angles = DVec3::from_array(rotation);
    let turn = if angles.is_finite() {
        kreate_scene::animation::from_euler(angles)
    } else {
        DQuat::IDENTITY
    };
    let d = doc.document_mut();
    let materials: Vec<Option<MaterialId>> = built
        .slots
        .iter()
        .map(|(n, look)| Some(material(d, n, *look)))
        .collect();
    let object_name = d.unique_object_name(built.name);
    let id = d.add_mesh_object(object_name, built.mesh);
    let mesh_id = d.object(id).and_then(|o| o.mesh()).expect("a mesh");
    d.set_mesh_slots(mesh_id, materials);
    let o = d.object_mut(id).expect("just added");
    o.transform.translation = DVec3::from_array(location);
    o.transform.rotation = turn;
    o.generator = Some(generator);
    d.selection.clear();
    d.selection.add(id);
    d.selection.active = Some(id);
    Ok(OpResult::Finished)
}

/// Sizes must be positive and finite, or there is nothing to build.
fn sizes(values: &[(&str, f64)]) -> Result<(), OpError> {
    for (name, v) in values {
        if !(*v > 0.0 && v.is_finite()) {
            return Err(OpError::Failed(format!("the {name} must be above zero")));
        }
    }
    Ok(())
}

fn made(name: &'static str, mesh: MeshData, slots: Vec<Slot>) -> Result<Built, OpError> {
    if mesh.num_faces() == 0 {
        return Err(OpError::Failed(format!(
            "these sizes make no {}",
            name.to_lowercase()
        )));
    }
    Ok(Built { name, mesh, slots })
}

macro_rules! element {
    (
        $(#[$doc:meta])*
        $op:ident, $params:ident, $id:literal, $label:literal,
        { $( $(#[$fdoc:meta])* $field:ident : $ty:ty = $default:expr ),* $(,)? },
        |$p:ident| $body:block
    ) => {
        #[derive(Clone, Debug, PartialEq, OpParams)]
        pub struct $params {
            $( $(#[$fdoc])* pub $field: $ty, )*
            /// Where it stands: the middle of its foot.
            #[param(unit = "length")]
            pub location: [f64; 3],
            /// How it is turned, as Euler XYZ; unturned it runs along X and
            /// faces −Y.
            #[param(unit = "angle")]
            pub rotation: [f64; 3],
        }

        impl Default for $params {
            fn default() -> Self {
                Self {
                    $( $field: $default, )*
                    location: [0.0; 3],
                    rotation: [0.0; 3],
                }
            }
        }

        $(#[$doc])*
        pub struct $op;

        impl $op {
            fn build($p: &$params) -> Result<Built, OpError> $body
        }

        impl Operator for $op {
            type Params = $params;
            fn id() -> &'static str {
                $id
            }
            fn label() -> &'static str {
                $label
            }
            fn execute(
                params: &Self::Params,
                doc: &mut DocMut<'_>,
                _ctx: &Context,
            ) -> Result<OpResult, OpError> {
                let built = Self::build(params)?;
                let line = generator_line($id, &params.to_map());
                place(doc, built, line, params.location, params.rotation)
            }
        }
    };
}

element!(
    /// A wall: a straight run, standing on the floor.
    AddWall, AddWallParams, "object.add_wall", "Add Wall",
    {
        #[param(min = 0.01, soft_max = 30.0, unit = "length")]
        length: f64 = 4.0,
        #[param(min = 0.01, soft_max = 10.0, unit = "length")]
        height: f64 = 2.7,
        #[param(min = 0.01, soft_max = 1.0, unit = "length")]
        thickness: f64 = 0.2,
    },
    |p| {
        sizes(&[("length", p.length), ("height", p.height), ("thickness", p.thickness)])?;
        let m = arch::wall(p.length as f32, p.height as f32, p.thickness as f32);
        made("Wall", m, vec![("Wall", painted)])
    }
);

element!(
    /// A door in its frame: a panelled leaf with a lever each side. Cut the
    /// wall's opening with a boolean, the opening's size `width` × `height`.
    AddDoor, AddDoorParams, "object.add_door", "Add Door",
    {
        /// The opening's width.
        #[param(min = 0.1, soft_max = 3.0, unit = "length")]
        width: f64 = 0.9,
        /// The opening's height.
        #[param(min = 0.1, soft_max = 4.0, unit = "length")]
        height: f64 = 2.1,
        /// The frame's depth: the wall's thickness.
        #[param(min = 0.02, soft_max = 1.0, unit = "length")]
        depth: f64 = 0.2,
        /// The frame's width round the opening.
        #[param(min = 0.01, soft_max = 0.3, unit = "length")]
        frame: f64 = 0.06,
        /// The leaf's thickness.
        #[param(min = 0.01, soft_max = 0.1, unit = "length")]
        leaf_thickness: f64 = 0.04,
        /// Raised panels on each face; zero for a flush door.
        #[param(min = 0, max = 8, unit = "count")]
        panels: u32 = 2,
        /// The handles' height above the floor.
        #[param(min = 0.05, soft_max = 2.0, unit = "length")]
        handle_height: f64 = 1.0,
        /// Which side the handles are on, seen from the front.
        #[param(variants("right", "left"))]
        handle_side: u32 = 0,
    },
    |p| {
        sizes(&[("width", p.width), ("height", p.height), ("depth", p.depth), ("frame", p.frame)])?;
        let m = arch::door(&arch::Door {
            width: p.width as f32,
            height: p.height as f32,
            depth: p.depth as f32,
            frame: p.frame as f32,
            leaf: p.leaf_thickness as f32,
            panels: p.panels,
            handle_height: p.handle_height as f32,
            handle_right: p.handle_side == 0,
        });
        made(
            "Door",
            m,
            vec![
                (arch::DOOR_SLOTS[0], wood),
                (arch::DOOR_SLOTS[1], painted),
                (arch::DOOR_SLOTS[2], metal),
            ],
        )
    }
);

element!(
    /// A window: a frame divided into panes of glass, on a sill.
    AddWindow, AddWindowParams, "object.add_window", "Add Window",
    {
        #[param(min = 0.1, soft_max = 5.0, unit = "length")]
        width: f64 = 1.2,
        #[param(min = 0.1, soft_max = 4.0, unit = "length")]
        height: f64 = 1.4,
        /// The frame's depth.
        #[param(min = 0.02, soft_max = 1.0, unit = "length")]
        depth: f64 = 0.15,
        /// The frame's width.
        #[param(min = 0.01, soft_max = 0.3, unit = "length")]
        frame: f64 = 0.06,
        /// How high its bottom is above the floor.
        #[param(min = 0.0, soft_max = 3.0, unit = "length")]
        sill_height: f64 = 0.9,
        /// Panes across.
        #[param(min = 1, max = 12, unit = "count")]
        columns: u32 = 2,
        /// Panes up.
        #[param(min = 1, max = 12, unit = "count")]
        rows: u32 = 1,
        /// The bars between the panes.
        #[param(min = 0.005, soft_max = 0.2, unit = "length")]
        bar_width: f64 = 0.036,
        #[param(min = 0.001, soft_max = 0.05, unit = "length")]
        glass_thickness: f64 = 0.008,
        /// How far the sill stands out in front; zero for none.
        #[param(min = 0.0, soft_max = 0.3, unit = "length")]
        sill_depth: f64 = 0.04,
    },
    |p| {
        sizes(&[("width", p.width), ("height", p.height), ("depth", p.depth), ("frame", p.frame)])?;
        if 2.0 * p.frame >= p.width.min(p.height) {
            return Err(OpError::Failed("the frame leaves no room for glass".into()));
        }
        let m = arch::window(&arch::Window {
            width: p.width as f32,
            height: p.height as f32,
            depth: p.depth as f32,
            frame: p.frame as f32,
            sill_height: p.sill_height.max(0.0) as f32,
            columns: p.columns,
            rows: p.rows,
            bar: p.bar_width as f32,
            glass: p.glass_thickness as f32,
            sill: p.sill_depth.max(0.0) as f32,
        });
        made(
            "Window",
            m,
            vec![(arch::WINDOW_SLOTS[0], painted), (arch::WINDOW_SLOTS[1], glass)],
        )
    }
);

element!(
    /// A straight flight of stairs, climbing along +Y.
    AddStairs, AddStairsParams, "object.add_stairs", "Add Stairs",
    {
        #[param(min = 0.1, soft_max = 5.0, unit = "length")]
        width: f64 = 1.0,
        /// The height it climbs: floor to floor.
        #[param(min = 0.05, soft_max = 10.0, unit = "length")]
        height: f64 = 2.7,
        /// How far it runs.
        #[param(min = 0.05, soft_max = 20.0, unit = "length")]
        length: f64 = 3.6,
        /// How many steps; fifteen of 18 cm suit a storey.
        #[param(min = 1, max = 200, unit = "count")]
        steps: u32 = 15,
        /// A solid stack under the treads, or open treads.
        solid: bool = true,
        /// An open tread's thickness.
        #[param(min = 0.005, soft_max = 0.2, unit = "length")]
        tread_thickness: f64 = 0.04,
    },
    |p| {
        sizes(&[("width", p.width), ("height", p.height), ("length", p.length)])?;
        let m = arch::stairs(&arch::Stairs {
            width: p.width as f32,
            height: p.height as f32,
            length: p.length as f32,
            steps: p.steps,
            solid: p.solid,
            tread: p.tread_thickness as f32,
        });
        made("Stairs", m, vec![("Concrete", concrete)])
    }
);

element!(
    /// A column, round or square, with a square base and capital.
    AddColumn, AddColumnParams, "object.add_column", "Add Column",
    {
        #[param(min = 0.02, soft_max = 2.0, unit = "length")]
        diameter: f64 = 0.3,
        #[param(min = 0.1, soft_max = 20.0, unit = "length")]
        height: f64 = 2.7,
        /// Round, or square.
        round: bool = true,
        /// A square base and capital.
        base: bool = true,
        /// How much wider than the shaft they are.
        #[param(min = 1.0, soft_max = 3.0)]
        base_scale: f64 = 1.35,
        /// Sides round a round shaft.
        #[param(min = 3, max = 256, unit = "count")]
        segments: u32 = 32,
    },
    |p| {
        sizes(&[("diameter", p.diameter), ("height", p.height)])?;
        let m = arch::column(&arch::Column {
            diameter: p.diameter as f32,
            height: p.height as f32,
            round: p.round,
            base: p.base,
            base_scale: p.base_scale as f32,
            segments: p.segments,
        });
        made("Column", m, vec![("Stone", stone)])
    }
);

element!(
    /// A railing: posts at its ends, a top and a bottom rail, balusters.
    AddRailing, AddRailingParams, "object.add_railing", "Add Railing",
    {
        #[param(min = 0.2, soft_max = 30.0, unit = "length")]
        length: f64 = 3.0,
        #[param(min = 0.2, soft_max = 3.0, unit = "length")]
        height: f64 = 1.0,
        /// About how far apart the balusters stand.
        #[param(min = 0.03, soft_max = 1.0, unit = "length")]
        spacing: f64 = 0.12,
        #[param(min = 0.005, soft_max = 0.3, unit = "length")]
        post_width: f64 = 0.06,
        #[param(min = 0.002, soft_max = 0.1, unit = "length")]
        baluster_width: f64 = 0.02,
        /// The top rail's width.
        #[param(min = 0.01, soft_max = 0.3, unit = "length")]
        rail_width: f64 = 0.07,
    },
    |p| {
        sizes(&[("length", p.length), ("height", p.height), ("spacing", p.spacing)])?;
        let m = arch::railing(&arch::Railing {
            length: p.length as f32,
            height: p.height as f32,
            spacing: p.spacing as f32,
            post: p.post_width as f32,
            baluster: p.baluster_width as f32,
            rail: p.rail_width as f32,
        });
        made("Railing", m, vec![("Railing Steel", steel)])
    }
);

element!(
    /// A floor slab, its top at its origin: the level it is the floor of.
    AddSlab, AddSlabParams, "object.add_slab", "Add Floor Slab",
    {
        #[param(min = 0.1, soft_max = 50.0, unit = "length")]
        width: f64 = 4.0,
        #[param(min = 0.1, soft_max = 50.0, unit = "length")]
        length: f64 = 4.0,
        #[param(min = 0.01, soft_max = 1.0, unit = "length")]
        thickness: f64 = 0.2,
    },
    |p| {
        sizes(&[("width", p.width), ("length", p.length), ("thickness", p.thickness)])?;
        let m = arch::slab(p.width as f32, p.length as f32, p.thickness as f32);
        made("Floor Slab", m, vec![("Concrete", concrete)])
    }
);

element!(
    /// A roof over a plan — gable, hip, shed or flat — its eaves at its
    /// origin's height: set it on the walls' tops.
    AddRoof, AddRoofParams, "object.add_roof", "Add Roof",
    {
        /// Its shape.
        #[param(variants("gable", "hip", "shed", "flat"))]
        kind: u32 = 0,
        /// The plan across the slopes.
        #[param(min = 0.1, soft_max = 50.0, unit = "length")]
        width: f64 = 6.0,
        /// The plan along the ridge.
        #[param(min = 0.1, soft_max = 50.0, unit = "length")]
        length: f64 = 8.0,
        /// From the eaves line up to the ridge.
        #[param(min = 0.01, soft_max = 10.0, unit = "length")]
        rise: f64 = 2.0,
        #[param(min = 0.01, soft_max = 1.0, unit = "length")]
        thickness: f64 = 0.15,
        /// How far it reaches past the walls.
        #[param(min = 0.0, soft_max = 2.0, unit = "length")]
        overhang: f64 = 0.3,
    },
    |p| {
        sizes(&[("width", p.width), ("length", p.length), ("rise", p.rise), ("thickness", p.thickness)])?;
        let kind = match p.kind {
            1 => arch::RoofKind::Hip,
            2 => arch::RoofKind::Shed,
            3 => arch::RoofKind::Flat,
            _ => arch::RoofKind::Gable,
        };
        let m = arch::roof(&arch::Roof {
            kind,
            width: p.width as f32,
            length: p.length as f32,
            rise: p.rise as f32,
            thickness: p.thickness as f32,
            overhang: p.overhang.max(0.0) as f32,
        });
        made("Roof", m, vec![("Roof Tiles", tiles)])
    }
);

/// What `op_id`'s parameters make, for `object.arch_set`.
fn build_line(op_id: &str, map: &PropertyMap) -> Result<Built, OpError> {
    match op_id {
        "object.add_wall" => AddWall::build(&AddWallParams::from_map(map)),
        "object.add_door" => AddDoor::build(&AddDoorParams::from_map(map)),
        "object.add_window" => AddWindow::build(&AddWindowParams::from_map(map)),
        "object.add_stairs" => AddStairs::build(&AddStairsParams::from_map(map)),
        "object.add_column" => AddColumn::build(&AddColumnParams::from_map(map)),
        "object.add_railing" => AddRailing::build(&AddRailingParams::from_map(map)),
        "object.add_slab" => AddSlab::build(&AddSlabParams::from_map(map)),
        "object.add_roof" => AddRoof::build(&AddRoofParams::from_map(map)),
        other => Err(OpError::Failed(format!(
            "`{other}` is not an architectural element"
        ))),
    }
}

/// An element's line read back: its operator and parameters.
pub fn read_line(line: &str) -> Result<crate::script::Call, OpError> {
    let mut registry = crate::operator::OperatorRegistry::new();
    register(&mut registry);
    let call =
        crate::script::parse_call(&registry, line).map_err(|e| OpError::Failed(e.to_string()))?;
    if !ELEMENTS.contains(&call.op_id) {
        return Err(OpError::Failed(format!(
            "`{}` is not an architectural element",
            call.op_id
        )));
    }
    Ok(call)
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ArchSetParams {
    /// The element's line with its new parameters, as its generator keeps
    /// it: `kreate.ops.object.add_door(width=1.0, panels=4)`.
    pub call: String,
}

/// Makes the active architectural element again from its line, changed:
/// its mesh in place, the object, its transform and its materials kept.
pub struct ArchSet;

impl Operator for ArchSet {
    type Params = ArchSetParams;
    fn id() -> &'static str {
        "object.arch_set"
    }
    fn label() -> &'static str {
        "Set Architecture"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|id| doc.object(id))
            .is_some_and(|o| o.generator.is_some() && o.mesh().is_some())
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let id = d
            .selection
            .active
            .filter(|id| d.object(*id).is_some_and(|o| o.generator.is_some()))
            .ok_or_else(|| OpError::Failed("select an architectural element".into()))?;
        let mesh_id = d
            .object(id)
            .and_then(|o| o.mesh())
            .ok_or_else(|| OpError::Failed("the element has no mesh".into()))?;
        let call = read_line(&params.call)?;
        let built = build_line(call.op_id, &call.params)?;
        let line = generator_line(call.op_id, &call.params);
        let old_slots = d.mesh_slots(mesh_id).to_vec();
        let d = doc.document_mut();
        // The materials it has keep their places, so a door repainted stays
        // repainted; slots it did not have get the element's own.
        let slots: Vec<Option<MaterialId>> = built
            .slots
            .iter()
            .enumerate()
            .map(|(i, (n, look))| {
                old_slots
                    .get(i)
                    .copied()
                    .flatten()
                    .or_else(|| Some(material(d, n, *look)))
            })
            .collect();
        *d.mesh_mut(mesh_id).expect("the element's mesh") = built.mesh;
        d.set_mesh_slots(mesh_id, slots);
        d.object_mut(id).expect("the element").generator = Some(line);
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyValue};

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

    fn setup() -> (OperatorRegistry, History, Document) {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        (r, History::new(), Document::new())
    }

    /// Every element is added as a closed solid, selected, with its
    /// materials and the line that made it; two windows share one glass.
    #[test]
    fn every_element_is_added_with_its_materials() {
        let (r, mut h, mut doc) = setup();
        for id in ELEMENTS.iter().chain(["object.add_window"].iter()) {
            run(&r, &mut h, &mut doc, id, &[]).unwrap_or_else(|e| panic!("{id}: {e:?}"));
            let o = doc.object(doc.selection.active.unwrap()).unwrap();
            let mesh_id = o.mesh().unwrap();
            let mp = kreate_mesh::measure::mass_properties(doc.mesh(mesh_id).unwrap(), |p| p);
            assert!(mp.closed && mp.volume > 0.0, "{id}");
            assert!(doc.mesh_slots(mesh_id).iter().all(|s| s.is_some()), "{id}");
            let line = o.generator.as_deref().expect("its line kept");
            assert!(line.contains(id) && !line.contains("location"), "{line}");
        }
        assert_eq!(doc.object_count(), 9);
        let glass = doc.materials().filter(|(_, m)| m.name == "Glass").count();
        assert_eq!(glass, 1, "one glass for both windows");
    }

    /// Its sizes are its parameters: a wider door, and nonsense refused.
    #[test]
    fn sizes_shape_the_element_and_nonsense_is_refused() {
        let (r, mut h, mut doc) = setup();
        run(
            &r,
            &mut h,
            &mut doc,
            "object.add_door",
            &[
                ("width", PropertyValue::Float(1.2)),
                ("location", PropertyValue::Vec3([2.0, 0.0, 0.0])),
            ],
        )
        .unwrap();
        let id = doc.selection.active.unwrap();
        let (lo, hi) = doc.object_world_bounds(id).unwrap();
        assert!((hi.x - lo.x - 1.32).abs() < 1e-4 && (lo.x + hi.x - 4.0).abs() < 1e-4);
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "object.add_window",
            &[("frame", PropertyValue::Float(0.8))],
        )
        .is_err());
    }

    /// A door made, moved and repainted, then made wider with four panels
    /// from its kept line: the same object, still where it was moved to,
    /// still repainted, its mesh the new one — and one undo puts it back.
    #[test]
    fn an_element_is_changed_after_from_its_line() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "object.add_door", &[]).unwrap();
        let id = doc.selection.active.unwrap();
        doc.object_mut(id).unwrap().transform.translation = DVec3::new(5.0, 0.0, 0.0);
        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        let painted = doc.add_material(Material {
            name: "Green Paint".into(),
            ..Default::default()
        });
        let mut slots = doc.mesh_slots(mesh_id).to_vec();
        slots[0] = Some(painted);
        doc.set_mesh_slots(mesh_id, slots);
        let before = doc.mesh(mesh_id).unwrap().num_faces();

        let line = doc.object(id).unwrap().generator.clone().unwrap();
        let mut call = read_line(&line).unwrap();
        call.params.set("width", PropertyValue::Float(1.2));
        call.params.set("panels", PropertyValue::Int(4));
        let changed = call.params.to_script_call(call.op_id);
        run(
            &r,
            &mut h,
            &mut doc,
            "object.arch_set",
            &[("call", PropertyValue::Text(changed))],
        )
        .unwrap();
        assert_eq!(doc.object_count(), 1, "the same object");
        let o = doc.object(id).unwrap();
        assert_eq!(o.mesh(), Some(mesh_id), "its mesh, made again");
        assert_eq!(o.transform.translation, DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(doc.mesh_slots(mesh_id)[0], Some(painted), "its paint kept");
        let (lo, hi) = doc.object_world_bounds(id).unwrap();
        assert!((hi.x - lo.x - 1.32).abs() < 1e-4, "{lo} {hi}");
        assert!(
            doc.mesh(mesh_id).unwrap().num_faces() > before,
            "more panels"
        );
        let kept = read_line(o.generator.as_deref().unwrap()).unwrap();
        assert_eq!(kept.params.get("panels"), Some(&PropertyValue::Int(4)));

        assert!(h.undo(&mut doc));
        let (lo, hi) = doc.object_world_bounds(id).unwrap();
        assert!((hi.x - lo.x - 1.02).abs() < 1e-4, "undone: {lo} {hi}");
    }

    /// A line naming something else is refused, and nothing changes.
    #[test]
    fn a_line_that_is_not_an_element_is_refused() {
        let (r, mut h, mut doc) = setup();
        run(&r, &mut h, &mut doc, "object.add_wall", &[]).unwrap();
        let err = run(
            &r,
            &mut h,
            &mut doc,
            "object.arch_set",
            &[(
                "call",
                PropertyValue::Text("kreate.ops.object.delete()".into()),
            )],
        );
        assert!(err.is_err());
        assert_eq!(doc.object_count(), 1);
    }
}
