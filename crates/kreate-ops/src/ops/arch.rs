//! Architecture: walls, doors, windows, stairs, columns, railings, slabs and
//! roofs, each one operator whose sizes are its parameters — so the last
//! operation panel shapes it, and the history replays it.
//!
//! The geometry is `kreate_mesh::arch`. The materials an element comes with
//! (a door's wood and brass, a window's glass) are found by name in the
//! document and made only the first time, so a house of twenty windows has
//! one glass, which is how it is edited.

use kreate_math::{DQuat, DVec3, SceneRgb};
use kreate_mesh::{arch, MeshData};
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
}

/// A material slot an element comes with: its name, and how a new one looks.
type Slot = (&'static str, fn(&mut Material));

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

/// Puts `mesh` in as a new object named `name` at `location`, turned by
/// `rotation`, its slots holding the named materials; it is selected.
fn place(
    doc: &mut DocMut<'_>,
    name: &str,
    mesh: MeshData,
    slots: &[Slot],
    location: [f64; 3],
    rotation: [f64; 3],
) -> Result<OpResult, OpError> {
    if mesh.num_faces() == 0 {
        return Err(OpError::Failed(format!(
            "these sizes make no {}",
            name.to_lowercase()
        )));
    }
    let angles = DVec3::from_array(rotation);
    let turn = if angles.is_finite() {
        kreate_scene::animation::from_euler(angles)
    } else {
        DQuat::IDENTITY
    };
    let d = doc.document_mut();
    let materials: Vec<Option<MaterialId>> = slots
        .iter()
        .map(|(n, look)| Some(material(d, n, *look)))
        .collect();
    let object_name = d.unique_object_name(name);
    let id = d.add_mesh_object(object_name, mesh);
    let mesh_id = d.object(id).and_then(|o| o.mesh()).expect("a mesh");
    d.set_mesh_slots(mesh_id, materials);
    let o = d.object_mut(id).expect("just added");
    o.transform.translation = DVec3::from_array(location);
    o.transform.rotation = turn;
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

macro_rules! element {
    (
        $(#[$doc:meta])*
        $op:ident, $params:ident, $id:literal, $label:literal,
        { $( $(#[$fdoc:meta])* $field:ident : $ty:ty = $default:expr ),* $(,)? },
        |$p:ident, $d:ident| $body:block
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
                let $p = params;
                let $d = doc;
                $body
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
    |p, d| {
        sizes(&[("length", p.length), ("height", p.height), ("thickness", p.thickness)])?;
        let m = arch::wall(p.length as f32, p.height as f32, p.thickness as f32);
        place(d, "Wall", m, &[("Wall", painted)], p.location, p.rotation)
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
    },
    |p, d| {
        sizes(&[("width", p.width), ("height", p.height), ("depth", p.depth), ("frame", p.frame)])?;
        let m = arch::door(p.width as f32, p.height as f32, p.depth as f32, p.frame as f32);
        let slots: [Slot; 3] = [
            (arch::DOOR_SLOTS[0], wood),
            (arch::DOOR_SLOTS[1], painted),
            (arch::DOOR_SLOTS[2], metal),
        ];
        place(d, "Door", m, &slots, p.location, p.rotation)
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
    },
    |p, d| {
        sizes(&[("width", p.width), ("height", p.height), ("depth", p.depth), ("frame", p.frame)])?;
        if 2.0 * p.frame >= p.width.min(p.height) {
            return Err(OpError::Failed("the frame leaves no room for glass".into()));
        }
        let m = arch::window(
            p.width as f32,
            p.height as f32,
            p.depth as f32,
            p.frame as f32,
            p.sill_height.max(0.0) as f32,
            p.columns,
            p.rows,
        );
        let slots: [Slot; 2] =
            [(arch::WINDOW_SLOTS[0], painted), (arch::WINDOW_SLOTS[1], glass)];
        place(d, "Window", m, &slots, p.location, p.rotation)
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
    },
    |p, d| {
        sizes(&[("width", p.width), ("height", p.height), ("length", p.length)])?;
        let m = arch::stairs(p.width as f32, p.height as f32, p.length as f32, p.steps);
        place(d, "Stairs", m, &[("Concrete", concrete)], p.location, p.rotation)
    }
);

element!(
    /// A column with a square base and capital, round or square.
    AddColumn, AddColumnParams, "object.add_column", "Add Column",
    {
        #[param(min = 0.02, soft_max = 2.0, unit = "length")]
        diameter: f64 = 0.3,
        #[param(min = 0.1, soft_max = 20.0, unit = "length")]
        height: f64 = 2.7,
        /// Round, or square.
        round: bool = true,
    },
    |p, d| {
        sizes(&[("diameter", p.diameter), ("height", p.height)])?;
        let m = arch::column(p.diameter as f32, p.height as f32, p.round);
        place(d, "Column", m, &[("Stone", stone)], p.location, p.rotation)
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
    },
    |p, d| {
        sizes(&[("length", p.length), ("height", p.height), ("spacing", p.spacing)])?;
        let m = arch::railing(p.length as f32, p.height as f32, p.spacing as f32);
        place(d, "Railing", m, &[("Railing Steel", steel)], p.location, p.rotation)
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
    |p, d| {
        sizes(&[("width", p.width), ("length", p.length), ("thickness", p.thickness)])?;
        let m = arch::slab(p.width as f32, p.length as f32, p.thickness as f32);
        place(d, "Floor Slab", m, &[("Concrete", concrete)], p.location, p.rotation)
    }
);

element!(
    /// A gable roof over a plan, its eaves at its origin's height: set it on
    /// the walls' tops.
    AddRoof, AddRoofParams, "object.add_roof", "Add Roof",
    {
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
    |p, d| {
        sizes(&[("width", p.width), ("length", p.length), ("rise", p.rise), ("thickness", p.thickness)])?;
        let m = arch::roof(
            p.width as f32,
            p.length as f32,
            p.rise as f32,
            p.thickness as f32,
            p.overhang.max(0.0) as f32,
        );
        place(d, "Roof", m, &[("Roof Tiles", tiles)], p.location, p.rotation)
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap, PropertyValue};

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

    /// Every element is added as a closed solid, selected, with its
    /// materials; two windows share one glass.
    #[test]
    fn every_element_is_added_with_its_materials() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        for id in [
            "object.add_wall",
            "object.add_door",
            "object.add_window",
            "object.add_window",
            "object.add_stairs",
            "object.add_column",
            "object.add_railing",
            "object.add_slab",
            "object.add_roof",
        ] {
            run(&r, &mut h, &mut doc, id, &[]).unwrap_or_else(|e| panic!("{id}: {e:?}"));
            let o = doc.object(doc.selection.active.unwrap()).unwrap();
            let mesh_id = o.mesh().unwrap();
            let mp = kreate_mesh::measure::mass_properties(doc.mesh(mesh_id).unwrap(), |p| p);
            assert!(mp.closed && mp.volume > 0.0, "{id}");
            assert!(doc.mesh_slots(mesh_id).iter().all(|s| s.is_some()), "{id}");
        }
        assert_eq!(doc.object_count(), 9);
        let glass = doc.materials().filter(|(_, m)| m.name == "Glass").count();
        assert_eq!(glass, 1, "one glass for both windows");
        assert!(doc.objects().any(|(_, o)| o.name == "Window.001"));
    }

    /// Its sizes are its parameters: a wider door, and nonsense refused.
    #[test]
    fn sizes_shape_the_element_and_nonsense_is_refused() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
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
}
