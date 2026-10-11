//! URDF: a robot as ROS describes it — links, the joints between them, and
//! the meshes and shapes each link looks like.
//!
//! # What a robot becomes
//!
//! A tree of empties, which is what a URDF is. Each link is an empty named
//! after it; each joint is an empty between its parent link and its child,
//! where the joint's origin puts it — **turned so that the joint's axis is
//! its local Z**. So a revolute joint moves by turning its empty about its
//! own Z, and a prismatic one by sliding it along its own Z: the robot is
//! posed in Kreate the way it moves, one rotation per joint, and the child
//! link sits back in the frame the file gave it. A fixed joint is an empty
//! that is never meant to move.
//!
//! A link's visuals are children of its empty, each where its own origin
//! says: a mesh file (STL, OBJ, COLLADA, glTF, PLY) read and put in place, or
//! a box, a cylinder or a sphere made here. A visual's material — a colour by
//! name or given in place — goes on a mesh that brings none of its own.
//!
//! URDF is Z up and in metres, as Kreate is, so nothing is converted; a mesh
//! file is read in its own axes and units where it says what they are (glTF,
//! COLLADA) and taken as metres and Z up where it does not (STL, OBJ), as ROS
//! takes them.
//!
//! # What is not kept, and said
//!
//! Collision shapes (a robot is looked at here, not simulated), inertia,
//! joint limits, mimic joints and the `<gazebo>` / `<transmission>` blocks.
//! A mesh that cannot be found is a note, not a failure: the rest of the
//! robot still comes in.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use kreate_math::{Quat, Vec3};

use crate::ir::{
    ImportMaterial, ImportMesh, ImportNode, ImportScene, ImportTransform, LossyNote, SourceMetadata,
};
use crate::xml::{self, Element};

/// Why a robot description could not be read at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UrdfError(pub String);

impl std::fmt::Display for UrdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "URDF: {}", self.0)
    }
}

impl std::error::Error for UrdfError {}

/// Reads a mesh file a visual names, already resolved to a path: the part,
/// in Kreate's axes and metres.
pub type LoadPart<'a> = dyn Fn(&Path) -> Result<ImportScene, String> + 'a;

/// A robot from its description. `base` is the folder the file is in, which
/// relative mesh paths and `package://` paths are found from; `load` reads a
/// mesh file.
pub fn parse_urdf(text: &str, base: &Path, load: &LoadPart<'_>) -> Result<ImportScene, UrdfError> {
    let root = xml::parse(text).map_err(|e| UrdfError(e.to_string()))?;
    if root.name != "robot" {
        return Err(UrdfError(format!(
            "the file is a `<{}>`, not a `<robot>`",
            root.name
        )));
    }
    let mut scene = ImportScene::new(SourceMetadata::new("urdf"));
    scene.source.generator = root.attr("name").map(|n| format!("robot `{n}`"));

    // Materials by name, as the robot declares them at its top.
    let mut named: HashMap<String, usize> = HashMap::new();
    for m in root.all("material") {
        if let Some(material) = material_of(m) {
            let name = material.name.clone();
            named.insert(name, scene.materials.len());
            scene.materials.push(material);
        }
    }

    let links: Vec<&Element> = root.all("link").collect();
    let joints: Vec<&Element> = root.all("joint").collect();
    if links.is_empty() {
        return Err(UrdfError("the robot has no links".into()));
    }
    // Which joint each link hangs from.
    let mut parent_joint: HashMap<&str, &Element> = HashMap::new();
    for j in &joints {
        if let Some(child) = j.child("child").and_then(|c| c.attr("link")) {
            parent_joint.insert(child, j);
        }
    }
    let by_name: HashMap<&str, &Element> = links
        .iter()
        .filter_map(|l| Some((l.attr("name")?, *l)))
        .collect();

    // Links placed from the roots down, so a parent is always made first.
    let mut link_node: HashMap<String, usize> = HashMap::new();
    let mut parts: HashMap<PathBuf, ImportScene> = HashMap::new();
    let mut queue: Vec<&str> = links
        .iter()
        .filter_map(|l| l.attr("name"))
        .filter(|n| !parent_joint.contains_key(n))
        .collect();
    if queue.is_empty() {
        return Err(UrdfError(
            "every link hangs from a joint: the tree has no root".into(),
        ));
    }
    let mut placed = 0;
    while let Some(name) = queue.pop() {
        if link_node.contains_key(name) || placed > links.len() {
            continue;
        }
        placed += 1;
        let Some(link) = by_name.get(name) else {
            scene.source.notes.push(LossyNote::Unsupported {
                what: format!("a joint names a link `{name}` the robot does not have"),
            });
            continue;
        };
        // The joint it hangs from, between its parent's empty and its own.
        let mut parent = None;
        let mut link_transform = ImportTransform::default();
        if let Some(j) = parent_joint.get(name) {
            let parent_link = j.child("parent").and_then(|p| p.attr("link")).unwrap_or("");
            let mut joint = ImportNode::new(j.attr("name").unwrap_or("joint"));
            joint.parent = link_node.get(parent_link).copied();
            let origin = origin_of(j);
            let kind = j.attr("type").unwrap_or("fixed");
            let moves = matches!(kind, "revolute" | "continuous" | "prismatic");
            let axis = j
                .child("axis")
                .and_then(|a| a.attr("xyz"))
                .map(vec3)
                .and_then(|a| a.try_normalize())
                .unwrap_or(Vec3::X);
            // The joint's axis turned onto its local Z, and turned back for
            // the link: rotating the joint about Z is moving the joint.
            let align = if moves {
                Quat::from_rotation_arc(Vec3::Z, axis)
            } else {
                Quat::IDENTITY
            };
            joint.transform = ImportTransform {
                translation: origin.translation,
                rotation: (origin.rotation * align).normalize(),
                scale: Vec3::ONE,
            };
            link_transform.rotation = align.inverse();
            if kind == "floating" || kind == "planar" {
                scene.source.notes.push(LossyNote::Unsupported {
                    what: format!(
                        "joint `{}` is {kind}: it is an empty to move by hand",
                        joint.name
                    ),
                });
            }
            parent = Some(scene.nodes.len());
            scene.nodes.push(joint);
        }
        let mut node = ImportNode::new(name);
        node.parent = parent;
        node.transform = link_transform;
        let at = scene.nodes.len();
        scene.nodes.push(node);
        link_node.insert(name.to_string(), at);

        for (k, visual) in link.all("visual").enumerate() {
            add_visual(
                &mut scene, visual, name, k, at, &named, base, load, &mut parts,
            );
        }
        if link.child("collision").is_some() && link.child("visual").is_none() {
            scene.source.notes.push(LossyNote::Unsupported {
                what: format!("link `{name}` has only a collision shape, which is not shown"),
            });
        }
        // Its children next.
        for j in &joints {
            if j.child("parent").and_then(|p| p.attr("link")) == Some(name) {
                if let Some(child) = j.child("child").and_then(|c| c.attr("link")) {
                    queue.push(child);
                }
            }
        }
    }
    if root.child("transmission").is_some() || root.child("gazebo").is_some() {
        scene.source.notes.push(LossyNote::Unsupported {
            what: "transmissions and simulator settings are not read".into(),
        });
    }
    if joints.iter().any(|j| j.child("limit").is_some()) {
        scene.source.notes.push(LossyNote::Unsupported {
            what: "joint limits are not kept: each joint turns freely about its empty's Z".into(),
        });
    }
    Ok(scene)
}

/// A file's robot, its meshes read from beside it.
pub fn parse_urdf_file(path: &Path, load: &LoadPart<'_>) -> Result<ImportScene, UrdfError> {
    let text = std::fs::read_to_string(path).map_err(|e| UrdfError(format!("cannot read: {e}")))?;
    let base = path.parent().unwrap_or(Path::new("."));
    parse_urdf(&text, base, load)
}

#[allow(clippy::too_many_arguments)]
fn add_visual(
    scene: &mut ImportScene,
    visual: &Element,
    link: &str,
    index: usize,
    under: usize,
    named: &HashMap<String, usize>,
    base: &Path,
    load: &LoadPart<'_>,
    parts: &mut HashMap<PathBuf, ImportScene>,
) {
    let Some(geometry) = visual.child("geometry") else {
        return;
    };
    let mut node = ImportNode::new(visual.attr("name").map(str::to_string).unwrap_or_else(|| {
        if index == 0 {
            format!("{link}_visual")
        } else {
            format!("{link}_visual_{index}")
        }
    }));
    node.parent = Some(under);
    node.transform = origin_of(visual);
    // Its material: one by name, or one given in place.
    let material = visual.child("material").and_then(|m| {
        if let Some(own) =
            material_of(m).filter(|_| m.child("color").is_some() || m.child("texture").is_some())
        {
            scene.materials.push(own);
            Some(scene.materials.len() - 1)
        } else {
            m.attr("name").and_then(|n| named.get(n).copied())
        }
    });
    let shape = |mesh: kreate_mesh::MeshData, name: &str| -> ImportScene {
        let mut part = ImportScene::new(SourceMetadata::new("urdf"));
        part.meshes.push(ImportMesh {
            name: name.to_string(),
            positions: mesh.positions().to_vec(),
            faces: mesh.faces().map(<[u32]>::to_vec).collect(),
            ..ImportMesh::default()
        });
        part
    };
    let part = if let Some(b) = geometry.child("box") {
        let size = b.attr("size").map(vec3).unwrap_or(Vec3::ONE);
        let mut m = kreate_mesh::primitives::cube(1.0);
        for p in m.positions_mut() {
            *p *= size;
        }
        Some(shape(m, "box"))
    } else if let Some(c) = geometry.child("cylinder") {
        let r = number(c.attr("radius"), 0.5);
        let l = number(c.attr("length"), 1.0);
        Some(shape(
            kreate_mesh::primitives::cylinder(r, l, 32),
            "cylinder",
        ))
    } else if let Some(s) = geometry.child("sphere") {
        let r = number(s.attr("radius"), 0.5);
        Some(shape(
            kreate_mesh::primitives::uv_sphere(r, 32, 16),
            "sphere",
        ))
    } else if let Some(m) = geometry.child("mesh") {
        let filename = m.attr("filename").unwrap_or("");
        if let Some(scale) = m.attr("scale") {
            node.transform.scale = vec3(scale);
        }
        match resolve(filename, base) {
            Some(path) => {
                let read = match parts.get(&path) {
                    Some(p) => Ok(p.clone()),
                    None => load(&path).inspect(|p| {
                        parts.insert(path.clone(), p.clone());
                    }),
                };
                match read {
                    Ok(p) => Some(p),
                    Err(why) => {
                        scene.source.notes.push(LossyNote::Unsupported {
                            what: format!("link `{link}`: `{filename}` could not be read: {why}"),
                        });
                        None
                    }
                }
            }
            None => {
                scene.source.notes.push(LossyNote::Unsupported {
                    what: format!("link `{link}`: `{filename}` was not found"),
                });
                None
            }
        }
    } else {
        None
    };
    let at = scene.nodes.len();
    scene.nodes.push(node);
    let Some(mut part) = part else { return };
    // The visual's colour, on what has none of its own.
    if let Some(material) = material {
        let local = part.materials.len();
        part.materials.push(scene.materials[material].clone());
        for mesh in &mut part.meshes {
            let bare = mesh
                .face_materials
                .as_ref()
                .is_none_or(|f| f.iter().all(|m| *m < 0));
            if bare {
                mesh.face_materials = Some(vec![local as i32; mesh.faces.len()]);
            }
        }
    }
    scene.graft(part, Some(at));
}

/// Where a mesh file is: beside the description, under a ROS package found
/// by walking up from it (or on `ROS_PACKAGE_PATH`), or a `file://` path.
pub fn resolve(filename: &str, base: &Path) -> Option<PathBuf> {
    if let Some(path) = filename.strip_prefix("file://") {
        let p = PathBuf::from(path);
        return p.is_file().then_some(p);
    }
    if let Some(rest) = filename.strip_prefix("package://") {
        let (package, inside) = rest.split_once('/')?;
        let mut roots: Vec<PathBuf> = base.ancestors().map(Path::to_path_buf).collect();
        if let Some(paths) = std::env::var_os("ROS_PACKAGE_PATH") {
            roots.extend(std::env::split_paths(&paths));
        }
        for root in roots {
            // The package as a folder here, or this folder the package.
            let candidates = [
                root.join(package).join(inside),
                root.file_name()
                    .filter(|n| n.to_string_lossy() == package)
                    .map(|_| root.join(inside))
                    .unwrap_or_default(),
            ];
            if let Some(found) = candidates.into_iter().find(|c| c.is_file()) {
                return Some(found);
            }
        }
        // A copy that kept the files and lost the package: beside the file.
        let beside = base.join(inside);
        return beside.is_file().then_some(beside);
    }
    let p = base.join(filename);
    if p.is_file() {
        return Some(p);
    }
    // Case differs on a case-sensitive disk: `pelvis.stl` for `pelvis.STL`.
    let dir = p.parent()?;
    let want = p.file_name()?.to_string_lossy().to_lowercase();
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|e| {
            e.file_name()
                .is_some_and(|n| n.to_string_lossy().to_lowercase() == want)
        })
}

/// An element's `<origin xyz rpy>`: roll, pitch and yaw about the fixed X,
/// Y and Z axes, in that order.
fn origin_of(e: &Element) -> ImportTransform {
    let Some(o) = e.child("origin") else {
        return ImportTransform::default();
    };
    let xyz = o.attr("xyz").map(vec3).unwrap_or(Vec3::ZERO);
    let rpy = o.attr("rpy").map(vec3).unwrap_or(Vec3::ZERO);
    ImportTransform {
        translation: xyz,
        rotation: Quat::from_euler(kreate_math::glam::EulerRot::ZYX, rpy.z, rpy.y, rpy.x),
        scale: Vec3::ONE,
    }
}

fn material_of(m: &Element) -> Option<ImportMaterial> {
    let mut out = ImportMaterial {
        name: m.attr("name").unwrap_or("material").to_string(),
        ..ImportMaterial::default()
    };
    if let Some(rgba) = m.child("color").and_then(|c| c.attr("rgba")) {
        let v = xml::numbers(rgba);
        if v.len() >= 3 {
            out.base_color = [
                v[0] as f32,
                v[1] as f32,
                v[2] as f32,
                v.get(3).copied().unwrap_or(1.0) as f32,
            ];
        }
    }
    Some(out)
}

fn vec3(text: &str) -> Vec3 {
    let v = xml::numbers(text);
    Vec3::new(
        v.first().copied().unwrap_or(0.0) as f32,
        v.get(1).copied().unwrap_or(0.0) as f32,
        v.get(2).copied().unwrap_or(0.0) as f32,
    )
}

fn number(text: Option<&str>, default: f32) -> f32 {
    text.and_then(|t| t.trim().parse().ok()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARM: &str = r#"<robot name="arm">
      <material name="blue"><color rgba="0 0 1 1"/></material>
      <link name="base">
        <visual><geometry><box size="1 2 0.5"/></geometry><material name="blue"/></visual>
      </link>
      <link name="upper">
        <visual>
          <origin xyz="0 0 0.5"/>
          <geometry><cylinder radius="0.1" length="1"/></geometry>
          <material name="red"><color rgba="1 0 0 1"/></material>
        </visual>
        <collision><geometry><cylinder radius="0.1" length="1"/></geometry></collision>
      </link>
      <link name="hand"><visual><geometry><mesh filename="package://arm_pkg/meshes/hand.stl" scale="0.001 0.001 0.001"/></geometry></visual></link>
      <joint name="shoulder" type="revolute">
        <parent link="base"/><child link="upper"/>
        <origin xyz="0 0 0.25" rpy="0 0 1.5707963"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" effort="10" velocity="1"/>
      </joint>
      <joint name="wrist" type="fixed">
        <parent link="upper"/><child link="hand"/>
        <origin xyz="0 0 1"/>
      </joint>
    </robot>"#;

    fn world(scene: &ImportScene, name: &str) -> kreate_math::DAffine3 {
        let at = scene.nodes.iter().position(|n| n.name == name).unwrap();
        scene.world_transform(at).unwrap()
    }

    #[test]
    fn links_hang_from_joints_whose_z_is_their_axis() {
        let dir = std::env::temp_dir().join(format!("kreate-urdf-{}", std::process::id()));
        let meshes = dir.join("arm_pkg").join("meshes");
        std::fs::create_dir_all(&meshes).unwrap();
        std::fs::write(meshes.join("hand.stl"), b"solid x\nendsolid x\n").unwrap();
        let base = dir.join("arm_pkg").join("urdf");
        std::fs::create_dir_all(&base).unwrap();
        let asked = std::cell::RefCell::new(Vec::new());
        let load = |p: &Path| -> Result<ImportScene, String> {
            asked.borrow_mut().push(p.to_path_buf());
            let mut part = ImportScene::new(SourceMetadata::new("stl"));
            part.meshes.push(ImportMesh {
                name: "hand".into(),
                positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
                faces: vec![vec![0, 1, 2]],
                ..ImportMesh::default()
            });
            Ok(part)
        };
        let scene = parse_urdf(ARM, &base, &load).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            asked.borrow().len(),
            1,
            "the package was found by walking up"
        );

        // The shoulder's local Z is its axis (the file's Y, turned 90° about
        // Z by its rpy: world -X).
        let shoulder = world(&scene, "shoulder");
        let z = shoulder.transform_vector3(kreate_math::DVec3::Z);
        assert!(
            (z - kreate_math::DVec3::new(-1.0, 0.0, 0.0)).length() < 1e-5,
            "{z}"
        );
        assert!((shoulder.translation.z - 0.25).abs() < 1e-6);
        // And the link below it is back in the frame the file gave it.
        let upper = world(&scene, "upper");
        let x = upper.transform_vector3(kreate_math::DVec3::X);
        assert!((x - kreate_math::DVec3::Y).length() < 1e-5, "{x}");
        // The hand a metre up the arm, its mesh at a thousandth.
        let hand = world(&scene, "hand");
        assert!((hand.translation.z - 1.25).abs() < 1e-5);
        let visual = world(&scene, "hand_visual");
        assert!((visual.transform_vector3(kreate_math::DVec3::X).length() - 0.001).abs() < 1e-7);

        // Shapes and colours.
        assert_eq!(scene.meshes.len(), 3);
        let blue = scene
            .materials
            .iter()
            .position(|m| m.name == "blue")
            .unwrap() as i32;
        let box_mesh = scene.meshes.iter().find(|m| m.name == "box").unwrap();
        assert!(box_mesh
            .face_materials
            .as_ref()
            .unwrap()
            .iter()
            .all(|f| scene.materials[*f as usize].base_color
                == scene.materials[blue as usize].base_color));
        let extent = box_mesh
            .positions
            .iter()
            .fold(Vec3::ZERO, |a, p| a.max(p.abs()));
        assert!((extent - Vec3::new(0.5, 1.0, 0.25)).length() < 1e-6);
        assert!(scene
            .materials
            .iter()
            .any(|m| m.name == "red" && m.base_color == [1.0, 0.0, 0.0, 1.0]));
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what.contains("limits"))));
    }

    #[test]
    fn what_is_not_a_robot_is_said() {
        let load = |_: &Path| -> Result<ImportScene, String> { Err("no".into()) };
        assert!(parse_urdf("<model/>", Path::new("."), &load).is_err());
        assert!(parse_urdf("<robot name='x'/>", Path::new("."), &load).is_err());
        // A missing mesh is a note; the robot still comes.
        let scene = parse_urdf(
            r#"<robot name="r"><link name="a"><visual><geometry><mesh filename="gone.stl"/></geometry></visual></link></robot>"#,
            Path::new("/nonexistent"),
            &load,
        )
        .unwrap();
        assert_eq!(scene.nodes.len(), 2);
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what.contains("not found"))));
    }
}
