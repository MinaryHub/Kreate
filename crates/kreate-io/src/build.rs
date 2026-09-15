//! Turning a read scene into a document.
//!
//! The step that did not exist: an [`ImportScene`] became a list of
//! `(name, MeshData)` pairs, every one of them placed at the origin, and a
//! light could not be imported at all because a light *is* a placement and
//! there was nowhere to put one.
//!
//! Here rather than in the application, because it is the other half of the
//! fidelity contract §5.7 asks for: what a file said and what the document ends
//! up holding are compared in tests, and a conversion living in the shell could
//! not be.

use crate::ir::{ImportLightKind, ImportNodeData, ImportScene};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::Document;

/// What was lost or guessed while building the document.
///
/// Returned rather than logged, for the same reason [`crate::ir::LossyNote`] is:
/// a fidelity contract that reports into a log is a contract nothing can test.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildNote {
    /// A node named a parent that is not in the file, or a chain that loops.
    /// The node is placed at the root with its own transform.
    BrokenParent { node: String },
    /// A node pointed at a mesh, light or camera the file does not contain.
    MissingData { node: String },
    /// A spot's cone came through; its blend was derived from the two angles
    /// glTF writes, which is not the same parameterisation.
    ConeBlend { light: String },
}

/// Adds everything in `scene` to `doc`, placed as the file placed it.
///
/// Returns the objects it made, in node order, so a caller can select what it
/// just imported.
pub fn build_into(scene: &ImportScene, doc: &mut Document) -> (Vec<ObjectId>, Vec<BuildNote>) {
    let mut notes = Vec::new();

    // The datablocks first, because a node refers to one by index and a
    // document refers to one by id.
    let meshes: Vec<_> = {
        let mut out = Vec::with_capacity(scene.meshes.len());
        let (converted, _) = scene.clone().into_mesh_data();
        for (_, mesh) in converted {
            out.push(doc.add_mesh(mesh));
        }
        out
    };

    let lights: Vec<_> = scene
        .lights
        .iter()
        .map(|light| {
            let kind = match light.kind {
                ImportLightKind::Point => kreate_scene::LightKind::Point {
                    // The file says nothing about how big the bulb is, and a
                    // source of no size casts a shadow nothing real casts. The
                    // datablock's own default is what a lamp looks like.
                    radius: match kreate_scene::LightKind::BULB {
                        kreate_scene::LightKind::Point { radius } => radius,
                        _ => 0.0,
                    },
                },
                ImportLightKind::Directional => kreate_scene::LightKind::SUN,
                ImportLightKind::Spot { inner, outer } => {
                    notes.push(BuildNote::ConeBlend {
                        light: light.name.clone(),
                    });
                    kreate_scene::LightKind::Spot {
                        radius: 0.05,
                        // glTF writes half-angles; the datablock stores the
                        // full cone, which is what a cone is called everywhere
                        // a person meets one.
                        cone_angle: (outer * 2.0) as f64,
                        // And a blend as the fraction of the cone that fades,
                        // which glTF expresses as a second angle instead. The
                        // two parameterisations carry the same shape and the
                        // conversion is exact in both directions.
                        blend: if outer > 0.0 {
                            ((outer - inner) / outer).clamp(0.0, 1.0) as f64
                        } else {
                            0.0
                        },
                    }
                }
            };
            let mut out = kreate_scene::Light::of_kind(light.name.clone(), kind);
            out.intensity = light.intensity as f64;
            // Linear on both sides: glTF's colour is a tint in linear space and
            // so is the datablock's, so this is a copy and not a decode.
            out.color = kreate_math::SceneRgb::new(light.color[0], light.color[1], light.color[2]);
            doc.add_light(out)
        })
        .collect();

    let cameras: Vec<_> = scene
        .cameras
        .iter()
        .map(|camera| {
            let mut out = kreate_scene::Camera::new(camera.name.clone());
            match (camera.fov_y, camera.ortho_width) {
                (Some(fov), _) => {
                    out.projection = kreate_scene::Projection::Perspective;
                    // The file says an angle and the datablock stores a lens.
                    // The sensor is the datablock's own — a file that writes a
                    // field of view is not saying anything about a sensor, and
                    // the focal length that follows is the one that gives that
                    // angle on the sensor the document has.
                    out.focal_length_mm =
                        (out.sensor_height_mm * 0.5) / (fov as f64 * 0.5).tan();
                }
                (None, Some(width)) => {
                    out.projection = kreate_scene::Projection::Orthographic;
                    out.ortho_scale = (width * scene.source.unit_scale) as f64;
                }
                (None, None) => {}
            }
            out.clip_near = (camera.clip_near * scene.source.unit_scale) as f64;
            if let Some(far) = camera.clip_far {
                out.clip_far = (far * scene.source.unit_scale) as f64;
            }
            doc.add_camera(out)
        })
        .collect();

    // A file with no hierarchy at all — OBJ, STL — still has meshes, and they
    // belong at the origin because that is where the file put them. Inventing a
    // node each would claim a structure the file does not have.
    if scene.nodes.is_empty() {
        let made = meshes
            .iter()
            .zip(&scene.meshes)
            .map(|(id, mesh)| {
                doc.add_object(Object::new(mesh.name.clone(), ObjectData::Mesh(*id)))
            })
            .collect();
        return (made, notes);
    }

    let made: Vec<ObjectId> = scene
        .nodes
        .iter()
        .map(|node| {
            let data = match node.data {
                ImportNodeData::Mesh(i) => match meshes.get(i) {
                    Some(id) => ObjectData::Mesh(*id),
                    None => {
                        notes.push(BuildNote::MissingData {
                            node: node.name.clone(),
                        });
                        ObjectData::Empty
                    }
                },
                ImportNodeData::Light(i) => match lights.get(i) {
                    Some(id) => ObjectData::Light(*id),
                    None => {
                        notes.push(BuildNote::MissingData {
                            node: node.name.clone(),
                        });
                        ObjectData::Empty
                    }
                },
                ImportNodeData::Camera(i) => match cameras.get(i) {
                    Some(id) => ObjectData::Camera(*id),
                    None => {
                        notes.push(BuildNote::MissingData {
                            node: node.name.clone(),
                        });
                        ObjectData::Empty
                    }
                },
                ImportNodeData::Empty => ObjectData::Empty,
            };
            // Placed at the origin for now. The transforms go on *after* the
            // parents, and the order is load-bearing: `set_parent` remembers
            // where the object was so that reparenting does not move it, which
            // is right when a user drags one thing onto another and wrong here.
            // A file's child transform is relative to its parent already.
            //
            // Parenting everything while it is still at the origin makes that
            // memory an identity, and then the transforms mean what the file
            // meant.
            doc.add_object(Object::new(node.name.clone(), data))
        })
        .collect();

    for (index, node) in scene.nodes.iter().enumerate() {
        let Some(parent) = node.parent else { continue };
        let (Some(child_id), Some(parent_id)) = (made.get(index), made.get(parent)) else {
            notes.push(BuildNote::BrokenParent {
                node: node.name.clone(),
            });
            continue;
        };
        // `set_parent` refuses a cycle, which a file can express and this
        // cannot fix. Refused means the node stays at the root with its own
        // transform, which is the honest partial answer.
        if doc.set_parent(*child_id, Some(*parent_id)).is_err() {
            notes.push(BuildNote::BrokenParent {
                node: node.name.clone(),
            });
        }
    }

    // And now the placements, with every parent link already in place.
    for (index, node) in scene.nodes.iter().enumerate() {
        let Some(id) = made.get(index) else { continue };
        if let Some(object) = doc.object_mut(*id) {
            object.transform = to_transform(node.transform.to_kreate(&scene.source));
        }
    }

    (made, notes)
}

/// A transform in the shape the document stores.
fn to_transform(affine: kreate_math::DAffine3) -> kreate_math::DTransform {
    let (scale, rotation, translation) = affine.to_scale_rotation_translation();
    kreate_math::DTransform {
        translation,
        rotation,
        scale,
    }
}
