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

use crate::ir::{
    ExportNode, ExportScene, ImportCamera, ImportImage, ImportLight, ImportLightKind,
    ImportMaterial, ImportNodeData, ImportScene, LossyNote,
};
use kreate_scene::document::{ImageId, MaterialId, Object, ObjectData, ObjectId};
use kreate_scene::{ColorSpace, Document, MaterialMap};

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
    /// A material named a texture that is not in the file, or one that could
    /// not be read. The slot is left empty.
    MissingImage { material: String },
}

/// Adds everything in `scene` to `doc`, placed as the file placed it.
///
/// Returns the objects it made, in node order, so a caller can select what it
/// just imported.
pub fn build_into(scene: &ImportScene, doc: &mut Document) -> (Vec<ObjectId>, Vec<BuildNote>) {
    let mut notes = Vec::new();

    // The datablocks first, because a node refers to one by index and a
    // document refers to one by id.
    let materials = build_materials(scene, doc, &mut notes);

    // A face in the file names a material in the *file's* table; a face in the
    // document names a slot in its *mesh's* list (§7.5). So each mesh gets the
    // materials it actually uses as its slots, in the order of their file
    // numbers, and its faces are renumbered into that list. Handing every mesh
    // the whole table would work and would give a one-material prop fifty
    // empty-looking slots in a file with fifty materials.
    let mut working = scene.clone();
    let mut slot_lists: Vec<Vec<Option<MaterialId>>> = Vec::with_capacity(working.meshes.len());
    for mesh in &mut working.meshes {
        let Some(faces) = &mut mesh.face_materials else {
            slot_lists.push(Vec::new());
            continue;
        };
        let mut used: Vec<i32> = faces.iter().copied().filter(|m| *m >= 0).collect();
        used.sort_unstable();
        used.dedup();
        for face in faces.iter_mut() {
            *face = used.binary_search(face).map(|slot| slot as i32).unwrap_or(-1);
        }
        slot_lists.push(
            used.iter()
                .map(|m| materials.get(*m as usize).copied())
                .collect(),
        );
    }
    let meshes: Vec<_> = {
        let mut out = Vec::with_capacity(working.meshes.len());
        let (converted, _) = working.into_mesh_data();
        for ((_, mesh), slots) in converted.into_iter().zip(slot_lists) {
            let id = doc.add_mesh(mesh);
            if !slots.is_empty() {
                doc.set_mesh_slots(id, slots);
            }
            out.push(id);
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

/// The file's materials as datablocks, in the file's order, with their images.
///
/// An image becomes one datablock per *space* it is used in: glTF lets one
/// picture be both a base colour and a roughness map, and the document
/// declares a space per image (§2.1), so the two uses are two datablocks. The
/// usual case — each image in one slot — is one each.
fn build_materials(
    scene: &ImportScene,
    doc: &mut Document,
    notes: &mut Vec<BuildNote>,
) -> Vec<MaterialId> {
    let mut made: std::collections::HashMap<(usize, ColorSpace), ImageId> =
        std::collections::HashMap::new();
    scene
        .materials
        .iter()
        .map(|m| {
            let mut image_for = |index: Option<usize>, space: ColorSpace| -> Option<ImageId> {
                let index = index?;
                if let Some(id) = made.get(&(index, space)) {
                    return Some(*id);
                }
                let built = scene.images.get(index).and_then(|image| {
                    if image.pixels.is_empty() {
                        return None;
                    }
                    crate::image_import::image_from_rgba8(
                        &image.name,
                        image.width,
                        image.height,
                        &image.pixels,
                        space,
                    )
                    .ok()
                });
                match built {
                    Some(image) => {
                        let id = doc.add_image(image);
                        made.insert((index, space), id);
                        Some(id)
                    }
                    None => {
                        notes.push(BuildNote::MissingImage {
                            material: m.name.clone(),
                        });
                        None
                    }
                }
            };
            let base_color_image =
                image_for(m.base_color_texture, MaterialMap::BaseColor.color_space());
            let metallic_roughness_image = image_for(
                m.metallic_roughness_texture,
                MaterialMap::MetallicRoughness.color_space(),
            );
            let normal_image = image_for(m.normal_texture, MaterialMap::Normal.color_space());
            doc.add_material(kreate_scene::Material {
                name: m.name.clone(),
                // Linear on both sides — glTF's factor is linear, and so is the
                // datablock's colour — so this is a copy, not a decode.
                base_color: kreate_math::SceneRgb::new(
                    m.base_color[0],
                    m.base_color[1],
                    m.base_color[2],
                ),
                alpha: m.base_color[3],
                metallic: m.metallic,
                roughness: m.roughness,
                base_color_image,
                metallic_roughness_image,
                normal_image,
                normal_strength: m.normal_scale,
                ..Default::default()
            })
        })
        .collect()
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

/// Turns a document into a scene an exporter can write.
///
/// The inverse of [`build_into`], and the half that did not exist: an export
/// took a list of `(name, &MeshData)` pairs, so a lit scene could arrive and
/// could not leave. §5.7's fidelity contract only held in one direction, which
/// is not a contract — a round trip either closes or it is a one-way door with
/// a hopeful name.
///
/// Notes are [`LossyNote`]s rather than [`BuildNote`]s because this is the
/// export direction and they are what an exporter already returns: a caller
/// writing a file gets one list to show, not two.
pub fn scene_from_document(doc: &Document) -> (ExportScene<'_>, Vec<LossyNote>) {
    let mut notes = Vec::new();

    // The material table, and where each id landed in it. Every material, not
    // only the used ones: a document's material list is authored data and a
    // slot pointing at one nothing happens to use today is still the author's.
    let mut material_index: std::collections::HashMap<MaterialId, usize> =
        std::collections::HashMap::new();
    let mut materials = Vec::new();
    // Each image once, however many materials and slots use it, so a file
    // carries one copy of a shared texture — the sharing the document had.
    let mut images: Vec<ImportImage> = Vec::new();
    let mut image_index: std::collections::HashMap<ImageId, usize> =
        std::collections::HashMap::new();
    let mut texture = |map: MaterialMap, id: Option<ImageId>, notes: &mut Vec<LossyNote>| {
        let id = id?;
        if let Some(index) = image_index.get(&id) {
            return Some(*index);
        }
        let image = doc.image(id)?;
        if image.format == kreate_scene::PixelFormat::Rgba16F {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: a float image was written as eight bits, and lost its range above one",
                    image.name
                ),
            });
        }
        let index = images.len();
        images.push(ImportImage {
            name: image.name.clone(),
            width: image.width(),
            height: image.height(),
            pixels: crate::image_import::rgba8_of(image, map.color_space()),
        });
        image_index.insert(id, index);
        Some(index)
    };
    for (id, material) in doc.materials() {
        if material.graph.is_some() {
            // glTF has no node graphs; a material is a standard surface and
            // nothing else. What goes out is the flat surface the material
            // had, which is what the graph was made from — and may no longer
            // be what it draws.
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: uses a node graph, which glTF cannot hold; its flat surface was written",
                    material.name
                ),
            });
        }
        material_index.insert(id, materials.len());
        materials.push(ImportMaterial {
            name: material.name.clone(),
            base_color: [
                material.base_color.r,
                material.base_color.g,
                material.base_color.b,
                material.alpha,
            ],
            metallic: material.metallic,
            roughness: material.roughness,
            // Not a field on the datablock yet: §2.3's node graph is where
            // emission will live, and inventing a value here would make the
            // file claim one.
            emissive: glam::Vec3::ZERO,
            base_color_texture: texture(MaterialMap::BaseColor, material.base_color_image, &mut notes),
            metallic_roughness_texture: texture(
                MaterialMap::MetallicRoughness,
                material.metallic_roughness_image,
                &mut notes,
            ),
            normal_texture: texture(MaterialMap::Normal, material.normal_image, &mut notes),
            normal_scale: material.normal_strength,
        });
    }

    let mut meshes: Vec<(String, &kreate_mesh::MeshData)> = Vec::new();
    let mut mesh_slots: Vec<Vec<Option<usize>>> = Vec::new();
    let mut mesh_index = std::collections::HashMap::new();
    let mut lights = Vec::new();
    let mut light_index = std::collections::HashMap::new();
    let mut cameras = Vec::new();
    let mut camera_index = std::collections::HashMap::new();
    let mut nodes: Vec<ExportNode> = Vec::new();
    let mut node_of = std::collections::HashMap::new();

    for (id, object) in doc.objects() {
        node_of.insert(id, nodes.len());
        let data = match object.data {
            // A mesh id the document no longer has. Resolved here rather than
            // when the list is handed over, because dropping an entry later
            // would shift every index after it and hang the wrong geometry on
            // every node that followed.
            ObjectData::Mesh(mesh) => match doc.mesh(mesh) {
                None => {
                    notes.push(LossyNote::Unsupported {
                        what: format!("{}: its mesh is not in the document", object.name),
                    });
                    ImportNodeData::Empty
                }
                // Deduplicated on the id, which is what a linked duplicate is:
                // two objects and one mesh. glTF says the same thing the same
                // way — several nodes pointing at one mesh — so the sharing
                // survives the file rather than becoming two copies.
                Some(data) => {
                    let index = *mesh_index.entry(mesh).or_insert_with(|| {
                        let slots = doc
                            .mesh_slots(mesh)
                            .iter()
                            .map(|slot| slot.and_then(|id| material_index.get(&id).copied()))
                            .collect();
                        mesh_slots.push(slots);
                        meshes.push((object.name.clone(), data));
                        meshes.len() - 1
                    });
                    ImportNodeData::Mesh(index)
                }
            },
            ObjectData::Light(light) => {
                let index = *light_index.entry(light).or_insert_with(|| {
                    lights.push(match doc.light(light) {
                        Some(light) => export_light(light, &mut notes),
                        None => {
                            notes.push(LossyNote::Unsupported {
                                what: format!(
                                    "{}: its light is not in the document",
                                    object.name
                                ),
                            });
                            ImportLight {
                                name: object.name.clone(),
                                kind: ImportLightKind::Point,
                                color: [1.0; 3],
                                intensity: 0.0,
                                range: None,
                            }
                        }
                    });
                    lights.len() - 1
                });
                ImportNodeData::Light(index)
            }
            ObjectData::Camera(camera) => {
                let index = *camera_index.entry(camera).or_insert_with(|| {
                    cameras.push(match doc.camera(camera) {
                        Some(camera) => export_camera(camera),
                        None => {
                            notes.push(LossyNote::Unsupported {
                                what: format!(
                                    "{}: its camera is not in the document",
                                    object.name
                                ),
                            });
                            ImportCamera {
                            name: object.name.clone(),
                            fov_y: Some(std::f32::consts::FRAC_PI_4),
                            ortho_width: None,
                            clip_near: 0.1,
                            clip_far: None,
                            }
                        }
                    });
                    cameras.len() - 1
                });
                ImportNodeData::Camera(index)
            }
            ObjectData::Empty => ImportNodeData::Empty,
        };
        nodes.push(ExportNode {
            name: object.name.clone(),
            // Filled in below: an object may name a parent that comes after it
            // in the arena, and an index into a half-built list would be wrong.
            parent: None,
            // **Not `object.transform`.** The document composes a world
            // transform as `parent_world · parent_inverse · local`, where the
            // inverse is where the parent stood at the moment of parenting —
            // that is what keeps dragging one object onto another from
            // teleporting it. A file's child transform is relative to its
            // parent with nothing in between, so the two have to be folded
            // together here or every parented object leaves at the wrong place.
            transform: if object.parent.is_some() {
                kreate_math::DTransform::from_affine(
                    object.parent_inverse * object.transform.to_affine(),
                )
            } else {
                object.transform
            },
            data,
        });
        if !object.visible {
            // Written anyway. Visibility is a viewport state and not a property
            // of the model, and a user who hid a wall to work behind it would
            // not expect the wall to be missing from the file.
            notes.push(LossyNote::Unsupported {
                what: format!("{}: hidden in the viewport, and written anyway", object.name),
            });
        }
    }

    for (index, (_, object)) in doc.objects().enumerate() {
        let Some(parent) = object.parent else { continue };
        nodes[index].parent = node_of.get(&parent).copied();
        if nodes[index].parent.is_none() {
            notes.push(LossyNote::Unsupported {
                what: format!("{}: its parent is not in the document", object.name),
            });
        }
    }

    (
        ExportScene {
            meshes,
            materials,
            images,
            mesh_slots,
            lights,
            cameras,
            nodes,
        },
        notes,
    )
}

/// One light, in the units an interchange format writes.
///
/// §2.2's decision paying off in the direction it was made for: the datablock's
/// candela and lux go out unconverted, so a file that comes back says the same
/// numbers it arrived with.
fn export_light(light: &kreate_scene::Light, notes: &mut Vec<LossyNote>) -> ImportLight {
    let kind = match light.kind {
        kreate_scene::LightKind::Point { .. } => ImportLightKind::Point,
        kreate_scene::LightKind::Sun { .. } => ImportLightKind::Directional,
        kreate_scene::LightKind::Spot {
            cone_angle, blend, ..
        } => {
            let outer = (cone_angle * 0.5) as f32;
            ImportLightKind::Spot {
                // The inverse of what `build_into` does on the way in: a full
                // cone and a blend fraction become two half-angles. Exact in
                // both directions, which is why the shape survives even though
                // the parameterisation does not.
                inner: outer * (1.0 - blend.clamp(0.0, 1.0) as f32),
                outer,
            }
        }
        kreate_scene::LightKind::Area { .. } => {
            // glTF has no area light, and the nearest thing that is not a lie
            // is the point light §7.3 already reduces it to for the raster
            // pass: the same axial candela from the same place. What is lost is
            // the shape, and with it the soft shadow that was the reason to
            // reach for an area light.
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: an area light was written as a point light of the same axial intensity",
                    light.name
                ),
            });
            ImportLightKind::Point
        }
    };
    if light.exposure != 0.0 {
        // glTF has one number where the datablock has two. Folded rather than
        // dropped — the amount of light is what the file is for — and said,
        // because a round trip brings back the product with the stops at zero.
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{}: its {:+.2} stops were folded into the intensity",
                light.name, light.exposure
            ),
        });
    }
    if light.kind.is_soft() {
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{}: the source's size was dropped; glTF's lights have no size",
                light.name
            ),
        });
    }
    ImportLight {
        name: light.name.clone(),
        kind,
        color: [light.color.r, light.color.g, light.color.b],
        // An area light's nits become candela here, which is the one kind whose
        // unit changes on the way out — because the kind it is being written as
        // is a different kind.
        intensity: light.axial_candela().unwrap_or(light.effective_intensity()) as f32,
        // No cutoff. The datablock has no artist-set distance yet (§7.8), and
        // the one the application derives from the display's black point is a
        // property of how the picture is being shown, not of the lamp.
        range: None,
    }
}

/// One camera, as much of it as glTF holds.
fn export_camera(camera: &kreate_scene::Camera) -> ImportCamera {
    // The camera's own sensor is the frame to measure the angle against. Any
    // other aspect would be the aspect of whatever this session happens to be
    // rendering, which is not a property of the camera.
    let aspect = camera.sensor_width_mm / camera.sensor_height_mm;
    let perspective = matches!(camera.projection, kreate_scene::Projection::Perspective);
    ImportCamera {
        name: camera.name.clone(),
        fov_y: perspective.then(|| camera.fov_y(aspect) as f32),
        ortho_width: (!perspective).then_some(camera.ortho_scale as f32),
        clip_near: camera.clip_near as f32,
        clip_far: Some(camera.clip_far as f32),
    }
}
