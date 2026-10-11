//! glTF 2.0 export.
//!
//! Written by hand on `gltf-json` rather than through a convenience writer,
//! because the parts that matter here are exactly the parts a convenience
//! writer decides for you: how n-gons are triangulated, where vertices are
//! split, which UV origin is used, and what happens to the face-to-material
//! partition.
//!
//! # The three lossy conversions, and what is done about each
//!
//! glTF is a runtime delivery format. It stores triangles, one attribute value
//! per vertex, and one material per primitive. Kreate stores n-gons, attribute
//! values per *corner*, and one material per face. Every export therefore has
//! to perform three conversions, and each one loses something unless it is
//! handled deliberately:
//!
//! | Kreate | glTF | What this module does |
//! |---|---|---|
//! | n-gon faces | triangles only | Triangulate, and record the original faces in [`NGON_EXTENSION`] |
//! | corner attributes | per-vertex attributes | Split a vertex wherever its corners disagree |
//! | material per face | material per primitive | One primitive per material actually used |
//!
//! The vertex split is the inverse of what [`crate::ir::ImportMesh::weld`] does
//! on the way in, and it is not optional: a cube with per-face normals has to
//! come out with 24 vertices or it shades as a sphere, and a UV seam that is
//! not split puts a smear of texture across it.
//!
//! # `KREATE_ngons`
//!
//! Triangulation is the one conversion that cannot be undone by inspection —
//! given a triangulated quad there is no way to know whether the author drew a
//! quad or two triangles, and guessing (merging coplanar neighbours) is wrong
//! about as often as it is right. So the original face list travels alongside
//! the triangles in a vendor extension, and [`crate::gltf_import`] reads it
//! back.
//!
//! It is listed in `extensionsUsed` but deliberately **not** in
//! `extensionsRequired`: every other reader ignores it and still gets correct,
//! complete triangles. An extension that makes a file unreadable to everyone
//! else in exchange for a round trip with ourselves would be a bad trade.
//!
//! The extension stores the faces explicitly — a size per face and a vertex per
//! corner — rather than something cleverer derived from the triangle order. The
//! clever version costs a third less space and depends on the triangulator never
//! changing its output order; recovering an ordered polygon ring from an
//! unordered set of triangles is a boundary walk that has its own failure cases
//! on degenerate input. Explicit is worth the third.

use crate::ir::{
    ExportScene, ImportLightKind, ImportMaterial, ImportNodeData, ImportTransform, LossyNote,
    SourceMetadata, UvOrigin,
};
use glam::{Vec2, Vec3};
use gltf::json;
use json::validation::{Checked, USize64};
use kreate_math::AxisConvention;
use kreate_mesh::attribute::names;
use kreate_mesh::MeshData;
use std::collections::HashMap;

/// The vendor extension carrying the pre-triangulation face list.
pub const NGON_EXTENSION: &str = "KREATE_ngons";

/// Kreate's default auto-smooth angle, 30 degrees.
///
/// Duplicated from `kreate-render` rather than shared, because `kreate-io` must
/// not depend on the renderer. The value matters for agreement, not for
/// correctness: an export whose normals were averaged over a different angle
/// than the viewport used would shade differently in the file than on screen.
const DEFAULT_AUTO_SMOOTH: f32 = std::f32::consts::PI / 6.0;

#[derive(Clone, Copy, Debug)]
pub struct GltfExportOptions {
    /// Write a binary `.glb` rather than a `.gltf` with a base64 data URI.
    ///
    /// On by default. The JSON form is roughly a third larger and exists for
    /// looking at the file, not for shipping it.
    pub binary: bool,
    /// Record the original faces in [`NGON_EXTENSION`].
    pub record_ngons: bool,
    /// Write a `NORMAL` attribute.
    ///
    /// On by default. Without it a reader computes its own, which loses every
    /// sharp edge and smoothing decision the mesh carries.
    pub normals: bool,
    /// The angle beyond which adjacent faces stop sharing a normal.
    pub auto_smooth_angle: f32,
}

impl Default for GltfExportOptions {
    fn default() -> Self {
        Self {
            binary: true,
            record_ngons: true,
            normals: true,
            auto_smooth_angle: DEFAULT_AUTO_SMOOTH,
        }
    }
}

/// One vertex of the exported, split vertex array.
///
/// Bit patterns rather than floats so the key is hashable and the comparison is
/// exact. Exact is what is wanted: two corners either carry the same value and
/// share a vertex, or they do not. An epsilon here would silently merge a seam
/// that the author placed one texel apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct VertexKey {
    vertex: u32,
    uv: [u32; 2],
    normal: [u32; 3],
    color: [u32; 4],
}

/// Writes a glTF 2.0 file.
///
/// `materials` may be empty: a mesh whose faces carry material indices with no
/// table behind them still needs somewhere for those indices to point, so a
/// placeholder is emitted for each one in use. That keeps the face-to-material
/// partition — which is real authored data — intact through the round trip even
/// when the material parameters are not available.
pub fn write_gltf(
    meshes: &[(String, &MeshData)],
    materials: &[ImportMaterial],
    options: &GltfExportOptions,
) -> (Vec<u8>, Vec<LossyNote>) {
    let mut scene = ExportScene::new(meshes.to_vec());
    scene.materials = materials.to_vec();
    write_gltf_scene(&scene, options)
}

/// Writes a glTF 2.0 file: geometry, hierarchy, lights and cameras.
///
/// [`write_gltf`] is this with nothing but meshes, which is all a format
/// without a hierarchy can say and all the converter has to hand.
///
/// # The file's convention is the file's
///
/// The scene arrives in Kreate space and leaves in glTF's — Y-up, one unit per
/// metre — and that change happens here and nowhere else, which is the rule
/// [`crate::ir`] opens with. There is deliberately no option for it: this
/// writer emits glTF, and a glTF file that is not Y-up is not a glTF file.
pub fn write_gltf_scene(
    scene: &ExportScene<'_>,
    options: &GltfExportOptions,
) -> (Vec<u8>, Vec<LossyNote>) {
    let mut notes = Vec::new();
    let mut root = json::Root {
        asset: json::Asset {
            version: "2.0".to_string(),
            generator: Some("Kreate".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut bin: Vec<u8> = Vec::new();
    // Scene material index to glTF material index. The *scene's*, not the
    // face's: a face carries a per-mesh slot and the slot table resolves it
    // before it arrives here, so that two meshes cannot collide on a number
    // that means something different in each.
    //
    // Only ever looked up and inserted into, never iterated: the emitted
    // table's order comes from the order materials are first referenced, which
    // is the face grouping's order below, which is sorted. That is where the
    // determinism lives.
    let mut material_map: std::collections::BTreeMap<usize, u32> =
        std::collections::BTreeMap::new();
    let mut used_ngons = false;

    // Kreate is Z-up; glTF is Y-up. Applied here and nowhere else. Both are
    // right-handed, so this rotation does not flip winding — asserted in the
    // tests rather than assumed, because a silent winding flip is the classic
    // way for an exporter to turn a model inside out.
    let to_gltf = AxisConvention::GLTF.from_kreate();

    // Which glTF mesh each of the scene's meshes became. Not the same numbering:
    // a mesh with no faces is skipped, so a node that assumed the two lists ran
    // in step would hang its name on the next mesh along.
    let mut mesh_indices: std::collections::BTreeMap<usize, u32> =
        std::collections::BTreeMap::new();

    // The textures before the meshes, so that a material emitted while the
    // meshes are written can name one by the scene's own image index.
    write_textures(&scene.images, &mut root, &mut bin, &mut notes);

    for (index, (name, mesh)) in scene.meshes.iter().enumerate() {
        let mesh_index = root.meshes.len() as u32;
        let primitives = write_mesh(
            name,
            mesh,
            scene,
            index,
            options,
            to_gltf,
            &mut root,
            &mut bin,
            &mut material_map,
            &mut notes,
            &mut used_ngons,
        );
        if primitives.is_empty() {
            notes.push(LossyNote::Unsupported {
                what: format!("{name}: no faces, so nothing was written"),
            });
            continue;
        }
        // Shape keys: the targets' names where Blender and three.js look for
        // them, and every target off — a node says how far its are turned.
        let targets = scene.morph_targets.get(index).filter(|t| !t.is_empty());
        root.meshes.push(json::Mesh {
            name: Some(name.clone()),
            primitives,
            extensions: None,
            extras: targets.and_then(|t| {
                serde_json::value::to_raw_value(&serde_json::json!({ "targetNames": t })).ok()
            }),
            weights: targets.map(|t| vec![0.0; t.len()]),
        });
        mesh_indices.insert(index, mesh_index);
    }

    write_lights(&scene.lights, &mut root);
    write_cameras(scene, &mut root, &mut notes);
    write_nodes(scene, &mesh_indices, &mut root, &mut notes);
    write_skins(scene, to_gltf, &mut root, &mut bin, &mut notes);
    write_animations(scene, &mut root, &mut bin, &mut notes);

    if used_ngons {
        root.extensions_used.push(NGON_EXTENSION.to_string());
    }
    note_dropped_attributes(&scene.meshes, options, &mut notes);

    // The buffer is declared even when empty so that a file with no geometry is
    // still a structurally valid glTF rather than one with dangling views.
    if !bin.is_empty() {
        root.buffers.push(json::Buffer {
            byte_length: USize64(bin.len() as u64),
            name: None,
            uri: if options.binary {
                None
            } else {
                Some(format!(
                    "data:application/octet-stream;base64,{}",
                    base64(&bin)
                ))
            },
            extensions: None,
            extras: Default::default(),
        });
    }

    let bytes = if options.binary {
        pack_glb(&root, &bin)
    } else {
        // Not `to_vec_pretty`: this is a delivery format, and the pretty form of
        // a mesh's worth of accessors is noise. The JSON path exists to be read
        // structurally, which a formatter can do on demand.
        serde_json::to_vec(&root).expect("the json tree is built from owned data")
    };
    (bytes, notes)
}

/// Writes the lights, as `KHR_lights_punctual`.
///
/// glTF keeps its lights in a table on the root and a node points at one by
/// index — the same shape the IR uses, which is not a coincidence: §2.2 chose
/// this format's units and this format's model so that the boundary is a copy.
///
/// Listed in `extensionsUsed` and deliberately not in `extensionsRequired`, for
/// the reason [`NGON_EXTENSION`] gives: a reader that ignores it still gets the
/// whole model, and a file nobody else can open is a bad trade for a round trip
/// with ourselves.
fn write_lights(lights: &[crate::ir::ImportLight], root: &mut json::Root) {
    use json::extensions::scene::khr_lights_punctual as khr;
    if lights.is_empty() {
        return;
    }
    let table: Vec<khr::Light> = lights
        .iter()
        .map(|light| {
            let (kind, spot) = match light.kind {
                ImportLightKind::Point | ImportLightKind::Area { .. } => (khr::Type::Point, None),
                ImportLightKind::Directional => (khr::Type::Directional, None),
                ImportLightKind::Spot { inner, outer } => (
                    khr::Type::Spot,
                    Some(khr::Spot {
                        inner_cone_angle: inner,
                        outer_cone_angle: outer,
                    }),
                ),
            };
            khr::Light {
                color: light.color,
                extensions: None,
                extras: Default::default(),
                intensity: light.intensity,
                name: Some(light.name.clone()),
                range: light.range,
                spot,
                type_: Checked::Valid(kind),
            }
        })
        .collect();
    root.extensions
        .get_or_insert_with(Default::default)
        .khr_lights_punctual = Some(json::extensions::root::KhrLightsPunctual { lights: table });
    root.extensions_used.push(LIGHT_EXTENSION.to_string());
}

/// Writes the cameras, and says what glTF has no room for.
///
/// A camera goes out even though it cannot go out whole: the datablock has an
/// f-stop, a shutter and a sensor and glTF has an angle. Writing the angle and
/// noting the rest is the honest trade — dropping the camera entirely would
/// lose the one thing glTF *can* carry, which is where it was pointing.
fn write_cameras(scene: &ExportScene<'_>, root: &mut json::Root, notes: &mut Vec<LossyNote>) {
    for camera in &scene.cameras {
        let (perspective, orthographic, kind) = match (camera.fov_y, camera.ortho_width) {
            (Some(fov), _) => (
                Some(json::camera::Perspective {
                    // Left out on purpose. The aspect belongs to the image being
                    // rendered, not to the camera, and writing the one this
                    // session happens to have would make the file claim a frame
                    // it does not have. glTF's own answer for "unset" is to use
                    // the viewport's.
                    aspect_ratio: None,
                    yfov: fov,
                    zfar: camera.clip_far,
                    znear: camera.clip_near,
                    extensions: None,
                    extras: Default::default(),
                }),
                None,
                json::camera::Type::Perspective,
            ),
            (None, Some(width)) => (
                None,
                Some(json::camera::Orthographic {
                    // glTF writes half-widths, and separately for each axis. The
                    // IR carries one width because that is what the datablock
                    // has: an orthographic scale is a single number and the
                    // height follows from the frame's aspect.
                    xmag: width * 0.5,
                    ymag: width * 0.5,
                    // Not optional here, unlike the perspective form. A file
                    // with no far plane and an orthographic camera is invalid,
                    // so an unset one becomes a distance rather than nothing.
                    zfar: camera.clip_far.unwrap_or(1.0e4),
                    znear: camera.clip_near,
                    extensions: None,
                    extras: Default::default(),
                }),
                json::camera::Type::Orthographic,
            ),
            (None, None) => {
                notes.push(LossyNote::Unsupported {
                    what: format!("{}: the camera has no projection", camera.name),
                });
                continue;
            }
        };
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{}: the aperture, shutter and sensor were dropped; glTF stores an angle",
                camera.name
            ),
        });
        root.cameras.push(json::Camera {
            name: Some(camera.name.clone()),
            orthographic,
            perspective,
            type_: Checked::Valid(kind),
            extensions: None,
            extras: Default::default(),
        });
    }
}

/// Writes the hierarchy, with every node's own transform.
///
/// An empty node list means the format should place each mesh at the origin,
/// which is what a bare mesh list says and what this exporter did before it
/// could say anything else.
fn write_nodes(
    scene: &ExportScene<'_>,
    mesh_indices: &std::collections::BTreeMap<usize, u32>,
    root: &mut json::Root,
    notes: &mut Vec<LossyNote>,
) {
    if scene.nodes.is_empty() {
        for index in mesh_indices.values() {
            let node = root.nodes.len() as u32;
            root.nodes.push(json::Node {
                mesh: Some(json::Index::new(*index)),
                name: root.meshes[*index as usize].name.clone(),
                ..Default::default()
            });
            root_scene(root).nodes.push(json::Index::new(node));
        }
        return;
    }

    // What the file's own metadata would be if this file were being read back:
    // glTF, Y-up, one unit per metre. Built here so that the conversion is the
    // same function the importer inverts — see `ImportTransform::from_kreate`.
    let target = SourceMetadata {
        axis: AxisConvention::GLTF,
        ..SourceMetadata::new("gltf")
    };

    // Children first, because a glTF node lists its children and the IR's nodes
    // name their parent. A parent index that is out of range is a scene that
    // does not describe itself, and it is said rather than dropped.
    let mut children: Vec<Vec<json::Index<json::Node>>> = vec![Vec::new(); scene.nodes.len()];
    for (index, node) in scene.nodes.iter().enumerate() {
        let Some(parent) = node.parent else { continue };
        match children.get_mut(parent) {
            Some(list) => list.push(json::Index::new(index as u32)),
            None => notes.push(LossyNote::Unsupported {
                what: format!("{}: its parent is not in the scene", node.name),
            }),
        }
    }

    for (index, node) in scene.nodes.iter().enumerate() {
        let out = placed_out(scene, index, node.transform.to_affine(), &target);
        let mut json_node = json::Node {
            name: Some(node.name.clone()),
            translation: Some(out.translation.to_array()),
            rotation: Some(json::scene::UnitQuaternion(out.rotation.to_array())),
            scale: Some(out.scale.to_array()),
            ..Default::default()
        };
        if !children[index].is_empty() {
            json_node.children = Some(children[index].clone());
        }
        match node.data {
            ImportNodeData::Mesh(i) => match mesh_indices.get(&i) {
                Some(mesh) => json_node.mesh = Some(json::Index::new(*mesh)),
                // The mesh had no faces and was not written. The node still is:
                // it may be somebody's parent, and dropping it would move every
                // child to the root.
                None => notes.push(LossyNote::Unsupported {
                    what: format!("{}: its mesh had nothing to write", node.name),
                }),
            },
            ImportNodeData::Light(i) => {
                if i < scene.lights.len() {
                    json_node
                        .extensions
                        .get_or_insert_with(Default::default)
                        .khr_lights_punctual = Some(
                        json::extensions::scene::khr_lights_punctual::KhrLightsPunctual {
                            light: json::Index::new(i as u32),
                        },
                    );
                } else {
                    notes.push(LossyNote::Unsupported {
                        what: format!("{}: its light is not in the scene", node.name),
                    });
                }
            }
            ImportNodeData::Camera(i) => {
                if i < root.cameras.len() {
                    json_node.camera = Some(json::Index::new(i as u32));
                } else {
                    notes.push(LossyNote::Unsupported {
                        what: format!("{}: its camera is not in the scene", node.name),
                    });
                }
            }
            ImportNodeData::Empty => {}
        }
        if let Some(skin) = node.skin.filter(|s| *s < scene.skins.len()) {
            json_node.skin = Some(json::Index::new(skin as u32));
        }
        if !node.morph_weights.is_empty() {
            json_node.weights = Some(node.morph_weights.clone());
        }
        root.nodes.push(json_node);
    }

    // The scene names its roots only; everything else hangs off `children`.
    // Listing them all would make each child appear twice, which readers
    // variously ignore, draw twice, or reject.
    let roots: Vec<_> = scene
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.parent.is_none())
        .map(|(index, _)| json::Index::new(index as u32))
        .collect();
    root_scene(root).nodes.extend(roots);
}

/// Writes the skins: joints and inverse bind matrices, in glTF's axes.
///
/// The matrix is the IR's with the axis change on both sides — a matrix maps
/// a Kreate point to a Kreate point, and glTF's has to map a glTF point to a
/// glTF point. glTF has no unit here: every exported unit is a metre.
fn write_skins(
    scene: &ExportScene<'_>,
    to_gltf: glam::Mat3,
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    notes: &mut Vec<LossyNote>,
) {
    let back = to_gltf.transpose();
    for skin in &scene.skins {
        if skin.joints.is_empty() || skin.joints.iter().any(|j| *j >= scene.nodes.len()) {
            notes.push(LossyNote::Unsupported {
                what: format!("{}: a skin whose joints are not in the scene", skin.name),
            });
            root.skins.push(json::Skin {
                extensions: None,
                extras: Default::default(),
                inverse_bind_matrices: None,
                joints: Vec::new(),
                name: Some(skin.name.clone()),
                skeleton: None,
            });
            continue;
        }
        let matrices: Vec<f32> = skin
            .inverse_bind
            .iter()
            .flat_map(|m| {
                let linear = to_gltf * m.matrix3.as_mat3() * back;
                let t = to_gltf * m.translation.as_vec3();
                glam::Mat4::from_cols(
                    linear.x_axis.extend(0.0),
                    linear.y_axis.extend(0.0),
                    linear.z_axis.extend(0.0),
                    t.extend(1.0),
                )
                .to_cols_array()
            })
            .collect();
        let view = push_view(root, bin, bytemuck::cast_slice(&matrices), None);
        let accessor = push_accessor(
            root,
            view,
            skin.inverse_bind.len(),
            json::accessor::ComponentType::F32,
            json::accessor::Type::Mat4,
            None,
            None,
        );
        root.skins.push(json::Skin {
            extensions: None,
            extras: Default::default(),
            inverse_bind_matrices: Some(accessor),
            joints: skin
                .joints
                .iter()
                .map(|j| json::Index::new(*j as u32))
                .collect(),
            name: Some(skin.name.clone()),
            skeleton: None,
        });
    }
}

/// The joints and weights each vertex of `mesh` carries, for the skin the
/// scene binds it with: its four heaviest bones, normalised, as glTF allows
/// no more. `None` when no node skins this mesh.
fn vertex_skin(
    mesh: &MeshData,
    scene: &ExportScene<'_>,
    index: usize,
) -> Option<(crate::ir::VertexSkin, bool)> {
    let skin = scene
        .nodes
        .iter()
        .find(|n| n.data == ImportNodeData::Mesh(index) && n.skin.is_some())
        .and_then(|n| scene.skins.get(n.skin?))?;
    let n = mesh.num_verts() as usize;
    let layers: Vec<(u16, &[f32])> = skin
        .joint_names
        .iter()
        .enumerate()
        .filter_map(|(j, name)| {
            let w = mesh.attributes().get_f32(&format!("weight:{name}"))?;
            (w.len() == n).then_some((j as u16, w))
        })
        .collect();
    let mut joints = vec![[0u16; 4]; n];
    let mut weights = vec![[0.0f32; 4]; n];
    let mut more_than_four = false;
    for v in 0..n {
        let mut all: Vec<(u16, f32)> = layers
            .iter()
            .map(|(j, w)| (*j, w[v]))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        more_than_four |= all.len() > 4;
        all.truncate(4);
        let total: f32 = all.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            // No bone weighs on it: glTF has no "stays put", so it follows
            // the first joint fully — at rest, where it is.
            weights[v][0] = 1.0;
            continue;
        }
        for (k, (j, w)) in all.into_iter().enumerate() {
            joints[v][k] = j;
            weights[v][k] = w / total;
        }
    }
    Some((crate::ir::VertexSkin { joints, weights }, more_than_four))
}

/// The default scene, made on first use.
/// Writes the clips: per track, one shared list of times and a channel for
/// each part of the placement that moves.
///
/// The axis change is the same call a node's rest placement makes, sample by
/// sample, so a moving node and a still one cannot come out in two different
/// conventions. glTF's node indices are the scene's own, because `write_nodes`
/// writes every node in order.
/// A node's placement in the file: a camera or a light with nothing under it
/// by the axis it looks along, as the reader places it (see
/// `ImportTransform::to_kreate_facing`), anything else as a mesh is.
fn placed_out(
    scene: &ExportScene<'_>,
    index: usize,
    affine: kreate_math::DAffine3,
    target: &SourceMetadata,
) -> ImportTransform {
    let faces =
        scene.nodes.get(index).is_some_and(|n| {
            matches!(n.data, ImportNodeData::Camera(_) | ImportNodeData::Light(_))
        }) && !scene.nodes.iter().any(|n| n.parent == Some(index));
    if faces {
        ImportTransform::from_kreate_facing(affine, target)
    } else {
        ImportTransform::from_kreate(affine, target)
    }
}

fn write_animations(
    scene: &ExportScene<'_>,
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    notes: &mut Vec<LossyNote>,
) {
    use json::animation::{Channel, Interpolation, Property, Sampler, Target};
    if scene.nodes.is_empty() {
        return;
    }
    let target = SourceMetadata {
        axis: AxisConvention::GLTF,
        ..SourceMetadata::new("gltf")
    };
    for animation in &scene.animations {
        let mut out = json::Animation {
            extensions: None,
            extras: Default::default(),
            channels: Vec::new(),
            name: Some(animation.name.clone()),
            samplers: Vec::new(),
        };
        for track in &animation.tracks {
            let count = track.times.len().min(track.transforms.len());
            if track.node >= scene.nodes.len() || count == 0 {
                notes.push(LossyNote::Unsupported {
                    what: format!("{}: a track with nothing to write", animation.name),
                });
                continue;
            }
            let times = &track.times[..count];
            let placed: Vec<ImportTransform> = track.transforms[..count]
                .iter()
                .map(|t| placed_out(scene, track.node, t.to_affine(), &target))
                .collect();
            let input_view = push_view(root, bin, bytemuck::cast_slice(times), None);
            let (first, last) = (times[0], times[count - 1]);
            let input = push_accessor(
                root,
                input_view,
                count,
                json::accessor::ComponentType::F32,
                json::accessor::Type::Scalar,
                // Required by the specification for a sampler's input.
                Some(serde_json::json!([first])),
                Some(serde_json::json!([last])),
            );
            let interpolation = if track.step {
                Interpolation::Step
            } else {
                Interpolation::Linear
            };
            let channel = |root: &mut json::Root,
                           bin: &mut Vec<u8>,
                           out: &mut json::Animation,
                           data: Vec<f32>,
                           ty: json::accessor::Type,
                           path: Property| {
                let view = push_view(root, bin, bytemuck::cast_slice(&data), None);
                let output = push_accessor(
                    root,
                    view,
                    count,
                    json::accessor::ComponentType::F32,
                    ty,
                    None,
                    None,
                );
                let sampler = out.samplers.len() as u32;
                out.samplers.push(Sampler {
                    extensions: None,
                    extras: Default::default(),
                    input,
                    interpolation: Checked::Valid(interpolation),
                    output,
                });
                out.channels.push(Channel {
                    sampler: json::Index::new(sampler),
                    target: Target {
                        extensions: None,
                        extras: Default::default(),
                        node: json::Index::new(track.node as u32),
                        path: Checked::Valid(path),
                    },
                    extensions: None,
                    extras: Default::default(),
                });
            };
            if track.translation {
                let data = placed
                    .iter()
                    .flat_map(|p| p.translation.to_array())
                    .collect();
                channel(
                    root,
                    bin,
                    &mut out,
                    data,
                    json::accessor::Type::Vec3,
                    Property::Translation,
                );
            }
            if track.rotation {
                // Each quaternion on the same side as the one before: `q` and
                // `-q` are one rotation, and a reader slerps between neighbours
                // as written — so a sign flip between two samples a degree
                // apart is a full turn the long way round.
                let mut previous = glam::Quat::IDENTITY;
                let data = placed
                    .iter()
                    .flat_map(|p| {
                        // A sample scaled to nothing has no rotation to take
                        // apart, and NaN in a file is a file nobody opens: it
                        // keeps the one before, which is what it looked like.
                        let r = if p.rotation.is_finite() {
                            p.rotation
                        } else {
                            previous
                        };
                        let q = if r.dot(previous) < 0.0 { -r } else { r };
                        previous = q;
                        q.to_array()
                    })
                    .collect();
                channel(
                    root,
                    bin,
                    &mut out,
                    data,
                    json::accessor::Type::Vec4,
                    Property::Rotation,
                );
            }
            if track.scale {
                let data = placed.iter().flat_map(|p| p.scale.to_array()).collect();
                channel(
                    root,
                    bin,
                    &mut out,
                    data,
                    json::accessor::Type::Vec3,
                    Property::Scale,
                );
            }
        }
        // Shape keys, as morph target weights: every target's weight at
        // every time, one after another.
        for track in &animation.weights {
            let count = track.times.len().min(track.values.len());
            if track.node >= scene.nodes.len() || count == 0 {
                continue;
            }
            let times = &track.times[..count];
            let input_view = push_view(root, bin, bytemuck::cast_slice(times), None);
            let input = push_accessor(
                root,
                input_view,
                count,
                json::accessor::ComponentType::F32,
                json::accessor::Type::Scalar,
                Some(serde_json::json!([times[0]])),
                Some(serde_json::json!([times[count - 1]])),
            );
            let flat: Vec<f32> = track.values[..count].iter().flatten().copied().collect();
            let view = push_view(root, bin, bytemuck::cast_slice(&flat), None);
            let output = push_accessor(
                root,
                view,
                flat.len(),
                json::accessor::ComponentType::F32,
                json::accessor::Type::Scalar,
                None,
                None,
            );
            let sampler = out.samplers.len() as u32;
            out.samplers.push(Sampler {
                extensions: None,
                extras: Default::default(),
                input,
                interpolation: Checked::Valid(Interpolation::Linear),
                output,
            });
            out.channels.push(Channel {
                sampler: json::Index::new(sampler),
                target: Target {
                    extensions: None,
                    extras: Default::default(),
                    node: json::Index::new(track.node as u32),
                    path: Checked::Valid(Property::MorphTargetWeights),
                },
                extensions: None,
                extras: Default::default(),
            });
        }
        if !out.channels.is_empty() {
            root.animations.push(out);
        }
    }
}

fn root_scene(root: &mut json::Root) -> &mut json::Scene {
    if root.scenes.is_empty() {
        root.scenes.push(json::Scene {
            nodes: Vec::new(),
            name: None,
            extensions: None,
            extras: Default::default(),
        });
        root.scene = Some(json::Index::new(0));
    }
    &mut root.scenes[0]
}

/// Builds every primitive of one mesh, appending its buffers to `bin`.
#[allow(clippy::too_many_arguments)]
fn write_mesh(
    name: &str,
    mesh: &MeshData,
    scene: &ExportScene<'_>,
    index: usize,
    options: &GltfExportOptions,
    to_gltf: glam::Mat3,
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    material_map: &mut std::collections::BTreeMap<usize, u32>,
    notes: &mut Vec<LossyNote>,
    used_ngons: &mut bool,
) -> Vec<json::mesh::Primitive> {
    let positions = mesh.positions();
    let uvs = mesh.attributes().get_f32x2(names::UV);
    // The vertex colours, as COLOR_0: per corner, from a corner layer or a
    // point one.
    let colors: Option<Vec<glam::Vec4>> = mesh.attributes().get(names::COLOR).and_then(|l| {
        use kreate_mesh::attribute::{AttrData, Domain};
        let corners = mesh.corner_verts();
        match (&l.domain, &l.data) {
            (Domain::Corner, AttrData::F32x4(v)) => Some(v.to_vec()),
            (Domain::Point, AttrData::F32x4(v)) => {
                Some(corners.iter().map(|p| v[*p as usize]).collect())
            }
            (Domain::Corner, AttrData::F32x3(v)) => Some(v.iter().map(|c| c.extend(1.0)).collect()),
            (Domain::Point, AttrData::F32x3(v)) => {
                Some(corners.iter().map(|p| v[*p as usize].extend(1.0)).collect())
            }
            _ => None,
        }
    });
    let colors = colors.filter(|c| c.len() == mesh.num_corners() as usize);
    // Authored split normals win over computed ones, which is the whole point of
    // storing them: a mesh that carries them has had its shading decided by hand
    // and recomputing would overrule the author.
    let authored = mesh.attributes().get_f32x3(names::CUSTOM_NORMAL);
    let computed;
    let normals: Option<&[Vec3]> = if !options.normals {
        None
    } else if let Some(n) = authored {
        Some(n)
    } else {
        computed = mesh.compute_corner_normals(options.auto_smooth_angle);
        Some(&computed)
    };

    let face_materials = mesh.attributes().get_i32(names::MATERIAL_INDEX);
    let has_slots = scene
        .mesh_slots
        .get(index)
        .is_some_and(|slots| !slots.is_empty());
    let tri = mesh.triangulate();
    let skinning = vertex_skin(mesh, scene, index);
    if skinning.as_ref().is_some_and(|(_, more)| *more) {
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{name}: vertices weighed on by more than four bones kept their four heaviest"
            ),
        });
    }

    // Which faces go in which primitive. Sorted by material index so the output
    // is the same whichever order the faces happen to be in.
    let mut groups: std::collections::BTreeMap<i32, Vec<u32>> = std::collections::BTreeMap::new();
    for f in 0..mesh.num_faces() {
        if mesh.face_len(f) < 3 {
            notes.push(LossyNote::DroppedDegenerateFace {
                mesh: name.to_string(),
                face: f as usize,
            });
            continue;
        }
        let m = face_materials.map(|m| m[f as usize]).unwrap_or(-1);
        // A face with no slot number, or a negative one, is drawn in slot zero
        // by the renderer (`mesh_gpu`), so a mesh that *has* slots exports it
        // there too. Without this, a fresh cube with a material assigned — no
        // per-face numbers at all, which is every primitive — left the file
        // with no material, and the round trip lost every material there was.
        // A bare mesh list has no slots, and "no material" is still what it
        // means.
        let m = if m < 0 && has_slots { 0 } else { m };
        groups.entry(m).or_default().push(f);
    }

    let mut primitives = Vec::new();
    for (source_material, faces) in groups {
        // The split vertex array for this primitive, and the corner-to-vertex
        // map that both the triangles and the n-gon record are written through.
        let mut seen: HashMap<VertexKey, u32> = HashMap::new();
        let mut out_positions: Vec<Vec3> = Vec::new();
        let mut out_uvs: Vec<Vec2> = Vec::new();
        let mut out_normals: Vec<Vec3> = Vec::new();
        let mut out_colors: Vec<glam::Vec4> = Vec::new();
        let mut out_vertex: Vec<u32> = Vec::new();
        let mut out_joints: Vec<[u16; 4]> = Vec::new();
        let mut out_weights: Vec<[f32; 4]> = Vec::new();
        let mut corner_vertex: HashMap<u32, u32> = HashMap::new();

        let mut face_sizes: Vec<u32> = Vec::new();
        let mut face_vertices: Vec<u32> = Vec::new();

        for &f in &faces {
            let range = mesh.face_corner_range(f);
            face_sizes.push(range.len() as u32);
            for c in range {
                let vertex = mesh.corner_verts()[c];
                let uv = uvs
                    .map(|u| UvOrigin::TopLeft.from_kreate(u[c]))
                    .unwrap_or(Vec2::ZERO);
                let normal = normals.map(|n| to_gltf * n[c]).unwrap_or(Vec3::Y);
                let color = colors.as_ref().map_or(glam::Vec4::ONE, |k| k[c]);
                let key = VertexKey {
                    vertex,
                    uv: [uv.x.to_bits(), uv.y.to_bits()],
                    normal: [normal.x.to_bits(), normal.y.to_bits(), normal.z.to_bits()],
                    color: color.to_array().map(f32::to_bits),
                };
                let index = *seen.entry(key).or_insert_with(|| {
                    out_positions.push(to_gltf * positions[vertex as usize]);
                    out_uvs.push(uv);
                    out_normals.push(normal);
                    out_colors.push(color);
                    out_vertex.push(vertex);
                    if let Some((skin, _)) = &skinning {
                        out_joints.push(skin.joints[vertex as usize]);
                        out_weights.push(skin.weights[vertex as usize]);
                    }
                    (out_positions.len() - 1) as u32
                });
                corner_vertex.insert(c as u32, index);
                face_vertices.push(index);
            }
        }

        // Triangles, mapped back through the corners they came from so they
        // reference the split vertices rather than the mesh's own.
        let in_group: std::collections::HashSet<u32> = faces.iter().copied().collect();
        let mut indices: Vec<u32> = Vec::new();
        for t in 0..tri.num_triangles() {
            let f = tri.tri_to_face[t];
            if !in_group.contains(&f) {
                continue;
            }
            for k in 0..3 {
                let v = tri.indices[t * 3 + k];
                match corner_of(mesh, f, v).and_then(|c| corner_vertex.get(&c).copied()) {
                    Some(index) => indices.push(index),
                    None => {
                        // Only reachable if the triangulator emitted a vertex the
                        // face does not contain, which would be a bug there.
                        // Reported rather than papered over with a zero.
                        notes.push(LossyNote::Unsupported {
                            what: format!(
                                "{name}: face {f} was triangulated to a vertex it does not contain"
                            ),
                        });
                        indices.push(0);
                    }
                }
            }
        }
        if indices.is_empty() {
            continue;
        }

        let mut attributes = std::collections::BTreeMap::new();
        let view = push_view(
            root,
            bin,
            bytemuck::cast_slice(&out_positions),
            ARRAY_BUFFER,
        );
        let (min, max) = bounds(&out_positions);
        attributes.insert(
            Checked::Valid(json::mesh::Semantic::Positions),
            push_accessor(
                root,
                view,
                out_positions.len(),
                json::accessor::ComponentType::F32,
                json::accessor::Type::Vec3,
                // Required by the specification for POSITION, and actually used:
                // viewers frame the model from it without decoding the buffer.
                Some(serde_json::json!([min.x, min.y, min.z])),
                Some(serde_json::json!([max.x, max.y, max.z])),
            ),
        );
        if normals.is_some() {
            let view = push_view(root, bin, bytemuck::cast_slice(&out_normals), ARRAY_BUFFER);
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Normals),
                push_accessor(
                    root,
                    view,
                    out_normals.len(),
                    json::accessor::ComponentType::F32,
                    json::accessor::Type::Vec3,
                    None,
                    None,
                ),
            );
        }
        if uvs.is_some() {
            let view = push_view(root, bin, bytemuck::cast_slice(&out_uvs), ARRAY_BUFFER);
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::TexCoords(0)),
                push_accessor(
                    root,
                    view,
                    out_uvs.len(),
                    json::accessor::ComponentType::F32,
                    json::accessor::Type::Vec2,
                    None,
                    None,
                ),
            );
        }

        if colors.is_some() {
            let view = push_view(root, bin, bytemuck::cast_slice(&out_colors), ARRAY_BUFFER);
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Colors(0)),
                push_accessor(
                    root,
                    view,
                    out_colors.len(),
                    json::accessor::ComponentType::F32,
                    json::accessor::Type::Vec4,
                    None,
                    None,
                ),
            );
        }

        if skinning.is_some() {
            let view = push_view(root, bin, bytemuck::cast_slice(&out_joints), ARRAY_BUFFER);
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Joints(0)),
                push_accessor(
                    root,
                    view,
                    out_joints.len(),
                    json::accessor::ComponentType::U16,
                    json::accessor::Type::Vec4,
                    None,
                    None,
                ),
            );
            let view = push_view(root, bin, bytemuck::cast_slice(&out_weights), ARRAY_BUFFER);
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Weights(0)),
                push_accessor(
                    root,
                    view,
                    out_weights.len(),
                    json::accessor::ComponentType::F32,
                    json::accessor::Type::Vec4,
                    None,
                    None,
                ),
            );
        }

        let view = push_view(
            root,
            bin,
            bytemuck::cast_slice(&indices),
            ELEMENT_ARRAY_BUFFER,
        );
        let index_accessor = push_accessor(
            root,
            view,
            indices.len(),
            json::accessor::ComponentType::U32,
            json::accessor::Type::Scalar,
            None,
            None,
        );

        let extensions = if options.record_ngons && face_sizes.iter().any(|&n| n > 3) {
            *used_ngons = true;
            Some(ngon_extension(root, bin, &face_sizes, &face_vertices))
        } else {
            None
        };
        if !options.record_ngons && face_sizes.iter().any(|&n| n > 3) {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{name}: n-gons were triangulated and not recorded; the faces cannot be recovered"
                ),
            });
        }

        // Morph targets: each shape key's offsets, per split vertex.
        let targets: Vec<json::mesh::MorphTarget> = scene
            .morph_targets
            .get(index)
            .map(|names| {
                names
                    .iter()
                    .map(|n| {
                        let offsets = mesh.attributes().get_f32x3(&kreate_scene::shape::layer(n));
                        let moved: Vec<Vec3> = out_vertex
                            .iter()
                            .map(|v| {
                                offsets
                                    .and_then(|o| o.get(*v as usize))
                                    .map_or(Vec3::ZERO, |d| to_gltf * *d)
                            })
                            .collect();
                        let view = push_view(root, bin, bytemuck::cast_slice(&moved), ARRAY_BUFFER);
                        let (min, max) = bounds(&moved);
                        json::mesh::MorphTarget {
                            positions: Some(push_accessor(
                                root,
                                view,
                                moved.len(),
                                json::accessor::ComponentType::F32,
                                json::accessor::Type::Vec3,
                                Some(serde_json::json!([min.x, min.y, min.z])),
                                Some(serde_json::json!([max.x, max.y, max.z])),
                            )),
                            normals: None,
                            tangents: None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        primitives.push(json::mesh::Primitive {
            attributes,
            extensions,
            extras: Default::default(),
            indices: Some(index_accessor),
            // The face's number is a *slot*, and a slot is per-mesh (§7.5).
            // Resolved through the scene's table before it reaches the material
            // map, or the second mesh in a file would be given the first one's
            // materials — which looks right until a file has two of them.
            material: scene.material_of(index, source_material).map(|resolved| {
                json::Index::new(emit_material(
                    root,
                    &scene.materials,
                    material_map,
                    resolved,
                ))
            }),
            mode: Checked::Valid(json::mesh::Mode::Triangles),
            targets: (!targets.is_empty()).then_some(targets),
        });
    }
    primitives
}

/// Which corner of `face` sits on `vertex`.
///
/// Linear, because a face is small and building an index for it would cost more
/// than the scan. The first match wins: a face that names the same vertex twice
/// is already invalid, and picking either corner is as good an answer as exists.
fn corner_of(mesh: &MeshData, face: u32, vertex: u32) -> Option<u32> {
    let range = mesh.face_corner_range(face);
    let verts = mesh.corner_verts();
    range
        .clone()
        .find(|&c| verts[c] == vertex)
        .map(|c| c as u32)
}

/// Builds the `KREATE_ngons` payload for one primitive.
fn ngon_extension(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    face_sizes: &[u32],
    face_vertices: &[u32],
) -> json::extensions::mesh::Primitive {
    let sizes_view = push_view(root, bin, bytemuck::cast_slice(face_sizes), None);
    let sizes = push_accessor(
        root,
        sizes_view,
        face_sizes.len(),
        json::accessor::ComponentType::U32,
        json::accessor::Type::Scalar,
        None,
        None,
    );
    let verts_view = push_view(root, bin, bytemuck::cast_slice(face_vertices), None);
    let verts = push_accessor(
        root,
        verts_view,
        face_vertices.len(),
        json::accessor::ComponentType::U32,
        json::accessor::Type::Scalar,
        None,
        None,
    );
    let mut others = serde_json::Map::new();
    others.insert(
        NGON_EXTENSION.to_string(),
        serde_json::json!({
            "faceSizes": sizes.value(),
            "faceVertices": verts.value(),
        }),
    );
    json::extensions::mesh::Primitive { others }
}

/// Writes every image into the binary blob as PNG, with one texture each.
///
/// Texture `i` is image `i`, so a material's index into the scene's images is
/// already the glTF texture index — no table to keep in step.
///
/// **Inside the file, never beside it.** A `.glb` is one file by definition
/// and a `.gltf` here is already self-contained (its buffer is a data URI), so
/// an image written as a sibling `.png` would be the one piece that does not
/// travel when somebody emails the model.
///
/// PNG because it is lossless and every reader has it. The pixels are the
/// bytes the document stores; whether they are colour or numbers is said by
/// the slot that points at them, which is glTF's rule and the document's.
fn write_textures(
    images: &[crate::ir::ImportImage],
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    notes: &mut Vec<LossyNote>,
) {
    if images.is_empty() {
        return;
    }
    // One sampler for all of them: repeat, with mipmaps — what a material
    // map on a surface means, and what the viewport does.
    root.samplers.push(json::texture::Sampler {
        mag_filter: Some(Checked::Valid(json::texture::MagFilter::Linear)),
        min_filter: Some(Checked::Valid(json::texture::MinFilter::LinearMipmapLinear)),
        wrap_s: Checked::Valid(json::texture::WrappingMode::Repeat),
        wrap_t: Checked::Valid(json::texture::WrappingMode::Repeat),
        ..Default::default()
    });
    for (index, image) in images.iter().enumerate() {
        let png = image::RgbaImage::from_raw(image.width, image.height, image.pixels.clone())
            .and_then(|pixels| {
                let mut out = std::io::Cursor::new(Vec::new());
                image::DynamicImage::ImageRgba8(pixels)
                    .write_to(&mut out, image::ImageFormat::Png)
                    .ok()
                    .map(|_| out.into_inner())
            });
        // An image that will not encode still gets its slot in the table, as a
        // one-pixel white stand-in: dropping it would shift every later
        // texture index onto its neighbour's picture.
        let png = png.unwrap_or_else(|| {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: the image could not be encoded and was written blank",
                    image.name
                ),
            });
            let mut out = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
                1,
                1,
                image::Rgba([255, 255, 255, 255]),
            ))
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("a one-pixel image encodes");
            out.into_inner()
        });
        let view = push_view(root, bin, &png, None);
        root.images.push(json::Image {
            buffer_view: Some(view),
            mime_type: Some(json::image::MimeType("image/png".to_string())),
            name: Some(image.name.clone()),
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
        root.textures.push(json::Texture {
            name: None,
            sampler: Some(json::Index::new(0)),
            source: json::Index::new(index as u32),
            extensions: None,
            extras: Default::default(),
        });
    }
}

/// A texture reference, for a slot that has one.
fn texture_info(index: Option<usize>) -> Option<json::texture::Info> {
    index.map(|i| json::texture::Info {
        index: json::Index::new(i as u32),
        tex_coord: 0,
        extensions: None,
        extras: Default::default(),
    })
}

/// Emits a material for a source index the first time it is referenced.
fn emit_material(
    root: &mut json::Root,
    materials: &[ImportMaterial],
    map: &mut std::collections::BTreeMap<usize, u32>,
    source: usize,
) -> u32 {
    if let Some(&existing) = map.get(&source) {
        return existing;
    }
    let index = root.materials.len() as u32;
    let described = materials.get(source);
    root.materials.push(json::Material {
        name: Some(
            described
                .map(|m| m.name.clone())
                .unwrap_or_else(|| format!("material{source}")),
        ),
        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
            base_color_factor: json::material::PbrBaseColorFactor(
                described
                    .map(|m| m.base_color)
                    .unwrap_or([0.8, 0.8, 0.8, 1.0]),
            ),
            metallic_factor: json::material::StrengthFactor(
                described.map(|m| m.metallic).unwrap_or(0.0),
            ),
            roughness_factor: json::material::StrengthFactor(
                described.map(|m| m.roughness).unwrap_or(0.5),
            ),
            base_color_texture: texture_info(described.and_then(|m| m.base_color_texture)),
            metallic_roughness_texture: texture_info(
                described.and_then(|m| m.metallic_roughness_texture),
            ),
            ..Default::default()
        },
        normal_texture: described.and_then(|m| {
            m.normal_texture.map(|i| json::material::NormalTexture {
                index: json::Index::new(i as u32),
                scale: m.normal_scale,
                tex_coord: 0,
                extensions: None,
                extras: Default::default(),
            })
        }),
        occlusion_texture: described.and_then(|m| {
            m.occlusion_texture
                .map(|i| json::material::OcclusionTexture {
                    index: json::Index::new(i as u32),
                    strength: json::material::StrengthFactor(m.occlusion_strength),
                    tex_coord: 0,
                    extensions: None,
                    extras: Default::default(),
                })
        }),
        emissive_texture: described
            .and_then(|m| m.emissive_texture)
            .map(|i| json::texture::Info {
                index: json::Index::new(i as u32),
                tex_coord: 0,
                extensions: None,
                extras: Default::default(),
            }),
        emissive_factor: json::material::EmissiveFactor(
            described.map(|m| m.emissive.to_array()).unwrap_or_default(),
        ),
        // A cut-out is glTF's MASK, and the threshold its cutoff.
        alpha_mode: json::validation::Checked::Valid(
            if described.is_some_and(|m| m.alpha_clip.is_some()) {
                json::material::AlphaMode::Mask
            } else {
                json::material::AlphaMode::Opaque
            },
        ),
        alpha_cutoff: described
            .and_then(|m| m.alpha_clip)
            .map(json::material::AlphaCutoff),
        extensions: described.and_then(|m| {
            let glows = m.emissive != Vec3::ZERO && m.emissive_strength != 1.0;
            let clear = m.transmission > 0.0;
            let bends = m.ior != 1.5;
            (glows || clear || bends).then(|| json::extensions::material::Material {
                emissive_strength: glows.then_some(json::extensions::material::EmissiveStrength {
                    emissive_strength: json::extensions::material::EmissiveStrengthFactor(
                        m.emissive_strength,
                    ),
                }),
                transmission: clear.then(|| json::extensions::material::Transmission {
                    transmission_factor: json::extensions::material::TransmissionFactor(
                        m.transmission,
                    ),
                    transmission_texture: None,
                    extras: Default::default(),
                }),
                ior: bends.then_some(json::extensions::material::Ior {
                    ior: json::extensions::material::IndexOfRefraction(m.ior),
                    extras: Default::default(),
                }),
                ..Default::default()
            })
        }),
        ..Default::default()
    });
    // Each extension a material uses, listed once.
    if let Some(ext) = root.materials.last().and_then(|m| m.extensions.as_ref()) {
        let used = [
            (
                ext.emissive_strength.is_some(),
                "KHR_materials_emissive_strength",
            ),
            (ext.transmission.is_some(), "KHR_materials_transmission"),
            (ext.ior.is_some(), "KHR_materials_ior"),
        ];
        for (on, name) in used {
            if on && !root.extensions_used.iter().any(|e| e == name) {
                root.extensions_used.push(name.to_string());
            }
        }
    }
    map.insert(source, index);
    index
}

/// The extension carrying the lights. Part of the specification's own set,
/// unlike [`NGON_EXTENSION`], so every reader that has heard of lights at all
/// reads it.
pub const LIGHT_EXTENSION: &str = "KHR_lights_punctual";

const ARRAY_BUFFER: Option<json::buffer::Target> = Some(json::buffer::Target::ArrayBuffer);
const ELEMENT_ARRAY_BUFFER: Option<json::buffer::Target> =
    Some(json::buffer::Target::ElementArrayBuffer);

/// Appends `data` to the binary blob and declares a view over it.
///
/// Every view starts on a four-byte boundary. The specification requires it for
/// accessor offsets, and a reader that maps the buffer instead of copying it
/// will fault on a misaligned view rather than merely reading slowly.
fn push_view(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    data: &[u8],
    target: Option<json::buffer::Target>,
) -> json::Index<json::buffer::View> {
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let offset = bin.len();
    bin.extend_from_slice(data);
    let index = root.buffer_views.len() as u32;
    root.buffer_views.push(json::buffer::View {
        buffer: json::Index::new(0),
        byte_length: USize64(data.len() as u64),
        byte_offset: Some(USize64(offset as u64)),
        byte_stride: None,
        name: None,
        target: target.map(Checked::Valid),
        extensions: None,
        extras: Default::default(),
    });
    json::Index::new(index)
}

fn push_accessor(
    root: &mut json::Root,
    view: json::Index<json::buffer::View>,
    count: usize,
    component: json::accessor::ComponentType,
    ty: json::accessor::Type,
    min: Option<serde_json::Value>,
    max: Option<serde_json::Value>,
) -> json::Index<json::accessor::Accessor> {
    let index = root.accessors.len() as u32;
    root.accessors.push(json::Accessor {
        buffer_view: Some(view),
        byte_offset: Some(USize64(0)),
        count: USize64(count as u64),
        component_type: Checked::Valid(json::accessor::GenericComponentType(component)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(ty),
        min,
        max,
        name: None,
        normalized: false,
        sparse: None,
    });
    json::Index::new(index)
}

fn bounds(positions: &[Vec3]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    if positions.is_empty() {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

/// Reports what the mesh carries that glTF has no place for.
///
/// Said rather than dropped silently: a user whose creases and seams do not come
/// back finds out from the file, not from the next time they open it.
fn note_dropped_attributes(
    meshes: &[(String, &MeshData)],
    options: &GltfExportOptions,
    notes: &mut Vec<LossyNote>,
) {
    for (name, mesh) in meshes {
        let attrs = mesh.attributes();
        for (layer, what) in [
            (names::CREASE, "edge creases"),
            (names::SHARP_EDGE, "sharp edges"),
            (names::SEAM, "UV seams"),
        ] {
            if attrs.get(layer).is_some() {
                notes.push(LossyNote::Unsupported {
                    what: format!("{name}: {what} were dropped; glTF stores none"),
                });
            }
        }
        // A second UV set is data the author made on purpose, and glTF can carry
        // it as TEXCOORD_1 — this exporter simply does not yet, which is worth
        // saying plainly rather than leaving to be discovered.
        let extra_uvs = attrs
            .iter()
            .filter(|l| l.name.starts_with("uv") && &*l.name != names::UV)
            .count();
        if extra_uvs > 0 {
            notes.push(LossyNote::Unsupported {
                what: format!("{name}: {extra_uvs} additional UV sets were dropped"),
            });
        }
        if !options.normals && attrs.get(names::SHADE_SMOOTH).is_some() {
            notes.push(LossyNote::Unsupported {
                what: format!("{name}: smoothing was dropped along with the normals"),
            });
        }
    }
}

/// Packs the JSON tree and binary blob into a `.glb`.
fn pack_glb(root: &json::Root, bin: &[u8]) -> Vec<u8> {
    const GLB_MAGIC: u32 = 0x46546C67;
    const CHUNK_JSON: u32 = 0x4E4F534A;
    const CHUNK_BIN: u32 = 0x004E4942;

    let mut js = serde_json::to_vec(root).expect("the json tree is built from owned data");
    // Chunks are padded to four bytes, JSON with spaces and binary with zeros —
    // the specification names the filler for each, and a reader that trims the
    // JSON chunk by length will choke on a trailing NUL.
    while !js.len().is_multiple_of(4) {
        js.push(b' ');
    }
    let mut blob = bin.to_vec();
    while !blob.len().is_multiple_of(4) {
        blob.push(0);
    }

    let mut out = Vec::with_capacity(28 + js.len() + blob.len());
    let total = 12 + 8 + js.len() + if blob.is_empty() { 0 } else { 8 + blob.len() };
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&js);
    if !blob.is_empty() {
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&blob);
    }
    out
}

/// Standard base64, for the `.gltf` data URI.
///
/// Hand-rolled rather than pulling a crate in for twenty lines that are used on
/// one code path, in a format that has not changed since 1987.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_view_starts_on_a_four_byte_boundary_after_an_odd_one() {
        // Tested here rather than through a written file, because every buffer
        // this exporter currently emits is already a multiple of four bytes —
        // positions and normals are twelve, UVs eight, every index four. A test
        // that reads the alignment off a real file therefore passes whether the
        // padding exists or not, which makes it a test of nothing.
        //
        // The padding is a guard for the attribute that is not here yet: a byte
        // colour, a half-float UV, a u16 index. This calls it directly so the
        // guard has a test that can actually fail.
        let mut root = json::Root::default();
        let mut bin = Vec::new();
        push_view(&mut root, &mut bin, &[1, 2, 3], None);
        let second = push_view(&mut root, &mut bin, &[4, 5, 6, 7], None);
        let offset = root.buffer_views[second.value()].byte_offset.unwrap().0;
        assert_eq!(offset, 4, "the second view landed at {offset}");
        assert_eq!(bin.len(), 8);
    }

    #[test]
    fn base64_matches_the_known_answers() {
        // The padding cases are the ones that go wrong, so all three lengths are
        // here. Values from RFC 4648's own test vectors.
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        // The high bits, which is where a wrong alphabet index or a sign
        // extension shows up and lowercase ASCII never would.
        assert_eq!(base64(&[0xff, 0xef, 0xbf]), "/++/");
    }
}
