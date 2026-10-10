//! glTF 2.0 import.
//!
//! The `gltf` crate does the decoding — glb containers, external and embedded
//! buffers, sparse accessors, every component type an accessor may use. That
//! part is tedious, well specified and easy to get subtly wrong, and there is
//! nothing to be gained by writing it again.
//!
//! What it does *not* do, and what this module is, is the semantics. glTF's
//! conventions differ from Kreate's in four ways that all have the same failure
//! mode — the model looks almost right — and each has to be handled exactly
//! once, here at the boundary:
//!
//! | | glTF | Kreate |
//! |---|---|---|
//! | Up axis | Y | Z |
//! | UV origin | top left | bottom left |
//! | Units | metres | metres |
//! | Faces | triangles only | n-gons |
//!
//! # The V flip, and why it has its own test
//!
//! glTF puts texture coordinate `(0, 0)` at the top left; Kreate, OBJ and most
//! of the rest of the world put it at the bottom left. Getting this wrong is
//! invisible on a model with a symmetric texture and catastrophic on anything
//! with text on it — and it is a single subtraction in one place, so nothing
//! about the code makes it look important. The plan says to write a test for it,
//! and there is one.
//!
//! The subtraction is **not** here. This module records `uv_origin` on the scene
//! and leaves the coordinates as the file wrote them; `ImportScene::into_mesh_data`
//! converts, exactly as it converts positions through `axis`. Doing it in both
//! places is not twice as safe — it is one flip too many, which is what happened,
//! and it survived every test that read the IR instead of the mesh it produces.
//!
//! # Welding
//!
//! glTF stores indexed triangles, so it is not soup in the way STL is — but a
//! vertex is split wherever *any* attribute differs, so a cube with per-face
//! normals arrives with 24 vertices rather than 8, and a UV seam splits every
//! vertex along it. Those splits are exactly what the corner domain exists to
//! represent, so the importer welds by position and lets the corner attributes
//! carry what the split was for.
//!
//! # `KREATE_ngons`
//!
//! A file this project wrote carries its pre-triangulation faces in a vendor
//! extension (see [`crate::gltf_export`]). When it is present the faces are read
//! from it and the triangles are ignored, which is what makes a Kreate mesh
//! survive a round trip through glTF with its quads intact. When it is absent,
//! or malformed, the triangles are used — the extension is an accelerator for
//! our own files, never a requirement.
//!
//! # What is deliberately not read yet
//!
//! Skinning, morph targets, animation, cameras and lights are all present in the
//! format and absent here. They are recorded as notes rather than ignored, so a
//! file that loses them says so. The plan wants skin weights and shape keys
//! carried through from day one; the *place* for them is the attribute store,
//! and adding them is additive once there is anything to do with them.

use crate::gltf_export::NGON_EXTENSION;
use crate::ir::{
    ChannelInterpolation, ChannelValues, ImportAnimation, ImportCamera, ImportChannel, ImportImage,
    ImportLight, ImportLightKind, ImportMaterial, ImportMesh, ImportNode, ImportNodeData,
    ImportScene, ImportTransform, LossyNote, SourceMetadata, UvOrigin,
};
use glam::{Vec2, Vec3};
use kreate_math::AxisConvention;

#[derive(Clone, Copy, Debug)]
pub struct GltfOptions {
    /// How close two vertices must be to be treated as one.
    ///
    /// Small: glTF's splits are exact duplicates of a position, so this is
    /// guarding against float noise from a exporter that recomputed a coordinate,
    /// not against a modelling tolerance.
    pub weld_epsilon: f32,
    /// Whether to weld at all.
    ///
    /// On by default. Off is for the rare case of wanting the file's vertex
    /// layout preserved exactly — a diagnostic, not a workflow.
    pub weld: bool,
}

impl Default for GltfOptions {
    fn default() -> Self {
        Self {
            weld_epsilon: 1e-6,
            weld: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GltfError {
    #[error("not a readable glTF file: {0}")]
    Parse(#[from] gltf::Error),
    #[error("mesh {mesh:?} primitive {primitive} has no positions")]
    NoPositions { mesh: String, primitive: usize },
    #[error("the file names {0:?}, which is not a file beside it")]
    Outside(String),
}

/// Reads a `.gltf` or `.glb` from memory.
///
/// Self-contained files only: a `.gltf` that references external `.bin` buffers
/// or image files needs those resolved relative to its path, which a byte slice
/// cannot do — [`parse_gltf_file`] does. `import_slice` handles embedded base64
/// buffers and `.glb`.
pub fn parse_gltf(bytes: &[u8], options: &GltfOptions) -> Result<ImportScene, GltfError> {
    let (document, buffers, images) = gltf::import_slice(bytes)?;
    from_imported(document, buffers, images, options)
}

/// Reads a `.gltf` or `.glb` from disk, with the buffers and images it names
/// beside it — what an exporter writing a folder, and Poly Haven, hand over.
///
/// **Only beside it.** A file says where its parts are, and a file that came
/// from somebody else could name `../../.ssh/id_rsa` or an absolute path;
/// read as an image, that would put a private file into a document somebody
/// might share. So every reference is checked before anything is read: an
/// embedded `data:` URI, or a relative path that does not climb out.
pub fn parse_gltf_file(
    path: &std::path::Path,
    options: &GltfOptions,
) -> Result<ImportScene, GltfError> {
    let bytes = std::fs::read(path).map_err(|e| GltfError::Parse(gltf::Error::Io(e)))?;
    let parsed = gltf::Gltf::from_slice(&bytes)?;
    let uris = parsed
        .buffers()
        .filter_map(|b| match b.source() {
            gltf::buffer::Source::Uri(u) => Some(u.to_string()),
            gltf::buffer::Source::Bin => None,
        })
        .chain(parsed.images().filter_map(|i| match i.source() {
            gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
            gltf::image::Source::View { .. } => None,
        }));
    for uri in uris {
        if !beside(&uri) {
            return Err(GltfError::Outside(uri));
        }
    }
    let (document, buffers, images) = gltf::import(path)?;
    from_imported(document, buffers, images, options)
}

/// Whether a glTF reference stays with its file: embedded, or a relative
/// path that does not climb out of the file's folder.
fn beside(uri: &str) -> bool {
    if uri.starts_with("data:") {
        return true;
    }
    // A scheme (`file:`, `http:`), a drive letter, or a root.
    if uri.contains(':') || uri.starts_with('/') || uri.starts_with('\\') {
        return false;
    }
    !uri.split(['/', '\\'])
        .any(|part| part == ".." || part.eq_ignore_ascii_case("%2e%2e"))
}

fn from_imported(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
    options: &GltfOptions,
) -> Result<ImportScene, GltfError> {
    let mut source = SourceMetadata::new("gltf");
    // Applied exactly once, here at the boundary. Every position, and every
    // normal, passes through this and nothing downstream knows the file was
    // Y-up.
    source.axis = AxisConvention::GLTF;
    // glTF's unit is the metre by specification — one of the few formats that
    // says so, and worth noting precisely because the others do not.
    source.unit_scale = 1.0;
    source.uv_origin = UvOrigin::TopLeft;
    // Straight from the json root: the typed API exposes no accessor for the
    // asset block, and knowing which program wrote a file is the first thing
    // worth having when its geometry turns out to be strange.
    source.generator = document.as_json().asset.generator.clone();

    let mut scene = ImportScene::new(source);
    read_nodes(&document, &mut scene);
    read_skins(&document, &buffers, &mut scene);
    read_animations(&document, &buffers, &mut scene);

    // Every image, in the file's order, so a texture's `source` index is an
    // index here too. Decoded already — `import_slice` unpacks embedded PNG and
    // JPEG — and widened or narrowed to the one layout the document takes.
    for (index, (data, image)) in images.iter().zip(document.images()).enumerate() {
        let name = image
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("image{index}"));
        let pixels = match rgba8(data) {
            Some(pixels) => pixels,
            None => {
                // A float image as a material map: glTF's core has no such
                // thing, so this is an extension's, and keeping its range
                // would need a float path this reader does not have. Said,
                // and left as an empty slot rather than a wrong texture.
                scene.source.notes.push(LossyNote::Unsupported {
                    what: format!("{name}: a {:?} image was not read", data.format),
                });
                Vec::new()
            }
        };
        scene.images.push(ImportImage {
            name,
            width: if pixels.is_empty() { 0 } else { data.width },
            height: if pixels.is_empty() { 0 } else { data.height },
            pixels,
        });
    }

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        // An older file's specular-glossiness material — Sketchfab's of a
        // few years ago — gives its colour as diffuse; read as the base
        // colour, a fair stand-in, where the metallic-roughness one is the
        // default it would otherwise be.
        let gloss = material.pbr_specular_glossiness();
        let base = match &gloss {
            Some(g) => g.diffuse_factor(),
            None => pbr.base_color_factor(),
        };
        let material_name = material.name().unwrap_or("material").to_string();
        // Only the first coordinate set is read, so a texture that asks for
        // another would land on the wrong coordinates. Kept — the image is
        // right and the mapping is the approximation — and said.
        let mut texture = |slot: &str, tex_coord: u32, source: usize| {
            if tex_coord != 0 {
                scene.source.notes.push(LossyNote::Unsupported {
                    what: format!(
                        "{material_name}: the {slot} map uses texture coordinates {tex_coord}; \
                         only the first set is read"
                    ),
                });
            }
            Some(source)
        };
        let base_color_texture = match &gloss {
            Some(g) => g
                .diffuse_texture()
                .and_then(|t| texture("diffuse", t.tex_coord(), t.texture().source().index())),
            None => pbr
                .base_color_texture()
                .and_then(|t| texture("base colour", t.tex_coord(), t.texture().source().index())),
        };
        let metallic_roughness_texture = pbr.metallic_roughness_texture().and_then(|t| {
            texture(
                "metallic-roughness",
                t.tex_coord(),
                t.texture().source().index(),
            )
        });
        let emissive_texture = material
            .emissive_texture()
            .and_then(|t| texture("emission", t.tex_coord(), t.texture().source().index()));
        let (normal_texture, normal_scale) = match material.normal_texture() {
            Some(t) => (
                texture("normal", t.tex_coord(), t.texture().source().index()),
                t.scale(),
            ),
            None => (None, 1.0),
        };
        scene.materials.push(ImportMaterial {
            name: material
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("material{}", scene.materials.len())),
            base_color: [base[0], base[1], base[2], base[3]],
            metallic: if gloss.is_some() {
                0.0
            } else {
                pbr.metallic_factor()
            },
            roughness: match &gloss {
                Some(g) => 1.0 - g.glossiness_factor(),
                None => pbr.roughness_factor(),
            },
            emissive: Vec3::from_array(material.emissive_factor()),
            base_color_texture,
            metallic_roughness_texture,
            normal_texture,
            emissive_texture,
            normal_scale,
            emissive_strength: material.emissive_strength().unwrap_or(1.0),
            // `BLEND` is see-through in a way a rasteriser blends; here it is
            // glass, which is what a model's blended part — a clock's front, a
            // lamp's shade — almost always is. Unless the file already says
            // how much passes through.
            transmission: material.transmission().map_or(
                if material.alpha_mode() == gltf::material::AlphaMode::Blend {
                    1.0
                } else {
                    0.0
                },
                |t| t.transmission_factor(),
            ),
            ior: material.ior().unwrap_or(1.5),
            alpha_clip: (material.alpha_mode() == gltf::material::AlphaMode::Mask)
                .then(|| material.alpha_cutoff().unwrap_or(0.5)),
        });
    }

    for mesh in document.meshes() {
        let name = mesh
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("mesh{}", mesh.index()));

        // Primitives are merged into one mesh, with the material carried per
        // face. A primitive is a material batch, not a separate object — glTF
        // splits by material because a draw call must, and importing each as its
        // own mesh would fragment a single model into a dozen pieces the user
        // then has to rejoin.
        let mut positions: Vec<Vec3> = Vec::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        let mut uvs: Vec<Vec2> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut face_materials: Vec<i32> = Vec::new();
        let mut any_uvs = false;
        let mut any_normals = false;
        let mut vertex_skin = crate::ir::VertexSkin::default();
        let mut any_skin = false;
        // Morph targets: names from `extras.targetNames`, where Blender
        // writes them, and a displacement per vertex for each.
        let target_names: Vec<String> = mesh
            .extras()
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw.get()).ok())
            .and_then(|v| {
                v.get("targetNames")?.as_array().map(|a| {
                    a.iter()
                        .map(|n| n.as_str().unwrap_or_default().to_string())
                        .collect()
                })
            })
            .unwrap_or_default();
        let target_count = mesh
            .primitives()
            .map(|p| p.morph_targets().len())
            .max()
            .unwrap_or(0);
        let mut morphs: Vec<Vec<Vec3>> = vec![Vec::new(); target_count];

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                scene.source.notes.push(LossyNote::Unsupported {
                    what: format!(
                        "{name}: a {:?} primitive was skipped; only triangles are read",
                        primitive.mode()
                    ),
                });
                continue;
            }
            let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let Some(vertices) = reader.read_positions() else {
                return Err(GltfError::NoPositions {
                    mesh: name.clone(),
                    primitive: primitive.index(),
                });
            };
            let base = positions.len() as u32;
            let vertices: Vec<Vec3> = vertices.map(Vec3::from_array).collect();
            let count = vertices.len();
            positions.extend(vertices);
            // Joints and weights per vertex, kept in step with the positions
            // across primitives whether or not this one has them.
            let joints: Vec<[u16; 4]> = reader
                .read_joints(0)
                .map(|j| j.into_u16().collect())
                .unwrap_or_default();
            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|w| w.into_f32().collect())
                .unwrap_or_default();
            any_skin |= !joints.is_empty() && !weights.is_empty();
            let mut read: Vec<Vec<Vec3>> = reader
                .read_morph_targets()
                .map(|(positions, _, _)| {
                    positions
                        .map(|p| p.map(Vec3::from_array).collect())
                        .unwrap_or_default()
                })
                .collect();
            read.resize(target_count, Vec::new());
            for (target, out) in read.iter().zip(morphs.iter_mut()) {
                for v in 0..count {
                    out.push(target.get(v).copied().unwrap_or(Vec3::ZERO));
                }
            }
            for v in 0..count {
                vertex_skin
                    .joints
                    .push(joints.get(v).copied().unwrap_or([0; 4]));
                vertex_skin
                    .weights
                    .push(weights.get(v).copied().unwrap_or([0.0; 4]));
            }

            let primitive_uvs: Option<Vec<Vec2>> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().map(Vec2::from_array).collect());
            let primitive_normals: Option<Vec<Vec3>> = reader
                .read_normals()
                .map(|n| n.map(Vec3::from_array).collect());
            any_uvs |= primitive_uvs.is_some();
            any_normals |= primitive_normals.is_some();

            // Indices are optional: without them the vertices are consumed in
            // order, three at a time.
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..count as u32).collect(),
            };
            let material = primitive.material().index().map_or(-1, |i| i as i32);

            // The faces this primitive contributes, indexed locally. One per
            // triangle normally; the original n-gons when the file records them.
            let local_faces: Vec<Vec<u32>> = match ngon_faces(
                &document,
                &buffers,
                mesh.index(),
                primitive.index(),
                &name,
                &mut scene.source.notes,
            ) {
                Some(ngons) => ngons,
                None => indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| t.to_vec())
                    .collect(),
            };

            for face in local_faces {
                // A file can index past its own vertex array. Dropped rather
                // than clamped: a clamped index makes a face that is silently
                // in the wrong place, which is harder to notice than a missing
                // one and impossible to explain. This covers the extension's
                // vertex list too, which is no more trustworthy than the
                // triangles it replaces.
                if face.iter().any(|i| *i as usize >= count) {
                    scene.source.notes.push(LossyNote::OutOfRangeIndex {
                        mesh: name.clone(),
                        face: faces.len(),
                    });
                    continue;
                }
                faces.push(face.iter().map(|i| base + i).collect());
                face_materials.push(material);
                for i in &face {
                    let local = *i as usize;
                    // Corner arrays are filled for *every* corner even when this
                    // primitive lacks the attribute, so they stay aligned with
                    // the face list across primitives that differ. A mesh whose
                    // first primitive has UVs and second does not would otherwise
                    // shift every UV after the join.
                    // Left in the file's own origin. `uv_origin` is declared on
                    // the scene and `ImportScene::into_mesh_data` applies it,
                    // the same way it applies the axis convention to positions —
                    // flipping here as well meant the flip happened twice and
                    // every glTF arrived with its V inverted.
                    uvs.push(
                        primitive_uvs
                            .as_ref()
                            .and_then(|u| u.get(local).copied())
                            .unwrap_or(Vec2::ZERO),
                    );
                    normals.push(
                        primitive_normals
                            .as_ref()
                            .and_then(|n| n.get(local).copied())
                            .unwrap_or(Vec3::Z),
                    );
                }
            }
        }

        let mut import = ImportMesh {
            name,
            positions,
            faces,
            corner_uvs: any_uvs.then_some(uvs),
            corner_normals: any_normals.then_some(normals),
            face_materials: (!face_materials.is_empty()).then_some(face_materials),
            smoothing_groups: None,
        };
        if options.weld {
            // glTF splits a vertex wherever any attribute differs. Those splits
            // are what the corner domain represents, so welding by position
            // recovers the surface the author modelled while the UVs and normals
            // keep the distinctions the split was made for.
            if let Some(map) = import.weld_mapped(options.weld_epsilon, &mut scene.source.notes) {
                // Morph targets follow the same way, first copy wins.
                let n = import.positions.len();
                for target in morphs.iter_mut() {
                    let mut out = vec![Vec3::ZERO; n];
                    let mut done = vec![false; n];
                    for (old, new) in map.iter().enumerate() {
                        let new = *new as usize;
                        if new < n && !done[new] && old < target.len() {
                            out[new] = target[old];
                            done[new] = true;
                        }
                    }
                    *target = out;
                }
                // A merged vertex keeps the first copy's joints: the copies of
                // one skinned vertex carry the same ones.
                let n = import.positions.len();
                let mut out = crate::ir::VertexSkin {
                    joints: vec![[0; 4]; n],
                    weights: vec![[0.0; 4]; n],
                };
                let mut done = vec![false; n];
                for (old, new) in map.iter().enumerate() {
                    let new = *new as usize;
                    if new < n && !done[new] && old < vertex_skin.joints.len() {
                        out.joints[new] = vertex_skin.joints[old];
                        out.weights[new] = vertex_skin.weights[old];
                        done[new] = true;
                    }
                }
                vertex_skin = out;
            }
        }
        scene.vertex_skins.push(any_skin.then_some(vertex_skin));
        scene.vertex_morphs.push(
            morphs
                .into_iter()
                .enumerate()
                .map(|(i, d)| {
                    let name = target_names
                        .get(i)
                        .filter(|n| !n.is_empty())
                        .cloned()
                        .unwrap_or_else(|| format!("Key {}", i + 1));
                    (name, d)
                })
                .collect(),
        );
        scene.meshes.push(import);
    }

    note_unread_features(&document, &mut scene);
    Ok(scene)
}

/// Reads the faces recorded in `KREATE_ngons`, if this primitive has them.
///
/// Returns `None` for every reason that is not "the faces are here and they add
/// up" — no extension, a malformed one, accessors of the wrong type. The caller
/// then uses the triangles, which are always present and always correct. That is
/// the point of putting this in an optional extension: a file with a damaged
/// record is a file that loses its quads, not a file that fails to open.
/// Reads the file's hierarchy, and the lights and cameras hanging off it.
///
/// Until this existed the importer walked `document.meshes()` and threw the
/// node tree away, so every mesh arrived at the origin and a light had nowhere
/// to be. That was survivable while everything imported had its transform baked
/// in, and stopped being survivable the moment a light was worth importing: a
/// light *is* a placement and a datablock, with nothing else to it.
///
/// `mesh_of` maps a glTF mesh index onto the index the IR gave it, because the
/// two need not agree — they do today, and relying on that would be a rule
/// nobody wrote down.
/// Every clip, channel by channel, as the file wrote it.
///
/// Morph target weights are said and skipped: the document has no shape keys
/// yet (Phase 11). A channel whose lists cannot be read, or that names a node
/// the file does not have, is skipped and said too — one broken channel is not
/// a reason to lose the rest of the clip.
fn read_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    scene: &mut ImportScene,
) {
    use gltf::animation::util::ReadOutputs;
    for (index, animation) in document.animations().enumerate() {
        let name = animation
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("Animation{index}"));
        let mut out = ImportAnimation {
            name: name.clone(),
            channels: Vec::new(),
        };
        for channel in animation.channels() {
            let node = channel.target().node().index();
            let reader = channel.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let times: Option<Vec<f32>> = reader.read_inputs().map(|i| i.collect());
            let values = match reader.read_outputs() {
                Some(ReadOutputs::Translations(v)) => {
                    Some(ChannelValues::Translation(v.map(Vec3::from).collect()))
                }
                Some(ReadOutputs::Rotations(v)) => Some(ChannelValues::Rotation(
                    v.into_f32().map(glam::Quat::from_array).collect(),
                )),
                Some(ReadOutputs::Scales(v)) => {
                    Some(ChannelValues::Scale(v.map(Vec3::from).collect()))
                }
                Some(ReadOutputs::MorphTargetWeights(w)) => {
                    if let Some(times) = reader.read_inputs().map(|i| i.collect()) {
                        scene.morph_channels.push(crate::ir::ImportMorphChannel {
                            animation: name.clone(),
                            node,
                            interpolation: match channel.sampler().interpolation() {
                                gltf::animation::Interpolation::Step => ChannelInterpolation::Step,
                                gltf::animation::Interpolation::Linear => {
                                    ChannelInterpolation::Linear
                                }
                                gltf::animation::Interpolation::CubicSpline => {
                                    ChannelInterpolation::CubicSpline
                                }
                            },
                            times,
                            values: w.into_f32().collect(),
                        });
                    }
                    continue;
                }
                None => None,
            };
            let (Some(times), Some(values)) = (times, values) else {
                scene.source.notes.push(LossyNote::Unsupported {
                    what: format!("{name}: a channel whose keys could not be read"),
                });
                continue;
            };
            if node >= scene.nodes.len() {
                continue;
            }
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Step => ChannelInterpolation::Step,
                gltf::animation::Interpolation::Linear => ChannelInterpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => ChannelInterpolation::CubicSpline,
            };
            out.channels.push(ImportChannel {
                node,
                interpolation,
                times,
                values,
            });
        }
        if !out.channels.is_empty() {
            scene.animations.push(out);
        }
    }
}

fn read_nodes(document: &gltf::Document, scene: &mut ImportScene) {
    // `lights()` answers nothing at all when the file does not declare the
    // extension, which is not the same as declaring it with no lights — and
    // flattening the two here is right, because the IR's answer to both is an
    // empty list.
    for light in document.lights().into_iter().flatten() {
        let kind = match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => ImportLightKind::Directional,
            gltf::khr_lights_punctual::Kind::Point => ImportLightKind::Point,
            gltf::khr_lights_punctual::Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => ImportLightKind::Spot {
                inner: inner_cone_angle,
                outer: outer_cone_angle,
            },
        };
        scene.lights.push(ImportLight {
            name: light
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("light{}", light.index())),
            kind,
            color: light.color(),
            // Candela for a point or a spot and lux for a directional light,
            // which is what the datablock stores. §2.2 chose these units so
            // that this line is a copy.
            intensity: light.intensity(),
            range: light.range(),
        });
    }

    for camera in document.cameras() {
        let (fov_y, ortho_width, near, far) = match camera.projection() {
            gltf::camera::Projection::Perspective(p) => (Some(p.yfov()), None, p.znear(), p.zfar()),
            gltf::camera::Projection::Orthographic(o) => {
                // glTF writes half-widths; the IR wants the whole width, which
                // is what an orthographic scale means everywhere else.
                (None, Some(o.xmag() * 2.0), o.znear(), Some(o.zfar()))
            }
        };
        scene.cameras.push(ImportCamera {
            name: camera
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("camera{}", camera.index())),
            fov_y,
            ortho_width,
            clip_near: near,
            clip_far: far,
        });
    }

    // glTF node indices are dense and the IR's are too, so one pass fills the
    // nodes and a second wires the parents — a child can be written before its
    // parent and an index into a half-built list would be wrong.
    let count = document.nodes().count();
    scene.nodes = document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            let mut out = ImportNode::new(
                node.name()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("node{}", node.index())),
            );
            out.transform = ImportTransform {
                translation: Vec3::from_array(translation),
                rotation: glam::Quat::from_array(rotation),
                scale: Vec3::from_array(scale),
            };
            out.skin = node.skin().map(|s| s.index());
            let weights = node
                .weights()
                .or_else(|| node.mesh().and_then(|m| m.weights()))
                .map(<[f32]>::to_vec)
                .unwrap_or_default();
            if scene.node_weights.len() <= node.index() {
                scene.node_weights.resize(node.index() + 1, Vec::new());
            }
            scene.node_weights[node.index()] = weights;
            out.data = if let Some(mesh) = node.mesh() {
                ImportNodeData::Mesh(mesh.index())
            } else if let Some(light) = node.light() {
                ImportNodeData::Light(light.index())
            } else if let Some(camera) = node.camera() {
                ImportNodeData::Camera(camera.index())
            } else {
                // An empty, and kept. A file's hierarchy is mostly empties that
                // group and get animated; dropping them flattens a rig into a
                // pile of parts.
                ImportNodeData::Empty
            };
            out
        })
        .collect();

    for node in document.nodes() {
        for child in node.children() {
            if child.index() < count {
                scene.nodes[child.index()].parent = Some(node.index());
            }
        }
    }
}

/// The skins: joints by node index and the inverse bind matrices, in the
/// file's space. The meshes' joints and weights are read with the meshes.
fn read_skins(document: &gltf::Document, buffers: &[gltf::buffer::Data], scene: &mut ImportScene) {
    for skin in document.skins() {
        let reader = skin.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        scene.skins.push(crate::ir::ImportSkin {
            name: skin
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("skin{}", skin.index())),
            joints: skin.joints().map(|n| n.index()).collect(),
            inverse_bind: reader
                .read_inverse_bind_matrices()
                .map(|m| m.map(|c| glam::Mat4::from_cols_array_2d(&c)).collect()),
        });
    }
}

fn ngon_faces(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    mesh_index: usize,
    primitive_index: usize,
    mesh_name: &str,
    notes: &mut Vec<LossyNote>,
) -> Option<Vec<Vec<u32>>> {
    let record = document
        .as_json()
        .meshes
        .get(mesh_index)?
        .primitives
        .get(primitive_index)?
        .extensions
        .as_ref()?
        .others
        .get(NGON_EXTENSION)?;
    let sizes = read_u32_accessor(
        document,
        buffers,
        record.get("faceSizes")?.as_u64()? as usize,
    )?;
    let vertices = read_u32_accessor(
        document,
        buffers,
        record.get("faceVertices")?.as_u64()? as usize,
    )?;

    // The two arrays have to agree, and a file where they do not is one where
    // the faces would be read shifted — every face after the mismatch made of
    // the wrong vertices. Falling back to the triangles is the only honest
    // response, and it is worth saying out loud.
    let total: u64 = sizes.iter().map(|&n| u64::from(n)).sum();
    if total != vertices.len() as u64 {
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{mesh_name}: the {NGON_EXTENSION} record claims {total} corners \
                 but lists {}; the triangles were used instead",
                vertices.len()
            ),
        });
        return None;
    }

    let mut faces = Vec::with_capacity(sizes.len());
    let mut at = 0usize;
    for size in sizes {
        let size = size as usize;
        if size >= 3 {
            faces.push(vertices[at..at + size].to_vec());
        }
        at += size;
    }
    Some(faces)
}

/// Reads a scalar `u32` accessor as a plain vector.
///
/// Deliberately strict about the component type rather than converting: this
/// reads an extension written by this project, which always writes `u32`
/// scalars. Anything else in that slot means the file is not what it claims,
/// and quietly widening a `u16` would be reading a file no one wrote.
fn read_u32_accessor(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    index: usize,
) -> Option<Vec<u32>> {
    let accessor = document.accessors().nth(index)?;
    if accessor.data_type() != gltf::accessor::DataType::U32
        || accessor.dimensions() != gltf::accessor::Dimensions::Scalar
    {
        return None;
    }
    let view = accessor.view()?;
    let data = &buffers.get(view.buffer().index())?.0;
    let stride = view.stride().unwrap_or(4);
    let start = view.offset() + accessor.offset();
    let mut out = Vec::with_capacity(accessor.count());
    for i in 0..accessor.count() {
        let at = start.checked_add(i.checked_mul(stride)?)?;
        let bytes = data.get(at..at.checked_add(4)?)?;
        out.push(u32::from_le_bytes(bytes.try_into().ok()?));
    }
    Some(out)
}

/// Records what the file contains and this importer does not read.
///
/// Not silence. A file whose rig or animation is dropped has to say so, or the
/// user discovers it when the export they made for someone else turns out to be
/// a static pose.
fn note_unread_features(document: &gltf::Document, scene: &mut ImportScene) {
    // Measured against what was read, not by what the file has: this list
    // once said every texture went unread long after textures were, and a
    // note that is always there is a note nobody reads.
    let mut notes = Vec::new();
    let mut mention = |what: String| notes.push(LossyNote::Unsupported { what });
    let missing = |present: usize, read: usize| present.saturating_sub(read);
    let n = missing(document.animations().len(), scene.animations.len());
    if n > 0 {
        mention(format!("{n} animations were not read"));
    }
    let n = missing(document.skins().len(), scene.skins.len());
    if n > 0 {
        mention(format!("{n} skins were not read"));
    }
    let n = missing(document.cameras().len(), scene.cameras.len());
    if n > 0 {
        mention(format!("{n} cameras were not read"));
    }
    // The maps a standard surface here has no slot for.
    let occlusion = document
        .materials()
        .filter(|m| m.occlusion_texture().is_some())
        .count();
    if occlusion > 0 {
        mention(format!("{occlusion} occlusion maps were not read"));
    }
    scene.source.notes.extend(notes);
}

/// A decoded glTF image as eight-bit RGBA, or `None` for a float one.
///
/// Sixteen-bit channels keep their high byte: a material map is sampled
/// through an eight-bit atlas anyway, and rounding here rather than there is
/// the same answer. The `gltf` crate hands sixteen-bit data over in the
/// machine's own byte order.
fn rgba8(data: &gltf::image::Data) -> Option<Vec<u8>> {
    use gltf::image::Format;
    let count = data.width as usize * data.height as usize;
    let wide = |i: usize| -> u8 {
        let at = i * 2;
        data.pixels
            .get(at..at + 2)
            .map(|b| (u16::from_ne_bytes([b[0], b[1]]) >> 8) as u8)
            .unwrap_or(0)
    };
    let narrow = |i: usize| data.pixels.get(i).copied().unwrap_or(0);
    let mut out = Vec::with_capacity(count * 4);
    for p in 0..count {
        let texel = match data.format {
            Format::R8 => {
                let v = narrow(p);
                [v, v, v, 255]
            }
            Format::R8G8 => [narrow(p * 2), narrow(p * 2 + 1), 0, 255],
            Format::R8G8B8 => [narrow(p * 3), narrow(p * 3 + 1), narrow(p * 3 + 2), 255],
            Format::R8G8B8A8 => [
                narrow(p * 4),
                narrow(p * 4 + 1),
                narrow(p * 4 + 2),
                narrow(p * 4 + 3),
            ],
            Format::R16 => {
                let v = wide(p);
                [v, v, v, 255]
            }
            Format::R16G16 => [wide(p * 2), wide(p * 2 + 1), 0, 255],
            Format::R16G16B16 => [wide(p * 3), wide(p * 3 + 1), wide(p * 3 + 2), 255],
            Format::R16G16B16A16 => [
                wide(p * 4),
                wide(p * 4 + 1),
                wide(p * 4 + 2),
                wide(p * 4 + 3),
            ],
            Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => return None,
        };
        out.extend_from_slice(&texel);
    }
    Some(out)
}
