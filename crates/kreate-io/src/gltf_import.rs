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
    ImportCamera, ImportLight, ImportLightKind, ImportMaterial, ImportMesh, ImportNode,
    ImportNodeData, ImportScene, ImportTransform, LossyNote, SourceMetadata, UvOrigin,
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
}

/// Reads a `.gltf` or `.glb` from memory.
///
/// Self-contained files only: a `.gltf` that references external `.bin` buffers
/// or image files needs those resolved relative to its path, which a byte slice
/// cannot do. `import_slice` handles embedded base64 buffers and `.glb`, which
/// covers the overwhelming majority of files that get passed around.
pub fn parse_gltf(bytes: &[u8], options: &GltfOptions) -> Result<ImportScene, GltfError> {
    let (document, buffers, _images) = gltf::import_slice(bytes)?;

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

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base = pbr.base_color_factor();
        scene.materials.push(ImportMaterial {
            name: material
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("material{}", scene.materials.len())),
            base_color: [base[0], base[1], base[2], base[3]],
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: Vec3::from_array(material.emissive_factor()),
            // The name only. Reading the pixels is a separate concern, and a
            // material that remembers which texture it wanted can be reconnected
            // later; one that forgot cannot.
            base_color_texture: pbr
                .base_color_texture()
                .and_then(|t| t.texture().source().name().map(str::to_string)),
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
            import.weld(options.weld_epsilon, &mut scene.source.notes);
        }
        scene.meshes.push(import);
    }

    note_unread_features(&document, &mut scene.source.notes);
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
            gltf::camera::Projection::Perspective(p) => {
                (Some(p.yfov()), None, p.znear(), p.zfar())
            }
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
fn note_unread_features(document: &gltf::Document, notes: &mut Vec<LossyNote>) {
    let mut mention = |what: String| notes.push(LossyNote::Unsupported { what });

    if document.animations().len() > 0 {
        mention(format!(
            "{} animations were not read",
            document.animations().len()
        ));
    }
    if document.skins().len() > 0 {
        mention(format!("{} skins were not read", document.skins().len()));
    }
    if document.cameras().len() > 0 {
        mention(format!(
            "{} cameras were not read",
            document.cameras().len()
        ));
    }
    let morph_targets: usize = document
        .meshes()
        .flat_map(|m| m.primitives())
        .map(|p| p.morph_targets().len())
        .sum();
    if morph_targets > 0 {
        mention(format!("{morph_targets} morph targets were not read"));
    }
    if document.textures().len() > 0 {
        mention(format!(
            "{} textures were not read; materials keep their factors only",
            document.textures().len()
        ));
    }
}
