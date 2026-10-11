//! USD, as text: `.usda`.
//!
//! The scene as a USD stage — meshes, the hierarchy, materials as
//! `UsdPreviewSurface` with their pictures, lights as `UsdLux`, cameras —
//! written as the ASCII layer every USD tool reads, or packed as `.usdz`.
//! Reading is [`crate::usd_read`]'s.
//!
//! A rig goes out as UsdSkel: an armature as a `Skeleton` (its bones'
//! binds and rests) with a `SkelAnimation` of its pose over time, a mesh it
//! bends with its joints, weights and bind transform, and shape keys as
//! `BlendShape`s whose weights the animation keys — a mesh with shape keys
//! and no armature getting a skeleton of no joints to play them, as UsdSkel
//! asks. The stage's root is then a `SkelRoot`.
//!
//! **The stage is Z-up and in metres**, as the document is, so nothing is
//! converted: `upAxis = "Z"` says so, and a reader that wants Y-up rotates
//! the stage rather than every point. Transforms are written as matrices, the
//! one form every reader agrees on.
//!
//! What is lost is said, as every writer here says it: an area light goes
//! out as the point the IR made of it, a light's units are translated into
//! UsdLux's, a camera keeps its angle and not its lens.

use std::fmt::Write as _;

use crate::ir::{
    ExportScene, ExportTrack, ImportLightKind, ImportNodeData, LossyNote, PointSamples,
};

/// Time codes a second: the stage's clock. Every time written is seconds
/// times this, so the scene's own frame rate does not have to match.
const TIME_CODES: f32 = 24.0;
use kreate_math::Vec3;
use kreate_mesh::attribute::names;

/// The stage, and what it could not say. Pictures its materials use are
/// named `textures/…png` beside it; [`write_usd`] gives them too.
pub fn write_usda(scene: &ExportScene<'_>) -> (String, Vec<LossyNote>) {
    let package = write_usd(scene);
    (package.text, package.notes)
}

/// A stage and the pictures it names, by the path it names them by.
pub struct UsdPackage {
    pub text: String,
    pub files: Vec<(String, Vec<u8>)>,
    pub notes: Vec<LossyNote>,
}

/// The stage as one `.usdz`: the stage first, then its pictures, each
/// stored as it is and on a 64-byte boundary, as the format asks.
pub fn write_usdz(scene: &ExportScene<'_>) -> (Vec<u8>, Vec<LossyNote>) {
    use std::io::Write;
    let package = write_usd(scene);
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .with_alignment(64);
    let mut notes = package.notes;
    let mut failed = None;
    let mut entries = vec![("scene.usda".to_string(), package.text.into_bytes())];
    entries.extend(package.files);
    for (name, bytes) in &entries {
        let result = zip
            .start_file(name.as_str(), options)
            .map_err(|e| e.to_string())
            .and_then(|_| zip.write_all(bytes).map_err(|e| e.to_string()));
        if let Err(why) = result {
            failed = Some(why);
            break;
        }
    }
    if let Some(why) = failed {
        notes.push(LossyNote::Unsupported {
            what: format!("the package could not be written whole: {why}"),
        });
    }
    let bytes = zip
        .finish()
        .map(std::io::Cursor::into_inner)
        .unwrap_or_default();
    (bytes, notes)
}

/// The stage, and the pictures its materials name.
pub fn write_usd(scene: &ExportScene<'_>) -> UsdPackage {
    let mut notes = Vec::new();
    // Each picture a material uses, as a PNG beside the stage.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut texture_paths: Vec<Option<String>> = vec![None; scene.images.len()];
    let used: Vec<usize> = scene
        .materials
        .iter()
        .flat_map(|m| {
            [
                m.base_color_texture,
                m.metallic_roughness_texture,
                m.normal_texture,
                m.emissive_texture,
                m.occlusion_texture,
            ]
        })
        .flatten()
        .collect();
    let image_names = unique(scene.images.iter().map(|i| i.name.as_str()));
    for i in used {
        if texture_paths.get(i).is_none_or(Option::is_some) {
            continue;
        }
        let image = &scene.images[i];
        let Some(rgba) =
            image::RgbaImage::from_raw(image.width, image.height, image.pixels.clone())
        else {
            continue;
        };
        let mut png = Vec::new();
        if rgba
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .is_ok()
        {
            let path = format!("textures/{}.png", image_names[i]);
            files.push((path.clone(), png));
            texture_paths[i] = Some(path);
        }
    }
    // The hierarchy, its prims' names and paths, and its rigs: worked out
    // first, because a rig makes the stage's root a skeleton root.
    let layout = Layout::of(scene);
    let rigs = Rigs::of(scene, &layout);
    let mut out = String::new();
    let _ = writeln!(out, "#usda 1.0");
    let _ = writeln!(out, "(");
    let _ = writeln!(out, "    defaultPrim = \"World\"");
    let _ = writeln!(out, "    metersPerUnit = 1");
    let _ = writeln!(out, "    upAxis = \"Z\"");
    let _ = writeln!(out, "    doc = \"Written by Kreate\"");
    // The range time samples cover, when anything moves.
    let last = scene
        .animations
        .iter()
        .flat_map(|a| a.tracks.iter().flat_map(|t| t.times.iter()))
        .chain(scene.point_samples.iter().flat_map(|s| s.times.iter()))
        .chain(
            scene
                .animations
                .iter()
                .flat_map(|a| a.weights.iter().flat_map(|w| w.times.iter())),
        )
        .fold(None, |m: Option<f32>, t| Some(m.map_or(*t, |m| m.max(*t))));
    if let Some(last) = last {
        let _ = writeln!(out, "    startTimeCode = 0");
        let _ = writeln!(out, "    endTimeCode = {}", (last * TIME_CODES).ceil());
        let _ = writeln!(out, "    timeCodesPerSecond = {TIME_CODES}");
        let _ = writeln!(out, "    framesPerSecond = {TIME_CODES}");
    }
    let _ = writeln!(out, ")");
    let _ = writeln!(out);
    // A skinned mesh and its skeleton must share a skeleton root; the whole
    // stage is one.
    let root_type = if rigs.any() { "SkelRoot" } else { "Xform" };
    let _ = writeln!(out, "def {root_type} \"World\"");
    let _ = writeln!(out, "{{");

    // Materials first, under one scope, so a mesh binds one by path.
    let material_names = unique(scene.materials.iter().map(|m| m.name.as_str()));
    if !scene.materials.is_empty() {
        let _ = writeln!(out, "    def Scope \"Looks\"");
        let _ = writeln!(out, "    {{");
        for (m, name) in scene.materials.iter().zip(&material_names) {
            write_material(&mut out, m, name, 2, &texture_paths);
        }
        let _ = writeln!(out, "    }}");
    }

    if scene.nodes.is_empty() {
        // A bare mesh list: each at the origin, as the file puts them.
        let names = unique(scene.meshes.iter().map(|(n, _)| n.as_str()));
        for (index, name) in names.iter().enumerate() {
            write_mesh(
                &mut out,
                scene,
                index,
                name,
                None,
                &material_names,
                1,
                &mut notes,
                None,
            );
        }
    } else {
        // Each node under its parent, by nesting. A rig's joints are in its
        // skeleton, not prims of their own.
        let Layout {
            children,
            roots,
            names,
            ..
        } = &layout;
        let mut stack: Vec<(usize, usize, bool)> =
            roots.iter().rev().map(|r| (*r, 1usize, false)).collect();
        while let Some((i, depth, closing)) = stack.pop() {
            let pad = "    ".repeat(depth);
            if closing {
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
            if rigs.joint[i] {
                continue;
            }
            write_node(
                &mut out,
                scene,
                i,
                &names[i],
                &material_names,
                depth,
                &mut notes,
                &rigs,
            );
            if let Some(skel) = rigs.skeletons.get(&i) {
                out.push_str(&skel.text(depth + 1));
            }
            stack.push((i, depth, true));
            for c in children[i].iter().rev() {
                stack.push((*c, depth + 1, false));
            }
        }
    }
    let _ = writeln!(out, "}}");
    UsdPackage {
        text: out,
        files,
        notes,
    }
}

/// A name USD accepts for a prim: letters, digits and underscores, not
/// starting with a digit.
pub fn prim_name(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    s
}

/// Prim names for `names`, made unique among themselves by a suffix.
fn unique<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    names
        .map(|n| {
            let base = prim_name(n);
            let mut name = base.clone();
            let mut k = 1;
            while !seen.insert(name.clone()) {
                name = format!("{base}_{k}");
                k += 1;
            }
            name
        })
        .collect()
}

fn matrix(m: &kreate_math::DAffine3) -> String {
    // USD's matrices are row vectors: each row is a basis vector, the last
    // the translation.
    let c = m.matrix3;
    let t = m.translation;
    format!(
        "( ({}, {}, {}, 0), ({}, {}, {}, 0), ({}, {}, {}, 0), ({}, {}, {}, 1) )",
        c.x_axis.x,
        c.x_axis.y,
        c.x_axis.z,
        c.y_axis.x,
        c.y_axis.y,
        c.y_axis.z,
        c.z_axis.x,
        c.z_axis.y,
        c.z_axis.z,
        t.x,
        t.y,
        t.z
    )
}

fn write_transform(
    out: &mut String,
    node: &crate::ir::ExportNode,
    pad: &str,
    track: Option<&ExportTrack>,
) {
    let _ = writeln!(
        out,
        "{pad}matrix4d xformOp:transform = {}",
        matrix(&node.transform.to_affine())
    );
    if let Some(track) = track {
        // Each time's whole placement; a step track is held to the next.
        let _ = writeln!(out, "{pad}matrix4d xformOp:transform.timeSamples = {{");
        for (t, at) in track.times.iter().zip(&track.transforms) {
            let _ = writeln!(
                out,
                "{pad}    {}: {},",
                t * TIME_CODES,
                matrix(&at.to_affine())
            );
        }
        let _ = writeln!(out, "{pad}}}");
    }
    let _ = writeln!(
        out,
        "{pad}uniform token[] xformOpOrder = [\"xformOp:transform\"]"
    );
}

/// The track that moves node `i`, if one does.
fn track_of<'a>(scene: &'a ExportScene<'_>, i: usize) -> Option<&'a ExportTrack> {
    scene
        .animations
        .iter()
        .flat_map(|a| &a.tracks)
        .find(|t| t.node == i)
}

#[allow(clippy::too_many_arguments)]
fn write_node(
    out: &mut String,
    scene: &ExportScene<'_>,
    i: usize,
    name: &str,
    materials: &[String],
    depth: usize,
    notes: &mut Vec<LossyNote>,
    rigs: &Rigs,
) {
    let node = &scene.nodes[i];
    let pad = "    ".repeat(depth);
    let inner = "    ".repeat(depth + 1);
    let track = track_of(scene, i);
    match node.data {
        ImportNodeData::Mesh(m) => {
            // An Xform holding the mesh, so the node's children nest under
            // a transform and not under geometry.
            let _ = writeln!(out, "{pad}def Xform \"{name}\"");
            let _ = writeln!(out, "{pad}{{");
            write_transform(out, node, &inner, track);
            // A rigged mesh moves by its rig, not frame by frame.
            let rig = rigs.meshes.get(&i);
            let samples = scene
                .point_samples
                .iter()
                .find(|s| s.node == i)
                .filter(|_| rig.is_none());
            write_mesh(
                out,
                scene,
                m,
                &format!("{name}_mesh"),
                samples,
                materials,
                depth + 1,
                notes,
                rig,
            );
        }
        ImportNodeData::Light(l) => {
            let Some(light) = scene.lights.get(l) else {
                let _ = writeln!(out, "{pad}def Xform \"{name}\"");
                let _ = writeln!(out, "{pad}{{");
                write_transform(out, node, &inner, track);
                return;
            };
            let (kind, intensity, extra) = match light.kind {
                // Candela from a small sphere: its surface's luminance.
                ImportLightKind::Point => {
                    let r = 0.05f32;
                    (
                        "SphereLight",
                        light.intensity / (std::f32::consts::PI * r * r),
                        format!("{inner}float inputs:radius = {r}\n"),
                    )
                }
                ImportLightKind::Area {
                    width,
                    height,
                    round,
                } => {
                    if round {
                        (
                            "DiskLight",
                            light.intensity,
                            format!("{inner}float inputs:radius = {}\n", width * 0.5),
                        )
                    } else {
                        (
                            "RectLight",
                            light.intensity,
                            format!(
                                "{inner}float inputs:width = {width}\n{inner}float inputs:height = {height}\n"
                            ),
                        )
                    }
                }
                ImportLightKind::Directional => (
                    "DistantLight",
                    light.intensity,
                    format!("{inner}float inputs:angle = 0.53\n"),
                ),
                ImportLightKind::Spot {
                    inner: cone_in,
                    outer,
                } => {
                    let r = 0.05f32;
                    (
                        "SphereLight",
                        light.intensity / (std::f32::consts::PI * r * r),
                        format!(
                            "{inner}float inputs:radius = {r}\n{inner}float inputs:shaping:cone:angle = {}\n{inner}float inputs:shaping:cone:softness = {}\n",
                            outer.to_degrees(),
                            if outer > 0.0 { ((outer - cone_in) / outer).clamp(0.0, 1.0) } else { 0.0 }
                        ),
                    )
                }
            };
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: its strength was translated into UsdLux's units",
                    light.name
                ),
            });
            let api = if matches!(light.kind, ImportLightKind::Spot { .. }) {
                " (\n".to_string()
                    + &pad
                    + "    prepend apiSchemas = [\"ShapingAPI\"]\n"
                    + &pad
                    + ")"
            } else {
                String::new()
            };
            let _ = writeln!(out, "{pad}def {kind} \"{name}\"{api}");
            let _ = writeln!(out, "{pad}{{");
            write_transform(out, node, &inner, track);
            let [r, g, b] = light.color;
            let _ = writeln!(out, "{inner}color3f inputs:color = ({r}, {g}, {b})");
            let _ = writeln!(out, "{inner}float inputs:intensity = {intensity}");
            out.push_str(&extra);
        }
        ImportNodeData::Camera(c) => {
            let _ = writeln!(out, "{pad}def Camera \"{name}\"");
            let _ = writeln!(out, "{pad}{{");
            write_transform(out, node, &inner, track);
            if let Some(camera) = scene.cameras.get(c) {
                // A 36 × 24 mm back, and the lens that gives the angle on it.
                let aperture = 24.0f32;
                match (camera.fov_y, camera.ortho_width) {
                    (Some(fov), _) => {
                        let focal = (aperture * 0.5) / (fov * 0.5).tan();
                        let _ = writeln!(out, "{inner}token projection = \"perspective\"");
                        let _ = writeln!(out, "{inner}float focalLength = {focal}");
                        let _ = writeln!(out, "{inner}float verticalAperture = {aperture}");
                        let _ = writeln!(out, "{inner}float horizontalAperture = 36");
                    }
                    (None, Some(width)) => {
                        // USD's apertures are tenths of a scene unit for an
                        // orthographic camera.
                        let _ = writeln!(out, "{inner}token projection = \"orthographic\"");
                        let _ = writeln!(out, "{inner}float horizontalAperture = {}", width * 10.0);
                        let _ = writeln!(
                            out,
                            "{inner}float verticalAperture = {}",
                            width * 10.0 * 2.0 / 3.0
                        );
                    }
                    (None, None) => {}
                }
                let _ = writeln!(
                    out,
                    "{inner}float2 clippingRange = ({}, {})",
                    camera.clip_near,
                    camera.clip_far.unwrap_or(1.0e6)
                );
            }
        }
        ImportNodeData::Empty => {
            let _ = writeln!(out, "{pad}def Xform \"{name}\"");
            let _ = writeln!(out, "{pad}{{");
            write_transform(out, node, &inner, track);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_mesh(
    out: &mut String,
    scene: &ExportScene<'_>,
    index: usize,
    name: &str,
    samples: Option<&PointSamples>,
    materials: &[String],
    depth: usize,
    notes: &mut Vec<LossyNote>,
    rig: Option<&MeshRig>,
) {
    let Some((label, mesh)) = scene.meshes.get(index) else {
        return;
    };
    let pad = "    ".repeat(depth);
    let inner = "    ".repeat(depth + 1);
    let slots = mesh.attributes().get_i32(names::MATERIAL_INDEX);
    let mut used: Vec<i32> = slots.map(|s| s.to_vec()).unwrap_or_default();
    used.sort_unstable();
    used.dedup();
    let bound = |slot: i32| {
        scene
            .material_of(index, slot)
            .and_then(|m| materials.get(m))
    };
    let mut schemas = Vec::new();
    if bound(0).is_some() || used.iter().any(|s| bound(*s).is_some()) {
        schemas.push("\"MaterialBindingAPI\"");
    }
    if rig.is_some() {
        schemas.push("\"SkelBindingAPI\"");
    }
    if schemas.is_empty() {
        let _ = writeln!(out, "{pad}def Mesh \"{name}\"");
    } else {
        let _ = writeln!(
            out,
            "{pad}def Mesh \"{name}\" (\n{pad}    prepend apiSchemas = [{}]\n{pad})",
            schemas.join(", ")
        );
    }
    let _ = writeln!(out, "{pad}{{");
    if let Some(rig) = rig {
        rig.write(out, scene, index, mesh, &inner, name);
    }
    let list = |v: &mut dyn Iterator<Item = String>| v.collect::<Vec<_>>().join(", ");
    let positions = mesh.positions();
    let _ = writeln!(
        out,
        "{inner}point3f[] points = [{}]",
        list(
            &mut positions
                .iter()
                .map(|p| format!("({}, {}, {})", p.x, p.y, p.z))
        )
    );
    if let Some(samples) = samples.filter(|s| s.points.iter().all(|p| p.len() == positions.len())) {
        let _ = writeln!(out, "{inner}point3f[] points.timeSamples = {{");
        for (t, points) in samples.times.iter().zip(&samples.points) {
            let _ = writeln!(
                out,
                "{inner}    {}: [{}],",
                t * TIME_CODES,
                list(
                    &mut points
                        .iter()
                        .map(|p| format!("({}, {}, {})", p.x, p.y, p.z))
                )
            );
        }
        let _ = writeln!(out, "{inner}}}");
    }
    let counts: Vec<String> = (0..mesh.num_faces())
        .map(|f| mesh.face_len(f).to_string())
        .collect();
    let _ = writeln!(
        out,
        "{inner}int[] faceVertexCounts = [{}]",
        counts.join(", ")
    );
    let _ = writeln!(
        out,
        "{inner}int[] faceVertexIndices = [{}]",
        list(&mut mesh.corner_verts().iter().map(|v| v.to_string()))
    );
    let normals: Vec<Vec3> = mesh.compute_corner_normals(30f32.to_radians());
    let _ =
        writeln!(
        out,
        "{inner}normal3f[] normals = [{}] (\n{inner}    interpolation = \"faceVarying\"\n{inner})",
        list(&mut normals.iter().map(|n| format!("({}, {}, {})", n.x, n.y, n.z)))
    );
    if let Some(uvs) = mesh.attributes().get_f32x2(names::UV) {
        // USD's st has its origin at the bottom left, as the document's does.
        let _ = writeln!(
            out,
            "{inner}texCoord2f[] primvars:st = [{}] (\n{inner}    interpolation = \"faceVarying\"\n{inner})",
            list(&mut uvs.iter().map(|uv| format!("({}, {})", uv.x, uv.y)))
        );
    }
    let _ = writeln!(out, "{inner}uniform token subdivisionScheme = \"none\"");
    // The first slot's material for the whole mesh, and where faces use
    // others, a subset of them each.
    let _ = (label, &notes);
    if let Some(m) = bound(0) {
        let _ = writeln!(out, "{inner}rel material:binding = </World/Looks/{m}>");
    }
    if let (Some(slots), true) = (slots, used.len() > 1) {
        let _ = writeln!(
            out,
            "{inner}uniform token subsetFamily:materialBind:familyType = \"nonOverlapping\""
        );
        let deep = "    ".repeat(depth + 2);
        for slot in &used {
            let Some(m) = bound(*slot) else { continue };
            let faces: Vec<String> = slots
                .iter()
                .enumerate()
                .filter(|(_, s)| *s == slot)
                .map(|(f, _)| f.to_string())
                .collect();
            let _ = writeln!(
                out,
                "{inner}def GeomSubset \"Slot{slot}\" (\n{inner}    prepend apiSchemas = [\"MaterialBindingAPI\"]\n{inner})"
            );
            let _ = writeln!(out, "{inner}{{");
            let _ = writeln!(out, "{deep}uniform token elementType = \"face\"");
            let _ = writeln!(out, "{deep}uniform token familyName = \"materialBind\"");
            let _ = writeln!(out, "{deep}int[] indices = [{}]", faces.join(", "));
            let _ = writeln!(out, "{deep}rel material:binding = </World/Looks/{m}>");
            let _ = writeln!(out, "{inner}}}");
        }
    }
    let _ = writeln!(out, "{pad}}}");
}

fn write_material(
    out: &mut String,
    m: &crate::ir::ImportMaterial,
    name: &str,
    depth: usize,
    textures: &[Option<String>],
) {
    let pad = "    ".repeat(depth);
    let inner = "    ".repeat(depth + 1);
    let deep = "    ".repeat(depth + 2);
    let picture = |i: Option<usize>| i.and_then(|i| textures.get(i).cloned().flatten());
    let base = picture(m.base_color_texture);
    let metal_rough = picture(m.metallic_roughness_texture);
    let normal = picture(m.normal_texture);
    let emission = picture(m.emissive_texture);
    let occlusion = picture(m.occlusion_texture);
    let at =
        |shader: &str, output: &str| format!("</World/Looks/{name}/{shader}.outputs:{output}>");
    let _ = writeln!(out, "{pad}def Material \"{name}\"");
    let _ = writeln!(out, "{pad}{{");
    let _ = writeln!(
        out,
        "{inner}token outputs:surface.connect = {}",
        at("Surface", "surface")
    );
    let _ = writeln!(out, "{inner}def Shader \"Surface\"");
    let _ = writeln!(out, "{inner}{{");
    let _ = writeln!(out, "{deep}uniform token info:id = \"UsdPreviewSurface\"");
    let [r, g, b, a] = m.base_color;
    match &base {
        Some(_) => {
            let _ = writeln!(
                out,
                "{deep}color3f inputs:diffuseColor.connect = {}",
                at("BaseColor", "rgb")
            );
        }
        None => {
            let _ = writeln!(out, "{deep}color3f inputs:diffuseColor = ({r}, {g}, {b})");
        }
    }
    let _ = writeln!(out, "{deep}float inputs:opacity = {a}");
    if let Some(clip) = m.alpha_clip {
        let _ = writeln!(out, "{deep}float inputs:opacityThreshold = {clip}");
    }
    // glTF's packing, which the importer keeps: roughness green, metal blue.
    match &metal_rough {
        Some(_) => {
            let _ = writeln!(
                out,
                "{deep}float inputs:metallic.connect = {}",
                at("MetalRough", "b")
            );
            let _ = writeln!(
                out,
                "{deep}float inputs:roughness.connect = {}",
                at("MetalRough", "g")
            );
        }
        None => {
            let _ = writeln!(out, "{deep}float inputs:metallic = {}", m.metallic);
            let _ = writeln!(out, "{deep}float inputs:roughness = {}", m.roughness);
        }
    }
    let e = m.emissive * m.emissive_strength;
    match &emission {
        Some(_) => {
            let _ = writeln!(
                out,
                "{deep}color3f inputs:emissiveColor.connect = {}",
                at("Emission", "rgb")
            );
        }
        None => {
            let _ = writeln!(
                out,
                "{deep}color3f inputs:emissiveColor = ({}, {}, {})",
                e.x, e.y, e.z
            );
        }
    }
    if normal.is_some() {
        let _ = writeln!(
            out,
            "{deep}normal3f inputs:normal.connect = {}",
            at("Normal", "rgb")
        );
    }
    if occlusion.is_some() {
        let _ = writeln!(
            out,
            "{deep}float inputs:occlusion.connect = {}",
            at("Occlusion", "r")
        );
    }
    let _ = writeln!(out, "{deep}float inputs:ior = {}", m.ior);
    let _ = writeln!(out, "{deep}token outputs:surface");
    let _ = writeln!(out, "{inner}}}");
    let any = [&base, &metal_rough, &normal, &emission, &occlusion]
        .iter()
        .any(|t| t.is_some());
    if any {
        // Where on the picture: the mesh's `st`.
        let _ = writeln!(out, "{inner}def Shader \"Reader\"");
        let _ = writeln!(out, "{inner}{{");
        let _ = writeln!(
            out,
            "{deep}uniform token info:id = \"UsdPrimvarReader_float2\""
        );
        let _ = writeln!(out, "{deep}string inputs:varname = \"st\"");
        let _ = writeln!(out, "{deep}float2 outputs:result");
        let _ = writeln!(out, "{inner}}}");
    }
    for (shader, file, space, scale, bias) in [
        ("BaseColor", &base, "sRGB", None, None),
        ("MetalRough", &metal_rough, "raw", None, None),
        // A normal map's colour is a direction: twice, less one.
        (
            "Normal",
            &normal,
            "raw",
            Some((2.0, 2.0, 2.0, 1.0)),
            Some((-1.0, -1.0, -1.0, 0.0)),
        ),
        (
            "Emission",
            &emission,
            "sRGB",
            Some((e.x, e.y, e.z, 1.0)),
            None,
        ),
        ("Occlusion", &occlusion, "raw", None, None),
    ] {
        let Some(file) = file else { continue };
        let _ = writeln!(out, "{inner}def Shader \"{shader}\"");
        let _ = writeln!(out, "{inner}{{");
        let _ = writeln!(out, "{deep}uniform token info:id = \"UsdUVTexture\"");
        let _ = writeln!(out, "{deep}asset inputs:file = @{file}@");
        let _ = writeln!(out, "{deep}token inputs:sourceColorSpace = \"{space}\"");
        let _ = writeln!(
            out,
            "{deep}float2 inputs:st.connect = {}",
            at("Reader", "result")
        );
        let _ = writeln!(out, "{deep}token inputs:wrapS = \"repeat\"");
        let _ = writeln!(out, "{deep}token inputs:wrapT = \"repeat\"");
        if let Some((x, y, z, w)) = scale {
            let _ = writeln!(out, "{deep}float4 inputs:scale = ({x}, {y}, {z}, {w})");
        }
        if let Some((x, y, z, w)) = bias {
            let _ = writeln!(out, "{deep}float4 inputs:bias = ({x}, {y}, {z}, {w})");
        }
        for o in ["rgb", "r", "g", "b", "a"] {
            let t = if o == "rgb" { "float3" } else { "float" };
            let _ = writeln!(out, "{deep}{t} outputs:{o}");
        }
        let _ = writeln!(out, "{inner}}}");
    }
    let _ = writeln!(out, "{pad}}}");
}

/// The node hierarchy as prims: each node's children, the roots, each
/// node's name among its siblings and its prim's path.
struct Layout {
    children: Vec<Vec<usize>>,
    roots: Vec<usize>,
    names: Vec<String>,
    paths: Vec<String>,
}

impl Layout {
    fn of(scene: &ExportScene<'_>) -> Self {
        let n = scene.nodes.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut roots = Vec::new();
        for (i, node) in scene.nodes.iter().enumerate() {
            match node.parent.filter(|p| *p < n) {
                Some(p) => children[p].push(i),
                None => roots.push(i),
            }
        }
        // Unique among siblings: USD names a prim by its path.
        let mut names = vec![String::new(); n];
        for group in std::iter::once(&roots).chain(children.iter()) {
            let made = unique(group.iter().map(|i| scene.nodes[*i].name.as_str()));
            for (i, name) in group.iter().zip(made) {
                names[*i] = name;
            }
        }
        let mut paths = vec![String::new(); n];
        let mut stack: Vec<(usize, String)> =
            roots.iter().map(|r| (*r, "/World".to_string())).collect();
        while let Some((i, above)) = stack.pop() {
            paths[i] = format!("{above}/{}", names[i]);
            for c in &children[i] {
                stack.push((*c, paths[i].clone()));
            }
        }
        Layout {
            children,
            roots,
            names,
            paths,
        }
    }
}

/// Where a node stands in the world, as it stands now.
fn world_of(scene: &ExportScene<'_>, mut i: usize) -> kreate_math::DAffine3 {
    let mut m = kreate_math::DAffine3::IDENTITY;
    for _ in 0..=scene.nodes.len() {
        let node = &scene.nodes[i];
        m = node.transform.to_affine() * m;
        match node.parent {
            Some(p) if p < scene.nodes.len() => i = p,
            _ => break,
        }
    }
    m
}

/// A transform's value at `t` on a track: the key at or before it.
fn sampled(track: &ExportTrack, t: f32) -> kreate_math::DTransform {
    let k = track
        .times
        .iter()
        .rposition(|x| *x <= t + 1e-6)
        .unwrap_or(0);
    track.transforms[k.min(track.transforms.len() - 1)]
}

/// One skeleton to write: under an armature's prim, or under a mesh that
/// has shape keys and no armature (UsdSkel plays blend shapes only through
/// a skeleton, so it gets one with no joints).
struct Skeleton {
    joints: Vec<String>,
    bind: Vec<kreate_math::DAffine3>,
    rest: Vec<kreate_math::DAffine3>,
    /// Per time code, each joint's turn, place and size.
    poses: Vec<(f32, Vec<kreate_math::DTransform>)>,
    /// Blend shapes and their weights, per time code.
    shapes: Vec<String>,
    weights: Vec<(f32, Vec<f32>)>,
}

impl Skeleton {
    fn text(&self, depth: usize) -> String {
        let pad = "    ".repeat(depth);
        let inner = "    ".repeat(depth + 1);
        let deep = "    ".repeat(depth + 2);
        let mut out = String::new();
        let tokens = |v: &[String]| {
            v.iter()
                .map(|j| format!("\"{j}\""))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let matrices =
            |v: &[kreate_math::DAffine3]| v.iter().map(matrix).collect::<Vec<_>>().join(", ");
        let _ = writeln!(
            out,
            "{pad}def Skeleton \"Skel\" (\n{pad}    prepend apiSchemas = [\"SkelBindingAPI\"]\n{pad})"
        );
        let _ = writeln!(out, "{pad}{{");
        let _ = writeln!(
            out,
            "{inner}uniform token[] joints = [{}]",
            tokens(&self.joints)
        );
        let _ = writeln!(
            out,
            "{inner}uniform matrix4d[] bindTransforms = [{}]",
            matrices(&self.bind)
        );
        let _ = writeln!(
            out,
            "{inner}uniform matrix4d[] restTransforms = [{}]",
            matrices(&self.rest)
        );
        let _ = writeln!(out, "{inner}rel skel:animationSource = <Anim>");
        let _ = writeln!(out, "{inner}def SkelAnimation \"Anim\"");
        let _ = writeln!(out, "{inner}{{");
        let _ = writeln!(
            out,
            "{deep}uniform token[] joints = [{}]",
            tokens(&self.joints)
        );
        if !self.joints.is_empty() {
            let part = |f: &dyn Fn(&kreate_math::DTransform) -> String| -> Vec<(f32, String)> {
                self.poses
                    .iter()
                    .map(|(t, pose)| (*t, pose.iter().map(f).collect::<Vec<_>>().join(", ")))
                    .collect()
            };
            let write = |out: &mut String, kind: &str, name: &str, rows: Vec<(f32, String)>| {
                if rows.len() == 1 {
                    let _ = writeln!(out, "{deep}{kind} {name} = [{}]", rows[0].1);
                } else {
                    let _ = writeln!(out, "{deep}{kind} {name}.timeSamples = {{");
                    for (t, row) in rows {
                        let _ = writeln!(out, "{deep}    {}: [{row}],", t * TIME_CODES);
                    }
                    let _ = writeln!(out, "{deep}}}");
                }
            };
            write(
                &mut out,
                "float3[]",
                "translations",
                part(&|p| {
                    format!(
                        "({}, {}, {})",
                        p.translation.x, p.translation.y, p.translation.z
                    )
                }),
            );
            // Real part first, as USD writes a quaternion.
            write(
                &mut out,
                "quatf[]",
                "rotations",
                part(&|p| {
                    let q = p.rotation.normalize();
                    format!("({}, {}, {}, {})", q.w, q.x, q.y, q.z)
                }),
            );
            write(
                &mut out,
                "half3[]",
                "scales",
                part(&|p| format!("({}, {}, {})", p.scale.x, p.scale.y, p.scale.z)),
            );
        }
        if !self.shapes.is_empty() {
            let _ = writeln!(
                out,
                "{deep}uniform token[] blendShapes = [{}]",
                tokens(&self.shapes)
            );
            let row = |w: &[f32]| {
                w.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            if self.weights.len() == 1 {
                let _ = writeln!(
                    out,
                    "{deep}float[] blendShapeWeights = [{}]",
                    row(&self.weights[0].1)
                );
            } else {
                let _ = writeln!(out, "{deep}float[] blendShapeWeights.timeSamples = {{");
                for (t, w) in &self.weights {
                    let _ = writeln!(out, "{deep}    {}: [{}],", t * TIME_CODES, row(w));
                }
                let _ = writeln!(out, "{deep}}}");
            }
        }
        let _ = writeln!(out, "{inner}}}");
        let _ = writeln!(out, "{pad}}}");
        out
    }
}

/// What a rigged mesh writes: the skeleton it follows, its joints and
/// weights when an armature bends it, its blend shapes when it has shape
/// keys.
struct MeshRig {
    skeleton: String,
    skin: Option<usize>,
    /// From the mesh's own space to the skeleton's, at the bind.
    geom_bind: kreate_math::DAffine3,
    /// Its shape keys, and the prim path its blend shapes go under.
    shapes: Vec<String>,
    shape_prefix: String,
}

impl MeshRig {
    fn write(
        &self,
        out: &mut String,
        scene: &ExportScene<'_>,
        index: usize,
        mesh: &kreate_mesh::MeshData,
        inner: &str,
        name: &str,
    ) {
        let _ = writeln!(out, "{inner}rel skel:skeleton = <{}>", self.skeleton);
        if self.skin.is_some() {
            if let Some((skin, _)) = crate::gltf_export::vertex_skin(mesh, scene, index) {
                let joints: Vec<String> = skin
                    .joints
                    .iter()
                    .flat_map(|j| j.iter().map(|v| v.to_string()))
                    .collect();
                let weights: Vec<String> = skin
                    .weights
                    .iter()
                    .flat_map(|w| w.iter().map(|v| v.to_string()))
                    .collect();
                let _ = writeln!(
                    out,
                    "{inner}int[] primvars:skel:jointIndices = [{}] (\n{inner}    elementSize = 4\n{inner}    interpolation = \"vertex\"\n{inner})",
                    joints.join(", ")
                );
                let _ = writeln!(
                    out,
                    "{inner}float[] primvars:skel:jointWeights = [{}] (\n{inner}    elementSize = 4\n{inner}    interpolation = \"vertex\"\n{inner})",
                    weights.join(", ")
                );
                let _ = writeln!(
                    out,
                    "{inner}matrix4d primvars:skel:geomBindTransform = {}",
                    matrix(&self.geom_bind)
                );
            }
        }
        if self.shapes.is_empty() {
            return;
        }
        let shape_names = unique(self.shapes.iter().map(String::as_str));
        let _ = writeln!(
            out,
            "{inner}uniform token[] skel:blendShapes = [{}]",
            self.shapes
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "{inner}rel skel:blendShapeTargets = [{}]",
            shape_names
                .iter()
                .map(|n| format!("<{}/{name}/{n}>", self.shape_prefix))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (key, prim) in self.shapes.iter().zip(&shape_names) {
            let Some(offsets) = mesh
                .attributes()
                .get_f32x3(&kreate_scene::shape::layer(key))
            else {
                continue;
            };
            let moved: Vec<(usize, Vec3)> = offsets
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, d)| d.length_squared() > 0.0)
                .collect();
            let _ = writeln!(out, "{inner}def BlendShape \"{prim}\"");
            let _ = writeln!(out, "{inner}{{");
            let _ = writeln!(
                out,
                "{inner}    uniform vector3f[] offsets = [{}]",
                moved
                    .iter()
                    .map(|(_, d)| format!("({}, {}, {})", d.x, d.y, d.z))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let _ = writeln!(
                out,
                "{inner}    uniform int[] pointIndices = [{}]",
                moved
                    .iter()
                    .map(|(i, _)| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let _ = writeln!(out, "{inner}}}");
        }
    }
}

/// Every rig in the scene: the joints (written in their skeletons, not as
/// prims), each skeleton by the node it goes under, each rigged mesh by its
/// node.
struct Rigs {
    joint: Vec<bool>,
    skeletons: std::collections::HashMap<usize, Skeleton>,
    meshes: std::collections::HashMap<usize, MeshRig>,
}

impl Rigs {
    fn any(&self) -> bool {
        !self.skeletons.is_empty()
    }

    fn of(scene: &ExportScene<'_>, layout: &Layout) -> Self {
        use kreate_math::{DAffine3, DTransform};
        let n = scene.nodes.len();
        let mut joint = vec![false; n];
        for skin in &scene.skins {
            for j in &skin.joints {
                if let Some(slot) = joint.get_mut(*j) {
                    *slot = true;
                }
            }
        }
        let mut skeletons = std::collections::HashMap::new();
        let mut meshes = std::collections::HashMap::new();
        // The times anything is keyed at, and a node's keys.
        let track = |node: usize| {
            scene
                .animations
                .iter()
                .flat_map(|a| &a.tracks)
                .find(|t| t.node == node)
        };
        let weights_of = |node: usize| {
            scene
                .animations
                .iter()
                .flat_map(|a| &a.weights)
                .find(|w| w.node == node)
        };
        let shapes_of = |node: usize| -> Vec<String> {
            match scene.nodes[node].data {
                ImportNodeData::Mesh(m) => scene.morph_targets.get(m).cloned().unwrap_or_default(),
                _ => Vec::new(),
            }
        };
        // Weights per time for one mesh's shapes, as the skeleton's
        // animation lists them.
        let shape_rows = |node: usize, shapes: &[String]| -> Vec<(f32, Vec<f32>)> {
            match weights_of(node) {
                Some(w) => w
                    .times
                    .iter()
                    .zip(&w.values)
                    .map(|(t, v)| {
                        (
                            *t,
                            (0..shapes.len())
                                .map(|k| v.get(k).copied().unwrap_or(0.0))
                                .collect(),
                        )
                    })
                    .collect(),
                None => vec![(
                    0.0,
                    (0..shapes.len())
                        .map(|k| {
                            scene.nodes[node]
                                .morph_weights
                                .get(k)
                                .copied()
                                .unwrap_or(0.0)
                        })
                        .collect(),
                )],
            }
        };
        // Armatures: one skeleton each, from the first skin that names its
        // bones.
        for (s, skin) in scene.skins.iter().enumerate() {
            let Some(&first) = skin.joints.first() else {
                continue;
            };
            let Some(armature) = scene.nodes.get(first).and_then(|j| j.parent) else {
                continue;
            };
            let Some(mesh_node) = scene.nodes.iter().position(|nd| nd.skin == Some(s)) else {
                continue;
            };
            let a_world = world_of(scene, armature);
            let m_world = world_of(scene, mesh_node);
            let skel_path = format!("{}/Skel", layout.paths[armature]);
            let shapes = shapes_of(mesh_node);
            meshes.insert(
                mesh_node,
                MeshRig {
                    skeleton: skel_path,
                    skin: Some(s),
                    geom_bind: a_world.inverse() * m_world,
                    shapes: shapes.clone(),
                    shape_prefix: layout.paths[mesh_node].clone(),
                },
            );
            let entry = skeletons.entry(armature).or_insert_with(|| {
                // Joint paths: each joint under its parent joint.
                let index_of = |node: usize| skin.joints.iter().position(|j| *j == node);
                let mut tokens = Vec::with_capacity(skin.joints.len());
                for j in &skin.joints {
                    let mut chain = vec![prim_name(&scene.nodes[*j].name)];
                    let mut at = *j;
                    while let Some(p) = scene.nodes[at].parent.filter(|p| index_of(*p).is_some()) {
                        chain.push(prim_name(&scene.nodes[p].name));
                        at = p;
                    }
                    chain.reverse();
                    tokens.push(chain.join("/"));
                }
                // At the bind, in the skeleton's space (the armature's).
                let bind: Vec<DAffine3> = skin
                    .inverse_bind
                    .iter()
                    .map(|ib| a_world.inverse() * m_world * ib.inverse())
                    .collect();
                let rest: Vec<DAffine3> = skin
                    .joints
                    .iter()
                    .enumerate()
                    .map(|(k, j)| match scene.nodes[*j].parent.and_then(index_of) {
                        Some(p) => bind[p].inverse() * bind[k],
                        None => bind[k],
                    })
                    .collect();
                // The pose at every keyed time, or as it stands.
                let mut times: Vec<f32> = skin
                    .joints
                    .iter()
                    .filter_map(|j| track(*j))
                    .flat_map(|t| t.times.iter().copied())
                    .collect();
                times.sort_by(f32::total_cmp);
                times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
                if times.is_empty() {
                    times.push(0.0);
                }
                let poses = times
                    .iter()
                    .map(|t| {
                        let pose: Vec<DTransform> = skin
                            .joints
                            .iter()
                            .map(|j| match track(*j) {
                                Some(tr) if !tr.transforms.is_empty() => sampled(tr, *t),
                                _ => scene.nodes[*j].transform,
                            })
                            .collect();
                        (*t, pose)
                    })
                    .collect();
                Skeleton {
                    joints: tokens,
                    bind,
                    rest,
                    poses,
                    shapes: Vec::new(),
                    weights: vec![(0.0, Vec::new())],
                }
            });
            // The mesh's shapes join its armature's animation.
            if !shapes.is_empty() {
                let rows = shape_rows(mesh_node, &shapes);
                merge_shapes(entry, &shapes, &rows);
            }
        }
        // Meshes with shape keys and no armature: a skeleton of their own.
        for (i, node) in scene.nodes.iter().enumerate() {
            if meshes.contains_key(&i) || !matches!(node.data, ImportNodeData::Mesh(_)) {
                continue;
            }
            let shapes = shapes_of(i);
            if shapes.is_empty() {
                continue;
            }
            let rows = shape_rows(i, &shapes);
            meshes.insert(
                i,
                MeshRig {
                    skeleton: format!("{}/Skel", layout.paths[i]),
                    skin: None,
                    geom_bind: DAffine3::IDENTITY,
                    shapes: shapes.clone(),
                    shape_prefix: layout.paths[i].clone(),
                },
            );
            skeletons.insert(
                i,
                Skeleton {
                    joints: Vec::new(),
                    bind: Vec::new(),
                    rest: Vec::new(),
                    poses: vec![(0.0, Vec::new())],
                    shapes,
                    weights: rows,
                },
            );
        }
        Rigs {
            joint,
            skeletons,
            meshes,
        }
    }
}

/// One mesh's shapes and their weights added to a skeleton's: the times
/// joined, each shape's weight at each.
fn merge_shapes(skel: &mut Skeleton, shapes: &[String], rows: &[(f32, Vec<f32>)]) {
    let fresh: Vec<&String> = shapes.iter().filter(|s| !skel.shapes.contains(s)).collect();
    if fresh.is_empty() {
        return;
    }
    let mut times: Vec<f32> = skel
        .weights
        .iter()
        .map(|(t, _)| *t)
        .chain(rows.iter().map(|(t, _)| *t))
        .collect();
    times.sort_by(f32::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let at = |list: &[(f32, Vec<f32>)], t: f32| -> Vec<f32> {
        list.iter()
            .rev()
            .find(|(x, _)| *x <= t + 1e-6)
            .or(list.first())
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    let old = std::mem::take(&mut skel.weights);
    let width = skel.shapes.len();
    skel.weights = times
        .iter()
        .map(|t| {
            let mut row = at(&old, *t);
            row.resize(width, 0.0);
            let theirs = at(rows, *t);
            for s in &fresh {
                let k = shapes.iter().position(|x| x == *s).unwrap_or(0);
                row.push(theirs.get(k).copied().unwrap_or(0.0));
            }
            (*t, row)
        })
        .collect();
    skel.shapes.extend(fresh.into_iter().cloned());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prim_name_is_an_identifier() {
        assert_eq!(prim_name("Cube.001"), "Cube_001");
        assert_eq!(prim_name("3 Lamps"), "_3_Lamps");
        assert_eq!(prim_name(""), "_");
        assert_eq!(
            unique(["A", "A", "a b"].into_iter()),
            vec!["A", "A_1", "a_b"]
        );
    }
}
