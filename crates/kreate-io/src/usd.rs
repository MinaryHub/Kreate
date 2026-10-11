//! USD, as text: `.usda`.
//!
//! The scene as a USD stage — meshes, the hierarchy, materials as
//! `UsdPreviewSurface`, lights as `UsdLux`, cameras — written as the ASCII
//! layer every USD tool reads. Write only: reading USD means reading its
//! composition (references, payloads, variants, layers), and that is a
//! library, not a parser.
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
        .fold(None, |m: Option<f32>, t| Some(m.map_or(*t, |m| m.max(*t))));
    if let Some(last) = last {
        let _ = writeln!(out, "    startTimeCode = 0");
        let _ = writeln!(out, "    endTimeCode = {}", (last * TIME_CODES).ceil());
        let _ = writeln!(out, "    timeCodesPerSecond = {TIME_CODES}");
        let _ = writeln!(out, "    framesPerSecond = {TIME_CODES}");
    }
    let _ = writeln!(out, ")");
    let _ = writeln!(out);
    let _ = writeln!(out, "def Xform \"World\"");
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
            );
        }
    } else {
        // Each node under its parent, by nesting.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); scene.nodes.len()];
        let mut roots = Vec::new();
        for (i, node) in scene.nodes.iter().enumerate() {
            match node.parent.filter(|p| *p < scene.nodes.len()) {
                Some(p) => children[p].push(i),
                None => roots.push(i),
            }
        }
        let names: Vec<String> = {
            // Unique among siblings: USD names a prim by its path.
            let mut out = vec![String::new(); scene.nodes.len()];
            for group in std::iter::once(&roots).chain(children.iter()) {
                let made = unique(group.iter().map(|i| scene.nodes[*i].name.as_str()));
                for (i, n) in group.iter().zip(made) {
                    out[*i] = n;
                }
            }
            out
        };
        let mut stack: Vec<(usize, usize, bool)> =
            roots.iter().rev().map(|r| (*r, 1usize, false)).collect();
        while let Some((i, depth, closing)) = stack.pop() {
            let pad = "    ".repeat(depth);
            if closing {
                let _ = writeln!(out, "{pad}}}");
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
            );
            stack.push((i, depth, true));
            for c in children[i].iter().rev() {
                stack.push((*c, depth + 1, false));
            }
        }
    }
    let _ = writeln!(out, "}}");
    if !scene.skins.is_empty() {
        notes.push(LossyNote::Unsupported {
            what: "skinned meshes went out as moving points, frame by frame, not as a UsdSkel rig"
                .into(),
        });
    }
    if scene.animations.iter().any(|a| !a.weights.is_empty()) {
        notes.push(LossyNote::Unsupported {
            what: "shape keys went out as moving points, not as blend shapes".into(),
        });
    }
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
            let samples = scene.point_samples.iter().find(|s| s.node == i);
            write_mesh(
                out,
                scene,
                m,
                &format!("{name}_mesh"),
                samples,
                materials,
                depth + 1,
                notes,
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
    if bound(0).is_some() || used.iter().any(|s| bound(*s).is_some()) {
        let _ = writeln!(
            out,
            "{pad}def Mesh \"{name}\" (\n{pad}    prepend apiSchemas = [\"MaterialBindingAPI\"]\n{pad})"
        );
    } else {
        let _ = writeln!(out, "{pad}def Mesh \"{name}\"");
    }
    let _ = writeln!(out, "{pad}{{");
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
