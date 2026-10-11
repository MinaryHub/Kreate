//! COLLADA, `.dae`: the XML scene format that robot descriptions name for
//! their meshes, and that older tools still write.
//!
//! What a scene needs: the node tree and its transforms (`matrix`,
//! `translate`, `rotate`, `scale`, in order), the geometries it instances —
//! `triangles`, `polylist` and `polygons`, with positions, normals, the
//! first set of texture coordinates and vertex colours — and the materials
//! bound to them, read from their effect's common profile: a diffuse colour
//! or picture, emission, transparency. The file's `<unit>` and `<up_axis>`
//! are its units and axes. Animation, skins, controllers, physics and the
//! programmable profiles are not read, and are said.

use std::collections::HashMap;
use std::path::Path;

use kreate_math::{AxisConvention, Vec2, Vec3};

use crate::ir::{
    ImportImage, ImportMaterial, ImportMesh, ImportNode, ImportNodeData, ImportScene,
    ImportTransform, LossyNote, SourceMetadata,
};
use crate::xml::{self, Element};

/// Why a COLLADA document could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DaeError(pub String);

impl std::fmt::Display for DaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "COLLADA: {}", self.0)
    }
}

impl std::error::Error for DaeError {}

/// How deep `instance_node`s may lead: a node that instances itself is a
/// file's mistake.
const MAX_DEPTH: usize = 64;

/// A file, its pictures found from beside it.
pub fn parse_dae_file(path: &Path) -> Result<ImportScene, DaeError> {
    let text = std::fs::read(path).map_err(|e| DaeError(format!("cannot read: {e}")))?;
    if text.starts_with(b"bplist") {
        return Err(DaeError(
            "this is an Apple SceneKit archive with a .dae name, not COLLADA".into(),
        ));
    }
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    parse_dae(&String::from_utf8_lossy(&text), &|name: &str| {
        let name = name.strip_prefix("file://").unwrap_or(name);
        let name = percent_decode(name);
        let p = base.join(&name);
        std::fs::read(&p).ok().or_else(|| {
            // Only the name, beside the file: what a copy that moved keeps.
            let bare = Path::new(&name).file_name()?;
            std::fs::read(base.join(bare)).ok()
        })
    })
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A document; `file` reads a picture it names.
pub fn parse_dae(
    text: &str,
    file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<ImportScene, DaeError> {
    let root = xml::parse(text).map_err(|e| DaeError(e.to_string()))?;
    if root.name != "COLLADA" {
        return Err(DaeError(format!(
            "the file is a `<{}>`, not COLLADA",
            root.name
        )));
    }
    let mut source = SourceMetadata::new("collada");
    if let Some(asset) = root.child("asset") {
        if let Some(m) = asset
            .child("unit")
            .and_then(|u| u.attr("meter"))
            .and_then(|m| m.trim().parse::<f32>().ok())
        {
            source.unit_scale = m;
        }
        source.axis = match asset.child("up_axis").map(|u| u.text.trim()) {
            Some("Z_UP") => AxisConvention::KREATE,
            _ => AxisConvention::GLTF,
        };
        source.generator = asset
            .child("contributor")
            .and_then(|c| c.child("authoring_tool"))
            .map(|t| t.text.trim().to_string());
    } else {
        source.axis = AxisConvention::GLTF;
    }
    // Everything with an id, to follow `#id` and `url`s to.
    let mut ids: HashMap<&str, &Element> = HashMap::new();
    for e in root.descendants() {
        if let Some(id) = e.attr("id") {
            ids.insert(id, e);
        }
    }
    let mut reader = Reader {
        scene: ImportScene::new(source),
        ids,
        file,
        materials: HashMap::new(),
        images: HashMap::new(),
        meshes: HashMap::new(),
    };
    let scene_url = root
        .child("scene")
        .and_then(|s| s.child("instance_visual_scene"))
        .and_then(|i| i.attr("url"));
    let visual = scene_url
        .and_then(|u| reader.find(u))
        .or_else(|| {
            root.child("library_visual_scenes")
                .and_then(|l| l.child("visual_scene"))
        })
        .ok_or_else(|| DaeError("the document has no scene".into()))?;
    for node in visual.all("node") {
        reader.node(node, None, 0);
    }
    if root.child("library_animations").is_some() {
        reader.say("animations are not read");
    }
    if root.child("library_controllers").is_some() {
        reader.say("skins and morphs (controllers) are not read; their meshes come in at rest where a node instances the geometry");
    }
    Ok(reader.scene)
}

/// A geometry, and which material each of its symbols is bound to.
type MeshKey = (String, Vec<(String, String)>);

struct Reader<'a> {
    scene: ImportScene,
    ids: HashMap<&'a str, &'a Element>,
    file: &'a dyn Fn(&str) -> Option<Vec<u8>>,
    materials: HashMap<String, Option<usize>>,
    images: HashMap<String, Option<usize>>,
    /// Meshes made, by geometry and the materials bound to it.
    meshes: HashMap<MeshKey, Option<usize>>,
}

impl<'a> Reader<'a> {
    fn say(&mut self, what: &str) {
        let note = LossyNote::Unsupported {
            what: what.to_string(),
        };
        if !self.scene.source.notes.contains(&note) {
            self.scene.source.notes.push(note);
        }
    }

    fn find(&self, url: &str) -> Option<&'a Element> {
        self.ids.get(url.trim_start_matches('#')).copied()
    }

    fn node(&mut self, e: &'a Element, parent: Option<usize>, depth: usize) {
        if depth > MAX_DEPTH {
            self.say("nodes nested too deep were left out");
            return;
        }
        let mut node = ImportNode::new(e.attr("name").or(e.attr("id")).unwrap_or("node"));
        node.parent = parent;
        node.transform = transform_of(e);
        if e.attr("type") == Some("JOINT") {
            // A skeleton's joint: an empty, as it is.
        }
        let at = self.scene.nodes.len();
        self.scene.nodes.push(node);
        // Geometry: each instance a child, so one node can carry several.
        let instances: Vec<&'a Element> = e.all("instance_geometry").collect();
        for (k, g) in instances.iter().enumerate() {
            let Some(mesh) = self.instance(g) else {
                continue;
            };
            if instances.len() == 1 && self.scene.nodes[at].data == ImportNodeData::Empty {
                self.scene.nodes[at].data = ImportNodeData::Mesh(mesh);
            } else {
                let mut child = ImportNode::new(format!("{}_{k}", self.scene.nodes[at].name));
                child.parent = Some(at);
                child.data = ImportNodeData::Mesh(mesh);
                self.scene.nodes.push(child);
            }
        }
        if e.child("instance_controller").is_some() {
            self.say("skinned meshes (instance_controller) are not read");
        }
        for c in e.all("node") {
            self.node(c, Some(at), depth + 1);
        }
        for i in e.all("instance_node") {
            if let Some(target) = i.attr("url").and_then(|u| self.find(u)) {
                self.node(target, Some(at), depth + 1);
            }
        }
    }

    fn instance(&mut self, g: &'a Element) -> Option<usize> {
        let url = g.attr("url")?;
        // The material each symbol in the geometry stands for here.
        let mut binding: Vec<(String, String)> = g
            .child("bind_material")
            .and_then(|b| b.child("technique_common"))
            .map(|t| {
                t.all("instance_material")
                    .filter_map(|m| {
                        Some((m.attr("symbol")?.to_string(), m.attr("target")?.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        binding.sort();
        let key = (url.to_string(), binding.clone());
        if let Some(made) = self.meshes.get(&key) {
            return *made;
        }
        let geometry = self.find(url);
        let made = geometry.and_then(|geo| self.mesh(geo, &binding));
        self.meshes.insert(key, made);
        made
    }

    /// A `<source>`'s numbers and how many make one item.
    fn source(&self, url: &str) -> Option<(Vec<f64>, usize)> {
        let s = self.find(url)?;
        let array = s.child("float_array")?;
        let numbers = xml::numbers(&array.text);
        let stride = s
            .child("technique_common")
            .and_then(|t| t.child("accessor"))
            .and_then(|a| a.attr("stride"))
            .and_then(|n| n.parse().ok())
            .unwrap_or(1usize)
            .max(1);
        Some((numbers, stride))
    }

    fn mesh(&mut self, geometry: &'a Element, binding: &[(String, String)]) -> Option<usize> {
        let mesh = geometry.child("mesh")?;
        let vertices = mesh.child("vertices");
        let position_url = vertices
            .and_then(|v| {
                v.all("input")
                    .find(|i| i.attr("semantic") == Some("POSITION"))
            })
            .and_then(|i| i.attr("source"))?;
        let (raw, stride) = self.source(position_url)?;
        let positions: Vec<Vec3> = raw
            .chunks(stride)
            .filter(|c| c.len() >= 3)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect();
        let mut out = ImportMesh {
            name: geometry
                .attr("name")
                .or(geometry.attr("id"))
                .unwrap_or("mesh")
                .to_string(),
            positions,
            ..ImportMesh::default()
        };
        let mut uvs: Vec<Vec2> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut colors: Vec<kreate_math::glam::Vec4> = Vec::new();
        let (mut any_uv, mut any_normal, mut any_color) = (false, false, false);
        let mut face_materials: Vec<i32> = Vec::new();
        for prim in mesh.children.iter() {
            let kind = prim.name.as_str();
            if !matches!(kind, "triangles" | "polylist" | "polygons") {
                if matches!(kind, "lines" | "linestrips" | "trifans" | "tristrips") {
                    self.say(&format!("`{}`: {kind} are not read", out.name));
                }
                continue;
            }
            let material = prim
                .attr("material")
                .and_then(|symbol| {
                    binding
                        .iter()
                        .find(|(s, _)| s == symbol)
                        .map(|(_, t)| t.clone())
                        .or_else(|| Some(symbol.to_string()))
                })
                .and_then(|target| self.material(&target))
                .map_or(-1, |m| m as i32);
            // The inputs: what each index in a corner's run is, and at
            // which offset.
            let inputs: Vec<(&str, usize, &str, u32)> = prim
                .all("input")
                .filter_map(|i| {
                    Some((
                        i.attr("semantic")?,
                        i.attr("offset")?.parse().ok()?,
                        i.attr("source")?,
                        i.attr("set").and_then(|s| s.parse().ok()).unwrap_or(0),
                    ))
                })
                .collect();
            let width = inputs.iter().map(|i| i.1).max().map_or(1, |m| m + 1);
            let vertex_at = inputs.iter().find(|i| i.0 == "VERTEX").map(|i| i.1)?;
            let first_uv_set = inputs
                .iter()
                .filter(|i| i.0 == "TEXCOORD")
                .map(|i| i.3)
                .min();
            let uv = inputs
                .iter()
                .find(|i| i.0 == "TEXCOORD" && Some(i.3) == first_uv_set)
                .and_then(|i| Some((i.1, self.source(i.2)?)));
            let normal = inputs
                .iter()
                .find(|i| i.0 == "NORMAL")
                .and_then(|i| Some((i.1, self.source(i.2)?)))
                // Normals given on the vertices themselves.
                .or_else(|| {
                    let url = vertices?
                        .all("input")
                        .find(|i| i.attr("semantic") == Some("NORMAL"))?
                        .attr("source")?;
                    Some((vertex_at, self.source(url)?))
                });
            let color = inputs
                .iter()
                .find(|i| i.0 == "COLOR")
                .and_then(|i| Some((i.1, self.source(i.2)?)));
            // The polygons' sizes, and their indices.
            let mut sizes: Vec<usize> = Vec::new();
            let mut indices: Vec<usize> = Vec::new();
            match kind {
                "triangles" => {
                    indices = numbers_usize(prim.child("p").map(|p| p.text.as_str()).unwrap_or(""));
                    sizes = vec![3; indices.len() / (3 * width)];
                }
                "polylist" => {
                    sizes =
                        numbers_usize(prim.child("vcount").map(|p| p.text.as_str()).unwrap_or(""));
                    indices = numbers_usize(prim.child("p").map(|p| p.text.as_str()).unwrap_or(""));
                }
                _ => {
                    for p in prim.all("p") {
                        let run = numbers_usize(&p.text);
                        sizes.push(run.len() / width);
                        indices.extend(run);
                    }
                    if prim.child("ph").is_some() {
                        self.say("polygons with holes are read without their holes");
                    }
                }
            }
            let mut at = 0;
            for &n in &sizes {
                if n < 3 || (at + n) * width > indices.len() {
                    at += n;
                    continue;
                }
                let mut face = Vec::with_capacity(n);
                for c in 0..n {
                    let run = &indices[(at + c) * width..(at + c + 1) * width];
                    face.push(run[vertex_at] as u32);
                    let item = |(offset, (data, stride)): &(usize, (Vec<f64>, usize))| -> Vec<f64> {
                        let i = run[*offset] * stride;
                        data.get(i..i + stride)
                            .map(<[f64]>::to_vec)
                            .unwrap_or_default()
                    };
                    if let Some(u) = &uv {
                        let v = item(u);
                        uvs.push(Vec2::new(
                            *v.first().unwrap_or(&0.0) as f32,
                            *v.get(1).unwrap_or(&0.0) as f32,
                        ));
                        any_uv = true;
                    } else {
                        uvs.push(Vec2::ZERO);
                    }
                    if let Some(nm) = &normal {
                        let v = item(nm);
                        normals.push(Vec3::new(
                            *v.first().unwrap_or(&0.0) as f32,
                            *v.get(1).unwrap_or(&0.0) as f32,
                            *v.get(2).unwrap_or(&1.0) as f32,
                        ));
                        any_normal = true;
                    } else {
                        normals.push(Vec3::ZERO);
                    }
                    if let Some(cl) = &color {
                        let v = item(cl);
                        colors.push(kreate_math::glam::Vec4::new(
                            *v.first().unwrap_or(&1.0) as f32,
                            *v.get(1).unwrap_or(&1.0) as f32,
                            *v.get(2).unwrap_or(&1.0) as f32,
                            *v.get(3).unwrap_or(&1.0) as f32,
                        ));
                        any_color = true;
                    } else {
                        colors.push(kreate_math::glam::Vec4::ONE);
                    }
                }
                out.faces.push(face);
                face_materials.push(material);
                at += n;
            }
        }
        if out.faces.is_empty() {
            return None;
        }
        let count = out.positions.len() as u32;
        if out.faces.iter().flatten().any(|i| *i >= count) {
            self.say(&format!(
                "`{}`: an index past its positions; the mesh was left out",
                out.name
            ));
            return None;
        }
        out.corner_uvs = any_uv.then_some(uvs);
        out.corner_normals = any_normal.then_some(normals);
        out.corner_colors = any_color.then_some(colors);
        if face_materials.iter().any(|m| *m >= 0) {
            out.face_materials = Some(face_materials);
        }
        self.scene.meshes.push(out);
        Some(self.scene.meshes.len() - 1)
    }

    fn material(&mut self, url: &str) -> Option<usize> {
        if let Some(m) = self.materials.get(url) {
            return *m;
        }
        let made = self.make_material(url);
        self.materials.insert(url.to_string(), made);
        made
    }

    fn make_material(&mut self, url: &str) -> Option<usize> {
        let material = self.find(url)?;
        let mut out = ImportMaterial {
            name: material
                .attr("name")
                .or(material.attr("id"))
                .unwrap_or("material")
                .to_string(),
            ..ImportMaterial::default()
        };
        let effect = material
            .child("instance_effect")
            .and_then(|i| i.attr("url"))
            .and_then(|u| self.find(u));
        let profile = effect.and_then(|e| e.child("profile_COMMON"));
        let technique = profile.and_then(|p| p.child("technique")).and_then(|t| {
            t.children
                .iter()
                .find(|c| matches!(c.name.as_str(), "phong" | "lambert" | "blinn" | "constant"))
        });
        if let Some(t) = technique {
            let colour = |name: &str| -> Option<[f32; 4]> {
                let v = xml::numbers(&t.child(name)?.child("color")?.text);
                (v.len() >= 3).then(|| {
                    [
                        v[0] as f32,
                        v[1] as f32,
                        v[2] as f32,
                        *v.get(3).unwrap_or(&1.0) as f32,
                    ]
                })
            };
            if let Some(c) = colour("diffuse") {
                out.base_color = c;
            }
            if let Some(c) = colour("emission").filter(|c| c[0] + c[1] + c[2] > 0.0) {
                out.emissive = Vec3::new(c[0], c[1], c[2]);
            }
            // The picture behind `diffuse`: a sampler, a surface, an image.
            if let Some(tex) = t
                .child("diffuse")
                .and_then(|d| d.child("texture"))
                .and_then(|x| x.attr("texture"))
            {
                if let Some(image) = self.texture(profile, tex) {
                    out.base_color_texture = Some(image);
                    out.base_color = [1.0, 1.0, 1.0, out.base_color[3]];
                }
            }
            if let Some(a) = t
                .child("transparency")
                .and_then(|x| x.child("float"))
                .and_then(|f| f.text.trim().parse::<f32>().ok())
            {
                // Most exporters write opacity here, whatever the spec's
                // default mode says; a value of zero would hide everything,
                // and is taken to mean "not transparent".
                if a > 0.0 && a < 1.0 {
                    out.base_color[3] = a;
                }
            }
            if let Some(s) = t
                .child("shininess")
                .and_then(|x| x.child("float"))
                .and_then(|f| f.text.trim().parse::<f32>().ok())
            {
                // Blinn-Phong's exponent as a roughness, roughly.
                out.roughness = (2.0 / (s.max(0.0) + 2.0)).sqrt().clamp(0.05, 1.0);
            }
        } else if effect.is_some() {
            self.say("effects without a common profile are read as plain grey");
        }
        self.scene.materials.push(out);
        Some(self.scene.materials.len() - 1)
    }

    /// A picture a texture names: through a `newparam` sampler and surface
    /// (1.4), or straight to an image (1.4 files from some tools, and 1.5).
    fn texture(&mut self, profile: Option<&'a Element>, name: &str) -> Option<usize> {
        let param = |sid: &str| {
            profile?
                .all("newparam")
                .find(|p| p.attr("sid") == Some(sid))
        };
        let mut image_id = name.to_string();
        if let Some(sampler) = param(name).and_then(|p| p.child("sampler2D")) {
            let surface = sampler.child("source").map(|s| s.text.trim().to_string());
            if let Some(init) = surface
                .as_deref()
                .and_then(param)
                .and_then(|p| p.child("surface"))
                .and_then(|s| s.child("init_from"))
            {
                image_id = init.text.trim().to_string();
            } else if let Some(i) = sampler.child("instance_image").and_then(|i| i.attr("url")) {
                image_id = i.trim_start_matches('#').to_string();
            }
        }
        let image = self.find(&image_id)?;
        let path = image.child("init_from").map(|i| {
            i.child("ref")
                .map_or(i.text.trim(), |r| r.text.trim())
                .to_string()
        })?;
        if let Some(known) = self.images.get(&path) {
            return *known;
        }
        let read = (self.file)(&path)
            .and_then(|bytes| image::load_from_memory(&bytes).ok())
            .map(|decoded| {
                let rgba = decoded.to_rgba8();
                self.scene.images.push(ImportImage {
                    name: Path::new(&path)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    width: rgba.width(),
                    height: rgba.height(),
                    pixels: rgba.into_raw(),
                });
                self.scene.images.len() - 1
            });
        if read.is_none() {
            self.say(&format!("the picture `{path}` was not found or not read"));
        }
        self.images.insert(path, read);
        read
    }
}

fn numbers_usize(text: &str) -> Vec<usize> {
    text.split_ascii_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// A node's transform elements, applied in the order written.
fn transform_of(e: &Element) -> ImportTransform {
    use kreate_math::glam::{DMat4, DQuat, DVec3};
    let mut m = DMat4::IDENTITY;
    for c in &e.children {
        let v = xml::numbers(&c.text);
        let op = match c.name.as_str() {
            "matrix" if v.len() >= 16 => {
                DMat4::from_cols_array(&v[..16].try_into().expect("sixteen")).transpose()
            }
            "translate" if v.len() >= 3 => DMat4::from_translation(DVec3::new(v[0], v[1], v[2])),
            "scale" if v.len() >= 3 => DMat4::from_scale(DVec3::new(v[0], v[1], v[2])),
            "rotate" if v.len() >= 4 => {
                let axis = DVec3::new(v[0], v[1], v[2]);
                match axis.try_normalize() {
                    Some(a) => DMat4::from_quat(DQuat::from_axis_angle(a, v[3].to_radians())),
                    None => continue,
                }
            }
            _ => continue,
        };
        m *= op;
    }
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    ImportTransform {
        translation: translation.as_vec3(),
        rotation: rotation.as_quat().normalize(),
        scale: scale.as_vec3(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="centimeter" meter="0.01"/><up_axis>Z_UP</up_axis></asset>
  <library_images><image id="img"><init_from>tex.png</init_from></image></library_images>
  <library_effects>
    <effect id="red-fx"><profile_COMMON>
      <newparam sid="surf"><surface type="2D"><init_from>img</init_from></surface></newparam>
      <newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>
      <technique sid="common"><phong>
        <diffuse><texture texture="samp" texcoord="UV"/></diffuse>
      </phong></technique>
    </profile_COMMON></effect>
    <effect id="blue-fx"><profile_COMMON><technique sid="common"><lambert>
      <diffuse><color>0 0 1 1</color></diffuse>
    </lambert></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="red" name="Red"><instance_effect url="#red-fx"/></material>
    <material id="blue" name="Blue"><instance_effect url="#blue-fx"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="quad" name="Quad"><mesh>
      <source id="pos"><float_array id="pos-a" count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
        <technique_common><accessor source="#pos-a" count="4" stride="3"/></technique_common></source>
      <source id="uv"><float_array id="uv-a" count="4">0 0 1 1</float_array>
        <technique_common><accessor source="#uv-a" count="2" stride="2"/></technique_common></source>
      <vertices id="verts"><input semantic="POSITION" source="#pos"/></vertices>
      <polylist material="m0" count="1">
        <input semantic="VERTEX" source="#verts" offset="0"/>
        <input semantic="TEXCOORD" source="#uv" offset="1" set="0"/>
        <vcount>4</vcount>
        <p>0 0 1 1 2 1 3 0</p>
      </polylist>
      <triangles material="m1" count="1">
        <input semantic="VERTEX" source="#verts" offset="0"/>
        <p>0 2 3</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="scene">
    <node id="n" name="Holder">
      <translate>0 0 100</translate>
      <rotate>0 0 1 90</rotate>
      <instance_geometry url="#quad"><bind_material><technique_common>
        <instance_material symbol="m0" target="#red"/>
        <instance_material symbol="m1" target="#blue"/>
      </technique_common></bind_material></instance_geometry>
    </node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>"##;

    #[test]
    fn a_scene_reads_with_its_units_axes_meshes_and_materials() {
        let mut png = Vec::new();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let scene = parse_dae(DOC, &|name| (name == "tex.png").then(|| png.clone())).unwrap();
        assert_eq!(scene.source.axis, AxisConvention::KREATE);
        assert_eq!(scene.source.unit_scale, 0.01);
        assert_eq!(scene.nodes.len(), 1);
        let world = scene.world_transform(0).unwrap();
        // A metre up (100 cm), turned a quarter about Z.
        assert!((world.translation.z - 1.0).abs() < 1e-6);
        let x = world.transform_vector3(kreate_math::DVec3::X).normalize();
        assert!((x - kreate_math::DVec3::Y).length() < 1e-6, "{x}");
        let mesh = &scene.meshes[0];
        assert_eq!(mesh.faces, vec![vec![0, 1, 2, 3], vec![0, 2, 3]]);
        let uvs = mesh.corner_uvs.as_ref().unwrap();
        assert_eq!(uvs[1], Vec2::ONE);
        let fm = mesh.face_materials.as_ref().unwrap();
        assert_eq!(scene.materials[fm[0] as usize].name, "Red");
        assert_eq!(scene.materials[fm[0] as usize].base_color_texture, Some(0));
        assert_eq!(
            scene.materials[fm[1] as usize].base_color,
            [0.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(scene.images[0].width, 2);
    }

    #[test]
    fn what_is_not_collada_is_said() {
        assert!(parse_dae("<model/>", &|_| None).is_err());
        assert!(parse_dae("<COLLADA/>", &|_| None).is_err());
    }
}
