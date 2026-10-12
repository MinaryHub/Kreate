//! COLLADA, `.dae`: the XML scene format that robot descriptions name for
//! their meshes, and that older tools still write.
//!
//! What a scene needs: the node tree and its transforms (`matrix`,
//! `translate`, `rotate`, `scale`, in order), the geometries it instances —
//! `triangles`, `polylist` and `polygons`, with positions, normals, the
//! first set of texture coordinates and vertex colours — and the materials
//! bound to them, read from their effect's common profile: a diffuse colour
//! or picture, emission, transparency. The file's `<unit>` and `<up_axis>`
//! are its units and axes.
//!
//! Rigs too: a skin controller binds its mesh to the joint nodes it names
//! (by `sid`, or by id), with their inverse bind matrices, its bind shape,
//! and its weights four to a point; a morph controller's targets become the
//! mesh's morph targets at its weights; and the animations keying a node's
//! transform elements — whole, or a part (`.ANGLE`, `.X`, `(i)(j)`) —
//! become its channels. Physics and the programmable profiles are not read.

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
        by_sid: HashMap::new(),
        by_id: HashMap::new(),
        elements: HashMap::new(),
        pending: Vec::new(),
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
    reader.controllers();
    if let Some(library) = root.child("library_animations") {
        reader.animations(library);
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
    /// Nodes made, by their scoped id (`sid`, what a skin names its joints
    /// by) and their id, and the element each was made from.
    by_sid: HashMap<String, usize>,
    by_id: HashMap<String, usize>,
    elements: HashMap<usize, &'a Element>,
    /// Controllers instanced, to bind once every joint is a node.
    pending: Vec<(usize, &'a Element)>,
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
        if let Some(sid) = e.attr("sid") {
            self.by_sid.entry(sid.to_string()).or_insert(at);
        }
        if let Some(id) = e.attr("id") {
            self.by_id.insert(id.to_string(), at);
        }
        if let Some(name) = e.attr("name") {
            self.by_sid.entry(name.to_string()).or_insert(at);
        }
        self.elements.insert(at, e);
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
        for c in e.all("instance_controller") {
            self.pending.push((at, c));
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

    fn binding_of(g: &Element) -> Vec<(String, String)> {
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
        binding
    }

    fn instance(&mut self, g: &'a Element) -> Option<usize> {
        let url = g.attr("url")?;
        // The material each symbol in the geometry stands for here.
        let binding = Self::binding_of(g);
        let key = (url.to_string(), binding.clone());
        if let Some(made) = self.meshes.get(&key) {
            return *made;
        }
        let geometry = self.find(url);
        let made = geometry.and_then(|geo| self.mesh(geo, &binding));
        self.meshes.insert(key, made);
        made
    }

    /// A `<source>`'s names: a `Name_array` or an `IDREF_array`.
    fn names(&self, url: &str) -> Vec<String> {
        let Some(s) = self.find(url) else {
            return Vec::new();
        };
        s.child("Name_array")
            .or_else(|| s.child("IDREF_array"))
            .map(|a| {
                a.text
                    .split_ascii_whitespace()
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every instanced controller: a skin bound to its joints, a morph's
    /// targets as morph targets — the mesh made fresh, as the controller
    /// changes it.
    fn controllers(&mut self) {
        use kreate_math::glam::Mat4;
        for (node, ic) in std::mem::take(&mut self.pending) {
            let Some(controller) = ic.attr("url").and_then(|u| self.find(u)) else {
                continue;
            };
            let binding = Self::binding_of(ic);
            let skin = controller.child("skin");
            // A skin's source may be a morph, whose source is the geometry.
            let mut morph = controller.child("morph");
            let mut source = skin.or(morph).and_then(|c| c.attr("source"));
            if let Some(inner) = source
                .and_then(|s| self.find(s))
                .and_then(|c| c.child("morph"))
            {
                morph = Some(inner);
                source = inner.attr("source");
            }
            let Some(geometry) = source
                .and_then(|s| self.find(s))
                .filter(|g| g.name == "geometry")
            else {
                continue;
            };
            let Some(mesh) = self.mesh(geometry, &binding) else {
                continue;
            };
            // Where the mesh goes: on the node, or a child when the node
            // already shows something.
            let at = if self.scene.nodes[node].data == ImportNodeData::Empty {
                node
            } else {
                let mut child = ImportNode::new(format!("{}_skin", self.scene.nodes[node].name));
                child.parent = Some(node);
                self.scene.nodes.push(child);
                self.scene.nodes.len() - 1
            };
            self.scene.nodes[at].data = ImportNodeData::Mesh(mesh);
            if let Some(m) = morph {
                self.morph(m, geometry, mesh, at);
            }
            let Some(skin) = skin else { continue };
            // The bind shape: where the mesh stood when it was bound.
            let bind_shape = skin
                .child("bind_shape_matrix")
                .map(|b| xml::numbers(&b.text))
                .filter(|n| n.len() >= 16)
                .map(|n| {
                    Mat4::from_cols_array(
                        &n[..16]
                            .iter()
                            .map(|v| *v as f32)
                            .collect::<Vec<_>>()
                            .try_into()
                            .expect("sixteen"),
                    )
                    .transpose()
                })
                .unwrap_or(Mat4::IDENTITY);
            {
                let normal = kreate_math::Mat3::from_mat4(bind_shape)
                    .inverse()
                    .transpose();
                let data = &mut self.scene.meshes[mesh];
                for p in &mut data.positions {
                    *p = bind_shape.transform_point3(*p);
                }
                if let Some(ns) = &mut data.corner_normals {
                    for n in ns {
                        *n = (normal * *n).normalize_or_zero();
                    }
                }
            }
            let joints_el = skin.child("joints");
            let input = |el: Option<&'a Element>, semantic: &str| {
                el.and_then(|j| {
                    j.all("input")
                        .find(|i| i.attr("semantic") == Some(semantic))
                })
                .and_then(|i| i.attr("source"))
            };
            let names = input(joints_el, "JOINT")
                .map(|u| self.names(u))
                .unwrap_or_default();
            let inverse: Vec<Mat4> = input(joints_el, "INV_BIND_MATRIX")
                .and_then(|u| self.source(u))
                .map(|(n, _)| {
                    n.as_chunks::<16>()
                        .0
                        .iter()
                        .map(|m| {
                            Mat4::from_cols_array(
                                &m.iter()
                                    .map(|v| *v as f32)
                                    .collect::<Vec<_>>()
                                    .try_into()
                                    .expect("sixteen"),
                            )
                            .transpose()
                        })
                        .collect()
                })
                .unwrap_or_default();
            let joints: Vec<Option<usize>> = names
                .iter()
                .map(|n| self.by_sid.get(n).or_else(|| self.by_id.get(n)).copied())
                .collect();
            if joints.is_empty() || joints.iter().any(Option::is_none) {
                self.say("a skin whose joints were not all found came in at rest");
                continue;
            }
            // The weights: per vertex, how many, then joint and weight
            // indices.
            let Some(vw) = skin.child("vertex_weights") else {
                continue;
            };
            let inputs: Vec<(&str, usize, &str)> = vw
                .all("input")
                .filter_map(|i| {
                    Some((
                        i.attr("semantic")?,
                        i.attr("offset")?.parse().ok()?,
                        i.attr("source")?,
                    ))
                })
                .collect();
            let width = inputs.iter().map(|i| i.1).max().map_or(1, |m| m + 1);
            let joint_at = inputs
                .iter()
                .find(|i| i.0 == "JOINT")
                .map(|i| i.1)
                .unwrap_or(0);
            let weight_input = inputs.iter().find(|i| i.0 == "WEIGHT");
            let weight_at = weight_input.map(|i| i.1).unwrap_or(1);
            let weights = weight_input
                .and_then(|i| self.source(i.2))
                .map(|(n, _)| n)
                .unwrap_or_default();
            let counts = numbers_usize(vw.child("vcount").map(|c| c.text.as_str()).unwrap_or(""));
            let v: Vec<i64> = vw
                .child("v")
                .map(|c| {
                    c.text
                        .split_ascii_whitespace()
                        .filter_map(|t| t.parse().ok())
                        .collect()
                })
                .unwrap_or_default();
            let mut data = crate::ir::VertexSkin::default();
            let mut at_v = 0usize;
            let points = self.scene.meshes[mesh].positions.len();
            for p in 0..points {
                let n = counts.get(p).copied().unwrap_or(0);
                let mut pairs: Vec<(usize, f32)> = (0..n)
                    .filter_map(|k| {
                        let run = v.get((at_v + k) * width..(at_v + k + 1) * width)?;
                        let j = usize::try_from(run[joint_at]).ok()?;
                        let w = *weights.get(usize::try_from(run[weight_at]).ok()?)? as f32;
                        Some((j, w))
                    })
                    .filter(|(_, w)| *w > 0.0)
                    .collect();
                at_v += n;
                pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
                pairs.truncate(4);
                let total: f32 = pairs.iter().map(|(_, w)| w).sum();
                let mut js = [0u16; 4];
                let mut ws = [0f32; 4];
                for (k, (j, w)) in pairs.iter().enumerate() {
                    js[k] = *j as u16;
                    ws[k] = if total > 0.0 { w / total } else { 0.0 };
                }
                data.joints.push(js);
                data.weights.push(ws);
            }
            if self.scene.vertex_skins.len() < self.scene.meshes.len() {
                self.scene
                    .vertex_skins
                    .resize(self.scene.meshes.len(), None);
            }
            self.scene.vertex_skins[mesh] = Some(data);
            self.scene.skins.push(crate::ir::ImportSkin {
                name: controller
                    .attr("name")
                    .or(controller.attr("id"))
                    .unwrap_or("skin")
                    .to_string(),
                joints: joints.into_iter().flatten().collect(),
                inverse_bind: (inverse.len() == names.len()).then_some(inverse),
            });
            self.scene.nodes[at].skin = Some(self.scene.skins.len() - 1);
        }
    }

    /// A morph's targets as the mesh's morph targets: each target
    /// geometry's points less the base's, and the controller's weights.
    fn morph(&mut self, morph: &'a Element, base: &'a Element, mesh: usize, node: usize) {
        let targets = morph.child("targets");
        let input = |semantic: &str| {
            targets
                .and_then(|t| {
                    t.all("input")
                        .find(|i| i.attr("semantic") == Some(semantic))
                })
                .and_then(|i| i.attr("source"))
        };
        let names = input("MORPH_TARGET")
            .map(|u| self.names(u))
            .unwrap_or_default();
        let weights = input("MORPH_WEIGHT")
            .and_then(|u| self.source(u))
            .map(|(n, _)| n)
            .unwrap_or_default();
        let positions = |g: &Element| -> Option<Vec<Vec3>> {
            let mesh = g.child("mesh")?;
            let url = mesh
                .child("vertices")?
                .all("input")
                .find(|i| i.attr("semantic") == Some("POSITION"))?
                .attr("source")?;
            let (raw, stride) = self.source(url)?;
            Some(
                raw.chunks(stride)
                    .filter(|c| c.len() >= 3)
                    .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
                    .collect(),
            )
        };
        let Some(rest) = positions(base) else { return };
        let mut morphs = Vec::new();
        for name in &names {
            let Some(target) = self.find(name) else {
                continue;
            };
            let Some(at) = positions(target).filter(|p| p.len() == rest.len()) else {
                continue;
            };
            let label = target.attr("name").unwrap_or(name).to_string();
            morphs.push((label, at.iter().zip(&rest).map(|(a, b)| *a - *b).collect()));
        }
        if morphs.is_empty() {
            return;
        }
        if self.scene.vertex_morphs.len() < self.scene.meshes.len() {
            self.scene
                .vertex_morphs
                .resize(self.scene.meshes.len(), Vec::new());
        }
        self.scene.vertex_morphs[mesh] = morphs;
        if self.scene.node_weights.len() <= node {
            self.scene.node_weights.resize(node + 1, Vec::new());
        }
        self.scene.node_weights[node] = weights.iter().map(|w| *w as f32).collect();
    }

    /// The animations: each node whose transform elements are keyed, its
    /// placement worked out at every key and given as channels.
    fn animations(&mut self, library: &'a Element) {
        use crate::ir::{ChannelInterpolation, ChannelValues, ImportAnimation, ImportChannel};
        // (node, element sid, member) → (times, values, values a key, step).
        type Keys = (Vec<f64>, Vec<f64>, usize, bool);
        let mut keyed: HashMap<usize, Vec<(String, String, Keys)>> = HashMap::new();
        for animation in library
            .descendants()
            .into_iter()
            .filter(|e| e.name == "animation")
        {
            for channel in animation.all("channel") {
                let (Some(source), Some(target)) = (channel.attr("source"), channel.attr("target"))
                else {
                    continue;
                };
                let Some((node_id, rest)) = target.split_once('/') else {
                    continue;
                };
                let Some(&node) = self.by_id.get(node_id) else {
                    continue;
                };
                let (sid, member) = match rest.split_once('.') {
                    Some((s, m)) => (s.to_string(), m.to_string()),
                    None => match rest.split_once('(') {
                        Some((s, m)) => (s.to_string(), format!("({m}")),
                        None => (rest.to_string(), String::new()),
                    },
                };
                let Some(sampler) = self.find(source) else {
                    continue;
                };
                let input = |semantic: &str| {
                    sampler
                        .all("input")
                        .find(|i| i.attr("semantic") == Some(semantic))
                        .and_then(|i| i.attr("source"))
                };
                let (Some((times, _)), Some((values, stride))) = (
                    input("INPUT").and_then(|u| self.source(u)),
                    input("OUTPUT").and_then(|u| self.source(u)),
                ) else {
                    continue;
                };
                let step = input("INTERPOLATION")
                    .map(|u| self.names(u))
                    .is_some_and(|n| n.iter().all(|i| i == "STEP"));
                keyed
                    .entry(node)
                    .or_default()
                    .push((sid, member, (times, values, stride, step)));
            }
        }
        if keyed.is_empty() {
            return;
        }
        let mut channels = Vec::new();
        let mut nodes: Vec<usize> = keyed.keys().copied().collect();
        nodes.sort_unstable();
        for node in nodes {
            let Some(element) = self.elements.get(&node).copied() else {
                continue;
            };
            let keys = &keyed[&node];
            let mut times: Vec<f64> = keys
                .iter()
                .flat_map(|(_, _, k)| k.0.iter().copied())
                .collect();
            times.sort_by(f64::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            let value_at = |k: &Keys, t: f64| -> Vec<f64> {
                let (ts, vs, stride, step) = k;
                let at = |i: usize| {
                    vs.get(i * stride..(i + 1) * stride)
                        .map(<[f64]>::to_vec)
                        .unwrap_or_default()
                };
                match ts.iter().position(|x| *x >= t - 1e-9) {
                    None => at(ts.len().saturating_sub(1)),
                    Some(0) => at(0),
                    Some(i) if *step || (ts[i] - t).abs() < 1e-9 => {
                        at(if (ts[i] - t).abs() < 1e-9 { i } else { i - 1 })
                    }
                    Some(i) => {
                        let f = (t - ts[i - 1]) / (ts[i] - ts[i - 1]);
                        at(i - 1)
                            .iter()
                            .zip(at(i))
                            .map(|(a, b)| a + (b - a) * f)
                            .collect()
                    }
                }
            };
            let mut translations = Vec::new();
            let mut rotations: Vec<kreate_math::Quat> = Vec::new();
            let mut scales = Vec::new();
            for t in &times {
                let over: Vec<(String, String, Vec<f64>)> = keys
                    .iter()
                    .map(|(sid, member, k)| (sid.clone(), member.clone(), value_at(k, *t)))
                    .collect();
                let at = transform_with(element, &over);
                let q = match rotations.last() {
                    Some(prev) if prev.dot(at.rotation) < 0.0 => -at.rotation,
                    _ => at.rotation,
                };
                translations.push(at.translation);
                rotations.push(q);
                scales.push(at.scale);
            }
            let seconds: Vec<f32> = times.iter().map(|t| *t as f32).collect();
            for values in [
                ChannelValues::Translation(translations),
                ChannelValues::Rotation(rotations),
                ChannelValues::Scale(scales),
            ] {
                channels.push(ImportChannel {
                    node,
                    interpolation: ChannelInterpolation::Linear,
                    times: seconds.clone(),
                    values,
                });
            }
        }
        self.scene.animations.push(ImportAnimation {
            name: "Take".into(),
            channels,
        });
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
    transform_with(e, &[])
}

/// The same, with some elements' values keyed: `(sid, member, value)`, the
/// member empty for the whole element, `X`/`Y`/`Z`/`ANGLE` for one part,
/// `(i)` or `(i)(j)` for one number.
fn transform_with(e: &Element, keyed: &[(String, String, Vec<f64>)]) -> ImportTransform {
    use kreate_math::glam::{DMat4, DQuat, DVec3};
    let mut m = DMat4::IDENTITY;
    for c in &e.children {
        let mut v = xml::numbers(&c.text);
        if let Some(sid) = c.attr("sid") {
            for (s, member, value) in keyed.iter().filter(|k| k.0 == sid) {
                let slot = match member.as_str() {
                    "" => {
                        if value.len() == v.len() {
                            v = value.clone();
                        }
                        continue;
                    }
                    "X" => Some(0),
                    "Y" => Some(1),
                    "Z" => Some(2),
                    "ANGLE" => Some(3),
                    indices => {
                        let nums: Vec<usize> = indices
                            .split(['(', ')'])
                            .filter_map(|n| n.parse().ok())
                            .collect();
                        match nums.as_slice() {
                            [i] => Some(*i),
                            // A matrix element by row and column.
                            [r, col] => Some(r * 4 + col),
                            _ => None,
                        }
                    }
                };
                let _ = s;
                if let (Some(i), Some(x)) = (slot, value.first()) {
                    if let Some(slot) = v.get_mut(i) {
                        *slot = *x;
                    }
                }
            }
        }
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

    const RIG: &str = r##"<?xml version="1.0"?>
<COLLADA version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="stick"><mesh>
      <source id="p"><float_array count="12">0 0 0 0.1 0 0 0 0 2 0.1 0 2</float_array>
        <technique_common><accessor count="4" stride="3"/></technique_common></source>
      <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
      <polylist count="1"><input semantic="VERTEX" source="#v" offset="0"/><vcount>4</vcount><p>0 1 3 2</p></polylist>
    </mesh></geometry>
    <geometry id="stick_bent"><mesh>
      <source id="pb"><float_array count="12">0 0 0 0.1 0 0 0.5 0 2 0.6 0 2</float_array>
        <technique_common><accessor count="4" stride="3"/></technique_common></source>
      <vertices id="vb"><input semantic="POSITION" source="#pb"/></vertices>
      <polylist count="1"><input semantic="VERTEX" source="#vb" offset="0"/><vcount>4</vcount><p>0 1 3 2</p></polylist>
    </mesh></geometry>
  </library_geometries>
  <library_controllers>
    <controller id="lean"><morph source="#stick" method="RELATIVE">
      <source id="lt"><IDREF_array count="1">stick_bent</IDREF_array></source>
      <source id="lw"><float_array count="1">0.25</float_array></source>
      <targets><input semantic="MORPH_TARGET" source="#lt"/><input semantic="MORPH_WEIGHT" source="#lw"/></targets>
    </morph></controller>
    <controller id="skin"><skin source="#lean">
      <bind_shape_matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</bind_shape_matrix>
      <source id="jn"><Name_array count="2">hip knee</Name_array></source>
      <source id="ib"><float_array count="32">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1  1 0 0 0 0 1 0 0 0 0 1 -1 0 0 0 1</float_array>
        <technique_common><accessor count="2" stride="16"/></technique_common></source>
      <source id="w"><float_array count="2">1 0.5</float_array></source>
      <joints><input semantic="JOINT" source="#jn"/><input semantic="INV_BIND_MATRIX" source="#ib"/></joints>
      <vertex_weights count="4">
        <input semantic="JOINT" source="#jn" offset="0"/><input semantic="WEIGHT" source="#w" offset="1"/>
        <vcount>1 1 2 1</vcount>
        <v>0 0 0 0 0 1 1 1 1 0</v>
      </vertex_weights>
    </skin></controller>
  </library_controllers>
  <library_animations>
    <animation id="bend">
      <source id="t"><float_array count="2">0 1</float_array></source>
      <source id="a"><float_array count="2">0 90</float_array></source>
      <sampler id="s"><input semantic="INPUT" source="#t"/><input semantic="OUTPUT" source="#a"/></sampler>
      <channel source="#s" target="knee/rot.ANGLE"/>
    </animation>
  </library_animations>
  <library_visual_scenes><visual_scene id="vs">
    <node id="hip" sid="hip" type="JOINT">
      <node id="knee" sid="knee" type="JOINT">
        <translate>0 0 1</translate>
        <rotate sid="rot">1 0 0 0</rotate>
      </node>
    </node>
    <node id="body" name="Body">
      <instance_controller url="#skin"><skeleton>#hip</skeleton></instance_controller>
    </node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#vs"/></scene>
</COLLADA>"##;

    #[test]
    fn a_skin_its_morph_and_its_animation_come_in() {
        let scene = parse_dae(RIG, &|_| None).unwrap();
        assert_eq!(scene.skins.len(), 1);
        let names: Vec<&str> = scene.skins[0]
            .joints
            .iter()
            .map(|j| scene.nodes[*j].name.as_str())
            .collect();
        assert_eq!(names, vec!["hip", "knee"]);
        let body = scene.nodes.iter().position(|n| n.name == "Body").unwrap();
        assert_eq!(scene.nodes[body].skin, Some(0));
        let weights = scene.vertex_skins[0].as_ref().unwrap();
        assert_eq!(weights.joints[2], [0, 1, 0, 0]);
        assert_eq!(weights.weights[2], [0.5, 0.5, 0.0, 0.0]);
        // The morph: the bent stick's points less the straight one's.
        let (name, offsets) = &scene.vertex_morphs[0][0];
        assert_eq!(name, "stick_bent");
        assert_eq!(offsets[2], Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(scene.node_weights[body], vec![0.25]);
        // The knee turns a quarter about X over a second.
        let knee = scene.nodes.iter().position(|n| n.name == "knee").unwrap();
        let turn = scene.animations[0]
            .channels
            .iter()
            .find(|c| c.node == knee && matches!(c.values, crate::ir::ChannelValues::Rotation(_)))
            .unwrap();
        let crate::ir::ChannelValues::Rotation(q) = &turn.values else {
            unreachable!()
        };
        assert!((q[1].to_axis_angle().1 - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        // And it builds: an armature, the mesh bound to it.
        let mut doc = kreate_scene::Document::new();
        crate::build::build_into(&scene, &mut doc);
        assert!(doc
            .objects()
            .any(|(_, o)| matches!(o.data, kreate_scene::ObjectData::Armature(_))));
        assert!(doc
            .objects()
            .any(|(_, o)| o.skin.is_some() && !o.shape_keys.is_empty()));
    }

    #[test]
    fn what_is_not_collada_is_said() {
        assert!(parse_dae("<model/>", &|_| None).is_err());
        assert!(parse_dae("<COLLADA/>", &|_| None).is_err());
    }
}
