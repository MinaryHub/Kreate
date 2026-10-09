//! 3MF: the 3D printing format. A zip holding an XML model.
//!
//! Unlike STL it says what unit it is in, keeps one object per part with its
//! name, and is always triangles, wound outward, which is what a slicer
//! wants. Kreate writes each part as it stands — modifiers applied, placed in
//! the world — in the scene's unit, so a part drawn as 40 mm arrives as 40 mm
//! whatever the slicer assumes.
//!
//! Only what printing needs: the meshes. No materials, colours or build
//! plate settings.

use std::io::{Cursor, Write};

use kreate_math::DAffine3;
use kreate_mesh::MeshData;

/// The units 3MF knows, by Kreate's length unit tag; anything else is
/// written in millimetres.
pub fn unit_name(metres_per_unit: f64) -> (&'static str, f64) {
    const UNITS: [(&str, f64); 6] = [
        ("micron", 1e-6),
        ("millimeter", 1e-3),
        ("centimeter", 1e-2),
        ("inch", 0.0254),
        ("foot", 0.3048),
        ("meter", 1.0),
    ];
    UNITS
        .iter()
        .find(|(_, m)| (m - metres_per_unit).abs() < 1e-12)
        .copied()
        .unwrap_or(("millimeter", 1e-3))
}

/// Writes `parts` — a name, a mesh and where it is in the world — as a 3MF
/// file in the unit that is `metres_per_unit` metres.
pub fn write_3mf(parts: &[(String, &MeshData, DAffine3)], metres_per_unit: f64) -> Vec<u8> {
    let (unit, scale) = unit_name(metres_per_unit);
    let mut model = String::new();
    model.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    model.push_str(&format!(
        "<model unit=\"{unit}\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
    ));
    model.push_str(" <resources>\n");
    let mut ids = Vec::new();
    for (i, (name, mesh, world)) in parts.iter().enumerate() {
        let id = i + 1;
        let tris = mesh.triangulate();
        if tris.indices.is_empty() {
            continue;
        }
        ids.push(id);
        model.push_str(&format!(
            "  <object id=\"{id}\" type=\"model\" name=\"{}\">\n   <mesh>\n    <vertices>\n",
            escape(name)
        ));
        for p in mesh.positions() {
            let w = world.transform_point3(p.as_dvec3()) / scale;
            model.push_str(&format!(
                "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                number(w.x),
                number(w.y),
                number(w.z)
            ));
        }
        model.push_str("    </vertices>\n    <triangles>\n");
        // A mirrored placement turns the surface inside out; the triangles are
        // turned back so they still face outward.
        let flip = world.matrix3.determinant() < 0.0;
        for t in tris.indices.as_chunks::<3>().0 {
            let (a, b, c) = if flip {
                (t[0], t[2], t[1])
            } else {
                (t[0], t[1], t[2])
            };
            model.push_str(&format!(
                "     <triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"/>\n"
            ));
        }
        model.push_str("    </triangles>\n   </mesh>\n  </object>\n");
    }
    model.push_str(" </resources>\n <build>\n");
    for id in ids {
        model.push_str(&format!("  <item objectid=\"{id}\"/>\n"));
    }
    model.push_str(" </build>\n</model>\n");

    let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n\
 <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n\
 <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n\
</Types>\n";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n\
 <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/>\n\
</Relationships>\n";

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .last_modified_time(zip::DateTime::default());
        for (path, body) in [
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", rels),
            ("3D/3dmodel.model", model.as_str()),
        ] {
            // Writing to memory cannot fail but for running out of it.
            zip.start_file(path, options).expect("in memory");
            zip.write_all(body.as_bytes()).expect("in memory");
        }
        zip.finish().expect("in memory");
    }
    buffer.into_inner()
}

/// A coordinate, with what precision a printer can use and no more.
fn number(v: f64) -> String {
    let text = format!("{v:.6}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" || text.is_empty() {
        "0".to_string()
    } else {
        text.to_string()
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// --- reading -------------------------------------------------------------

/// Why a 3MF file could not be read.
#[derive(Debug, thiserror::Error)]
pub enum ThreeMfError {
    #[error("not a 3MF file: {0}")]
    Zip(String),
    #[error("the package has no model part")]
    NoModel,
    #[error("the model is not text: {0}")]
    Encoding(String),
}

/// One XML tag, as far as 3MF needs: its name without a prefix, its
/// attributes, and whether it opens, closes or is both.
#[derive(Debug)]
struct Tag {
    name: String,
    attributes: Vec<(String, String)>,
    closing: bool,
    empty: bool,
}

impl Tag {
    fn get(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Every tag in `xml`, in order. Text between tags is not wanted — 3MF puts
/// everything in attributes — so it is passed over, as are comments,
/// declarations and processing instructions.
///
/// Not a general XML parser, and it does not need to be one: what it reads
/// is a model part, whose structure the specification fixes.
fn tags(xml: &str) -> Vec<Tag> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(open) = rest.find('<') {
        rest = &rest[open + 1..];
        if let Some(after) = rest.strip_prefix("!--") {
            rest = after.find("-->").map_or("", |e| &after[e + 3..]);
            continue;
        }
        if rest.starts_with('?') || rest.starts_with('!') {
            rest = rest.find('>').map_or("", |e| &rest[e + 1..]);
            continue;
        }
        // The end of the tag: the first `>` not inside a quoted value.
        let mut quote = None;
        let mut end = None;
        for (k, c) in rest.char_indices() {
            match (quote, c) {
                (None, '"' | '\'') => quote = Some(c),
                (Some(q), c) if c == q => quote = None,
                (None, '>') => {
                    end = Some(k);
                    break;
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        let body = &rest[..end];
        rest = &rest[end + 1..];
        let closing = body.starts_with('/');
        let empty = body.ends_with('/');
        let body = body.trim_start_matches('/').trim_end_matches('/');
        let name_end = body.find(|c: char| c.is_whitespace()).unwrap_or(body.len());
        let name = local(&body[..name_end]).to_string();
        let mut attributes = Vec::new();
        let mut a = &body[name_end..];
        while let Some(eq) = a.find('=') {
            let key = local(a[..eq].trim()).to_string();
            let v = a[eq + 1..].trim_start();
            let Some(q) = v.chars().next().filter(|c| *c == '"' || *c == '\'') else {
                break;
            };
            let Some(close) = v[1..].find(q) else { break };
            attributes.push((key, unescape(&v[1..1 + close])));
            a = &v[close + 2..];
        }
        out.push(Tag {
            name,
            attributes,
            closing,
            empty,
        });
    }
    out
}

/// A name without its namespace prefix.
fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn unescape(v: &str) -> String {
    v.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// 3MF's twelve numbers, as the matrix they are: rows of a row-vector
/// transform, the last row the translation.
fn matrix(text: Option<&str>) -> glam::Mat4 {
    let m: Vec<f32> = text
        .unwrap_or("")
        .split_ascii_whitespace()
        .filter_map(|w| w.parse().ok())
        .collect();
    if m.len() != 12 {
        return glam::Mat4::IDENTITY;
    }
    glam::Mat4::from_cols(
        glam::Vec4::new(m[0], m[1], m[2], 0.0),
        glam::Vec4::new(m[3], m[4], m[5], 0.0),
        glam::Vec4::new(m[6], m[7], m[8], 0.0),
        glam::Vec4::new(m[9], m[10], m[11], 1.0),
    )
}

/// An sRGB `#RRGGBB` or `#RRGGBBAA`, linear.
fn colour(text: &str) -> Option<[f32; 4]> {
    let hex = text.strip_prefix('#')?;
    let byte = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
    let linear = |b: u8| kreate_math::srgb_to_linear(b as f32 / 255.0);
    let alpha = if hex.len() >= 8 {
        byte(6)? as f32 / 255.0
    } else {
        1.0
    };
    Some([linear(byte(0)?), linear(byte(2)?), linear(byte(4)?), alpha])
}

/// An object of the model: a mesh, or parts of other objects placed — each
/// part by the model part it is in (`None` for this one), its id there, and
/// its placement.
#[derive(Default)]
struct Object {
    name: String,
    mesh: Option<crate::ir::ImportMesh>,
    components: Vec<(Option<String>, String, glam::Mat4)>,
}

/// What one model part holds: its objects by id, what it builds, and its
/// unit.
struct Part {
    objects: Vec<(String, Object)>,
    items: Vec<(String, glam::Mat4)>,
    unit: f32,
}

/// Reads one model part. Its base materials join `materials`; its groups'
/// ids are its own, as the specification scopes them.
fn read_part(
    xml: &str,
    materials: &mut Vec<crate::ir::ImportMaterial>,
    notes: &mut Vec<crate::ir::LossyNote>,
) -> Part {
    use crate::ir::{ImportMaterial, ImportMesh, LossyNote};
    let mut part = Part {
        objects: Vec::new(),
        items: Vec::new(),
        unit: 1e-3,
    };
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut current: Option<(String, Object)> = None;
    // The object's own material, for triangles that do not name one.
    let mut default_material: Option<i32> = None;
    let mut group: Option<(String, Vec<usize>)> = None;
    // A component's part, by its path, made relative to the package.
    let path_of = |tag: &Tag| {
        tag.get("path")
            .map(|p| p.trim_start_matches('/').to_string())
    };
    for tag in tags(xml) {
        match (tag.name.as_str(), tag.closing) {
            ("model", false) => {
                part.unit = match tag.get("unit").unwrap_or("millimeter") {
                    "micron" => 1e-6,
                    "centimeter" => 1e-2,
                    "inch" => 0.0254,
                    "foot" => 0.3048,
                    "meter" => 1.0,
                    _ => 1e-3,
                };
            }
            ("basematerials", false) => {
                group = Some((tag.get("id").unwrap_or("").to_string(), Vec::new()));
                if tag.empty {
                    groups.extend(group.take());
                }
            }
            ("basematerials", true) => groups.extend(group.take()),
            ("base", false) => {
                if let Some((_, entries)) = &mut group {
                    entries.push(materials.len());
                    materials.push(ImportMaterial {
                        name: tag.get("name").unwrap_or("Material").to_string(),
                        base_color: tag
                            .get("displaycolor")
                            .and_then(colour)
                            .unwrap_or([0.8, 0.8, 0.8, 1.0]),
                        ..Default::default()
                    });
                }
            }
            ("object", false) => {
                let id = tag.get("id").unwrap_or("").to_string();
                let name = tag
                    .get("name")
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("Object {id}"));
                default_material = material_of(&groups, tag.get("pid"), tag.get("pindex"));
                current = Some((
                    id,
                    Object {
                        name,
                        ..Default::default()
                    },
                ));
                if tag.empty {
                    part.objects.extend(current.take());
                }
            }
            ("object", true) => part.objects.extend(current.take()),
            ("mesh", false) => {
                if let Some((_, object)) = &mut current {
                    object.mesh = Some(ImportMesh {
                        name: object.name.clone(),
                        positions: Vec::new(),
                        faces: Vec::new(),
                        corner_uvs: None,
                        corner_normals: None,
                        face_materials: None,
                        smoothing_groups: None,
                    });
                }
            }
            ("vertex", false) => {
                if let Some(mesh) = current.as_mut().and_then(|(_, o)| o.mesh.as_mut()) {
                    let at = |k: &str| tag.get(k).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    mesh.positions
                        .push(glam::Vec3::new(at("x"), at("y"), at("z")));
                }
            }
            ("triangle", false) => {
                if let Some(mesh) = current.as_mut().and_then(|(_, o)| o.mesh.as_mut()) {
                    let at = |k: &str| tag.get(k).and_then(|v| v.parse::<u32>().ok());
                    let (Some(a), Some(b), Some(c)) = (at("v1"), at("v2"), at("v3")) else {
                        continue;
                    };
                    let n = mesh.positions.len() as u32;
                    let face = mesh.faces.len();
                    if a >= n || b >= n || c >= n {
                        notes.push(LossyNote::OutOfRangeIndex {
                            mesh: mesh.name.clone(),
                            face,
                        });
                        continue;
                    }
                    let material =
                        material_of(&groups, tag.get("pid"), tag.get("p1")).or(default_material);
                    if let Some(m) = material {
                        mesh.face_materials
                            .get_or_insert_with(|| vec![-1; face])
                            .push(m);
                    } else if let Some(list) = &mut mesh.face_materials {
                        list.push(-1);
                    }
                    mesh.faces.push(vec![a, b, c]);
                }
            }
            ("component", false) => {
                if let Some((_, object)) = &mut current {
                    object.components.push((
                        path_of(&tag),
                        tag.get("objectid").unwrap_or("").to_string(),
                        matrix(tag.get("transform")),
                    ));
                }
            }
            ("item", false) => part.items.push((
                tag.get("objectid").unwrap_or("").to_string(),
                matrix(tag.get("transform")),
            )),
            _ => {}
        }
    }
    part
}

/// Reads a 3MF package into the IR: each object with a mesh a mesh, each
/// build item and each component a node placed as the file says, in the
/// unit the model declares.
///
/// With the production extension's parts too: a slicer's package keeps each
/// object in a model part of its own under `3D/Objects`, and the build names
/// them by path.
pub fn parse_3mf(bytes: &[u8]) -> Result<crate::ir::ImportScene, ThreeMfError> {
    use crate::ir::{ImportNode, ImportNodeData, ImportScene, ImportTransform, SourceMetadata};
    use std::collections::HashMap;
    use std::io::Read;
    let mut zip =
        zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| ThreeMfError::Zip(e.to_string()))?;
    let mut text_of = |name: &str| -> Option<String> {
        let mut entry = zip.by_name(name.trim_start_matches('/')).ok()?;
        let mut text = String::new();
        entry.read_to_string(&mut text).ok()?;
        Some(text)
    };
    // The model part is wherever the package's relationships say; the
    // conventional place when they do not.
    let target = text_of("_rels/.rels")
        .and_then(|rels| {
            tags(&rels)
                .into_iter()
                .find(|t| {
                    t.name == "Relationship"
                        && t.get("Type").is_some_and(|ty| ty.ends_with("/3dmodel"))
                })
                .and_then(|t| t.get("Target").map(str::to_string))
        })
        .unwrap_or_else(|| "3D/3dmodel.model".to_string())
        .trim_start_matches('/')
        .to_string();
    let xml = text_of(&target).ok_or(ThreeMfError::NoModel)?;

    let mut source = SourceMetadata::new("3mf");
    let mut materials = Vec::new();
    let main = read_part(&xml, &mut materials, &mut source.notes);
    source.unit_scale = main.unit;
    let items = main.items.clone();
    let mut parts: HashMap<String, Part> = HashMap::new();
    parts.insert(target.clone(), main);
    // Every part a component names, and every part those name in turn.
    let mut wanted: Vec<String> = Vec::new();
    loop {
        for part in parts.values() {
            for (_, object) in &part.objects {
                for (path, _, _) in &object.components {
                    if let Some(path) = path {
                        if !parts.contains_key(path) && !wanted.contains(path) {
                            wanted.push(path.clone());
                        }
                    }
                }
            }
        }
        let Some(path) = wanted.pop() else { break };
        let part = match text_of(&path) {
            Some(text) => read_part(&text, &mut materials, &mut source.notes),
            None => Part {
                objects: Vec::new(),
                items: Vec::new(),
                unit: source.unit_scale,
            },
        };
        parts.insert(path, part);
        if parts.len() > 4096 {
            break;
        }
    }

    let mut scene = ImportScene::new(source);
    scene.materials = materials;
    // Each object's mesh once, however many places use it.
    let mut meshes: HashMap<(String, String), usize> = HashMap::new();
    struct Placing<'a> {
        parts: &'a mut HashMap<String, Part>,
        meshes: &'a mut HashMap<(String, String), usize>,
        scene: &'a mut ImportScene,
    }
    impl Placing<'_> {
        fn place(
            &mut self,
            path: &str,
            id: &str,
            transform: glam::Mat4,
            parent: Option<usize>,
            depth: usize,
        ) {
            if depth > 32 {
                return;
            }
            let Some(part) = self.parts.get_mut(path) else {
                return;
            };
            let Some(object) = part
                .objects
                .iter_mut()
                .find(|(k, _)| k == id)
                .map(|(_, o)| o)
            else {
                return;
            };
            let key = (path.to_string(), id.to_string());
            let mesh = match self.meshes.get(&key) {
                Some(m) => Some(*m),
                None => object.mesh.take().map(|mesh| {
                    self.scene.meshes.push(mesh);
                    let m = self.scene.meshes.len() - 1;
                    self.meshes.insert(key, m);
                    m
                }),
            };
            let name = object.name.clone();
            let components = object.components.clone();
            let (scale, rotation, translation) = transform.to_scale_rotation_translation();
            let mut node = ImportNode::new(name);
            node.parent = parent;
            node.transform = ImportTransform {
                translation,
                rotation,
                scale,
            };
            if let Some(mesh) = mesh {
                node.data = ImportNodeData::Mesh(mesh);
            }
            self.scene.nodes.push(node);
            let me = self.scene.nodes.len() - 1;
            for (child_path, child, m) in components {
                let child_path = child_path.unwrap_or_else(|| path.to_string());
                self.place(&child_path, &child, m, Some(me), depth + 1);
            }
        }
    }
    let mut placing = Placing {
        parts: &mut parts,
        meshes: &mut meshes,
        scene: &mut scene,
    };
    for (id, transform) in &items {
        placing.place(&target, id, *transform, None, 0);
    }
    Ok(scene)
}

/// The scene material a `pid` and an index into its group name.
fn material_of(
    groups: &[(String, Vec<usize>)],
    pid: Option<&str>,
    index: Option<&str>,
) -> Option<i32> {
    let (_, entries) = groups.iter().find(|(id, _)| Some(id.as_str()) == pid)?;
    let index: usize = index.unwrap_or("0").parse().ok()?;
    entries.get(index).map(|m| *m as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn model_of(bytes: &[u8]) -> String {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).expect("a zip");
        let mut text = String::new();
        zip.by_name("3D/3dmodel.model")
            .expect("the model")
            .read_to_string(&mut text)
            .unwrap();
        assert!(zip.by_name("[Content_Types].xml").is_ok());
        assert!(zip.by_name("_rels/.rels").is_ok());
        text
    }

    #[test]
    fn a_part_is_written_in_its_unit_where_it_stands() {
        let cube = kreate_mesh::primitives::cube(0.02); // 20 mm
        let at = DAffine3::from_translation(kreate_math::DVec3::new(0.1, 0.0, 0.0));
        let bytes = write_3mf(&[("Part <1>".to_string(), &cube, at)], 0.001);
        let model = model_of(&bytes);
        assert!(model.contains("unit=\"millimeter\""));
        assert!(model.contains("name=\"Part &lt;1&gt;\""));
        // 100 mm along X, give or take the half width.
        assert!(
            model.contains("x=\"110\"") && model.contains("x=\"90\""),
            "{model}"
        );
        assert_eq!(model.matches("<triangle ").count(), 12);
        assert_eq!(model.matches("<vertex ").count(), 8);
        assert!(model.contains("<item objectid=\"1\"/>"));
    }

    #[test]
    fn a_mirrored_part_still_faces_out() {
        let cube = kreate_mesh::primitives::cube(1.0);
        let plain = model_of(&write_3mf(&[("a".into(), &cube, DAffine3::IDENTITY)], 1.0));
        let mirrored = model_of(&write_3mf(
            &[(
                "a".into(),
                &cube,
                DAffine3::from_scale(kreate_math::DVec3::new(-1.0, 1.0, 1.0)),
            )],
            1.0,
        ));
        let first = |m: &str| {
            m.split("<triangle ")
                .nth(1)
                .unwrap()
                .split("/>")
                .next()
                .unwrap()
                .to_string()
        };
        assert_ne!(
            first(&plain),
            first(&mirrored),
            "the winding turned with the mirror"
        );
    }

    #[test]
    fn an_unknown_unit_is_millimetres() {
        assert_eq!(unit_name(0.5).0, "millimeter");
        assert_eq!(unit_name(0.0254).0, "inch");
    }

    #[test]
    fn a_written_package_reads_back_in_its_unit_and_place() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let mesh = cube;
        let placed = DAffine3::from_translation(kreate_math::DVec3::new(0.01, 0.0, 0.0));
        let bytes = write_3mf(&[("Box".into(), &mesh, placed)], 1e-3);
        let scene = parse_3mf(&bytes).expect("reads");
        assert_eq!(scene.source.unit_scale, 1e-3);
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].faces.len(), 12, "a cube in triangles");
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.nodes[0].name, "Box");
    }

    #[test]
    fn components_materials_and_a_transform_come_through() {
        let model = r##"<?xml version="1.0" encoding="UTF-8"?>
<model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <!-- a comment with a <tag> in it -->
  <resources>
    <basematerials id="5">
      <base name="Red" displaycolor="#FF0000" />
      <base name="Blue &amp; Grey" displaycolor="#0000FFFF" />
    </basematerials>
    <object id="1" type="model" pid="5" pindex="0" name="Tri">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0" /><vertex x="1" y="0" z="0" /><vertex x="0" y="1" z="0" />
          <vertex x="0" y="0" z="1" />
        </vertices>
        <triangles>
          <triangle v1="0" v2="1" v3="2" />
          <triangle v1="0" v2="1" v3="3" pid="5" p1="1" />
          <triangle v1="0" v2="1" v3="9" />
        </triangles>
      </mesh>
    </object>
    <object id="2" type="model" name="Pair">
      <components>
        <component objectid="1" />
        <component objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0" />
      </components>
    </object>
  </resources>
  <build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 3" /></build>
</model>"##;
        let mut buffer = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buffer));
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("3D/3dmodel.model", options).expect("a part");
            zip.write_all(model.as_bytes()).expect("written");
            zip.finish().expect("finished");
        }
        let scene = parse_3mf(&buffer).expect("reads");
        assert_eq!(scene.source.unit_scale, 1e-2);
        assert_eq!(scene.materials.len(), 2);
        assert_eq!(scene.materials[1].name, "Blue & Grey");
        assert!(scene.materials[0].base_color[0] > 0.99 && scene.materials[0].base_color[2] < 0.01);
        let mesh = &scene.meshes[0];
        assert_eq!(mesh.faces.len(), 2, "the bad triangle is dropped");
        assert_eq!(mesh.face_materials, Some(vec![0, 1]));
        // The pair and its two parts, one moved along X.
        assert_eq!(scene.nodes.len(), 3);
        assert_eq!(scene.nodes[0].transform.translation.z, 3.0);
        assert_eq!(scene.nodes[2].parent, Some(0));
        assert_eq!(scene.nodes[2].transform.translation.x, 5.0);
    }
}
