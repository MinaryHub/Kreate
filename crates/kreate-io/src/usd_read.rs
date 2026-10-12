//! Reading USD: `.usda`, `.usdc`, `.usd` (either) and `.usdz` (a zip of
//! them and their pictures), composed into one stage and read into the IR.
//!
//! # Composition, as far as a scene needs it
//!
//! A USD file is rarely the whole of what it shows: it brings in other
//! layers by sub-layering, references and payloads, and picks among variants.
//! Each is done here, strongest opinion first — the layer's own, then its
//! variants, then what it references, then its payloads, then its
//! sub-layers — merged prim by prim and property by property. A reference
//! inside a referenced file is found from that file's folder, as USD finds
//! it. Inherits and specializes (classes) are not followed, and are noted.
//!
//! # What becomes what
//!
//! Every defined prim becomes a node, its `xformOp`s its placement: meshes,
//! the implicit shapes (cube, sphere, cylinder, cone, capsule, plane),
//! cameras and lights — and every other prim an empty, so a hierarchy keeps
//! its groups. A mesh keeps its polygons as they are, its normals, its UVs
//! (any `texCoord2f` primvar, `st` first), its display colour and its
//! `GeomSubset`s' materials. A material is read through its
//! `UsdPreviewSurface`: colours, metal, roughness, emission, opacity, and
//! the pictures `UsdUVTexture`s give them.
//!
//! What moves, moves: time-sampled `xformOp`s become an animation, at the
//! stage's `timeCodesPerSecond`. A skeleton becomes an armature — its joints'
//! rest and bind transforms, its meshes' joint indices and weights (four a
//! point, the strongest), and its `SkelAnimation` as the joints' channels. A
//! point instancer places its prototypes at each instance, the prototypes'
//! meshes shared. Curves come in as thin tubes; rectangle and disc lights as
//! area lights; a dome light's picture as the world's light.
//!
//! A mesh's blend shapes become its morph targets (shape keys), keyed by the
//! skeleton's animation; a mesh whose points move becomes a run of them,
//! one a time, keyed so it moves between them as USD draws it. A point
//! cloud comes in as small diamonds of its widths.
//!
//! The stage's `upAxis` and `metersPerUnit` are the source's axes and units —
//! Y and centimetres when it does not say, as USD's own fallbacks are.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use kreate_math::{AxisConvention, Quat, Vec2, Vec3};

use crate::ir::{
    ChannelInterpolation, ChannelValues, ImportAnimation, ImportCamera, ImportChannel, ImportImage,
    ImportLight, ImportLightKind, ImportMaterial, ImportMesh, ImportNode, ImportNodeData,
    ImportScene, ImportTransform, LossyNote, SourceMetadata,
};

/// How many layers deep references may go: a loop of references is a
/// file's mistake, and must not be Kreate's hang.
const MAX_ARCS: usize = 32;

/// `def`, `over` or `class`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Specifier {
    #[default]
    Def,
    Over,
    Class,
}

/// A reference or a payload: a layer, and a prim in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Arc {
    /// Empty for a reference into the same layer.
    pub asset: String,
    pub prim_path: Option<String>,
}

/// A value: numbers, flattened (a `point3f[]` is three a point), or text.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Nums(Vec<f64>),
    Texts(Vec<String>),
}

/// An attribute or a relationship.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Property {
    pub name: String,
    /// `point3f[]`, `token`, … or `rel`.
    pub type_name: String,
    pub value: Option<Val>,
    pub interpolation: Option<String>,
    pub element_size: Option<i64>,
    /// A relationship's targets, or an attribute's connections.
    pub targets: Vec<String>,
    /// Its values over time, by time code, in order — empty when it does not
    /// move. [`Property::value`] is its value at rest: the default, or else
    /// the first of these.
    pub samples: Vec<(f64, Val)>,
    /// `uniform`: one value for all time.
    pub uniform: bool,
}

impl Property {
    pub fn nums(&self) -> Option<&[f64]> {
        match &self.value {
            Some(Val::Nums(n)) => Some(n),
            _ => None,
        }
    }

    pub fn texts(&self) -> Option<&[String]> {
        match &self.value {
            Some(Val::Texts(t)) => Some(t),
            _ => None,
        }
    }

    fn text(&self) -> Option<&str> {
        self.texts().and_then(|t| t.first()).map(String::as_str)
    }
}

/// A prim as one layer, or the composed stage, has it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prim {
    pub name: String,
    pub type_name: String,
    pub specifier: Specifier,
    pub properties: Vec<Property>,
    pub children: Vec<Prim>,
    pub references: Vec<Arc>,
    pub payloads: Vec<Arc>,
    pub inherits: Vec<String>,
    pub variant_selection: Vec<(String, String)>,
    pub variant_sets: Vec<(String, Vec<(String, Prim)>)>,
    pub active: Option<bool>,
    pub kind: Option<String>,
    pub api_schemas: Vec<String>,
    /// Where the layer this prim came from is, for the assets it names.
    pub base: String,
}

impl Prim {
    pub fn property(&self, name: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.name == name)
    }

    fn nums(&self, name: &str) -> Option<&[f64]> {
        self.property(name).and_then(Property::nums)
    }

    fn num(&self, name: &str) -> Option<f64> {
        self.nums(name).and_then(|n| n.first().copied())
    }

    fn text(&self, name: &str) -> Option<&str> {
        self.property(name).and_then(Property::text)
    }
}

/// One layer: its stage metadata and its prims.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layer {
    pub up_axis: Option<String>,
    pub meters_per_unit: Option<f64>,
    pub default_prim: Option<String>,
    /// Time codes a second; 24 when the layer does not say.
    pub time_codes_per_second: Option<f64>,
    /// The time codes the stage plays between, when it says.
    pub start_time_code: Option<f64>,
    pub end_time_code: Option<f64>,
    pub sublayers: Vec<String>,
    pub prims: Vec<Prim>,
}

/// Why a stage could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsdError(pub String);

impl std::fmt::Display for UsdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "USD: {}", self.0)
    }
}

impl std::error::Error for UsdError {}

// --- where the files are ------------------------------------------------------

/// Where layers and pictures are read from: the disk, or a `.usdz`'s zip.
pub struct Assets {
    /// A `.usdz`'s files, by their names inside it.
    zip: Option<HashMap<String, Vec<u8>>>,
}

impl Assets {
    /// `name` relative to `base`, as a path with its `.` and `..` resolved.
    fn join(base: &str, name: &str) -> String {
        let name = name.replace('\\', "/");
        if name.contains(":/") {
            return name;
        }
        let (absolute, mut parts): (bool, Vec<&str>) = if name.starts_with('/') {
            (true, Vec::new())
        } else {
            (
                base.starts_with('/'),
                base.split('/').filter(|p| !p.is_empty()).collect(),
            )
        };
        for part in name.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                p => parts.push(p),
            }
        }
        let joined = parts.join("/");
        if absolute {
            format!("/{joined}")
        } else {
            joined
        }
    }

    /// A file's bytes, and the folder it is in (the base its own references
    /// are found from).
    fn read(&self, base: &str, name: &str) -> Option<(Vec<u8>, String)> {
        // A `.usdz` names a file inside itself as `archive.usdz[inner]`.
        let name = name
            .find('[')
            .and_then(|o| name.strip_suffix(']').map(|n| &n[o + 1..]))
            .unwrap_or(name);
        let path = Self::join(base, name);
        let folder = path
            .rsplit_once('/')
            .map(|(d, _)| d.to_string())
            .unwrap_or_default();
        let folder = if path.starts_with('/') && folder.is_empty() {
            "/".to_string()
        } else {
            folder
        };
        if let Some(zip) = &self.zip {
            if let Some(bytes) = zip.get(&path) {
                return Some((bytes.clone(), folder));
            }
            // Pictures are often named from the archive's root.
            let bare = Self::join("", name);
            return zip.get(&bare).map(|b| (b.clone(), folder));
        }
        std::fs::read(&path).ok().map(|b| (b, folder))
    }
}

/// A layer from its bytes, whichever form they are in.
pub fn layer_from_bytes(bytes: &[u8]) -> Result<Layer, UsdError> {
    if bytes.starts_with(b"PXR-USDC") {
        crate::usdc::parse_usdc(bytes).map_err(|e| UsdError(e.to_string()))
    } else {
        let text = String::from_utf8_lossy(bytes);
        crate::usda::parse_usda(&text).map_err(|e| UsdError(e.to_string()))
    }
}

// --- composition --------------------------------------------------------------

fn set_base(prim: &mut Prim, base: &str) {
    if prim.base.is_empty() {
        prim.base = base.to_string();
    }
    for c in &mut prim.children {
        set_base(c, base);
    }
    for (_, variants) in &mut prim.variant_sets {
        for (_, v) in variants {
            set_base(v, base);
        }
    }
}

/// `weak`'s opinions under `strong`'s: what `strong` says stands, what only
/// `weak` says is added.
fn merge(strong: &mut Prim, weak: Prim) {
    if strong.type_name.is_empty() {
        strong.type_name = weak.type_name;
    }
    if weak.specifier == Specifier::Def && strong.specifier == Specifier::Over {
        strong.specifier = Specifier::Def;
    }
    if strong.active.is_none() {
        strong.active = weak.active;
    }
    if strong.kind.is_none() {
        strong.kind = weak.kind;
    }
    strong.api_schemas.extend(weak.api_schemas);
    for w in weak.properties {
        match strong.properties.iter_mut().find(|p| p.name == w.name) {
            Some(s) => {
                if s.value.is_none() {
                    s.value = w.value;
                    if s.interpolation.is_none() {
                        s.interpolation = w.interpolation;
                    }
                }
                if s.targets.is_empty() {
                    s.targets = w.targets;
                }
                if s.samples.is_empty() {
                    s.samples = w.samples;
                }
                if s.type_name.is_empty() {
                    s.type_name = w.type_name;
                }
            }
            None => strong.properties.push(w),
        }
    }
    for c in weak.children {
        match strong.children.iter_mut().find(|s| s.name == c.name) {
            Some(s) => merge(s, c),
            None => strong.children.push(c),
        }
    }
}

/// The prim at `path` among a layer's prims.
fn find<'a>(prims: &'a [Prim], path: &str) -> Option<&'a Prim> {
    let mut parts = path
        .trim_start_matches('/')
        .split('/')
        .filter(|p| !p.is_empty());
    let first = parts.next()?;
    let mut at = prims.iter().find(|p| p.name == first)?;
    for part in parts {
        at = at.children.iter().find(|c| c.name == part)?;
    }
    Some(at)
}

struct Composer<'a> {
    assets: &'a Assets,
    notes: Vec<LossyNote>,
    depth: usize,
    /// Layers read, by where they are: a part referenced fifty times is read
    /// once.
    layers: HashMap<String, Option<Layer>>,
}

impl Composer<'_> {
    /// A layer, read and with its sub-layers merged in, its prims knowing
    /// where they came from.
    fn layer(&mut self, base: &str, name: &str) -> Option<Layer> {
        let key = Assets::join(base, name);
        if let Some(known) = self.layers.get(&key) {
            return known.clone();
        }
        let read = self.assets.read(base, name).and_then(|(bytes, folder)| {
            match layer_from_bytes(&bytes) {
                Ok(mut layer) => {
                    for p in &mut layer.prims {
                        set_base(p, &folder);
                    }
                    // Sub-layers: weaker, in order.
                    for sub in std::mem::take(&mut layer.sublayers) {
                        if self.depth > MAX_ARCS {
                            break;
                        }
                        self.depth += 1;
                        let weak = self.layer(&folder, &sub);
                        self.depth -= 1;
                        match weak {
                            Some(weak) => {
                                for p in weak.prims {
                                    match layer.prims.iter_mut().find(|s| s.name == p.name) {
                                        Some(s) => merge(s, p),
                                        None => layer.prims.push(p),
                                    }
                                }
                                layer.up_axis = layer.up_axis.or(weak.up_axis);
                                layer.meters_per_unit =
                                    layer.meters_per_unit.or(weak.meters_per_unit);
                            }
                            None => self.notes.push(LossyNote::Unsupported {
                                what: format!("the sub-layer `{sub}` was not found"),
                            }),
                        }
                    }
                    Some(layer)
                }
                Err(e) => {
                    self.notes.push(LossyNote::Unsupported {
                        what: format!("`{name}` could not be read: {}", e.0),
                    });
                    None
                }
            }
        });
        self.layers.insert(key, read.clone());
        read
    }

    /// `prim` with its variants, references and payloads composed in, and
    /// its children's. `roots` is its own layer's prims, for references
    /// inside that layer.
    fn compose(&mut self, mut prim: Prim, roots: &[Prim]) -> Prim {
        // Variants: the one selected, or else the first.
        for (set, variants) in std::mem::take(&mut prim.variant_sets) {
            let chosen = prim
                .variant_selection
                .iter()
                .find(|(s, _)| *s == set)
                .map(|(_, v)| v.clone());
            let body = match chosen {
                Some(name) => variants
                    .into_iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, b)| b),
                None => variants.into_iter().next().map(|(_, b)| b),
            };
            if let Some(body) = body {
                let body = self.compose(body, roots);
                merge(&mut prim, body);
            }
        }
        let arcs: Vec<Arc> = std::mem::take(&mut prim.references)
            .into_iter()
            .chain(std::mem::take(&mut prim.payloads))
            .collect();
        for arc in arcs {
            if self.depth > MAX_ARCS {
                self.notes.push(LossyNote::Unsupported {
                    what: format!("references nest too deep under `{}`", prim.name),
                });
                break;
            }
            self.depth += 1;
            let target = if arc.asset.is_empty() {
                arc.prim_path
                    .as_deref()
                    .and_then(|p| find(roots, p))
                    .cloned()
                    .map(|t| self.compose(t, roots))
            } else {
                self.layer(&prim.base.clone(), &arc.asset)
                    .and_then(|layer| {
                        let target = match &arc.prim_path {
                            Some(p) => find(&layer.prims, p),
                            None => layer
                                .default_prim
                                .as_deref()
                                .and_then(|d| find(&layer.prims, d))
                                .or_else(|| {
                                    layer.prims.iter().find(|p| p.specifier == Specifier::Def)
                                }),
                        }
                        .cloned();
                        target.map(|t| self.compose(t, &layer.prims))
                    })
            };
            self.depth -= 1;
            match target {
                Some(t) => merge(&mut prim, t),
                None => self.notes.push(LossyNote::Unsupported {
                    what: format!(
                        "`{}` references `{}{}`, which was not found",
                        prim.name,
                        arc.asset,
                        arc.prim_path.as_deref().unwrap_or("")
                    ),
                }),
            }
        }
        if !prim.inherits.is_empty() {
            self.notes.push(LossyNote::Unsupported {
                what: format!(
                    "`{}` inherits from a class, which is not followed",
                    prim.name
                ),
            });
            prim.inherits.clear();
        }
        let children = std::mem::take(&mut prim.children);
        prim.children = children
            .into_iter()
            .map(|c| self.compose(c, roots))
            .collect();
        prim
    }
}

// --- reading -------------------------------------------------------------------

/// A USD file, composed and read.
pub fn parse_usd_file(path: &Path) -> Result<ImportScene, UsdError> {
    let bytes = std::fs::read(path).map_err(|e| UsdError(format!("cannot read: {e}")))?;
    let folder = path
        .parent()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let folder = if folder.is_empty() {
        std::env::current_dir()
            .map(|d| d.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default()
    } else {
        folder
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if bytes.starts_with(b"PK") {
        // A `.usdz`: its first file is the stage, the rest what it names.
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&bytes))
            .map_err(|e| UsdError(format!("the archive: {e}")))?;
        let mut files = HashMap::new();
        let mut first = None;
        for i in 0..archive.len() {
            use std::io::Read;
            let mut item = archive.by_index(i).map_err(|e| UsdError(e.to_string()))?;
            let Some(inner) = item
                .enclosed_name()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
            else {
                continue;
            };
            let mut data = Vec::new();
            item.by_ref()
                .take(1 << 30)
                .read_to_end(&mut data)
                .map_err(|e| UsdError(e.to_string()))?;
            let lower = inner.to_ascii_lowercase();
            if first.is_none()
                && (lower.ends_with(".usda") || lower.ends_with(".usdc") || lower.ends_with(".usd"))
            {
                first = Some(inner.clone());
            }
            files.insert(inner, data);
        }
        let first = first.ok_or_else(|| UsdError("the archive holds no stage".into()))?;
        let assets = Assets { zip: Some(files) };
        return read_stage(&assets, "", &first);
    }
    let assets = Assets { zip: None };
    read_stage(&assets, &folder, &name)
}

/// A stage's text, read with nothing around it to reference.
pub fn parse_usda_text(text: &str) -> Result<ImportScene, UsdError> {
    let mut files = HashMap::new();
    files.insert("stage.usda".to_string(), text.as_bytes().to_vec());
    read_stage(&Assets { zip: Some(files) }, "", "stage.usda")
}

fn read_stage(assets: &Assets, base: &str, name: &str) -> Result<ImportScene, UsdError> {
    let mut composer = Composer {
        assets,
        notes: Vec::new(),
        depth: 0,
        layers: HashMap::new(),
    };
    let root = match composer.layer(base, name) {
        Some(l) => l,
        None => {
            return Err(composer
                .notes
                .pop()
                .map(|n| match n {
                    LossyNote::Unsupported { what } => UsdError(what),
                    other => UsdError(format!("{other:?}")),
                })
                .unwrap_or_else(|| UsdError("the stage could not be read".into())))
        }
    };
    let roots = root.prims.clone();
    let prims: Vec<Prim> = root
        .prims
        .into_iter()
        .map(|p| composer.compose(p, &roots))
        .collect();

    let mut source = SourceMetadata::new("usd");
    source.axis = match root.up_axis.as_deref() {
        Some("Z") => AxisConvention::KREATE,
        _ => AxisConvention::GLTF,
    };
    source.unit_scale = root.meters_per_unit.unwrap_or(0.01) as f32;
    if root.up_axis.is_none() {
        source.notes.push(LossyNote::AssumedAxisConvention {
            assumed: "Y up (the stage does not say)".into(),
        });
    }
    if root.meters_per_unit.is_none() {
        source.notes.push(LossyNote::AssumedUnitScale {
            assumed: "centimetres (the stage does not say)".into(),
        });
    }
    source.notes.extend(composer.notes);
    let mut reader = Reader {
        scene: ImportScene::new(source),
        assets,
        materials: HashMap::new(),
        images: HashMap::new(),
        colours: HashMap::new(),
        rate: root
            .time_codes_per_second
            .filter(|r| *r > 0.0)
            .unwrap_or(24.0),
        skeletons: HashMap::new(),
        pending: Vec::new(),
        shared: HashMap::new(),
        all: HashMap::new(),
        said: Vec::new(),
    };
    fn index<'a>(prims: &'a [Prim], at: &str, all: &mut HashMap<String, &'a Prim>) {
        for p in prims {
            let path = format!("{at}/{}", p.name);
            index(&p.children, &path, all);
            all.insert(path, p);
        }
    }
    let mut all = HashMap::new();
    index(&prims, "", &mut all);
    reader.all = all;
    for p in &prims {
        reader.prim(p, None, &format!("/{}", p.name));
    }
    reader.bind_skins();
    Ok(reader.scene)
}

/// A skeleton as read: its skin, its node, its joints' names and nodes.
/// A skeleton as read: its skin, its node, its joints' names and nodes, and
/// the path of the animation it plays.
type SkeletonRead = (usize, usize, Vec<String>, Vec<usize>, Option<String>);

struct Reader<'a> {
    scene: ImportScene,
    assets: &'a Assets,
    all: HashMap<String, &'a Prim>,
    /// Materials by their prim's path; pictures by where they are; plain
    /// display colours by their bits.
    materials: HashMap<String, Option<usize>>,
    images: HashMap<String, Option<usize>>,
    colours: HashMap<[u32; 3], usize>,
    /// Time codes a second.
    rate: f64,
    /// The mesh each geometry prim became, by path: an instanced prototype
    /// is read once.
    shared: HashMap<String, Option<usize>>,
    /// Skeletons read, by path: their skin, their node, and their joints'
    /// names and nodes.
    skeletons: HashMap<String, SkeletonRead>,
    /// Skinned meshes to bind once every skeleton is read: the mesh, its
    /// node, its prim's path.
    pending: Vec<(usize, usize, String)>,
    /// What has been said already, so a thousand prims of one unsupported
    /// kind make one note.
    said: Vec<String>,
}

impl<'a> Reader<'a> {
    fn say(&mut self, what: String) {
        if !self.said.contains(&what) {
            self.said.push(what.clone());
            self.scene
                .source
                .notes
                .push(LossyNote::Unsupported { what });
        }
    }

    fn prim(&mut self, prim: &'a Prim, parent: Option<usize>, path: &str) {
        if prim.specifier != Specifier::Def || prim.active == Some(false) {
            return;
        }
        if matches!(prim.text("visibility"), Some("invisible"))
            || matches!(prim.text("purpose"), Some("guide" | "proxy"))
        {
            return;
        }
        let t = prim.type_name.as_str();
        match t {
            "Material" | "Shader" | "NodeGraph" | "GeomSubset" => return,
            "SkelAnimation" => return,
            "BlendShape" => {
                self.say("blend shapes are not read".into());
                return;
            }
            "DomeLight" => {
                self.dome(prim);
                return;
            }
            _ => {}
        }
        let mut node = ImportNode::new(prim.name.clone());
        node.parent = parent;
        node.transform = self.transform(prim);
        node.data = match t {
            // A prim instanced many times is one mesh, shared.
            "Mesh" | "Cube" | "Sphere" | "Cylinder" | "Cone" | "Capsule" | "Plane"
            | "Cylinder_1" | "Capsule_1" | "BasisCurves" | "NurbsCurves" | "HermiteCurves"
            | "Points" => {
                let made = match self.shared.get(path) {
                    Some(m) => *m,
                    None => {
                        let m = match t {
                            "Mesh" => self.mesh(prim, path),
                            "BasisCurves" | "NurbsCurves" | "HermiteCurves" => {
                                self.curves(prim, path)
                            }
                            "Points" => self.point_cloud(prim, path),
                            _ => self.shape(prim, path),
                        };
                        self.shared.insert(path.to_string(), m);
                        m
                    }
                };
                made.map_or(ImportNodeData::Empty, ImportNodeData::Mesh)
            }
            "Camera" => ImportNodeData::Camera(self.camera(prim)),
            "DistantLight" | "SphereLight" | "RectLight" | "DiskLight" | "CylinderLight" => {
                ImportNodeData::Light(self.light(prim))
            }
            _ => ImportNodeData::Empty,
        };
        let at = self.scene.nodes.len();
        self.scene.nodes.push(node);
        self.animate(prim, at);
        if t == "Skeleton" {
            self.skeleton(prim, path, at);
        }
        if let (ImportNodeData::Mesh(m), true) = (
            self.scene.nodes[at].data,
            prim.property("primvars:skel:jointIndices").is_some()
                || prim.property("skel:blendShapes").is_some(),
        ) {
            self.pending.push((m, at, path.to_string()));
        }
        if let (ImportNodeData::Mesh(m), "Mesh") = (self.scene.nodes[at].data, t) {
            if prim.property("points").is_some_and(|p| p.samples.len() > 1)
                && self.scene.vertex_morphs.get(m).is_none_or(Vec::is_empty)
            {
                self.point_cache(prim, m, at);
            }
        }
        if t == "PointInstancer" {
            // Its children are the prototypes, drawn where it says, not
            // where they stand.
            self.instances(prim, at, path);
            return;
        }
        for c in &prim.children {
            self.prim(c, Some(at), &format!("{path}/{}", c.name));
        }
    }

    /// `xformOpOrder`'s operations, in order, as one placement — at rest.
    fn transform(&mut self, prim: &Prim) -> ImportTransform {
        trs(xform_at(prim, None).unwrap_or(kreate_math::glam::DMat4::IDENTITY))
    }

    /// A prim whose placement moves: a channel for each part of it, keyed
    /// at every time any of its operations is.
    fn animate(&mut self, prim: &Prim, node: usize) {
        let Some(order) = prim.property("xformOpOrder").and_then(Property::texts) else {
            return;
        };
        let mut times: Vec<f64> = order
            .iter()
            .filter_map(|op| prim.property(op.trim_start_matches("!invert!")))
            .flat_map(|p| p.samples.iter().map(|(t, _)| *t))
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        if times.len() < 2 {
            return;
        }
        let mut translations = Vec::with_capacity(times.len());
        let mut rotations: Vec<Quat> = Vec::with_capacity(times.len());
        let mut scales = Vec::with_capacity(times.len());
        for t in &times {
            let at = trs(xform_at(prim, Some(*t)).unwrap_or(kreate_math::glam::DMat4::IDENTITY));
            // The same way round as the key before, so the turn between
            // them is the short one.
            let q = match rotations.last() {
                Some(prev) if prev.dot(at.rotation) < 0.0 => -at.rotation,
                _ => at.rotation,
            };
            translations.push(at.translation);
            rotations.push(q);
            scales.push(at.scale);
        }
        let seconds: Vec<f32> = times.iter().map(|t| (*t / self.rate) as f32).collect();
        let animation = self.take();
        for values in [
            ChannelValues::Translation(translations),
            ChannelValues::Rotation(rotations),
            ChannelValues::Scale(scales),
        ] {
            animation.channels.push(ImportChannel {
                node,
                interpolation: ChannelInterpolation::Linear,
                times: seconds.clone(),
                values,
            });
        }
    }

    /// The stage's one animation, made when first needed.
    fn take(&mut self) -> &mut ImportAnimation {
        if self.scene.animations.is_empty() {
            self.scene.animations.push(ImportAnimation {
                name: "Stage".into(),
                channels: Vec::new(),
            });
        }
        self.scene.animations.last_mut().expect("made")
    }

    /// A skeleton: a node for each joint under its own, a skin whose
    /// binds are its `bindTransforms`, and its animation if it has one.
    fn skeleton(&mut self, prim: &'a Prim, path: &str, node: usize) {
        use kreate_math::glam::DMat4;
        // No joints is a skeleton too: one that plays only blend shapes. An
        // empty list reads as no numbers, so it is taken as no names.
        let joints: &[String] = prim
            .property("joints")
            .and_then(Property::texts)
            .unwrap_or(&[]);
        let matrices = |name: &str| -> Option<Vec<DMat4>> {
            let n = prim.nums(name)?;
            (n.len() == joints.len() * 16).then(|| {
                n.as_chunks::<16>()
                    .0
                    .iter()
                    .map(DMat4::from_cols_array)
                    .collect()
            })
        };
        let bind = matrices("bindTransforms");
        let rest = matrices("restTransforms");
        let parent_of = |i: usize| -> Option<usize> {
            let (up, _) = joints[i].rsplit_once('/')?;
            joints.iter().position(|j| j == up)
        };
        let mut nodes: Vec<usize> = Vec::with_capacity(joints.len());
        for i in 0..joints.len() {
            let name = joints[i]
                .rsplit('/')
                .next()
                .unwrap_or(&joints[i])
                .to_string();
            let parent = parent_of(i);
            // At rest: as the skeleton says, or from the binds.
            let local = match (&rest, &bind) {
                (Some(r), _) => r[i],
                (None, Some(b)) => match parent {
                    Some(p) => b[p].inverse() * b[i],
                    None => b[i],
                },
                _ => DMat4::IDENTITY,
            };
            let mut joint = ImportNode::new(name);
            joint.parent = Some(parent.and_then(|p| nodes.get(p).copied()).unwrap_or(node));
            joint.transform = trs(local);
            nodes.push(self.scene.nodes.len());
            self.scene.nodes.push(joint);
        }
        self.scene.skins.push(crate::ir::ImportSkin {
            name: prim.name.clone(),
            joints: nodes.clone(),
            // From the skeleton's space to each joint's at the bind.
            inverse_bind: bind.map(|b| b.iter().map(|m| m.inverse().as_mat4()).collect()),
        });
        let skin = self.scene.skins.len() - 1;
        // Its animation: named on it, or on the skeleton root above it.
        let mut at = path.to_string();
        let mut source = prim
            .property("skel:animationSource")
            .and_then(|r| r.targets.first().cloned());
        while source.is_none() {
            let Some((up, _)) = at.rsplit_once('/') else {
                break;
            };
            at = up.to_string();
            if at.is_empty() {
                break;
            }
            source = self
                .all
                .get(&at)
                .and_then(|p| p.property("skel:animationSource"))
                .and_then(|r| r.targets.first().cloned());
        }
        let anim_path = source.map(|s| absolute(&s, path));
        if let Some(anim) = anim_path.as_ref().and_then(|s| self.all.get(s).copied()) {
            self.skel_animation(anim, joints, &nodes);
        }
        self.skeletons.insert(
            path.to_string(),
            (skin, node, joints.to_vec(), nodes.clone(), anim_path),
        );
    }

    /// A SkelAnimation's turns, moves and sizes, a channel each per joint.
    fn skel_animation(&mut self, anim: &Prim, skeleton: &[String], nodes: &[usize]) {
        let Some(names) = anim.property("joints").and_then(Property::texts) else {
            return;
        };
        let target: Vec<Option<usize>> = names
            .iter()
            .map(|n| {
                skeleton
                    .iter()
                    .position(|s| s == n)
                    .and_then(|i| nodes.get(i).copied())
            })
            .collect();
        for (attr, width) in [("translations", 3), ("rotations", 4), ("scales", 3)] {
            let Some(p) = anim.property(attr) else {
                continue;
            };
            let times: Vec<f64> = if p.samples.is_empty() {
                vec![0.0]
            } else {
                p.samples.iter().map(|(t, _)| *t).collect()
            };
            let per_time: Vec<Vec<f64>> = times
                .iter()
                .filter_map(|t| value_at(p, (!p.samples.is_empty()).then_some(*t)))
                .collect();
            if per_time.len() != times.len() {
                continue;
            }
            let seconds: Vec<f32> = times.iter().map(|t| (*t / self.rate) as f32).collect();
            for (k, node) in target.iter().enumerate() {
                let Some(node) = *node else { continue };
                let values: Option<Vec<&[f64]>> = per_time
                    .iter()
                    .map(|v| v.get(k * width..(k + 1) * width))
                    .collect();
                let Some(values) = values else { continue };
                let values = if attr == "rotations" {
                    let mut out: Vec<Quat> = Vec::with_capacity(values.len());
                    for v in &values {
                        // Real part first, as USD writes them.
                        let q = Quat::from_xyzw(v[1] as f32, v[2] as f32, v[3] as f32, v[0] as f32)
                            .normalize();
                        let q = match out.last() {
                            Some(prev) if prev.dot(q) < 0.0 => -q,
                            _ => q,
                        };
                        out.push(q);
                    }
                    ChannelValues::Rotation(out)
                } else {
                    let v3: Vec<Vec3> = values
                        .iter()
                        .map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
                        .collect();
                    if attr == "scales" {
                        ChannelValues::Scale(v3)
                    } else {
                        ChannelValues::Translation(v3)
                    }
                };
                self.take().channels.push(ImportChannel {
                    node,
                    interpolation: ChannelInterpolation::Linear,
                    times: seconds.clone(),
                    values,
                });
            }
        }
    }

    /// Every skinned mesh bound to its skeleton: hung from it, its points
    /// in the skeleton's space, its joints and weights four to a point —
    /// and its blend shapes as morph targets, keyed by the skeleton's
    /// animation.
    fn bind_skins(&mut self) {
        use kreate_math::glam::DMat4;
        for (mesh, node, path) in std::mem::take(&mut self.pending) {
            let Some(prim) = self.all.get(&path).copied() else {
                continue;
            };
            let shapes = self.blend_shapes(prim, mesh);
            // Its skeleton: named on it or above it, or the first under the
            // skeleton root it is in.
            let mut at = path.clone();
            let mut skel = prim
                .property("skel:skeleton")
                .and_then(|r| r.targets.first().cloned());
            let mut root = None;
            while skel.is_none() {
                let Some((up, _)) = at.rsplit_once('/') else {
                    break;
                };
                at = up.to_string();
                if at.is_empty() {
                    break;
                }
                let p = self.all.get(&at).copied();
                if root.is_none() && p.is_some_and(|p| p.type_name == "SkelRoot") {
                    root = Some(at.clone());
                }
                skel = p
                    .and_then(|p| p.property("skel:skeleton"))
                    .and_then(|r| r.targets.first().cloned());
            }
            let skel = skel.map(|s| absolute(&s, &path)).or_else(|| {
                let root = root?;
                self.skeletons
                    .keys()
                    .filter(|k| k.starts_with(&format!("{root}/")))
                    .min()
                    .cloned()
            });
            let Some((skin, skel_node, names, _, anim)) =
                skel.and_then(|s| self.skeletons.get(&s).cloned())
            else {
                if prim.property("primvars:skel:jointIndices").is_some() {
                    self.say(format!(
                        "`{}`: its skeleton was not found; it comes in at rest",
                        prim.name
                    ));
                }
                continue;
            };
            // Its blend shapes' weights, keyed as the animation says.
            let anim = anim.as_ref().and_then(|a| self.all.get(a).copied());
            if let (Some(shapes), Some(anim)) = (&shapes, anim) {
                self.shape_weights(anim, shapes, node);
            }
            let (Some(indices), Some(weights)) = (
                prim.nums("primvars:skel:jointIndices"),
                prim.nums("primvars:skel:jointWeights"),
            ) else {
                continue;
            };
            let property = prim.property("primvars:skel:jointIndices");
            let element = property.and_then(|p| p.element_size).unwrap_or(1).max(1) as usize;
            let constant = property.and_then(|p| p.interpolation.as_deref()) == Some("constant");
            // A mesh may name its own subset of the joints.
            let remap: Option<Vec<usize>> = prim
                .property("skel:joints")
                .and_then(Property::texts)
                .map(|own| {
                    own.iter()
                        .map(|n| names.iter().position(|s| s == n).unwrap_or(0))
                        .collect()
                });
            let points = self.scene.meshes[mesh].positions.len();
            let mut skin_data = crate::ir::VertexSkin::default();
            for p in 0..points {
                let base = if constant { 0 } else { p * element };
                let mut pairs: Vec<(usize, f32)> = (0..element)
                    .filter_map(|k| {
                        let j = *indices.get(base + k)? as usize;
                        let w = *weights.get(base + k)? as f32;
                        let j = remap.as_ref().map_or(j, |r| r.get(j).copied().unwrap_or(0));
                        Some((j, w))
                    })
                    .filter(|(_, w)| *w > 0.0)
                    .collect();
                pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
                pairs.truncate(4);
                let total: f32 = pairs.iter().map(|(_, w)| w).sum();
                let mut js = [0u16; 4];
                let mut ws = [0f32; 4];
                for (k, (j, w)) in pairs.iter().enumerate() {
                    js[k] = *j as u16;
                    ws[k] = if total > 0.0 { w / total } else { 0.0 };
                }
                skin_data.joints.push(js);
                skin_data.weights.push(ws);
            }
            if element > 4 {
                self.say(
                    "skin weights past four a point were let go, and the rest made to add up"
                        .into(),
                );
            }
            // Its points where the skeleton sees them at the bind — and its
            // shapes' offsets turned the same way.
            if let Some(m) = prim
                .nums("primvars:skel:geomBindTransform")
                .or_else(|| prim.nums("skel:geomBindTransform"))
                .filter(|m| m.len() >= 16)
            {
                let bind = DMat4::from_cols_array(&m[..16].try_into().expect("sixteen")).as_mat4();
                let linear = kreate_math::Mat3::from_mat4(bind);
                let normal = linear.inverse().transpose();
                let data = &mut self.scene.meshes[mesh];
                for q in &mut data.positions {
                    *q = bind.transform_point3(*q);
                }
                if let Some(ns) = &mut data.corner_normals {
                    for n in ns {
                        *n = (normal * *n).normalize_or_zero();
                    }
                }
                if let Some(targets) = self.scene.vertex_morphs.get_mut(mesh) {
                    for (_, offsets) in targets {
                        for d in offsets {
                            *d = linear * *d;
                        }
                    }
                }
            }
            if self.scene.vertex_skins.len() < self.scene.meshes.len() {
                self.scene
                    .vertex_skins
                    .resize(self.scene.meshes.len(), None);
            }
            self.scene.vertex_skins[mesh] = Some(skin_data);
            let n = &mut self.scene.nodes[node];
            n.parent = Some(skel_node);
            n.transform = ImportTransform::default();
            n.skin = Some(skin);
        }
    }

    /// A mesh's blend shapes as its morph targets: each target's offsets,
    /// for the points it names or for all of them. Their names, in the
    /// mesh's order, when it has any.
    fn blend_shapes(&mut self, prim: &Prim, mesh: usize) -> Option<Vec<String>> {
        let names = prim
            .property("skel:blendShapes")
            .and_then(Property::texts)?
            .to_vec();
        let targets = prim.property("skel:blendShapeTargets")?.targets.clone();
        let count = self.scene.meshes[mesh].positions.len();
        let mut morphs = Vec::new();
        for (name, target) in names.iter().zip(&targets) {
            let Some(shape) = self.all.get(&absolute(target, "/")).copied() else {
                continue;
            };
            let Some(offsets) = shape.nums("offsets") else {
                continue;
            };
            let offsets: Vec<Vec3> = offsets
                .as_chunks::<3>()
                .0
                .iter()
                .map(|o| Vec3::new(o[0] as f32, o[1] as f32, o[2] as f32))
                .collect();
            let mut full = vec![Vec3::ZERO; count];
            match shape.nums("pointIndices") {
                Some(at) => {
                    for (i, d) in at.iter().zip(&offsets) {
                        if let Some(slot) = full.get_mut(*i as usize) {
                            *slot = *d;
                        }
                    }
                }
                None => {
                    for (slot, d) in full.iter_mut().zip(&offsets) {
                        *slot = *d;
                    }
                }
            }
            if shape
                .properties
                .iter()
                .any(|p| p.name.starts_with("inbetweens:"))
            {
                self.say("blend shapes' in-between shapes are not read".into());
            }
            morphs.push((name.clone(), full));
        }
        if morphs.is_empty() {
            return None;
        }
        if self.scene.vertex_morphs.len() < self.scene.meshes.len() {
            self.scene
                .vertex_morphs
                .resize(self.scene.meshes.len(), Vec::new());
        }
        self.scene.vertex_morphs[mesh] = morphs;
        Some(names)
    }

    /// The animation's blend-shape weights for one mesh's shapes, in their
    /// order: a morph channel on its node.
    fn shape_weights(&mut self, anim: &Prim, shapes: &[String], node: usize) {
        let Some(keyed) = anim.property("blendShapes").and_then(Property::texts) else {
            return;
        };
        let Some(weights) = anim.property("blendShapeWeights") else {
            return;
        };
        let column: Vec<Option<usize>> = shapes
            .iter()
            .map(|s| keyed.iter().position(|k| k == s))
            .collect();
        let times: Vec<f64> = if weights.samples.is_empty() {
            vec![0.0]
        } else {
            weights.samples.iter().map(|(t, _)| *t).collect()
        };
        let mut values = Vec::with_capacity(times.len() * shapes.len());
        for t in &times {
            let Some(row) = value_at(weights, (!weights.samples.is_empty()).then_some(*t)) else {
                return;
            };
            for c in &column {
                values.push(c.and_then(|c| row.get(c)).copied().unwrap_or(0.0) as f32);
            }
        }
        if self.scene.node_weights.len() <= node {
            self.scene.node_weights.resize(node + 1, Vec::new());
        }
        self.scene.node_weights[node] = values[..shapes.len()].to_vec();
        let name = self.take().name.clone();
        let rate = self.rate;
        self.scene
            .morph_channels
            .push(crate::ir::ImportMorphChannel {
                animation: name,
                node,
                interpolation: ChannelInterpolation::Linear,
                times: times.iter().map(|t| (*t / rate) as f32).collect(),
                values,
            });
    }

    /// A mesh whose points move: a morph target for each time after the
    /// first, keyed one at a time — so between two times the points move in
    /// the straight line between them, as USD draws them.
    fn point_cache(&mut self, prim: &Prim, mesh: usize, node: usize) {
        const MAX_VALUES: usize = 6_000_000;
        let Some(p) = prim.property("points") else {
            return;
        };
        let count = self.scene.meshes[mesh].positions.len();
        if p.samples.len() < 2 {
            return;
        }
        if p.samples.len() * count > MAX_VALUES {
            self.say(format!(
                "`{}`: too many moving points to keep; read at their first time",
                prim.name
            ));
            return;
        }
        let rest = self.scene.meshes[mesh].positions.clone();
        let mut targets = Vec::new();
        for (k, (_, v)) in p.samples.iter().enumerate().skip(1) {
            let Val::Nums(n) = v else {
                return;
            };
            if n.len() != count * 3 {
                self.say(format!(
                    "`{}`: its point count changes over time; read at the first",
                    prim.name
                ));
                return;
            }
            let offsets: Vec<Vec3> = n
                .as_chunks::<3>()
                .0
                .iter()
                .zip(&rest)
                .map(|(q, r)| Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32) - *r)
                .collect();
            targets.push((format!("frame_{k}"), offsets));
        }
        let n = targets.len();
        if self.scene.vertex_morphs.len() < self.scene.meshes.len() {
            self.scene
                .vertex_morphs
                .resize(self.scene.meshes.len(), Vec::new());
        }
        self.scene.vertex_morphs[mesh] = targets;
        // At each time, its own target full and the rest off.
        let mut values = Vec::with_capacity(p.samples.len() * n);
        for k in 0..p.samples.len() {
            for j in 0..n {
                values.push(if k == j + 1 { 1.0 } else { 0.0 });
            }
        }
        if self.scene.node_weights.len() <= node {
            self.scene.node_weights.resize(node + 1, Vec::new());
        }
        self.scene.node_weights[node] = vec![0.0; n];
        let name = self.take().name.clone();
        let times = p
            .samples
            .iter()
            .map(|(t, _)| (*t / self.rate) as f32)
            .collect();
        self.scene
            .morph_channels
            .push(crate::ir::ImportMorphChannel {
                animation: name,
                node,
                interpolation: ChannelInterpolation::Linear,
                times,
                values,
            });
    }

    /// A point cloud as small diamonds, each as wide as its point says.
    fn point_cloud(&mut self, prim: &Prim, path: &str) -> Option<usize> {
        const MAX_POINTS: usize = 200_000;
        let points = prim.nums("points")?;
        let widths = prim.nums("widths").unwrap_or(&[]);
        let count = (points.len() / 3).min(MAX_POINTS);
        if points.len() / 3 > MAX_POINTS {
            self.say(format!(
                "`{}`: the first {MAX_POINTS} of its points were read",
                prim.name
            ));
        }
        let mut positions = Vec::with_capacity(count * 6);
        let mut faces = Vec::with_capacity(count * 8);
        for i in 0..count {
            let c = Vec3::new(
                points[i * 3] as f32,
                points[i * 3 + 1] as f32,
                points[i * 3 + 2] as f32,
            );
            let r = match widths.len() {
                0 => 0.01,
                1 => widths[0] as f32 * 0.5,
                _ => widths.get(i).copied().unwrap_or(widths[0]) as f32 * 0.5,
            };
            let base = positions.len() as u32;
            for d in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
                positions.push(c + d * r);
            }
            for [a, b, e] in [
                [0, 2, 4],
                [2, 1, 4],
                [1, 3, 4],
                [3, 0, 4],
                [2, 0, 5],
                [1, 2, 5],
                [3, 1, 5],
                [0, 3, 5],
            ] {
                faces.push(vec![base + a, base + b, base + e]);
            }
        }
        if faces.is_empty() {
            return None;
        }
        let material = self.binding(prim, path).or_else(|| {
            prim.nums("primvars:displayColor")
                .filter(|c| c.len() >= 3)
                .map(|c| self.colour(c[0] as f32, c[1] as f32, c[2] as f32))
        });
        self.scene.meshes.push(ImportMesh {
            name: prim.name.clone(),
            positions,
            face_materials: material.map(|m| vec![m as i32; faces.len()]),
            faces,
            ..ImportMesh::default()
        });
        Some(self.scene.meshes.len() - 1)
    }

    /// A primvar's values spread to every corner, by its interpolation.
    fn corners(
        &self,
        prim: &Prim,
        name: &str,
        width: usize,
        counts: &[usize],
        indices: &[u32],
        default_interp: &str,
    ) -> Option<Vec<Vec<f64>>> {
        let p = prim.property(name)?;
        let raw = p.nums()?;
        if raw.is_empty() || width == 0 {
            return None;
        }
        let values: Vec<&[f64]> = raw.chunks_exact(width).collect();
        let index: Option<Vec<usize>> = prim
            .nums(&format!("{name}:indices"))
            .map(|i| i.iter().map(|v| *v as usize).collect());
        let pick = |k: usize| -> Option<Vec<f64>> {
            let k = match &index {
                Some(ix) => *ix.get(k)?,
                None => k,
            };
            values.get(k).map(|v| v.to_vec())
        };
        let interp = p.interpolation.as_deref().unwrap_or(default_interp);
        let mut out = Vec::with_capacity(indices.len());
        let mut corner = 0;
        for (f, &n) in counts.iter().enumerate() {
            for c in 0..n {
                let v = match interp {
                    "constant" => pick(0),
                    "uniform" => pick(f),
                    "faceVarying" => pick(corner + c),
                    _ => pick(*indices.get(corner + c)? as usize),
                }?;
                out.push(v);
            }
            corner += n;
        }
        Some(out)
    }

    fn mesh(&mut self, prim: &Prim, path: &str) -> Option<usize> {
        let points = prim.nums("points")?;
        let counts: Vec<usize> = prim
            .nums("faceVertexCounts")?
            .iter()
            .map(|c| *c as usize)
            .collect();
        let indices: Vec<u32> = prim
            .nums("faceVertexIndices")?
            .iter()
            .map(|i| *i as u32)
            .collect();
        if counts.iter().sum::<usize>() != indices.len() {
            self.say(format!(
                "`{}`: its face counts and indices disagree",
                prim.name
            ));
            return None;
        }
        let positions: Vec<Vec3> = points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
            .collect();
        let left = prim.text("orientation") == Some("leftHanded");
        // Faces, turned round for a left-handed mesh, and the corner order
        // that goes with them.
        let mut faces = Vec::with_capacity(counts.len());
        let mut order: Vec<usize> = Vec::with_capacity(indices.len());
        let mut start = 0;
        for &n in &counts {
            let mut corner_ids: Vec<usize> = (start..start + n).collect();
            if left {
                corner_ids.reverse();
            }
            faces.push(corner_ids.iter().map(|&k| indices[k]).collect::<Vec<u32>>());
            order.extend(corner_ids);
            start += n;
        }
        let reorder =
            |v: Vec<Vec<f64>>| -> Vec<Vec<f64>> { order.iter().map(|&k| v[k].clone()).collect() };
        let normals = self
            .corners(prim, "primvars:normals", 3, &counts, &indices, "vertex")
            .or_else(|| self.corners(prim, "normals", 3, &counts, &indices, "vertex"))
            .map(reorder)
            .map(|v| {
                v.iter()
                    .map(|n| Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32))
                    .collect::<Vec<Vec3>>()
            });
        // UVs: `st`, or any texture-coordinate primvar.
        let uv_name = [
            "primvars:st",
            "primvars:st0",
            "primvars:UVMap",
            "primvars:uv",
            "primvars:map1",
        ]
        .into_iter()
        .find(|n| prim.property(n).is_some())
        .map(str::to_string)
        .or_else(|| {
            prim.properties
                .iter()
                .find(|p| {
                    p.type_name.starts_with("texCoord2")
                        && p.name.starts_with("primvars:")
                        && !p.name.ends_with(":indices")
                })
                .map(|p| p.name.clone())
        });
        let uvs = uv_name
            .and_then(|n| self.corners(prim, &n, 2, &counts, &indices, "faceVarying"))
            .map(reorder)
            .map(|v| {
                v.iter()
                    .map(|t| Vec2::new(t[0] as f32, t[1] as f32))
                    .collect::<Vec<Vec2>>()
            });
        // Display colour: a colour per corner when it varies; a material when
        // it is one colour and nothing else is bound.
        let display = prim.property("primvars:displayColor");
        let constant = display
            .map(|d| {
                matches!(d.interpolation.as_deref(), None | Some("constant"))
                    || d.nums().is_some_and(|n| n.len() == 3)
            })
            .unwrap_or(false);
        let colors = if display.is_some() && !constant {
            self.corners(
                prim,
                "primvars:displayColor",
                3,
                &counts,
                &indices,
                "vertex",
            )
            .map(reorder)
            .map(|v| {
                v.iter()
                    .map(|c| {
                        kreate_math::glam::Vec4::new(c[0] as f32, c[1] as f32, c[2] as f32, 1.0)
                    })
                    .collect::<Vec<_>>()
            })
        } else {
            None
        };
        // Materials: the mesh's own binding, its subsets', or an ancestor's.
        let own = self.binding(prim, path);
        let mut face_materials: Option<Vec<i32>> = own.map(|m| vec![m as i32; counts.len()]);
        for subset in prim.children.iter().filter(|c| c.type_name == "GeomSubset") {
            if subset
                .text("familyName")
                .is_some_and(|f| f != "materialBind")
            {
                continue;
            }
            let Some(m) = self.binding(subset, &format!("{path}/{}", subset.name)) else {
                continue;
            };
            let fm = face_materials.get_or_insert_with(|| vec![-1; counts.len()]);
            for f in subset.nums("indices").unwrap_or(&[]) {
                if let Some(slot) = fm.get_mut(*f as usize) {
                    *slot = m as i32;
                }
            }
        }
        if face_materials.is_none() && constant {
            if let Some(c) = display.and_then(Property::nums).filter(|c| c.len() >= 3) {
                let m = self.colour(c[0] as f32, c[1] as f32, c[2] as f32);
                face_materials = Some(vec![m as i32; counts.len()]);
            }
        }
        if prim.text("subdivisionScheme").is_some_and(|s| s != "none") {
            self.say("subdivision surfaces come in as their cage; add a Subdivision modifier to smooth them".into());
        }
        self.scene.meshes.push(ImportMesh {
            name: prim.name.clone(),
            positions,
            faces,
            corner_uvs: uvs,
            corner_normals: normals,
            corner_colors: colors,
            face_materials,
            smoothing_groups: None,
        });
        Some(self.scene.meshes.len() - 1)
    }

    /// An implicit shape, made as a mesh at its size, along its axis.
    fn shape(&mut self, prim: &Prim, path: &str) -> Option<usize> {
        use kreate_mesh::primitives as p;
        let num = |n: &str, d: f64| prim.num(n).unwrap_or(d) as f32;
        let mesh = match prim.type_name.as_str() {
            "Cube" => p::cube(num("size", 2.0)),
            "Sphere" => p::uv_sphere(num("radius", 1.0), 32, 16),
            "Cylinder" | "Cylinder_1" => p::cylinder(num("radius", 1.0), num("height", 2.0), 32),
            "Cone" => p::cone(num("radius", 1.0), num("height", 2.0), 32),
            "Capsule" | "Capsule_1" => {
                self.say("capsules come in as cylinders of their length".into());
                let r = num("radius", 0.5);
                p::cylinder(r, num("height", 1.0) + 2.0 * r, 32)
            }
            "Plane" => {
                let (w, l) = (num("width", 2.0), num("length", 2.0));
                let mut m = p::grid(1.0, 1, 1);
                for q in m.positions_mut() {
                    q.x *= w;
                    q.y *= l;
                }
                m
            }
            _ => return None,
        };
        // The shapes are made along Z; USD's may be along X or Y.
        let turn = match prim.text("axis") {
            Some("X") => Some(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
            Some("Y") => Some(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
            _ => None,
        };
        let positions: Vec<Vec3> = mesh
            .positions()
            .iter()
            .map(|q| turn.map_or(*q, |t| t * *q))
            .collect();
        let faces: Vec<Vec<u32>> = mesh.faces().map(<[u32]>::to_vec).collect();
        let material = self.binding(prim, path).or_else(|| {
            prim.nums("primvars:displayColor")
                .filter(|c| c.len() >= 3)
                .map(|c| self.colour(c[0] as f32, c[1] as f32, c[2] as f32))
        });
        self.scene.meshes.push(ImportMesh {
            name: prim.name.clone(),
            positions,
            face_materials: material.map(|m| vec![m as i32; faces.len()]),
            faces,
            ..ImportMesh::default()
        });
        Some(self.scene.meshes.len() - 1)
    }

    /// A plain material of one colour, shared by everything that colour.
    fn colour(&mut self, r: f32, g: f32, b: f32) -> usize {
        let key = [r.to_bits(), g.to_bits(), b.to_bits()];
        if let Some(m) = self.colours.get(&key) {
            return *m;
        }
        self.scene.materials.push(ImportMaterial {
            name: "displayColor".into(),
            base_color: [r, g, b, 1.0],
            ..ImportMaterial::default()
        });
        let at = self.scene.materials.len() - 1;
        self.colours.insert(key, at);
        at
    }

    /// The material bound to `prim`, or to the nearest ancestor that binds
    /// one.
    fn binding(&mut self, prim: &Prim, path: &str) -> Option<usize> {
        let target = prim
            .property("material:binding")
            .and_then(|b| b.targets.first().cloned())
            .or_else(|| {
                let mut at = path.to_string();
                while let Some((up, _)) = at.rsplit_once('/') {
                    at = up.to_string();
                    if at.is_empty() {
                        break;
                    }
                    if let Some(t) = self
                        .all
                        .get(&at)
                        .and_then(|p| p.property("material:binding"))
                        .and_then(|b| b.targets.first())
                    {
                        return Some(t.clone());
                    }
                }
                None
            })?;
        let target = absolute(&target, path);
        if let Some(m) = self.materials.get(&target) {
            return *m;
        }
        let made = self.material(&target);
        self.materials.insert(target, made);
        made
    }

    /// The prim a connection names, and the output it names on it.
    fn connected(&self, p: &Property, from: &str) -> Option<(&'a Prim, String, String)> {
        let target = absolute(p.targets.first()?, from);
        let (prim_path, output) = target.rsplit_once('.')?;
        Some((
            *self.all.get(prim_path)?,
            prim_path.to_string(),
            output.to_string(),
        ))
    }

    fn material(&mut self, path: &str) -> Option<usize> {
        let material = *self.all.get(path)?;
        // The surface: `outputs:surface`, or the render context's own.
        let surface = [
            "outputs:surface",
            "outputs:mtlx:surface",
            "outputs:ri:surface",
        ]
        .into_iter()
        .filter_map(|o| material.property(o))
        .find_map(|o| self.connected(o, path));
        let mut out = ImportMaterial {
            name: material.name.clone(),
            ..ImportMaterial::default()
        };
        let Some((mut shader, mut shader_path, _)) = surface else {
            self.scene.materials.push(out);
            return Some(self.scene.materials.len() - 1);
        };
        // Through a node graph's output to the shader behind it.
        for _ in 0..4 {
            if shader.type_name != "NodeGraph" {
                break;
            }
            let Some(next) = shader
                .properties
                .iter()
                .find(|p| p.name.starts_with("outputs:"))
                .and_then(|o| self.connected(o, &shader_path))
            else {
                break;
            };
            (shader, shader_path) = (next.0, next.1);
        }
        let id = shader.text("info:id").unwrap_or("");
        if id != "UsdPreviewSurface" {
            self.say(format!(
                "material `{}` uses `{id}`, which is read only for what it shares with UsdPreviewSurface",
                material.name
            ));
        }
        // An input's value, or the picture connected to it.
        let input = |me: &mut Self, name: &str| -> (Option<Vec<f64>>, Option<(usize, String)>) {
            let Some(p) = shader.property(&format!("inputs:{name}")) else {
                return (None, None);
            };
            let texture = me
                .connected(p, &shader_path)
                .and_then(|(tex, tex_path, output)| {
                    let file = tex.property("inputs:file")?.text()?.to_string();
                    if file.is_empty() {
                        return None;
                    }
                    let image = me.image(&tex.base, &file, &tex_path)?;
                    Some((image, output))
                });
            (p.nums().map(<[f64]>::to_vec), texture)
        };
        let (diffuse, diffuse_tex) = input(self, "diffuseColor");
        if let Some(c) = diffuse.filter(|c| c.len() >= 3) {
            out.base_color = [c[0] as f32, c[1] as f32, c[2] as f32, 1.0];
        }
        if let Some((t, _)) = diffuse_tex {
            out.base_color_texture = Some(t);
            out.base_color = [1.0, 1.0, 1.0, out.base_color[3]];
        }
        let (opacity, _) = input(self, "opacity");
        if let Some(o) = opacity.and_then(|o| o.first().copied()) {
            out.base_color[3] = o as f32;
        }
        if let Some(t) = shader.num("inputs:opacityThreshold").filter(|t| *t > 0.0) {
            out.alpha_clip = Some(t as f32);
        }
        let (metallic, metallic_tex) = input(self, "metallic");
        let (roughness, roughness_tex) = input(self, "roughness");
        if let Some(m) = metallic.and_then(|m| m.first().copied()) {
            out.metallic = m as f32;
        }
        if let Some(r) = roughness.and_then(|r| r.first().copied()) {
            out.roughness = r as f32;
        }
        // One picture for both, as glTF packs them — roughness in green,
        // metal in blue — or two made into one.
        match (metallic_tex, roughness_tex) {
            (Some((m, mo)), Some((r, ro)))
                if m == r && mo.ends_with(":b") && ro.ends_with(":g") =>
            {
                out.metallic_roughness_texture = Some(m);
                out.metallic = 1.0;
                out.roughness = 1.0;
            }
            (m, r) if m.is_some() || r.is_some() => {
                if let Some(packed) = self.pack(m, r) {
                    out.metallic_roughness_texture = Some(packed);
                    out.metallic = 1.0;
                    out.roughness = 1.0;
                }
            }
            _ => {}
        }
        let (emissive, emissive_tex) = input(self, "emissiveColor");
        if let Some(e) = emissive.filter(|e| e.len() >= 3) {
            out.emissive = Vec3::new(e[0] as f32, e[1] as f32, e[2] as f32);
        }
        if let Some((t, _)) = emissive_tex {
            out.emissive_texture = Some(t);
            if out.emissive == Vec3::ZERO {
                out.emissive = Vec3::ONE;
            }
        }
        let (_, normal_tex) = input(self, "normal");
        out.normal_texture = normal_tex.map(|(t, _)| t);
        let (_, occlusion_tex) = input(self, "occlusion");
        out.occlusion_texture = occlusion_tex.map(|(t, _)| t);
        if let Some(ior) = shader.num("inputs:ior") {
            out.ior = ior as f32;
        }
        self.scene.materials.push(out);
        Some(self.scene.materials.len() - 1)
    }

    /// Separate metal and roughness pictures made into one, glTF's way:
    /// roughness in green, metal in blue. Each is read from its red.
    fn pack(
        &mut self,
        metal: Option<(usize, String)>,
        rough: Option<(usize, String)>,
    ) -> Option<usize> {
        let channel = |o: &str| match o.rsplit(':').next() {
            Some("g") => 1,
            Some("b") => 2,
            Some("a") => 3,
            _ => 0,
        };
        let size = metal
            .as_ref()
            .or(rough.as_ref())
            .map(|(i, _)| (self.scene.images[*i].width, self.scene.images[*i].height))?;
        for (i, _) in metal.iter().chain(rough.iter()) {
            let img = &self.scene.images[*i];
            if (img.width, img.height) != size {
                self.say("metal and roughness pictures of different sizes: the material keeps its numbers instead".into());
                return None;
            }
        }
        let (w, h) = size;
        let mut pixels = vec![255u8; (w * h * 4) as usize];
        for px in 0..(w * h) as usize {
            if let Some((i, o)) = &rough {
                pixels[px * 4 + 1] = self.scene.images[*i].pixels[px * 4 + channel(o)];
            }
            if let Some((i, o)) = &metal {
                pixels[px * 4 + 2] = self.scene.images[*i].pixels[px * 4 + channel(o)];
            } else {
                pixels[px * 4 + 2] = 0;
            }
        }
        self.scene.images.push(ImportImage {
            name: "metallicRoughness".into(),
            width: w,
            height: h,
            pixels,
        });
        Some(self.scene.images.len() - 1)
    }

    /// A picture, read once however many materials use it.
    fn image(&mut self, base: &str, file: &str, from: &str) -> Option<usize> {
        let key = Assets::join(base, file);
        if let Some(known) = self.images.get(&key) {
            return *known;
        }
        let read =
            self.assets.read(base, file).and_then(|(bytes, _)| {
                match image::load_from_memory(&bytes) {
                    Ok(decoded) => {
                        let rgba = decoded.to_rgba8();
                        let name = PathBuf::from(file)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "texture".into());
                        self.scene.images.push(ImportImage {
                            name,
                            width: rgba.width(),
                            height: rgba.height(),
                            pixels: rgba.into_raw(),
                        });
                        Some(self.scene.images.len() - 1)
                    }
                    Err(_) => None,
                }
            });
        if read.is_none() {
            self.say(format!(
                "`{from}`: the picture `{file}` was not found or not read"
            ));
        }
        self.images.insert(key, read);
        read
    }

    fn camera(&mut self, prim: &Prim) -> usize {
        let focal = prim.num("focalLength").unwrap_or(50.0);
        let vertical = prim.num("verticalAperture").unwrap_or(15.2908);
        let horizontal = prim.num("horizontalAperture").unwrap_or(20.955);
        let clip = prim.nums("clippingRange").unwrap_or(&[1.0, 1_000_000.0]);
        let ortho = prim.text("projection") == Some("orthographic");
        self.scene.cameras.push(ImportCamera {
            name: prim.name.clone(),
            fov_y: (!ortho).then(|| (2.0 * ((vertical * 0.5) / focal.max(1e-6)).atan()) as f32),
            // Apertures are in tenths of a unit.
            ortho_width: ortho.then_some((horizontal * 0.1) as f32),
            clip_near: clip.first().copied().unwrap_or(1.0) as f32,
            clip_far: clip.get(1).map(|f| *f as f32),
        });
        self.scene.cameras.len() - 1
    }

    /// A dome light's picture as the world's light.
    fn dome(&mut self, prim: &Prim) {
        let file = prim
            .text("inputs:texture:file")
            .or_else(|| prim.text("texture:file"))
            .map(str::to_string);
        let Some(file) = file.filter(|f| !f.is_empty()) else {
            self.say("a dome light with no picture: the world keeps its colour".into());
            return;
        };
        if prim
            .text("inputs:texture:format")
            .is_some_and(|f| f != "latlong" && f != "automatic")
        {
            self.say("a dome light's picture is read as a latitude-longitude one".into());
        }
        let get = |n: &str| prim.num(&format!("inputs:{n}")).or_else(|| prim.num(n));
        let strength =
            (get("intensity").unwrap_or(1.0) * 2f64.powf(get("exposure").unwrap_or(0.0))) as f32;
        match self.assets.read(&prim.base, &file) {
            Some((bytes, _)) => {
                self.scene.environment = Some(crate::ir::ImportEnvironment {
                    name: Path::new(&file)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "environment".into()),
                    bytes,
                    strength,
                })
            }
            None => self.say(format!("the dome light's picture `{file}` was not found")),
        }
    }

    /// A point instancer's instances: a node each, placed as it says, with
    /// its prototype under it — the prototype's meshes shared, not copied.
    fn instances(&mut self, prim: &'a Prim, node: usize, path: &str) {
        const MAX_INSTANCES: usize = 20_000;
        let prototypes: Vec<(String, &'a Prim)> = prim
            .property("prototypes")
            .map(|r| r.targets.clone())
            .unwrap_or_default()
            .iter()
            .filter_map(|t| {
                let at = absolute(t, path);
                let p = *self.all.get(&at)?;
                Some((at, p))
            })
            .collect();
        let Some(indices) = prim.nums("protoIndices") else {
            return;
        };
        let positions = prim.nums("positions").unwrap_or(&[]);
        let orientations = prim.nums("orientations").unwrap_or(&[]);
        let scales = prim.nums("scales").unwrap_or(&[]);
        let hidden: Vec<i64> = prim
            .nums("invisibleIds")
            .map(|i| i.iter().map(|v| *v as i64).collect())
            .unwrap_or_default();
        let ids: Option<Vec<i64>> = prim
            .nums("ids")
            .map(|i| i.iter().map(|v| *v as i64).collect());
        if indices.len() > MAX_INSTANCES {
            self.say(format!(
                "`{}`: the first {MAX_INSTANCES} of its {} instances were read",
                prim.name,
                indices.len()
            ));
        }
        for (i, proto) in indices.iter().enumerate().take(MAX_INSTANCES) {
            let id = ids
                .as_ref()
                .and_then(|v| v.get(i).copied())
                .unwrap_or(i as i64);
            if hidden.contains(&id) {
                continue;
            }
            let Some((proto_path, proto)) = prototypes.get(*proto as usize).cloned() else {
                continue;
            };
            let v3 = |a: &[f64], d: f32| {
                a.get(i * 3..i * 3 + 3)
                    .map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
                    .unwrap_or(Vec3::splat(d))
            };
            let rotation = orientations
                .get(i * 4..i * 4 + 4)
                .map(|q| {
                    Quat::from_xyzw(q[1] as f32, q[2] as f32, q[3] as f32, q[0] as f32).normalize()
                })
                .unwrap_or(Quat::IDENTITY);
            let mut instance = ImportNode::new(format!("{}_{i}", prim.name));
            instance.parent = Some(node);
            instance.transform = ImportTransform {
                translation: v3(positions, 0.0),
                rotation,
                scale: if scales.is_empty() {
                    Vec3::ONE
                } else {
                    v3(scales, 1.0)
                },
            };
            let at = self.scene.nodes.len();
            self.scene.nodes.push(instance);
            self.prim(proto, Some(at), &proto_path);
        }
    }

    /// Curves as thin tubes along their points: hair, wires, strokes. A
    /// cubic curve's points are its control points, followed straight.
    fn curves(&mut self, prim: &Prim, path: &str) -> Option<usize> {
        const MAX_SEGMENTS: usize = 400_000;
        const SIDES: usize = 6;
        let points: Vec<Vec3> = prim
            .nums("points")?
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
            .collect();
        let counts: Vec<usize> = prim
            .nums("curveVertexCounts")
            .map(|c| c.iter().map(|n| *n as usize).collect())
            .unwrap_or_else(|| vec![points.len()]);
        if prim.text("type").is_some_and(|t| t != "linear") || prim.type_name != "BasisCurves" {
            self.say("curves are followed through their control points, straight".into());
        }
        let widths = prim.nums("widths").unwrap_or(&[]);
        // A width per point, or one for all; or, not given, a hundredth of
        // the curves' size.
        let fallback = {
            let (lo, hi) = points
                .iter()
                .fold((Vec3::MAX, Vec3::MIN), |(l, h), p| (l.min(*p), h.max(*p)));
            ((hi - lo).length() * 0.005).max(1e-4)
        };
        let width = |k: usize| -> f32 {
            match widths.len() {
                0 => fallback * 2.0,
                1 => widths[0] as f32,
                _ => widths.get(k).copied().unwrap_or(widths[0]) as f32,
            }
        };
        let mut positions = Vec::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        let mut start = 0;
        let mut segments = 0;
        for &n in &counts {
            let run = start..(start + n).min(points.len());
            start += n;
            if run.len() < 2 || segments > MAX_SEGMENTS {
                continue;
            }
            segments += run.len() - 1;
            // A frame carried along the curve, so the tube does not twist.
            let mut normal = Vec3::ZERO;
            let first = positions.len() as u32;
            for k in run.clone() {
                let prev = points[k.max(run.start + 1) - 1];
                let next = points[(k + 1).min(run.end - 1)];
                let along = (next - prev).normalize_or_zero();
                if normal == Vec3::ZERO {
                    normal = along.any_orthonormal_vector();
                }
                normal = (normal - along * normal.dot(along))
                    .normalize_or(along.any_orthonormal_vector());
                let side = along.cross(normal);
                let r = width(k) * 0.5;
                for s in 0..SIDES {
                    let a = std::f32::consts::TAU * s as f32 / SIDES as f32;
                    positions.push(points[k] + (normal * a.cos() + side * a.sin()) * r);
                }
            }
            for k in 0..run.len() as u32 - 1 {
                for s in 0..SIDES as u32 {
                    let a = first + k * SIDES as u32 + s;
                    let b = first + k * SIDES as u32 + (s + 1) % SIDES as u32;
                    faces.push(vec![a, b, b + SIDES as u32, a + SIDES as u32]);
                }
            }
        }
        if segments > MAX_SEGMENTS {
            self.say(format!(
                "`{}`: its curves were read up to {MAX_SEGMENTS} segments",
                prim.name
            ));
        }
        if faces.is_empty() {
            return None;
        }
        let material = self.binding(prim, path).or_else(|| {
            prim.nums("primvars:displayColor")
                .filter(|c| c.len() >= 3)
                .map(|c| self.colour(c[0] as f32, c[1] as f32, c[2] as f32))
        });
        self.scene.meshes.push(ImportMesh {
            name: prim.name.clone(),
            positions,
            face_materials: material.map(|m| vec![m as i32; faces.len()]),
            faces,
            ..ImportMesh::default()
        });
        Some(self.scene.meshes.len() - 1)
    }

    fn light(&mut self, prim: &Prim) -> usize {
        // `inputs:` since USD 21; plain names before.
        let get = |n: &str| prim.num(&format!("inputs:{n}")).or_else(|| prim.num(n));
        let colour = prim
            .nums("inputs:color")
            .or_else(|| prim.nums("color"))
            .filter(|c| c.len() >= 3)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .unwrap_or([1.0; 3]);
        let strength =
            (get("intensity").unwrap_or(1.0) * 2f64.powf(get("exposure").unwrap_or(0.0))) as f32;
        let pi = std::f32::consts::PI;
        let (kind, intensity) = match prim.type_name.as_str() {
            "DistantLight" => (ImportLightKind::Directional, strength),
            // A glowing shape, in nits, as it is.
            "RectLight" => (
                ImportLightKind::Area {
                    width: get("width").unwrap_or(1.0) as f32,
                    height: get("height").unwrap_or(1.0) as f32,
                    round: false,
                },
                strength,
            ),
            "DiskLight" => {
                let d = 2.0 * get("radius").unwrap_or(0.5) as f32;
                (
                    ImportLightKind::Area {
                        width: d,
                        height: d,
                        round: true,
                    },
                    strength,
                )
            }
            t => {
                // A glowing surface's brightness times its area: candela.
                let area = match t {
                    "SphereLight" => {
                        let r = get("radius").unwrap_or(0.5) as f32;
                        pi * r * r
                    }
                    "RectLight" => {
                        (get("width").unwrap_or(1.0) * get("height").unwrap_or(1.0)) as f32
                    }
                    "DiskLight" => {
                        let r = get("radius").unwrap_or(0.5) as f32;
                        pi * r * r
                    }
                    _ => {
                        let r = get("radius").unwrap_or(0.5) as f32;
                        2.0 * r * get("length").unwrap_or(1.0) as f32
                    }
                };
                if t == "CylinderLight" {
                    self.say("cylinder lights come in as point lights of the same strength".into());
                }
                let cone = prim
                    .num("inputs:shaping:cone:angle")
                    .or_else(|| prim.num("shaping:cone:angle"))
                    .filter(|a| *a < 90.0);
                let kind = match cone {
                    Some(angle) => {
                        let outer = (angle as f32).to_radians();
                        let soft = prim.num("inputs:shaping:cone:softness").unwrap_or(0.0) as f32;
                        ImportLightKind::Spot {
                            inner: outer * (1.0 - soft.clamp(0.0, 1.0)),
                            outer,
                        }
                    }
                    None => ImportLightKind::Point,
                };
                (kind, strength * area)
            }
        };
        self.scene.lights.push(ImportLight {
            name: prim.name.clone(),
            kind,
            color: colour,
            intensity,
            range: None,
        });
        self.scene.lights.len() - 1
    }
}

/// A property's value at `time` — between its samples, a straight line;
/// before or after them, the nearest — or its value at rest.
fn value_at(p: &Property, time: Option<f64>) -> Option<Vec<f64>> {
    let rest = || p.nums().map(<[f64]>::to_vec);
    let Some(t) = time else { return rest() };
    let nums = |v: &Val| match v {
        Val::Nums(n) => Some(n.clone()),
        _ => None,
    };
    if p.samples.is_empty() {
        return rest();
    }
    let after = p.samples.iter().position(|(s, _)| *s >= t);
    match after {
        None => nums(&p.samples.last()?.1),
        Some(0) => nums(&p.samples[0].1),
        Some(k) => {
            let (t0, v0) = (&p.samples[k - 1].0, nums(&p.samples[k - 1].1)?);
            let (t1, v1) = (&p.samples[k].0, nums(&p.samples[k].1)?);
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            Some(v0.iter().zip(&v1).map(|(a, b)| a + (b - a) * f).collect())
        }
    }
}

/// A prim's `xformOpOrder` as one matrix, at `time` or at rest; `None` when
/// it has none.
fn xform_at(prim: &Prim, time: Option<f64>) -> Option<kreate_math::glam::DMat4> {
    use kreate_math::glam::{DMat4, DQuat, DVec3};
    let order = prim.property("xformOpOrder").and_then(Property::texts)?;
    let mut m = DMat4::IDENTITY;
    for op in order {
        if op == "!resetXformStack!" {
            m = DMat4::IDENTITY;
            continue;
        }
        let (invert, name) = match op.strip_prefix("!invert!") {
            Some(n) => (true, n),
            None => (false, op.as_str()),
        };
        let Some(v) = prim.property(name).and_then(|p| value_at(p, time)) else {
            continue;
        };
        let kind = name
            .strip_prefix("xformOp:")
            .unwrap_or(name)
            .split(':')
            .next()
            .unwrap_or("");
        let v3 = || DVec3::new(v[0], *v.get(1).unwrap_or(&0.0), *v.get(2).unwrap_or(&0.0));
        let r = |a: f64| a.to_radians();
        let op_m = match kind {
            "translate" if v.len() >= 3 => DMat4::from_translation(v3()),
            "scale" if v.len() >= 3 => DMat4::from_scale(v3()),
            "scale" if v.len() == 1 => DMat4::from_scale(DVec3::splat(v[0])),
            "rotateX" if !v.is_empty() => DMat4::from_rotation_x(r(v[0])),
            "rotateY" if !v.is_empty() => DMat4::from_rotation_y(r(v[0])),
            "rotateZ" if !v.is_empty() => DMat4::from_rotation_z(r(v[0])),
            // `rotateXYZ` turns about X first: the matrix is Z·Y·X. The value
            // is (x, y, z) whatever the order; the name says which is first.
            k if k.starts_with("rotate") && k.len() == 9 && v.len() >= 3 => {
                let mut q = DQuat::IDENTITY;
                for a in k[6..].chars() {
                    let turn = match a {
                        'X' => DQuat::from_rotation_x(r(v[0])),
                        'Y' => DQuat::from_rotation_y(r(v[1])),
                        _ => DQuat::from_rotation_z(r(v[2])),
                    };
                    q = turn * q;
                }
                DMat4::from_quat(q)
            }
            // A quaternion is written real part first.
            "orient" if v.len() >= 4 => {
                DMat4::from_quat(DQuat::from_xyzw(v[1], v[2], v[3], v[0]).normalize())
            }
            // Rows of a row-vector matrix: glam's columns.
            "transform" if v.len() >= 16 => {
                DMat4::from_cols_array(&v[..16].try_into().expect("sixteen"))
            }
            _ => continue,
        };
        let op_m = if invert { op_m.inverse() } else { op_m };
        // Earlier operations in the order are outermost.
        m *= op_m;
    }
    Some(m)
}

fn trs(m: kreate_math::glam::DMat4) -> ImportTransform {
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    ImportTransform {
        translation: translation.as_vec3(),
        rotation: rotation.as_quat().normalize(),
        scale: scale.as_vec3(),
    }
}

/// A target path made absolute: `../Looks/M` from where it was written.
pub(crate) fn absolute(target: &str, from: &str) -> String {
    if target.starts_with('/') {
        return target.to_string();
    }
    let mut parts: Vec<&str> = from.split('/').filter(|p| !p.is_empty()).collect();
    for part in target.split('/') {
        match part {
            ".." => {
                parts.pop();
            }
            "." | "" => {}
            p => parts.push(p),
        }
    }
    format!("/{}", parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    double3 xformOp:translate = (1, 0, 0)
    float3 xformOp:rotateXYZ = (0, 0, 90)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        texCoord2f[] primvars:st = [(0, 0), (1, 1)] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [0, 1, 1, 0]
        normal3f[] normals = [(0, 0, 1)] (
            interpolation = "constant"
        )
        rel material:binding = </World/Looks/Red>
    }

    def Cube "Box" (
        references = </World/Template>
    )
    {
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    class "Template" {}

    def Scope "Hidden" (
        active = false
    )
    {
        def Mesh "Gone" {}
    }

    def Scope "Looks"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Looks/Red/S.outputs:surface>
            def Shader "S"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 0, 0)
                float inputs:roughness = 0.25
                float inputs:metallic = 1
                token outputs:surface
            }
        }
    }

    def SphereLight "Lamp"
    {
        float inputs:intensity = 100
        float inputs:radius = 0.1
    }

    def Camera "Cam"
    {
        float focalLength = 50
        float verticalAperture = 24
    }
}
"#;

    #[test]
    fn a_stage_becomes_nodes_meshes_materials_lights_and_cameras() {
        let scene = parse_usda_text(STAGE).unwrap();
        assert_eq!(scene.source.axis, AxisConvention::KREATE);
        assert_eq!(scene.source.unit_scale, 1.0);
        let names: Vec<&str> = scene.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"Quad") && names.contains(&"Box") && names.contains(&"Lamp"));
        assert!(
            !names.contains(&"Gone"),
            "an inactive prim is not there: {names:?}"
        );
        assert!(!names.contains(&"Template"), "a class is not a thing");

        // Translate, then turn: the quad's corner (1, 0, 0) is at (1, 1, 0).
        let quad = scene.nodes.iter().position(|n| n.name == "Quad").unwrap();
        let world = scene.world_transform(quad).unwrap();
        let p = world.transform_point3(kreate_math::DVec3::X);
        assert!(
            (p - kreate_math::DVec3::new(1.0, 1.0, 0.0)).length() < 1e-6,
            "{p}"
        );

        let mesh = &scene.meshes[0];
        assert_eq!(mesh.faces, vec![vec![0, 1, 2, 3]]);
        assert_eq!(
            mesh.corner_uvs.as_ref().unwrap(),
            &vec![Vec2::ZERO, Vec2::ONE, Vec2::ONE, Vec2::ZERO],
            "indexed UVs"
        );
        assert_eq!(mesh.corner_normals.as_ref().unwrap().len(), 4);
        let m = &scene.materials[mesh.face_materials.as_ref().unwrap()[0] as usize];
        assert_eq!(m.base_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!((m.metallic, m.roughness), (1.0, 0.25));

        let lamp = &scene.lights[0];
        assert!((lamp.intensity - 100.0 * std::f32::consts::PI * 0.01).abs() < 1e-4);
        let cam = &scene.cameras[0];
        assert!((cam.fov_y.unwrap() - 2.0 * (12.0f32 / 50.0).atan()).abs() < 1e-5);
        // The cube, made at USD's size of two.
        let boxed = scene.meshes.iter().find(|m| m.name == "Box").unwrap();
        let extent = boxed
            .positions
            .iter()
            .fold(0.0f32, |a, p| a.max(p.abs().max_element()));
        assert!((extent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn references_and_variants_compose_strongest_first() {
        let dir = std::env::temp_dir().join(format!("kreate-usd-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("parts")).unwrap();
        std::fs::write(
            dir.join("parts").join("wheel.usda"),
            r#"#usda 1.0
(
    defaultPrim = "Wheel"
)
def Xform "Wheel"
{
    def Cylinder "Tyre" { double radius = 3 }
    def Sphere "Hub" (references = @./hub.usda@) {}
}
"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("parts").join("hub.usda"),
            "#usda 1.0\ndef Sphere \"Hub\" { double radius = 0.5 }\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("car.usda"),
            r#"#usda 1.0
(
    upAxis = "Y"
)
def Xform "Car" (
    variants = {
        string colour = "blue"
    }
)
{
    def "Front" (references = @parts/wheel.usda@)
    {
        over "Tyre" { double radius = 4 }
    }
    variantSet "colour" = {
        "red" { def Cube "RedBody" {} }
        "blue" { def Cube "BlueBody" {} }
    }
}
"#,
        )
        .unwrap();
        let scene = parse_usd_file(&dir.join("car.usda")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let names: Vec<&str> = scene.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(
            names.contains(&"BlueBody") && !names.contains(&"RedBody"),
            "{names:?}"
        );
        assert!(
            names.contains(&"Hub"),
            "a reference inside a reference, from its own folder"
        );
        let tyre = scene.meshes.iter().find(|m| m.name == "Tyre").unwrap();
        let r = tyre.positions.iter().fold(0.0f32, |a, p| a.max(p.x.abs()));
        assert!((r - 4.0).abs() < 1e-5, "the local opinion wins: {r}");
        assert_eq!(scene.source.axis, AxisConvention::GLTF);
        assert!(
            (scene.source.unit_scale - 0.01).abs() < 1e-9,
            "centimetres when unsaid"
        );
    }

    const ARM: &str = r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
    timeCodesPerSecond = 10
)
def SkelRoot "Rig"
{
    def Skeleton "Skel" (
        prepend apiSchemas = ["SkelBindingAPI"]
    )
    {
        uniform token[] joints = ["Upper", "Upper/Lower"]
        uniform matrix4d[] bindTransforms = [((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)), ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,1,1))]
        uniform matrix4d[] restTransforms = [((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)), ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,1,1))]
        rel skel:animationSource = </Rig/Skel/Bend>

        def SkelAnimation "Bend"
        {
            uniform token[] joints = ["Upper/Lower"]
            uniform token[] blendShapes = ["Bulge"]
            float[] blendShapeWeights.timeSamples = {
                0: [0],
                10: [1],
            }
            quatf[] rotations.timeSamples = {
                0: [(1, 0, 0, 0)],
                10: [(0.7071068, 0.7071068, 0, 0)],
            }
            float3[] translations = [(0, 0, 1)]
        }
    }
    def Mesh "Skin" (
        prepend apiSchemas = ["SkelBindingAPI"]
    )
    {
        point3f[] points = [(0,0,0), (0.2,0,0), (0,0,2), (0.2,0,2)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 3, 2]
        int[] primvars:skel:jointIndices = [0, 0, 0, 0, 1, 0, 1, 0] (
            elementSize = 2
            interpolation = "vertex"
        )
        float[] primvars:skel:jointWeights = [1, 0, 1, 0, 0.75, 0.25, 0.75, 0.25] (
            elementSize = 2
            interpolation = "vertex"
        )
        rel skel:skeleton = </Rig/Skel>
        uniform token[] skel:blendShapes = ["Bulge"]
        rel skel:blendShapeTargets = </Rig/Skin/Bulge>

        def BlendShape "Bulge"
        {
            uniform vector3f[] offsets = [(0.5, 0, 0)]
            uniform int[] pointIndices = [3]
        }
    }
}
"#;

    #[test]
    fn a_skeleton_binds_its_mesh_and_plays_its_animation() {
        let scene = parse_usda_text(ARM).unwrap();
        assert_eq!(scene.skins.len(), 1);
        let skin = &scene.skins[0];
        let names: Vec<&str> = skin
            .joints
            .iter()
            .map(|j| scene.nodes[*j].name.as_str())
            .collect();
        assert_eq!(names, vec!["Upper", "Lower"]);
        // The lower joint hangs from the upper, a metre up.
        let lower = skin.joints[1];
        assert_eq!(scene.nodes[lower].parent, Some(skin.joints[0]));
        assert!((scene.world_transform(lower).unwrap().translation.z - 1.0).abs() < 1e-6);
        let mesh_node = scene.nodes.iter().position(|n| n.name == "Skin").unwrap();
        assert_eq!(scene.nodes[mesh_node].skin, Some(0));
        let weights = scene.vertex_skins[0].as_ref().unwrap();
        assert_eq!(weights.joints[2], [1, 0, 0, 0]);
        assert_eq!(weights.weights[2], [0.75, 0.25, 0.0, 0.0]);
        // A quarter turn about X over a second.
        let turn = scene.animations[0]
            .channels
            .iter()
            .find(|c| c.node == lower && matches!(c.values, ChannelValues::Rotation(_)))
            .unwrap();
        assert_eq!(turn.times, vec![0.0, 1.0]);
        let ChannelValues::Rotation(q) = &turn.values else {
            unreachable!()
        };
        assert!((q[1].to_axis_angle().1 - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        // And it builds: an armature, the mesh following it.
        let mut doc = kreate_scene::Document::new();
        crate::build::build_into(&scene, &mut doc);
        assert!(doc
            .objects()
            .any(|(_, o)| matches!(o.data, kreate_scene::ObjectData::Armature(_))));
        assert!(doc
            .objects()
            .any(|(_, o)| o.name == "Skin" && o.skin.is_some()));
        // Its blend shape: one point out, keyed from nothing to all of it.
        let (name, offsets) = &scene.vertex_morphs[0][0];
        assert_eq!(name, "Bulge");
        assert_eq!(offsets[3], Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(offsets[0], Vec3::ZERO);
        let keyed = &scene.morph_channels[0];
        assert_eq!(
            (keyed.times.clone(), keyed.values.clone()),
            (vec![0.0, 1.0], vec![0.0, 1.0])
        );
    }

    #[test]
    fn moving_points_play_and_a_point_cloud_shows() {
        let scene = parse_usda_text(
            r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
    timeCodesPerSecond = 1
)
def Mesh "Flag"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        1: [(0, 0, 0), (1, 0, 1), (0, 1, 0)],
        2: [(0, 0, 0), (1, 0, 0), (0, 1, 2)],
    }
}
def Points "Dust"
{
    point3f[] points = [(0, 0, 0), (3, 0, 0)]
    float[] widths = [0.2, 0.4] (
        interpolation = "vertex"
    )
}
"#,
        )
        .unwrap();
        let flag = scene.meshes.iter().position(|m| m.name == "Flag").unwrap();
        let targets = &scene.vertex_morphs[flag];
        assert_eq!(targets.len(), 2, "a target for each time after the first");
        assert_eq!(targets[0].1[1], Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(targets[1].1[2], Vec3::new(0.0, 0.0, 2.0));
        let keyed = &scene.morph_channels[0];
        assert_eq!(keyed.times, vec![0.0, 1.0, 2.0]);
        assert_eq!(keyed.values, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        // They build into shape keys.
        let mut doc = kreate_scene::Document::new();
        crate::build::build_into(&scene, &mut doc);
        let keys = doc
            .objects()
            .find(|(_, o)| o.name == "Flag")
            .map(|(_, o)| o.shape_keys.len())
            .unwrap();
        assert_eq!(keys, 2);
        let dust = scene.meshes.iter().find(|m| m.name == "Dust").unwrap();
        assert_eq!(dust.faces.len(), 16);
        assert!(
            (dust.positions[6].x - 3.2).abs() < 1e-6,
            "{}",
            dust.positions[6]
        );
    }

    #[test]
    fn a_moving_prim_becomes_an_animation_in_seconds() {
        let scene = parse_usda_text(
            r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "Mover"
{
    double3 xformOp:translate.timeSamples = {
        0: (0, 0, 0),
        48: (2, 0, 0),
    }
    float xformOp:rotateZ = 90
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]
}
"#,
        )
        .unwrap();
        let channels = &scene.animations[0].channels;
        let moves = channels
            .iter()
            .find(|c| matches!(c.values, ChannelValues::Translation(_)))
            .unwrap();
        assert_eq!(moves.times, vec![0.0, 2.0], "24 a second when unsaid");
        let ChannelValues::Translation(t) = &moves.values else {
            unreachable!()
        };
        assert_eq!(t[1], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn instancers_domes_area_lights_and_curves_come_in() {
        let dir = std::env::temp_dir().join(format!("kreate-usd-more-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut png = Vec::new();
        image::RgbaImage::from_pixel(4, 2, image::Rgba([200, 220, 255, 255]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(dir.join("sky.png"), &png).unwrap();
        std::fs::write(
            dir.join("more.usda"),
            r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "World"
{
    def PointInstancer "Forest"
    {
        rel prototypes = [</World/Forest/Protos/Tree>, </World/Forest/Protos/Rock>]
        int[] protoIndices = [0, 1, 0]
        point3f[] positions = [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
        quath[] orientations = [(1, 0, 0, 0), (0.7071, 0, 0, 0.7071), (1, 0, 0, 0)]
        int64[] invisibleIds = [2]
        def Scope "Protos"
        {
            def Cube "Tree" { double size = 1 }
            def Sphere "Rock" { double radius = 0.5 }
        }
    }
    def DomeLight "Sky"
    {
        asset inputs:texture:file = @sky.png@
        float inputs:intensity = 2
    }
    def RectLight "Panel"
    {
        float inputs:width = 2
        float inputs:height = 0.5
        float inputs:intensity = 300
    }
    def BasisCurves "Wire"
    {
        uniform token type = "linear"
        int[] curveVertexCounts = [3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        float[] widths = [0.1] (
            interpolation = "constant"
        )
    }
}
"#,
        )
        .unwrap();
        let scene = parse_usd_file(&dir.join("more.usda")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        // Two instances shown (the third hidden), sharing their prototypes'
        // meshes.
        let instances: Vec<usize> = scene
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.name.starts_with("Forest_"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(instances.len(), 2);
        let rock = scene.nodes.iter().position(|n| n.name == "Rock").unwrap();
        let w = scene.world_transform(rock).unwrap();
        assert!((w.translation.x - 5.0).abs() < 1e-6);
        let shown_meshes: std::collections::HashSet<usize> = scene
            .nodes
            .iter()
            .filter_map(|n| match n.data {
                ImportNodeData::Mesh(m) => Some(m),
                _ => None,
            })
            .collect();
        assert_eq!(
            scene.meshes.iter().filter(|m| m.name == "Tree").count(),
            1,
            "read once"
        );
        assert!(shown_meshes.len() >= 2);
        // The sky.
        let env = scene.environment.as_ref().expect("a sky");
        assert_eq!(env.strength, 2.0);
        assert_eq!(&env.bytes[..4], &png[..4]);
        // The panel, as itself.
        let panel = scene.lights.iter().find(|l| l.name == "Panel").unwrap();
        assert_eq!(
            panel.kind,
            ImportLightKind::Area {
                width: 2.0,
                height: 0.5,
                round: false
            }
        );
        assert_eq!(panel.intensity, 300.0);
        // The wire, as a tube round its two segments.
        let wire = scene.meshes.iter().find(|m| m.name == "Wire").unwrap();
        assert_eq!(wire.faces.len(), 2 * 6);
        let r = wire.positions[..6]
            .iter()
            .map(|p| p.length())
            .fold(0.0f32, f32::max);
        assert!((r - 0.05).abs() < 1e-5, "{r}");
        // And they build: the world wears the sky, the panel is an area light.
        let mut doc = kreate_scene::Document::new();
        crate::build::build_into(&scene, &mut doc);
        let world = doc.active_world().and_then(|w| doc.world(w)).unwrap();
        assert!(matches!(
            world.background,
            kreate_scene::Background::Image(_)
        ));
        assert!(doc
            .lights()
            .any(|(_, l)| matches!(l.kind, kreate_scene::LightKind::Area { .. })));
    }

    /// The binary form says what the text does: each stage above, made a
    /// `.usdc` by Pixar's own `usdcat`, reads the same. Skipped where
    /// `usdcat` is not installed.
    #[test]
    fn binary_layers_read_as_their_text_does() {
        let dir = std::env::temp_dir().join(format!("kreate-usdc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (k, text) in [STAGE, ARM].into_iter().enumerate() {
            let a = dir.join(format!("{k}.usda"));
            let c = dir.join(format!("{k}.usdc"));
            std::fs::write(&a, text).unwrap();
            let made = std::process::Command::new("usdcat")
                .arg(&a)
                .arg("-o")
                .arg(&c)
                .output();
            if !made.is_ok_and(|o| o.status.success()) {
                eprintln!("usdcat is not here: skipped");
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
            let from_text = parse_usd_file(&a).unwrap();
            let from_binary = parse_usd_file(&c).unwrap();
            let names =
                |s: &ImportScene| s.nodes.iter().map(|n| n.name.clone()).collect::<Vec<_>>();
            assert_eq!(names(&from_text), names(&from_binary));
            for (x, y) in from_text.meshes.iter().zip(&from_binary.meshes) {
                assert_eq!(x.positions, y.positions);
                assert_eq!(x.faces, y.faces);
                assert_eq!(x.corner_uvs, y.corner_uvs);
            }
            assert_eq!(from_text.vertex_skins, from_binary.vertex_skins);
            assert_eq!(from_text.vertex_morphs, from_binary.vertex_morphs);
            assert_eq!(from_text.morph_channels, from_binary.morph_channels);
            assert_eq!(from_text.skins, from_binary.skins);
            let channels = |s: &ImportScene| {
                s.animations
                    .iter()
                    .flat_map(|a| a.channels.clone())
                    .collect::<Vec<_>>()
            };
            assert_eq!(channels(&from_text), channels(&from_binary));
            assert_eq!(from_text.lights.len(), from_binary.lights.len());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn paths_join_as_files_do() {
        assert_eq!(Assets::join("a/b", "../c.png"), "a/c.png");
        assert_eq!(Assets::join("/x/y", "./z.usda"), "/x/y/z.usda");
        assert_eq!(absolute("../Looks/M", "/World/Mesh"), "/World/Looks/M");
    }
}
