//! The asset browser's library: what can be added, from where, and the
//! fetching of it.
//!
//! Two sources. **Local folders** the user names, read for scene files,
//! environment images and folders of texture maps. **Poly Haven**, whose
//! models, textures and HDRIs are CC0 and whose API is free for any use,
//! asking only that requests say which program they come from and that the
//! program says where its assets come from — the panel's credit line.
//!
//! # Why curl
//!
//! For the reason `contact` gives: an HTTPS stack in Rust brings certificate
//! bundles the dependency policy does not allow, and `curl` is on every
//! machine Kreate runs on, using the system's own certificates.
//!
//! # Everything lands in the cache first
//!
//! A list, a thumbnail and an asset's files are each written under
//! [`cache_root`] and read from there, so a second look costs nothing and a
//! machine that has gone offline still has what it fetched. An asset's files
//! keep the layout its source gives them, which is what lets a glTF find the
//! textures it names beside it.
//!
//! What arrives over the network is not trusted: it is parsed by the same
//! bounded readers as a file somebody opened, and a download is capped.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// What an asset is, which decides what adding it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A scene file: added beside what is there.
    Model,
    /// A set of texture maps: a material, put on the active object.
    Material,
    /// An environment image: the world's light.
    Hdri,
}

impl Kind {
    pub const ALL: [Self; 3] = [Self::Model, Self::Material, Self::Hdri];

    fn polyhaven(self) -> &'static str {
        match self {
            Kind::Model => "models",
            Kind::Material => "textures",
            Kind::Hdri => "hdris",
        }
    }
}

/// Where the assets come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Local,
    PolyHaven,
    /// ambientCG: CC0 materials and HDRIs, each a zip of its maps.
    AmbientCg,
}

impl Source {
    pub const ALL: [Self; 3] = [Self::Local, Self::PolyHaven, Self::AmbientCg];

    /// Whether assets come over the network, and so in a size to choose.
    pub fn online(self) -> bool {
        self != Source::Local
    }
}

/// One asset, as the browser lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Unique within its source: Poly Haven's slug, or a local path.
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub source: Source,
    /// What a search matches beside the name: categories and tags.
    pub tags: Vec<String>,
    /// Who made it, for the tile's hover.
    pub authors: String,
    /// For a local asset, the file or the folder of maps.
    pub path: Option<PathBuf>,
}

impl Entry {
    /// Whether every word of `search` is in the name or a tag.
    pub fn matches(&self, search: &str) -> bool {
        let name = self.name.to_lowercase();
        search.split_whitespace().all(|word| {
            let word = word.to_lowercase();
            name.contains(&word) || self.tags.iter().any(|t| t.contains(&word))
        })
    }
}

/// An asset fetched and on disk, ready to add.
#[derive(Clone, Debug, PartialEq)]
pub enum Ready {
    /// A scene file.
    Model(PathBuf),
    /// Texture maps, each where it is on disk.
    Material(Maps),
    /// An environment image.
    Hdri(PathBuf),
}

/// A material's maps. Only the colour is required.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Maps {
    pub color: Option<PathBuf>,
    pub roughness: Option<PathBuf>,
    /// OpenGL-convention (+Y up), which is Kreate's.
    pub normal: Option<PathBuf>,
    pub metallic: Option<PathBuf>,
}

/// A picture as eight-bit RGBA rows: width, height, pixels.
pub type Picture = (u32, u32, Vec<u8>);

/// What a background job reports.
#[derive(Debug)]
pub enum Event {
    Listed {
        source: Source,
        kind: Kind,
        result: Result<Vec<Entry>, String>,
    },
    /// A thumbnail, as eight-bit RGBA rows; `None` when there is none.
    Thumbnail { id: String, image: Option<Picture> },
    /// Boxed: an entry and a set of paths are several times the size of the
    /// other events, and every event would pay for them.
    Fetched {
        entry: Box<Entry>,
        result: Box<Result<Ready, String>>,
    },
}

/// The sizes Poly Haven offers that are worth fetching by default.
pub const RESOLUTIONS: [&str; 3] = ["1k", "2k", "4k"];

/// Where fetched assets are kept: beside the autosaves, in the cache the
/// platform clears when it needs room.
pub fn cache_root() -> Option<PathBuf> {
    crate::docfile::autosave_root().and_then(|dir| dir.parent().map(|p| p.join("assets")))
}

/// The local folders the browser reads, one a line.
pub fn folders_path() -> Option<PathBuf> {
    crate::prefs::config_dir().map(|dir| dir.join("asset_folders.txt"))
}

pub fn load_folders() -> Vec<PathBuf> {
    folders_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|text| {
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

pub fn save_folders(folders: &[PathBuf]) -> Result<(), std::io::Error> {
    let Some(path) = folders_path() else {
        return Ok(());
    };
    let text: String = folders
        .iter()
        .map(|p| format!("{}\n", p.display()))
        .collect();
    crate::prefs::write_atomic(&path, &text)
}

// --- the network ------------------------------------------------------------

/// What every request says it is: Poly Haven asks for a name it can tell
/// apart from others.
fn user_agent() -> String {
    format!(
        "Kreate/{} (+https://github.com/knoxxr/Kreate)",
        crate::version::VERSION
    )
}

/// The most one download may be. Poly Haven's 4k models are well under it.
const MAX_BYTES: &str = "1073741824";

/// Fetches `url` into `dest`, through a temporary file renamed into place —
/// so an interrupted download is never mistaken for a finished one.
fn download(url: &str, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let partial = dest.with_extension("part");
    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    let out = crate::claude::quiet_command(curl)
        .args(["--silent", "--show-error", "--fail", "--location"])
        .args(["--max-time", "600", "--max-filesize", MAX_BYTES])
        .arg("--user-agent")
        .arg(user_agent())
        .arg("--output")
        .arg(&partial)
        .arg(url)
        .output()
        .map_err(|e| format!("cannot start curl: {e}"))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&partial);
        let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if why.is_empty() {
            format!("could not fetch {url}")
        } else {
            why
        });
    }
    std::fs::rename(&partial, dest).map_err(|e| e.to_string())
}

/// `url` as text, cached at `dest` for a day: an asset list changes when
/// somebody publishes, not by the minute.
fn cached_text(url: &str, dest: &Path) -> Result<String, String> {
    let fresh = std::fs::metadata(dest)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() < 24 * 3600);
    if !fresh {
        if let Err(e) = download(url, dest) {
            // Offline with an old copy: the old copy.
            if !dest.exists() {
                return Err(e);
            }
        }
    }
    std::fs::read_to_string(dest).map_err(|e| e.to_string())
}

// --- Poly Haven ---------------------------------------------------------------

const API: &str = "https://api.polyhaven.com";

/// Poly Haven's list of one kind, as entries, by name.
pub fn parse_polyhaven_list(json: &str, kind: Kind) -> Result<Vec<Entry>, String> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let map = value
        .as_object()
        .ok_or("the list is not an object of assets")?;
    let words = |v: &serde_json::Value| -> Vec<String> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str())
                    .map(str::to_lowercase)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut out: Vec<Entry> = map
        .iter()
        .map(|(id, a)| {
            let mut tags = words(&a["categories"]);
            tags.extend(words(&a["tags"]));
            let authors = a["authors"]
                .as_object()
                .map(|m| m.keys().cloned().collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            Entry {
                id: id.clone(),
                name: a["name"].as_str().unwrap_or(id).to_string(),
                kind,
                source: Source::PolyHaven,
                tags,
                authors,
                path: None,
            }
        })
        .collect();
    out.sort_by_key(|e| e.name.to_lowercase());
    Ok(out)
}

/// What to download for one asset: each file's place under the asset's
/// directory and its URL, and what the result is once they are there.
#[derive(Debug, PartialEq)]
pub struct Plan {
    pub files: Vec<(String, String)>,
    /// The result, with paths relative to the asset's directory.
    pub ready: Ready,
}

/// Reads Poly Haven's file list for one asset at `resolution`, falling back
/// to the nearest smaller one it has.
pub fn polyhaven_plan(json: &str, kind: Kind, resolution: &str) -> Result<Plan, String> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let at = |key: &str| -> Option<&serde_json::Value> {
        let sizes = value.get(key)?;
        let order = RESOLUTIONS
            .iter()
            .position(|r| *r == resolution)
            .unwrap_or(0);
        RESOLUTIONS[..=order]
            .iter()
            .rev()
            .find_map(|r| sizes.get(*r))
            .or_else(|| sizes.as_object()?.values().next())
    };
    let url = |v: &serde_json::Value| v["url"].as_str().map(str::to_string);
    let name_of = |u: &str| u.rsplit('/').next().unwrap_or("file").to_string();
    match kind {
        Kind::Model => {
            let gltf = at("gltf")
                .and_then(|g| g.get("gltf"))
                .ok_or("this model has no glTF")?;
            let main = url(gltf).ok_or("the glTF has no address")?;
            let main_name = name_of(&main);
            let mut files = vec![(main_name.clone(), main)];
            if let Some(include) = gltf["include"].as_object() {
                for (relative, file) in include {
                    // Only paths that stay inside the asset's directory: a
                    // name is the file's to give, and `..` is not one.
                    if relative
                        .split(['/', '\\'])
                        .any(|p| p == ".." || p.is_empty())
                    {
                        continue;
                    }
                    if let Some(u) = url(file) {
                        files.push((relative.clone(), u));
                    }
                }
            }
            Ok(Plan {
                files,
                ready: Ready::Model(PathBuf::from(main_name)),
            })
        }
        Kind::Material => {
            let mut files = Vec::new();
            let mut map = |key: &str| -> Option<PathBuf> {
                let u = url(at(key)?.get("jpg").or_else(|| at(key)?.get("png"))?)?;
                let name = name_of(&u);
                files.push((name.clone(), u));
                Some(PathBuf::from(name))
            };
            let maps = Maps {
                color: map("Diffuse"),
                roughness: map("Rough"),
                normal: map("nor_gl"),
                metallic: map("Metal"),
            };
            if maps.color.is_none() {
                return Err("this texture has no colour map".into());
            }
            Ok(Plan {
                files,
                ready: Ready::Material(maps),
            })
        }
        Kind::Hdri => {
            let file = at("hdri")
                .and_then(|h| h.get("hdr").or_else(|| h.get("exr")))
                .ok_or("this HDRI has no image")?;
            let u = url(file).ok_or("the image has no address")?;
            let name = name_of(&u);
            Ok(Plan {
                files: vec![(name.clone(), u)],
                ready: Ready::Hdri(PathBuf::from(name)),
            })
        }
    }
}

fn under(dir: &Path, ready: Ready) -> Ready {
    let join = |p: Option<PathBuf>| p.map(|p| dir.join(p));
    match ready {
        Ready::Model(p) => Ready::Model(dir.join(p)),
        Ready::Hdri(p) => Ready::Hdri(dir.join(p)),
        Ready::Material(m) => Ready::Material(Maps {
            color: join(m.color),
            roughness: join(m.roughness),
            normal: join(m.normal),
            metallic: join(m.metallic),
        }),
    }
}

// --- local folders ------------------------------------------------------------

/// Extensions read as models, and as environments.
const SCENES: &[&str] = &[
    "gltf", "glb", "fbx", "obj", "stl", "ply", "3mf", "off", "kreate",
];
const ENVIRONMENTS: &[&str] = &["hdr", "exr"];
const IMAGES: &[&str] = &["png", "jpg", "jpeg"];

/// Which map an image file is, by the words texture sites put in its name.
fn map_role(name: &str) -> Option<&'static str> {
    let n = name.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| n.contains(w));
    if has(&["nor_gl", "normal_gl", "normalgl", "_nrm", "normal", "_nor"])
        && !has(&["nor_dx", "normaldx"])
    {
        Some("normal")
    } else if has(&["rough"]) {
        Some("roughness")
    } else if has(&["metal"]) {
        Some("metallic")
    } else if has(&["diff", "albedo", "basecolor", "base_color", "color", "col_"]) {
        Some("color")
    } else {
        None
    }
}

/// The assets in `folders`, to a few levels down: each scene file a model,
/// each HDR or EXR an environment, and each folder holding a colour map a
/// material made of the maps beside it.
pub fn scan_folders(folders: &[PathBuf]) -> Vec<Entry> {
    let mut out = Vec::new();
    fn walk(dir: &Path, depth: usize, out: &mut Vec<Entry>) {
        let Ok(read) = std::fs::read_dir(dir) else {
            return;
        };
        let mut images = Vec::new();
        let mut children: Vec<PathBuf> = Vec::new();
        for item in read.flatten() {
            let path = item.path();
            if path.is_dir() {
                if depth < 4 {
                    children.push(path);
                }
                continue;
            }
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let kind = if SCENES.contains(&ext.as_str()) {
                Some(Kind::Model)
            } else if ENVIRONMENTS.contains(&ext.as_str()) {
                Some(Kind::Hdri)
            } else {
                if IMAGES.contains(&ext.as_str()) {
                    images.push(path.clone());
                }
                None
            };
            if let Some(kind) = kind {
                out.push(Entry {
                    id: path.display().to_string(),
                    name: stem.replace(['_', '-'], " "),
                    kind,
                    source: Source::Local,
                    tags: vec![ext],
                    authors: String::new(),
                    path: Some(path),
                });
            }
        }
        if images.iter().any(|p| {
            map_role(&p.file_name().unwrap_or_default().to_string_lossy()) == Some("color")
        }) {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().replace(['_', '-'], " "))
                .unwrap_or_else(|| "Material".into());
            out.push(Entry {
                id: dir.display().to_string(),
                name,
                kind: Kind::Material,
                source: Source::Local,
                tags: Vec::new(),
                authors: String::new(),
                path: Some(dir.to_path_buf()),
            });
        }
        children.sort();
        for child in children {
            walk(&child, depth + 1, out);
        }
    }
    for folder in folders {
        walk(folder, 0, &mut out);
    }
    out.sort_by_key(|e| e.name.to_lowercase());
    out
}

/// A local material's maps, from the folder it is.
pub fn local_maps(dir: &Path) -> Maps {
    let mut maps = Maps::default();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    files.sort();
    for path in files {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if !IMAGES.contains(&ext.as_str()) {
            continue;
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let slot = match map_role(&name) {
            Some("color") => &mut maps.color,
            Some("roughness") => &mut maps.roughness,
            Some("normal") => &mut maps.normal,
            Some("metallic") => &mut maps.metallic,
            _ => continue,
        };
        if slot.is_none() {
            *slot = Some(path);
        }
    }
    maps
}

/// A small picture of an environment image: its light squeezed into eight
/// bits, so a bright sky reads as a sky and not as white.
fn environment_thumbnail(path: &Path) -> Option<Picture> {
    let image = kreate_io::load_image_file(path, kreate_scene::ColorSpace::Raw).ok()?;
    let (w, h) = (image.width(), image.height());
    // The working-space pixels, as the world's light is uploaded: half floats.
    let halves = crate::environment::pixels_of(&image);
    let at = |x: u32, y: u32, c: usize| -> f32 {
        let i = (y as usize * w as usize + x as usize) * 8 + c * 2;
        halves
            .get(i..i + 2)
            .map_or(0.0, |b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
    };
    // Box-filtered down to fit 256 by 128, never up.
    let scale = (w as f32 / 256.0).max(h as f32 / 128.0).max(1.0);
    let (tw, th) = (
        ((w as f32 / scale).round() as u32).max(1),
        ((h as f32 / scale).round() as u32).max(1),
    );
    let mut out = Vec::with_capacity((tw * th * 4) as usize);
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, y0) = ((tx as f32 * scale) as u32, (ty as f32 * scale) as u32);
            let (x1, y1) = (
                (((tx + 1) as f32 * scale) as u32).clamp(x0 + 1, w),
                (((ty + 1) as f32 * scale) as u32).clamp(y0 + 1, h),
            );
            let mut sum = [0.0f32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    for (c, s) in sum.iter_mut().enumerate() {
                        *s += at(x, y, c);
                    }
                }
            }
            let n = ((x1 - x0) * (y1 - y0)) as f32;
            for s in sum {
                // Reinhard, then the sRGB curve: the picture, not the numbers.
                let v = (s / n).max(0.0);
                out.push((kreate_math::linear_to_srgb(v / (1.0 + v)) * 255.0).round() as u8);
            }
            out.push(255);
        }
    }
    Some((tw, th, out))
}

/// A small picture of an image file: what a local material's tile shows.
fn thumbnail_of(bytes: &[u8]) -> Option<Picture> {
    let image = image::load_from_memory(bytes)
        .ok()?
        // Twice the tile, for a screen of two pixels a point.
        .thumbnail(256, 256)
        .to_rgba8();
    Some((image.width(), image.height(), image.into_raw()))
}

// --- adding ------------------------------------------------------------------

/// A material made of texture maps, as a node graph: the colour into the
/// base colour, a grey map's red into roughness and metallic, and a normal
/// map through a Normal Map node. `None` when the colour map cannot be read.
///
/// The images and the material are added to `doc` directly — outside the
/// history, as an image picked for a material is; what the caller does with
/// the material is the undo step.
pub fn material_from_maps(
    doc: &mut kreate_scene::Document,
    name: &str,
    maps: &Maps,
) -> Option<kreate_scene::document::MaterialId> {
    use kreate_scene::shader::{Link, Node, NodeKind, ShaderGraph};
    use kreate_scene::ColorSpace;
    let load = |path: &Option<PathBuf>, space| {
        path.as_ref()
            .and_then(|p| kreate_io::load_image_file(p, space).ok())
    };
    let colour = load(&maps.color, ColorSpace::Srgb)?;
    let colour = doc.add_image(colour);
    let mut graph =
        ShaderGraph::standard(name, kreate_math::SceneRgb::new(0.8, 0.8, 0.8), 0.0, 0.5);
    let wire = |graph: &mut ShaderGraph, from_node, from_socket, to_node, to_socket| {
        let _ = graph.link(Link {
            from_node,
            from_socket,
            to_node,
            to_socket,
        });
    };
    if let Some(node) = graph.add_node(Node::new("Color", NodeKind::image(Some(colour)))) {
        wire(&mut graph, node, 0, 0, 0);
    }
    // A grey map's number is its red: through a channel split.
    for (path, label, socket) in [
        (&maps.roughness, "Roughness", 2),
        (&maps.metallic, "Metallic", 1),
    ] {
        let Some(image) = load(path, ColorSpace::Raw) else {
            continue;
        };
        let image = doc.add_image(image);
        let (Some(texture), Some(split)) = (
            graph.add_node(Node::new(label, NodeKind::image(Some(image)))),
            graph.add_node(Node::new(label, NodeKind::SeparateRgb)),
        ) else {
            continue;
        };
        wire(&mut graph, texture, 0, split, 0);
        wire(&mut graph, split, 0, 0, socket);
    }
    if let Some(image) = load(&maps.normal, ColorSpace::Raw) {
        let image = doc.add_image(image);
        if let (Some(texture), Some(decode)) = (
            graph.add_node(Node::new("Normal", NodeKind::image(Some(image)))),
            graph.add_node(Node::new(
                "Normal Map",
                NodeKind::NormalMap { strength: 1.0 },
            )),
        ) {
            wire(&mut graph, texture, 0, decode, 0);
            wire(&mut graph, decode, 0, 0, 3);
        }
    }
    graph.arrange();
    Some(doc.add_material(kreate_scene::Material {
        name: name.to_string(),
        graph: Some(graph),
        ..Default::default()
    }))
}

// --- the library ----------------------------------------------------------------

/// A thumbnail asked for: the entry's id, its source, and its path if local.
type Wanted = (String, Source, Option<PathBuf>);

/// Lists, thumbnails and fetches, each on a thread of its own, reporting
/// through [`Library::poll`].
pub struct Library {
    tx: Sender<Event>,
    rx: Receiver<Event>,
    wake: Arc<dyn Fn() + Send + Sync>,
    /// Thumbnails asked for and not yet sent, for the few workers to share.
    thumbs: Arc<Mutex<Vec<Wanted>>>,
    thumb_workers: usize,
}

impl Library {
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            wake: Arc::new(wake),
            thumbs: Arc::new(Mutex::new(Vec::new())),
            thumb_workers: 0,
        }
    }

    fn spawn(&self, job: impl FnOnce() -> Event + Send + 'static) {
        let (tx, wake) = (self.tx.clone(), Arc::clone(&self.wake));
        std::thread::spawn(move || {
            let _ = tx.send(job());
            wake();
        });
    }

    /// Lists one kind from one source.
    pub fn list(&self, source: Source, kind: Kind, folders: Vec<PathBuf>) {
        self.spawn(move || {
            let result = list_now(source, kind, &folders);
            Event::Listed {
                source,
                kind,
                result,
            }
        });
    }

    /// Asks for an entry's thumbnail; it arrives as an [`Event::Thumbnail`].
    pub fn thumbnail(&mut self, entry: &Entry) {
        if let Ok(mut queue) = self.thumbs.lock() {
            queue.push((entry.id.clone(), entry.source, entry.path.clone()));
        }
        // A few workers between them, newest request first: what is on the
        // screen now, not what scrolled past.
        while self.thumb_workers < 4 {
            self.thumb_workers += 1;
            let (queue, tx, wake) = (
                Arc::clone(&self.thumbs),
                self.tx.clone(),
                Arc::clone(&self.wake),
            );
            std::thread::spawn(move || loop {
                let next = queue.lock().ok().and_then(|mut q| q.pop());
                let Some((id, source, path)) = next else {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                };
                let image = match source {
                    Source::PolyHaven => cache_root().and_then(|root| {
                        let dest = root.join("polyhaven").join("thumbs").join(format!("{id}_256.png"));
                        if !dest.exists() {
                            download(
                                &format!(
                                    "https://cdn.polyhaven.com/asset_img/thumbs/{id}.png?width=256&height=256"
                                ),
                                &dest,
                            )
                            .ok()?;
                        }
                        thumbnail_of(&std::fs::read(dest).ok()?)
                    }),
                    Source::AmbientCg => cache_root().and_then(|root| {
                        let dest = root.join("ambientcg").join("thumbs").join(format!("{id}.png"));
                        if !dest.exists() {
                            download(
                                &format!("https://acg-media.struffelproductions.com/file/ambientCG-Web/media/thumbnail/256-PNG/{id}.png"),
                                &dest,
                            )
                            .ok()?;
                        }
                        thumbnail_of(&std::fs::read(dest).ok()?)
                    }),
                    Source::Local => path.and_then(|p| {
                        let file = if p.is_dir() { local_maps(&p).color? } else { p };
                        let ext = file.extension()?.to_string_lossy().to_lowercase();
                        if ENVIRONMENTS.contains(&ext.as_str()) {
                            return environment_thumbnail(&file);
                        }
                        if !IMAGES.contains(&ext.as_str()) {
                            return None;
                        }
                        thumbnail_of(&std::fs::read(file).ok()?)
                    }),
                };
                let _ = tx.send(Event::Thumbnail { id, image });
                wake();
            });
        }
    }

    /// Gets an asset's files onto the disk, for adding.
    pub fn fetch(&self, entry: Entry, resolution: &'static str) {
        self.spawn(move || {
            let result = fetch(&entry, resolution);
            Event::Fetched {
                entry: Box::new(entry),
                result: Box::new(result),
            }
        });
    }

    pub fn poll(&self) -> Vec<Event> {
        self.rx.try_iter().collect()
    }
}

/// One kind from one source, here and now: what [`Library::list`] does on a
/// thread, for a caller that waits — the connector, answering a search.
pub fn list_now(source: Source, kind: Kind, folders: &[PathBuf]) -> Result<Vec<Entry>, String> {
    match source {
        Source::Local => Ok(scan_folders(folders)
            .into_iter()
            .filter(|e| e.kind == kind)
            .collect()),
        Source::PolyHaven => cache_root()
            .ok_or_else(|| "no cache folder".to_string())
            .and_then(|root| {
                cached_text(
                    &format!("{API}/assets?t={}", kind.polyhaven()),
                    &root
                        .join("polyhaven")
                        .join(format!("{}.json", kind.polyhaven())),
                )
            })
            .and_then(|json| parse_polyhaven_list(&json, kind)),
        Source::AmbientCg => ambientcg_list(kind),
    }
}

impl Source {
    /// The name the connector uses.
    pub fn tag(self) -> &'static str {
        match self {
            Source::Local => "local",
            Source::PolyHaven => "polyhaven",
            Source::AmbientCg => "ambientcg",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.tag() == tag)
    }
}

impl Kind {
    /// The name the connector uses.
    pub fn tag(self) -> &'static str {
        match self {
            Kind::Model => "model",
            Kind::Material => "material",
            Kind::Hdri => "hdri",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.tag() == tag)
    }
}

fn fetch(entry: &Entry, resolution: &str) -> Result<Ready, String> {
    match entry.source {
        Source::Local => {
            let path = entry.path.clone().ok_or("the asset has no path")?;
            Ok(match entry.kind {
                Kind::Model => Ready::Model(path),
                Kind::Hdri => Ready::Hdri(path),
                Kind::Material => Ready::Material(local_maps(&path)),
            })
        }
        Source::PolyHaven => {
            // A slug from the list, and still checked: it becomes a folder.
            if entry.id.is_empty()
                || !entry
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(format!("`{}` is not an asset name", entry.id));
            }
            let root = cache_root().ok_or("no cache folder")?;
            let dir = root
                .join("polyhaven")
                .join(entry.kind.polyhaven())
                .join(&entry.id)
                .join(resolution);
            let json = cached_text(
                &format!("{API}/files/{}", entry.id),
                &root
                    .join("polyhaven")
                    .join("files")
                    .join(format!("{}.json", entry.id)),
            )?;
            let plan = polyhaven_plan(&json, entry.kind, resolution)?;
            for (relative, url) in &plan.files {
                let dest = dir.join(relative);
                if !dest.exists() {
                    download(url, &dest)?;
                }
            }
            Ok(under(&dir, plan.ready))
        }
        Source::AmbientCg => ambientcg_fetch(entry, resolution),
    }
}

// --- ambientCG --------------------------------------------------------------

const ACG_API: &str = "https://ambientcg.com/api/v2/full_json";

/// ambientCG's name for a kind; it has no models.
fn ambientcg_type(kind: Kind) -> Option<&'static str> {
    match kind {
        Kind::Material => Some("Material"),
        Kind::Hdri => Some("HDRI"),
        Kind::Model => None,
    }
}

/// One page of ambientCG's list as entries, and whether another follows.
pub fn parse_ambientcg_page(json: &str, kind: Kind) -> Result<(Vec<Entry>, bool), String> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let assets = value["foundAssets"]
        .as_array()
        .ok_or("the list has no assets")?;
    let entries = assets
        .iter()
        .filter_map(|a| {
            let id = a["assetId"].as_str()?.to_string();
            let mut tags: Vec<String> = a["tags"]
                .as_array()
                .map(|t| {
                    t.iter()
                        .filter_map(|s| s.as_str())
                        .map(str::to_lowercase)
                        .collect()
                })
                .unwrap_or_default();
            if let Some(category) = a["category"].as_str() {
                tags.push(category.to_lowercase());
            }
            Some(Entry {
                name: a["displayName"].as_str().unwrap_or(&id).to_string(),
                id,
                kind,
                source: Source::AmbientCg,
                tags,
                authors: "ambientCG".into(),
                path: None,
            })
        })
        .collect::<Vec<_>>();
    let more = !value["nextPageHttp"].is_null() && !entries.is_empty();
    Ok((entries, more))
}

/// Every ambientCG asset of a kind, each page cached for a day.
///
/// The first page says how many there are; the rest are asked for at once
/// rather than one after another — ambientCG takes seconds a page, and nine
/// in a row was half a minute before the first picture.
fn ambientcg_list(kind: Kind) -> Result<Vec<Entry>, String> {
    let Some(ty) = ambientcg_type(kind) else {
        return Ok(Vec::new());
    };
    let root = cache_root().ok_or("no cache folder")?;
    let page = move |offset: usize| -> Result<String, String> {
        cached_text(
            &format!("{ACG_API}?type={ty}&limit=250&offset={offset}&sort=popular&include=tagData"),
            &root.join("ambientcg").join(format!("{ty}_{offset}.json")),
        )
    };
    let first = page(0)?;
    let total = serde_json::from_str::<serde_json::Value>(&first)
        .ok()
        .and_then(|v| v["numberOfResults"].as_u64())
        .unwrap_or(0) as usize;
    let (mut out, _) = parse_ambientcg_page(&first, kind)?;
    // Bounded: a count from the network sizes no more than forty pages.
    let offsets: Vec<usize> = (1..total.div_ceil(250).min(40)).map(|n| n * 250).collect();
    let pages: Vec<std::thread::JoinHandle<Result<String, String>>> = offsets
        .into_iter()
        .map(|offset| {
            let page = page.clone();
            std::thread::spawn(move || page(offset))
        })
        .collect();
    for handle in pages {
        let json = handle
            .join()
            .map_err(|_| "a page of the list could not be read".to_string())??;
        out.extend(parse_ambientcg_page(&json, kind)?.0);
    }
    Ok(out)
}

/// Downloads an ambientCG asset's zip and takes out what Kreate reads: a
/// material's maps, or an HDRI's EXR.
fn ambientcg_fetch(entry: &Entry, resolution: &str) -> Result<Ready, String> {
    if entry.id.is_empty()
        || !entry
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(format!("`{}` is not an asset name", entry.id));
    }
    let size = resolution.to_ascii_uppercase();
    let file = match entry.kind {
        Kind::Material => format!("{}_{size}-JPG", entry.id),
        Kind::Hdri => format!("{}_{size}", entry.id),
        Kind::Model => return Err("ambientCG has no models".into()),
    };
    let root = cache_root().ok_or("no cache folder")?;
    let dir = root.join("ambientcg").join(&file);
    let zip = dir.join(format!("{file}.zip"));
    if !zip.exists() {
        download(&format!("https://ambientcg.com/get?file={file}.zip"), &zip)?;
    }
    let wanted = |name: &str| {
        let lower = name.to_ascii_lowercase();
        match entry.kind {
            Kind::Hdri => lower.ends_with(".exr") || lower.ends_with(".hdr"),
            _ => lower.ends_with(".jpg") || lower.ends_with(".png"),
        }
    };
    let extracted = unzip_flat(&zip, &dir, wanted)?;
    match entry.kind {
        Kind::Hdri => extracted
            .into_iter()
            .next()
            .map(Ready::Hdri)
            .ok_or_else(|| "the download has no HDR image".into()),
        _ => {
            let maps = local_maps(&dir);
            if maps.color.is_none() {
                return Err("the download has no colour map".into());
            }
            Ok(Ready::Material(maps))
        }
    }
}

/// Takes the files `wanted` from a zip into `dir`, by their names alone — a
/// path inside a downloaded zip is not one to write to — and returns where
/// they went. Already-extracted files are left as they are.
fn unzip_flat(
    zip: &Path,
    dir: &Path,
    wanted: impl Fn(&str) -> bool,
) -> Result<Vec<PathBuf>, String> {
    use std::io::Read;
    let file = std::fs::File::open(zip).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for i in 0..archive.len() {
        let mut item = archive.by_index(i).map_err(|e| e.to_string())?;
        let Some(name) = item
            .enclosed_name()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        else {
            continue;
        };
        if !wanted(&name) {
            continue;
        }
        let dest = dir.join(&name);
        if !dest.exists() {
            // Capped as a download is: a zip says how big its parts are, and
            // a part that claims a little and unpacks to a lot is not read on.
            let mut bytes = Vec::new();
            item.by_ref()
                .take(1 << 30)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            std::fs::write(&dest, bytes).map_err(|e| e.to_string())?;
        }
        out.push(dest);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIST: &str = r#"{
        "wooden_floor": {"name": "Wooden Floor", "categories": ["floor", "wood"], "tags": ["Planks"], "authors": {"Rob": "All"}},
        "aerial_rocks": {"name": "Aerial Rocks", "categories": ["rock"], "tags": [], "authors": {"Ana": "All", "Bo": "Photos"}}
    }"#;

    #[test]
    fn an_ambientcg_page_reads_and_says_whether_another_follows() {
        let page = r#"{"foundAssets": [
            {"assetId": "Wood049", "displayName": "Wood 049", "tags": ["Planks", "floor"], "category": "Wood"},
            {"assetId": "Rock030", "displayName": "Rock 030", "tags": [], "category": "Rock"}
        ], "nextPageHttp": "https://ambientcg.com/api/v2/full_json?offset=250"}"#;
        let (entries, more) = parse_ambientcg_page(page, Kind::Material).expect("parses");
        assert!(more);
        assert_eq!(entries[0].id, "Wood049");
        assert!(entries[0].matches("wood planks"));
        assert_eq!(entries[1].source, Source::AmbientCg);
        let last = r#"{"foundAssets": [], "nextPageHttp": null}"#;
        assert!(!parse_ambientcg_page(last, Kind::Hdri).expect("parses").1);
    }

    #[test]
    fn a_zip_gives_up_its_files_by_name_and_no_further() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("kreate-unzip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let zip_path = dir.join("pack.zip");
        {
            let mut zip = zip::ZipWriter::new(std::fs::File::create(&zip_path).expect("made"));
            let options = zip::write::SimpleFileOptions::default();
            for name in [
                "Wood_Color.jpg",
                "deep/Wood_NormalGL.jpg",
                "../escape.jpg",
                "Wood.blend",
            ] {
                zip.start_file(name, options).expect("a part");
                zip.write_all(b"x").expect("written");
            }
            zip.finish().expect("finished");
        }
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("made");
        let got = unzip_flat(&zip_path, &out, |n| n.ends_with(".jpg")).expect("unzipped");
        let names: Vec<String> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["Wood_Color.jpg", "Wood_NormalGL.jpg"]);
        assert!(!dir.join("escape.jpg").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_list_reads_by_name_and_searches_its_tags() {
        let entries = parse_polyhaven_list(LIST, Kind::Material).expect("parses");
        assert_eq!(entries[0].name, "Aerial Rocks");
        assert_eq!(entries[0].authors, "Ana, Bo");
        assert!(entries[1].matches("planks"));
        assert!(entries[1].matches("wood FLOOR"));
        assert!(!entries[1].matches("rock"));
    }

    #[test]
    fn a_models_plan_keeps_its_textures_beside_it_and_inside_its_folder() {
        let files = r#"{"gltf": {"1k": {"gltf": {"url": "https://dl/x/Chair.gltf", "include": {
            "Chair.bin": {"url": "https://dl/x/Chair.bin"},
            "textures/chair_diff_1k.jpg": {"url": "https://dl/x/chair_diff_1k.jpg"},
            "../../escape.txt": {"url": "https://dl/x/escape.txt"}
        }}}}}"#;
        let plan = polyhaven_plan(files, Kind::Model, "2k").expect("plans");
        assert_eq!(plan.ready, Ready::Model(PathBuf::from("Chair.gltf")));
        let names: Vec<&str> = plan.files.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"textures/chair_diff_1k.jpg"));
        assert!(!names.iter().any(|n| n.contains("..")), "{names:?}");
    }

    #[test]
    fn a_textures_plan_takes_its_maps_at_the_size_asked_or_smaller() {
        let files = r#"{
            "Diffuse": {"1k": {"jpg": {"url": "https://dl/a_diff_1k.jpg"}}, "2k": {"jpg": {"url": "https://dl/a_diff_2k.jpg"}}},
            "Rough": {"1k": {"jpg": {"url": "https://dl/a_rough_1k.jpg"}}},
            "nor_gl": {"2k": {"jpg": {"url": "https://dl/a_nor_gl_2k.jpg"}}}
        }"#;
        let plan = polyhaven_plan(files, Kind::Material, "2k").expect("plans");
        let Ready::Material(maps) = plan.ready else {
            panic!("a material")
        };
        assert_eq!(maps.color, Some(PathBuf::from("a_diff_2k.jpg")));
        assert_eq!(maps.roughness, Some(PathBuf::from("a_rough_1k.jpg")));
        assert_eq!(maps.normal, Some(PathBuf::from("a_nor_gl_2k.jpg")));
        assert_eq!(maps.metallic, None);
        assert!(polyhaven_plan("{}", Kind::Material, "1k").is_err());
    }

    #[test]
    fn an_hdris_plan_prefers_the_hdr() {
        let files = r#"{"hdri": {"1k": {"exr": {"url": "https://dl/s.exr"}, "hdr": {"url": "https://dl/s.hdr"}}}}"#;
        let plan = polyhaven_plan(files, Kind::Hdri, "1k").expect("plans");
        assert_eq!(plan.ready, Ready::Hdri(PathBuf::from("s.hdr")));
    }

    #[test]
    fn maps_make_a_material_whose_graph_reads_each_and_compiles() {
        let dir = std::env::temp_dir().join(format!("kreate-maps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let write = |name: &str| {
            let path = dir.join(name);
            image::RgbaImage::from_pixel(4, 4, image::Rgba([200, 100, 50, 255]))
                .save(&path)
                .expect("written");
            path
        };
        let maps = Maps {
            color: Some(write("wood_diff.png")),
            roughness: Some(write("wood_rough.png")),
            normal: Some(write("wood_nor_gl.png")),
            metallic: None,
        };
        let mut doc = kreate_scene::Document::new();
        let material = material_from_maps(&mut doc, "Wood", &maps).expect("made");
        let graph = doc
            .material(material)
            .and_then(|m| m.graph.clone())
            .expect("a graph");
        // Colour, roughness and normal each reach the surface.
        for socket in [0, 2, 3] {
            assert!(graph.incoming(0, socket).is_some(), "socket {socket}");
        }
        assert!(graph.incoming(0, 1).is_none(), "no metal map");
        let compiled = kreate_shader::compile_with(&graph, |id| {
            doc.image(id)
                .map(|i| i.color_space)
                .unwrap_or(kreate_scene::ColorSpace::Srgb)
        })
        .expect("compiles");
        assert_eq!(compiled.images.len(), 3);
        // The colour is a colour and the rest are numbers.
        assert_eq!(
            compiled.image_spaces,
            vec![
                kreate_scene::ColorSpace::Srgb,
                kreate_scene::ColorSpace::Raw,
                kreate_scene::ColorSpace::Raw
            ]
        );
        assert!(material_from_maps(&mut doc, "None", &Maps::default()).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_environment_file_has_a_picture_whose_bright_sky_is_not_white() {
        let dir = std::env::temp_dir().join(format!("kreate-envthumb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        // A sky four times brighter than white on top, a dim ground below.
        let (w, h) = (64u32, 32u32);
        let value = |top: f32, bottom: f32| -> Vec<f32> {
            (0..w * h)
                .map(|i| if i / w < h / 2 { top } else { bottom })
                .collect()
        };
        let (r, g, b) = (value(4.0, 0.05), value(4.0, 0.05), value(6.0, 0.05));
        let channels =
            [("R", &r), ("G", &g), ("B", &b)].map(|(name, values)| kreate_io::exr::Channel {
                name,
                values,
                full_float: false,
            });
        let bytes = kreate_io::exr::write(w, h, &channels, &[]);
        let path = dir.join("sky.exr");
        std::fs::write(&path, bytes).expect("written");
        let (tw, th, pixels) = environment_thumbnail(&path).expect("a picture");

        assert!(tw > 0 && th > 0);
        let top = &pixels[..4];
        let bottom = &pixels[pixels.len() - 4..];
        assert!(
            top[2] < 255 && top[2] > 200,
            "the sky is bright, not clipped: {top:?}"
        );
        assert!(bottom[0] < 80, "the ground is dark: {bottom:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_folder_is_read_for_models_environments_and_materials() {
        let root = std::env::temp_dir().join(format!("kreate-assets-{}", std::process::id()));
        let wood = root.join("textures").join("oak_planks");
        std::fs::create_dir_all(&wood).expect("made");
        for (path, bytes) in [
            (root.join("chair.glb"), &b"glb"[..]),
            (root.join("sky_1k.hdr"), b"hdr"),
            (wood.join("oak_planks_diff_1k.jpg"), b"jpg"),
            (wood.join("oak_planks_rough_1k.jpg"), b"jpg"),
            (wood.join("oak_planks_nor_gl_1k.jpg"), b"jpg"),
            (wood.join("oak_planks_nor_dx_1k.jpg"), b"jpg"),
            (root.join("notes.txt"), b"txt"),
        ] {
            std::fs::write(path, bytes).expect("written");
        }
        let entries = scan_folders(std::slice::from_ref(&root));
        let kinds: Vec<(Kind, &str)> = entries.iter().map(|e| (e.kind, e.name.as_str())).collect();
        assert!(kinds.contains(&(Kind::Model, "chair")), "{kinds:?}");
        assert!(kinds.contains(&(Kind::Hdri, "sky 1k")));
        assert!(kinds.contains(&(Kind::Material, "oak planks")));
        assert_eq!(entries.len(), 3);
        let maps = local_maps(&wood);
        assert!(maps
            .color
            .is_some_and(|p| p.ends_with("oak_planks_diff_1k.jpg")));
        assert!(maps
            .normal
            .is_some_and(|p| p.ends_with("oak_planks_nor_gl_1k.jpg")));
        assert!(maps.roughness.is_some());
        let _ = std::fs::remove_dir_all(&root);
    }
}
