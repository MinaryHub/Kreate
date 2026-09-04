//! `.kreate` — the document container.
//!
//! A zip64 archive of named entries. Zip rather than a bespoke container for
//! reasons that are all about the day something goes wrong: universal tooling,
//! random access to a single mesh without reading the rest, append-friendly,
//! and recoverable with `unzip` at two in the morning.
//!
//! ```text
//! manifest.json        format, version, generator, units, up axis
//! scene.json           objects, transforms, hierarchy, references by key
//! meshes/<key>.kmesh   binary, STORED so it can be memory mapped
//! (anything else)      preserved verbatim
//! ```
//!
//! # Three decisions that cannot be revisited later
//!
//! **Binary entries are STORED, never deflated.** A compressed entry cannot be
//! memory mapped, and the whole point of [`crate::kmesh`] is that loading a mesh
//! is a cast rather than a parse. Compressing them would save disk and give back
//! the load time the format exists to buy. The text entries are deflated, where
//! it costs nothing.
//!
//! **The bytes are deterministic.** Entries are written in a fixed order, and
//! every timestamp is zeroed — a zip's per-entry modification time is otherwise
//! filled from the clock, so two saves of an unchanged document would differ in
//! every entry and version control would store a whole new copy each time.
//!
//! **Unknown entries survive.** An archive written by a newer build, opened and
//! saved here, keeps every entry this build did not recognise. Without it, one
//! save in an older version silently discards whatever the newer one stored, and
//! the user is told nothing.
//!
//! # Why the file has its own record types
//!
//! The document's structs are not serialised directly. The plan rejects that
//! explicitly — deriving a format from in-memory layout welds the two together
//! forever, and then a field cannot be renamed without breaking every file ever
//! written. So [`ObjectRecord`] and friends are the *format*, converted to and
//! from the document at the boundary, and each carries an `extra` map so a field
//! added by a newer build comes back out untouched.

use kreate_math::DTransform;
use kreate_mesh::MeshData;
use kreate_scene::document::{Document, Linkage, Object, ObjectData};
use kreate_scene::{MeshId, ObjectId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, Write};

/// What this build writes.
pub const FORMAT: &str = "kreate";
pub const VERSION_MAJOR: u32 = 1;
pub const VERSION_MINOR: u32 = 0;

const MANIFEST: &str = "manifest.json";
const SCENE: &str = "scene.json";
const MESH_DIR: &str = "meshes/";

#[derive(Debug, thiserror::Error)]
pub enum ContainerError {
    #[error("not a kreate document: {0}")]
    NotAKreateDocument(String),
    #[error("document needs a reader of major version {needed}, this is {VERSION_MAJOR}")]
    TooNew { needed: u32 },
    #[error("archive error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{entry} is not valid json: {source}")]
    Json {
        entry: String,
        source: serde_json::Error,
    },
    #[error("mesh {key}: {source}")]
    Mesh {
        key: String,
        source: crate::kmesh::KmeshError,
    },
    #[error("object {object} references mesh {key}, which the archive does not contain")]
    MissingMesh { object: String, key: String },
    #[error("object {object} cannot be parented: {reason}")]
    BadHierarchy { object: String, reason: String },
}

/// The archive's own description of itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version_major: u32,
    pub version_minor: u32,
    /// The oldest major version that can still read this document correctly.
    pub min_reader_major: u32,
    pub generator: String,
    /// Metres per Kreate unit.
    pub unit_scale: f64,
    /// Always `Z` for a file this build writes; recorded so a future convention
    /// change is a stated fact in the file rather than an assumption.
    pub up_axis: String,
    /// Anything a newer build added.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            format: FORMAT.to_string(),
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            min_reader_major: 1,
            generator: format!("kreate {}", env!("CARGO_PKG_VERSION")),
            unit_scale: 1.0,
            up_axis: "Z".to_string(),
            extra: BTreeMap::new(),
        }
    }
}

/// One object, as the file describes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRecord {
    pub name: String,
    /// Index into [`SceneFile::objects`], not an arena id.
    ///
    /// Arena ids carry a generation and are only meaningful to the process that
    /// made them; writing one to a file would be writing down a pointer.
    pub parent: Option<usize>,
    pub translation: [f64; 3],
    /// Rotation as `xyzw`.
    pub rotation: [f64; 4],
    pub scale: [f64; 3],
    /// Row-major 4x3 affine, the inverse of the parent's world matrix when
    /// parenting happened. Stored because recomputing it would move the object.
    pub parent_inverse: [f64; 12],
    /// The mesh entry's key, for an object that has geometry.
    pub mesh: Option<String>,
    pub visible: bool,
    pub selectable: bool,
    /// `None` for a local datablock; otherwise the library and name it came
    /// from.
    pub linked: Option<[String; 2]>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// The scene graph, as the file describes it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneFile {
    pub objects: Vec<ObjectRecord>,
    /// Indices into `objects`, in user-visible order.
    pub roots: Vec<usize>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A document plus everything in the archive this build did not recognise.
pub struct Loaded {
    pub document: Document,
    pub manifest: Manifest,
    /// Entries kept so a save does not discard them. Hand back to [`write`].
    pub unknown: Vec<UnknownEntry>,
}

/// An archive entry this build has no meaning for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownEntry {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Serialises a document into a `.kreate` archive.
pub fn write(
    document: &Document,
    manifest: &Manifest,
    unknown: &[UnknownEntry],
) -> Result<Vec<u8>, ContainerError> {
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);

        // Zeroed, not "now". A zip stores a modification time per entry, and
        // taking it from the clock would make two saves of an unchanged document
        // differ in every entry — so version control would store a fresh copy of
        // a file whose contents never changed.
        //
        // Set explicitly even though `zip` currently defaults to a fixed date:
        // that default is only fixed because zip is built without its `time`
        // feature, and a feature is exactly the sort of thing another crate in
        // the graph can turn on without anyone noticing. Then the default
        // becomes the clock and determinism goes silently.
        let epoch = zip::DateTime::default();
        let text = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .last_modified_time(epoch);
        // STORED, because a deflated entry cannot be memory mapped and mapping
        // is the entire reason `.kmesh` is laid out the way it is.
        let binary = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .last_modified_time(epoch)
            .large_file(true);

        zip.start_file(MANIFEST, text)?;
        zip.write_all(&canonical_json(manifest)?)?;

        // Meshes keyed by their position in the document iteration, zero
        // padded so the order in the archive matches the order in the scene
        // whether it is read by a program or by a person running `unzip -l`.
        //
        // Position rather than arena index: an arena id carries a generation and
        // is only meaningful to the process that made it. Writing one to a file
        // would be writing down a pointer.
        let mut scene = SceneFile::default();
        let index_of: BTreeMap<ObjectId, usize> = document
            .objects()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        let keys: BTreeMap<MeshId, String> = document
            .meshes()
            .enumerate()
            .map(|(slot, (id, _))| (id, mesh_key(slot)))
            .collect();

        for (id, object) in document.objects() {
            let mesh = match object.data {
                ObjectData::Mesh(m) => keys.get(&m).cloned(),
                _ => None,
            };
            let t = &object.transform;
            let m = object.parent_inverse;
            scene.objects.push(ObjectRecord {
                name: object.name.clone(),
                parent: object.parent.and_then(|p| index_of.get(&p).copied()),
                translation: t.translation.to_array(),
                rotation: [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                scale: t.scale.to_array(),
                parent_inverse: affine_to_array(&m),
                mesh,
                visible: object.visible,
                selectable: object.selectable,
                linked: match &object.linkage {
                    Linkage::Local => None,
                    Linkage::Linked { library, name } => Some([library.clone(), name.clone()]),
                },
                extra: BTreeMap::new(),
            });
            let _ = id;
        }
        scene.roots = document
            .roots()
            .iter()
            .filter_map(|id| index_of.get(id).copied())
            .collect();

        zip.start_file(SCENE, text)?;
        zip.write_all(&canonical_json(&scene)?)?;

        for (slot, (_, mesh)) in document.meshes().enumerate() {
            zip.start_file(format!("{MESH_DIR}{}.kmesh", mesh_key(slot)), binary)?;
            zip.write_all(&crate::kmesh::write(mesh, &[]))?;
        }

        // Last, and in the order they arrived, so their presence does not
        // disturb the position of anything else.
        for entry in unknown {
            zip.start_file(&entry.name, text)?;
            zip.write_all(&entry.bytes)?;
        }
        zip.finish()?;
    }
    Ok(buffer.into_inner())
}

/// Parses a `.kreate` archive.
pub fn read<R: Read + Seek>(reader: R) -> Result<Loaded, ContainerError> {
    let mut zip = zip::ZipArchive::new(reader)?;

    let manifest: Manifest = {
        let mut file = zip
            .by_name(MANIFEST)
            .map_err(|_| ContainerError::NotAKreateDocument("no manifest.json".into()))?;
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        serde_json::from_str(&text).map_err(|source| ContainerError::Json {
            entry: MANIFEST.to_string(),
            source,
        })?
    };
    if manifest.format != FORMAT {
        return Err(ContainerError::NotAKreateDocument(format!(
            "format is {:?}",
            manifest.format
        )));
    }
    if manifest.min_reader_major > VERSION_MAJOR {
        return Err(ContainerError::TooNew {
            needed: manifest.min_reader_major,
        });
    }

    let scene: SceneFile = {
        let mut file = zip
            .by_name(SCENE)
            .map_err(|_| ContainerError::NotAKreateDocument("no scene.json".into()))?;
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        serde_json::from_str(&text).map_err(|source| ContainerError::Json {
            entry: SCENE.to_string(),
            source,
        })?
    };

    // Every entry, so unknown ones can be carried through.
    let names: Vec<String> = zip.file_names().map(str::to_string).collect();
    let mut meshes: BTreeMap<String, MeshData> = BTreeMap::new();
    let mut unknown = Vec::new();
    for name in names {
        if name == MANIFEST || name == SCENE {
            continue;
        }
        let mut file = zip.by_name(&name)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        if let Some(key) = name
            .strip_prefix(MESH_DIR)
            .and_then(|k| k.strip_suffix(".kmesh"))
        {
            let loaded = crate::kmesh::read(&bytes).map_err(|source| ContainerError::Mesh {
                key: key.to_string(),
                source,
            })?;
            meshes.insert(key.to_string(), loaded.mesh);
        } else if !name.ends_with('/') {
            // A directory entry carries nothing and is recreated by the writer;
            // anything else is content this build cannot interpret and must not
            // discard.
            unknown.push(UnknownEntry { name, bytes });
        }
    }

    let mut document = Document::new();
    let mut mesh_ids: BTreeMap<String, kreate_scene::MeshId> = BTreeMap::new();
    for (key, mesh) in meshes {
        mesh_ids.insert(key, document.add_mesh(mesh));
    }

    let mut object_ids: Vec<ObjectId> = Vec::with_capacity(scene.objects.len());
    for record in &scene.objects {
        let data = match &record.mesh {
            Some(key) => {
                let id = mesh_ids
                    .get(key)
                    .copied()
                    .ok_or_else(|| ContainerError::MissingMesh {
                        object: record.name.clone(),
                        key: key.clone(),
                    })?;
                ObjectData::Mesh(id)
            }
            None => ObjectData::Empty,
        };
        let mut object = Object::new(record.name.clone(), data);
        object.transform = DTransform {
            translation: glam::DVec3::from_array(record.translation),
            rotation: glam::DQuat::from_xyzw(
                record.rotation[0],
                record.rotation[1],
                record.rotation[2],
                record.rotation[3],
            ),
            scale: glam::DVec3::from_array(record.scale),
        };
        object.parent_inverse = affine_from_array(&record.parent_inverse);
        object.visible = record.visible;
        object.selectable = record.selectable;
        object.linkage = match &record.linked {
            None => Linkage::Local,
            Some([library, name]) => Linkage::Linked {
                library: library.clone(),
                name: name.clone(),
            },
        };
        object_ids.push(document.add_object(object));
    }

    // Parenting in a second pass: a record may name a parent that appears later
    // in the list, and the document assigns ids as objects are added.
    //
    // A cycle or a dangling index in a hand-edited file comes back as an error
    // from `set_parent` rather than as a hierarchy that loops — which would make
    // the first world-transform query run forever.
    for (record, id) in scene.objects.iter().zip(&object_ids) {
        if let Some(parent) = record.parent.and_then(|i| object_ids.get(i).copied()) {
            document
                .set_parent(*id, Some(parent))
                .map_err(|e| ContainerError::BadHierarchy {
                    object: record.name.clone(),
                    reason: e.to_string(),
                })?;
        }
    }

    // `set_parent` recomputes `parent_inverse` so the object does not move,
    // which is right when a user parents something and wrong when loading: the
    // file already recorded the value, and recomputing it from a transform that
    // has not been applied yet would shift the object. Put back what the file
    // said.
    for (record, id) in scene.objects.iter().zip(&object_ids) {
        if let Some(object) = document.object_mut(*id) {
            object.parent_inverse = affine_from_array(&record.parent_inverse);
        }
    }

    Ok(Loaded {
        document,
        manifest,
        unknown,
    })
}

/// The archive key for the mesh in a given document slot.
fn mesh_key(slot: usize) -> String {
    format!("{slot:06}")
}

/// JSON with sorted keys and a trailing newline.
///
/// Sorted because `serde_json`'s map is a `BTreeMap` unless `preserve_order` is
/// enabled, which it deliberately is not: key order that followed insertion
/// would make the bytes depend on the order a struct's fields happened to be
/// visited, and the file would differ between saves of an identical document.
fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, ContainerError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|source| ContainerError::Json {
        entry: "<writing>".to_string(),
        source,
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn affine_to_array(m: &glam::DAffine3) -> [f64; 12] {
    let c = m.to_cols_array();
    [
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11],
    ]
}

fn affine_from_array(a: &[f64; 12]) -> glam::DAffine3 {
    glam::DAffine3::from_cols_array(a)
}
