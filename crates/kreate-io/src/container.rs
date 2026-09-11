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
/// Bumped for an addition an older reader can ignore.
///
/// 1: datablock identities (`ObjectRecord::uid`, `SceneFile::mesh_uids`). A
/// reader that does not know them puts them in its `extra` map and writes them
/// back, so a round trip through an older build does not lose a scene's
/// animation targets.
///
/// 2: lights and the world (`SceneFile::lights`, `SceneFile::worlds`,
/// `ObjectRecord::light`). A version-1 reader loses an object's *link* to its
/// light — the object comes back as an empty and the light records travel in
/// its `extra` map — which is the honest limit of forwards compatibility for
/// something that changes what an object *is*. Nothing is discarded.
///
/// 3: materials (`SceneFile::materials`, `SceneFile::mesh_slots`,
/// `ObjectRecord::material_overrides`). An older reader keeps all three in its
/// `extra` maps and writes them back, so a scene round-tripped through one
/// keeps its materials — unlike a light, a material does not change what an
/// object *is*, so nothing about the object record has to be understood for
/// the link to survive.
pub const VERSION_MINOR: u32 = 3;

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
    /// The object's identity, which outlives this file.
    ///
    /// Not a position and not the name: a curve, a driver or a constraint
    /// stored in a scene has to find its target when the scene is opened
    /// again, and both of the other candidates move when the user does
    /// ordinary work. See `kreate_reflect::path`.
    ///
    /// Zero, or absent, in a scene written before identities existed. Such a
    /// scene has no curves in it either — nothing could have stored a path
    /// there was nothing to point at — so a fresh id on load loses nothing.
    #[serde(default)]
    pub uid: u64,
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
    /// Index into [`SceneFile::cameras`], for an object that is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<usize>,
    /// Index into [`SceneFile::lights`], for an object that is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light: Option<usize>,
    /// Per-slot materials that win over the mesh's, as indices into
    /// [`SceneFile::materials`].
    ///
    /// `null` in a position means "as the mesh says" and not "no material" —
    /// the list is positional, so overriding slot one needs an entry for slot
    /// zero and that entry must not erase it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub material_overrides: Vec<Option<usize>>,
    pub visible: bool,
    pub selectable: bool,
    /// `None` for a local datablock; otherwise the library and name it came
    /// from.
    pub linked: Option<[String; 2]>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// The scene graph, as the file describes it.
/// A camera datablock.
///
/// Physical parameters, in the units the instruments are labelled in — see
/// `kreate_scene::camera`, which is explicit that the focal length is stored
/// and the field of view is derived. Writing the angle instead would have made
/// this file the second source of truth about a camera's framing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraRecord {
    pub name: String,
    /// The datablock's identity, which outlives this file.
    #[serde(default)]
    pub uid: u64,
    /// `perspective` or `orthographic`.
    pub projection: String,
    pub focal_length_mm: f64,
    pub sensor_width_mm: f64,
    pub sensor_height_mm: f64,
    /// `auto`, `horizontal` or `vertical`.
    pub sensor_fit: String,
    pub ortho_scale: f64,
    pub clip_near: f64,
    pub clip_far: f64,
    /// Absent for a pinhole. Distinguished from a very large number because
    /// "no depth of field" and "almost none" are different code paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f_stop: Option<f64>,
    pub focus_distance: f64,
    /// A fraction of one frame's interval, not a duration.
    pub shutter: f64,
    /// Sensitivity, as an ISO number.
    ///
    /// Defaulted rather than required, so a scene written before the exposure
    /// existed loads at the base sensitivity instead of at zero — which would
    /// be a black frame.
    #[serde(default = "default_iso")]
    pub iso: f64,
    /// The ±EV dial, in stops.
    #[serde(default)]
    pub exposure_compensation: f64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// What a camera from before the exposure existed is read as.
fn default_iso() -> f64 {
    100.0
}

/// A light datablock.
///
/// Intensity is in the unit the *kind* uses — candela for a point or a spot,
/// lux for a sun, nits for an area light — which is what `KHR_lights_punctual`
/// holds, so a glTF export is a copy rather than a conversion. Writing a unit
/// of our own would put a rounding step in every round trip and leave nobody
/// able to say which side was authoritative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightRecord {
    pub name: String,
    /// The datablock's identity, which outlives this file.
    #[serde(default)]
    pub uid: u64,
    /// `point`, `sun`, `spot` or `area`.
    pub kind: String,
    /// The tint, linear and in the working space.
    ///
    /// Linear because that is what the document holds and what glTF holds;
    /// encoding it here would put a transfer function in the middle of a round
    /// trip for no gain. Not an amount of light — that is `intensity`.
    pub color: [f32; 3],
    pub intensity: f64,
    /// Stops on top of `intensity`. Separate because `UsdLux` has both.
    pub exposure: f64,

    /// Metres, for a point or a spot: what makes the shadow soft.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// Radians, for a sun: how wide the source appears.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_diameter: Option<f64>,
    /// Radians, for a spot: the full cone angle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cone_angle: Option<f64>,
    /// The fraction of a spot's cone that fades.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blend: Option<f64>,
    /// `rect`, `disc` or `ellipse`, for an area light.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    /// Metres. A disc uses `width` as its diameter and ignores `height`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    /// Radians, for an area light: how far off the normal it emits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,

    pub cast_shadow: bool,
    pub affect_diffuse: bool,
    pub affect_specular: bool,
    pub max_bounces: u8,

    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A world datablock: what light arrives from no object in particular.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldRecord {
    pub name: String,
    #[serde(default)]
    pub uid: u64,
    /// `flat` or `image`.
    pub background: String,
    /// The flat colour, linear — and for an image background, what to fall back
    /// to.
    ///
    /// A world's background is one thing or the other, so an image background
    /// has no flat colour of its own to write and the writer puts a neutral
    /// grey here. That matters because an image that cannot be resolved on
    /// load lands on this value, and black would read as a broken file rather
    /// than a missing texture.
    ///
    /// If a world ever holds both a colour *and* an image — which is how a
    /// shading graph would express it — this field becomes the real colour and
    /// nothing about the format changes.
    pub color: [f32; 3],
    /// The background image's *identity*, not its index.
    ///
    /// An identity because an image is a datablock in its own right and this
    /// file does not yet carry images — the archive's `images/` layout is still
    /// to come. Storing the uid now means the pointer survives every save in
    /// between, and starts resolving the day images are written, with no change
    /// to this format. An index into a list that does not exist would have had
    /// to be invented and re-invented.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_uid: Option<u64>,
    pub strength: f64,
    /// Radians about the up axis.
    pub rotation: f64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A material datablock.
///
/// The base colour is linear, like a light's tint and like glTF's own
/// `baseColorFactor`, so crossing this boundary is a copy rather than a
/// conversion. Alpha is beside it rather than in it, because a transfer
/// function applies to three channels and never to the fourth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialRecord {
    pub name: String,
    /// The datablock's identity, which outlives this file.
    #[serde(default)]
    pub uid: u64,
    /// Linear, bounded by one.
    pub base_color: [f32; 3],
    pub alpha: f32,
    pub metallic: f32,
    pub roughness: f32,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneFile {
    pub objects: Vec<ObjectRecord>,
    /// Indices into `objects`, in user-visible order.
    pub roots: Vec<usize>,
    /// Mesh datablock identities, by the key the archive files them under.
    ///
    /// Beside the objects rather than inside them, because a mesh is not owned
    /// by an object: two may share one, and one may belong to none at all.
    ///
    /// Empty in a scene written before identities existed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mesh_uids: BTreeMap<String, u64>,
    /// Camera datablocks, in the order the document holds them.
    ///
    /// An object refers to one by its index here, the same way it refers to a
    /// parent — a within-file link that is rebuilt on load, for which a
    /// position is right. What has to outlive the file is the *identity*, and
    /// that is `CameraRecord::uid`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<CameraRecord>,
    /// Light datablocks, in the order the document holds them. An object refers
    /// to one by its index here, like a camera.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lights: Vec<LightRecord>,
    /// World datablocks. More than one, because a file holding a day setup and
    /// a night setup is the ordinary case and switching between them should not
    /// mean deleting one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worlds: Vec<WorldRecord>,
    /// Material datablocks, in the order the document holds them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialRecord>,
    /// Each mesh's material slots, by the key the archive files the mesh
    /// under, as indices into `materials`.
    ///
    /// Beside the meshes rather than inside them, for the same reason their
    /// identities are: a slot list belongs to a mesh, and a mesh is not owned
    /// by an object.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mesh_slots: BTreeMap<String, Vec<Option<usize>>>,
    /// Index into `worlds`: which one is lighting the scene.
    ///
    /// `None` is a scene lit by nothing but its own lights, which is a real
    /// thing to want rather than a missing value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_world: Option<usize>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A light datablock as the file describes it.
///
/// The kind's parameters are flattened into optional fields rather than a
/// nested enum, so that a version-1 reader carries them through its `extra` map
/// and a person can read the file. Which fields are meaningful follows `kind`.
fn light_record(light: &kreate_scene::Light) -> LightRecord {
    use kreate_scene::{AreaShape, LightKind};

    let mut record = LightRecord {
        name: light.name.clone(),
        uid: light.uid.raw(),
        kind: light.kind.id().to_string(),
        color: light.color.to_array(),
        intensity: light.intensity,
        exposure: light.exposure,
        radius: None,
        angular_diameter: None,
        cone_angle: None,
        blend: None,
        shape: None,
        width: None,
        height: None,
        spread: None,
        cast_shadow: light.cast_shadow,
        affect_diffuse: light.affect_diffuse,
        affect_specular: light.affect_specular,
        max_bounces: light.max_bounces,
        extra: BTreeMap::new(),
    };
    match light.kind {
        LightKind::Point { radius } => record.radius = Some(radius),
        LightKind::Sun { angular_diameter } => record.angular_diameter = Some(angular_diameter),
        LightKind::Spot {
            radius,
            cone_angle,
            blend,
        } => {
            record.radius = Some(radius);
            record.cone_angle = Some(cone_angle);
            record.blend = Some(blend);
        }
        LightKind::Area { shape, spread } => {
            record.spread = Some(spread);
            let (name, width, height) = match shape {
                AreaShape::Rect { width, height } => ("rect", width, height),
                // A disc's diameter goes in `width`, so that a reader that only
                // understands rectangles gets a panel of about the right size
                // rather than one of side zero.
                AreaShape::Disc { radius } => ("disc", radius * 2.0, radius * 2.0),
                AreaShape::Ellipse { width, height } => ("ellipse", width, height),
            };
            record.shape = Some(name.to_string());
            record.width = Some(width);
            record.height = Some(height);
        }
    }
    record
}

/// The light a record describes.
///
/// An unknown kind, or a kind missing the parameters it needs, falls back to
/// that kind's default rather than failing the load: a scene from a later
/// version is worth opening with one lamp the wrong size, and is not worth
/// refusing. The colour and the intensity survive either way, which is what
/// the eye notices first.
fn light_from_record(record: &LightRecord) -> kreate_scene::Light {
    use kreate_scene::{AreaShape, LightKind};

    let kind = match record.kind.as_str() {
        "sun" => LightKind::Sun {
            angular_diameter: record
                .angular_diameter
                .unwrap_or(LightKind::SUN_ANGULAR_DIAMETER),
        },
        "spot" => LightKind::Spot {
            radius: record.radius.unwrap_or(0.05),
            cone_angle: record
                .cone_angle
                .unwrap_or(45.0 * std::f64::consts::PI / 180.0),
            blend: record.blend.unwrap_or(0.15),
        },
        "area" => {
            let width = record.width.unwrap_or(1.0);
            let height = record.height.unwrap_or(width);
            LightKind::Area {
                shape: match record.shape.as_deref() {
                    Some("disc") => AreaShape::Disc {
                        radius: width * 0.5,
                    },
                    Some("ellipse") => AreaShape::Ellipse { width, height },
                    _ => AreaShape::Rect { width, height },
                },
                spread: record.spread.unwrap_or(std::f64::consts::PI),
            }
        }
        // Including anything this build has never heard of.
        _ => LightKind::Point {
            radius: record.radius.unwrap_or(0.05),
        },
    };

    kreate_scene::Light {
        name: record.name.clone(),
        linkage: Linkage::Local,
        uid: kreate_scene::Uid::NONE,
        kind,
        color: kreate_math::SceneRgb::new(record.color[0], record.color[1], record.color[2]),
        intensity: record.intensity,
        exposure: record.exposure,
        cast_shadow: record.cast_shadow,
        affect_diffuse: record.affect_diffuse,
        affect_specular: record.affect_specular,
        max_bounces: record.max_bounces,
    }
}

fn world_record(world: &kreate_scene::World, document: &Document) -> WorldRecord {
    use kreate_scene::Background;

    let (background, color, image_uid) = match world.background {
        Background::Flat(color) => ("flat", color, None),
        // The colour a flat background would have is not stored for an image
        // one, so a neutral grey stands in — the field exists so that turning
        // an image off lands somewhere sensible rather than on black.
        Background::Image(id) => (
            "image",
            kreate_math::SceneRgb::new(0.05, 0.05, 0.05),
            document.image_uid(id).map(|uid| uid.raw()),
        ),
    };
    WorldRecord {
        name: world.name.clone(),
        uid: world.uid.raw(),
        background: background.to_string(),
        color: color.to_array(),
        image_uid,
        strength: world.strength,
        rotation: world.rotation,
        extra: BTreeMap::new(),
    }
}

/// The world a record describes.
///
/// An image background whose image is not in this document comes back as the
/// flat colour beside it. That is the ordinary case today rather than an error
/// case: the archive has no `images/` layout yet, so nothing an image
/// background points at survives a save. The *identity* does, which is why it
/// is stored — the day images are written this resolves with no change here.
fn world_from_record(record: &WorldRecord, document: &Document) -> kreate_scene::World {
    use kreate_scene::Background;

    let color = kreate_math::SceneRgb::new(record.color[0], record.color[1], record.color[2]);
    let background = match record.background.as_str() {
        "image" => record
            .image_uid
            .and_then(|raw| document.image_by_uid(kreate_scene::Uid::from_raw(raw)))
            .map(Background::Image)
            .unwrap_or(Background::Flat(color)),
        _ => Background::Flat(color),
    };
    kreate_scene::World {
        name: record.name.clone(),
        linkage: Linkage::Local,
        uid: kreate_scene::Uid::NONE,
        background,
        strength: record.strength,
        rotation: record.rotation,
    }
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
        scene.mesh_uids = keys
            .iter()
            .filter_map(|(id, key)| {
                let uid = document.mesh_uid(*id)?;
                (!uid.is_none()).then(|| (key.clone(), uid.raw()))
            })
            .collect();

        let camera_index: BTreeMap<kreate_scene::document::CameraId, usize> = document
            .cameras()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        scene.cameras = document
            .cameras()
            .map(|(_, camera)| CameraRecord {
                name: camera.name.clone(),
                uid: camera.uid.raw(),
                projection: match camera.projection {
                    kreate_scene::Projection::Perspective => "perspective".to_string(),
                    kreate_scene::Projection::Orthographic => "orthographic".to_string(),
                },
                focal_length_mm: camera.focal_length_mm,
                sensor_width_mm: camera.sensor_width_mm,
                sensor_height_mm: camera.sensor_height_mm,
                sensor_fit: camera.sensor_fit.id().to_string(),
                ortho_scale: camera.ortho_scale,
                clip_near: camera.clip_near,
                clip_far: camera.clip_far,
                f_stop: camera.f_stop,
                focus_distance: camera.focus_distance,
                shutter: camera.shutter,
                iso: camera.iso,
                exposure_compensation: camera.exposure_compensation,
                extra: BTreeMap::new(),
            })
            .collect();

        let material_index: BTreeMap<kreate_scene::MaterialId, usize> = document
            .materials()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        scene.materials = document
            .materials()
            .map(|(_, m)| {
                let [r, g, b] = m.base_color.to_array();
                MaterialRecord {
                    name: m.name.clone(),
                    uid: m.uid.raw(),
                    base_color: [r, g, b],
                    alpha: m.alpha,
                    metallic: m.metallic,
                    roughness: m.roughness,
                    extra: BTreeMap::new(),
                }
            })
            .collect();
        scene.mesh_slots = keys
            .iter()
            .filter_map(|(id, key)| {
                let slots = document.mesh_slots(*id);
                if slots.is_empty() {
                    return None;
                }
                Some((
                    key.clone(),
                    slots
                        .iter()
                        .map(|s| s.and_then(|m| material_index.get(&m).copied()))
                        .collect(),
                ))
            })
            .collect();

        let light_index: BTreeMap<kreate_scene::LightId, usize> = document
            .lights()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        scene.lights = document
            .lights()
            .map(|(_, light)| light_record(light))
            .collect();

        let world_index: BTreeMap<kreate_scene::WorldId, usize> = document
            .worlds()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        scene.worlds = document
            .worlds()
            .map(|(_, world)| world_record(world, document))
            .collect();
        scene.active_world = document
            .active_world()
            .and_then(|id| world_index.get(&id).copied());

        for (id, object) in document.objects() {
            let mesh = match object.data {
                ObjectData::Mesh(m) => keys.get(&m).cloned(),
                _ => None,
            };
            let camera = match object.data {
                ObjectData::Camera(c) => camera_index.get(&c).copied(),
                _ => None,
            };
            let light = match object.data {
                ObjectData::Light(l) => light_index.get(&l).copied(),
                _ => None,
            };
            let t = &object.transform;
            let m = object.parent_inverse;
            scene.objects.push(ObjectRecord {
                name: object.name.clone(),
                uid: object.uid.raw(),
                parent: object.parent.and_then(|p| index_of.get(&p).copied()),
                translation: t.translation.to_array(),
                rotation: [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                scale: t.scale.to_array(),
                parent_inverse: affine_to_array(&m),
                mesh,
                camera,
                light,
                material_overrides: object
                    .material_overrides
                    .iter()
                    .map(|s| s.and_then(|m| material_index.get(&m).copied()))
                    .collect(),
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
        // A scene written before identities existed has none to restore, and
        // gets fresh ones. It has no curves in it either, so nothing is
        // orphaned by that.
        let id = match scene.mesh_uids.get(&key) {
            Some(raw) => document.restore_mesh(mesh, kreate_scene::Uid::from_raw(*raw)),
            None => document.add_mesh(mesh),
        };
        mesh_ids.insert(key, id);
    }

    let mut camera_ids: Vec<kreate_scene::document::CameraId> =
        Vec::with_capacity(scene.cameras.len());
    for record in &scene.cameras {
        // An unknown projection or sensor fit falls back to the default rather
        // than failing the load. A scene from a later version is worth opening
        // with one camera framed wrongly; it is not worth refusing.
        let camera = kreate_scene::Camera {
            name: record.name.clone(),
            linkage: Linkage::Local,
            uid: kreate_scene::Uid::NONE,
            projection: match record.projection.as_str() {
                "orthographic" => kreate_scene::Projection::Orthographic,
                _ => kreate_scene::Projection::Perspective,
            },
            focal_length_mm: record.focal_length_mm,
            sensor_width_mm: record.sensor_width_mm,
            sensor_height_mm: record.sensor_height_mm,
            sensor_fit: kreate_scene::SensorFit::from_id(&record.sensor_fit).unwrap_or_default(),
            ortho_scale: record.ortho_scale,
            clip_near: record.clip_near,
            clip_far: record.clip_far,
            f_stop: record.f_stop,
            focus_distance: record.focus_distance,
            shutter: record.shutter,
            iso: record.iso,
            exposure_compensation: record.exposure_compensation,
        };
        camera_ids.push(document.restore_camera(camera, kreate_scene::Uid::from_raw(record.uid)));
    }

    let mut material_ids: Vec<kreate_scene::MaterialId> =
        Vec::with_capacity(scene.materials.len());
    for record in &scene.materials {
        let material = kreate_scene::Material {
            name: record.name.clone(),
            base_color: kreate_math::SceneRgb::new(
                record.base_color[0],
                record.base_color[1],
                record.base_color[2],
            ),
            alpha: record.alpha,
            metallic: record.metallic,
            roughness: record.roughness,
            linkage: Linkage::Local,
            uid: kreate_scene::Uid::NONE,
        };
        material_ids
            .push(document.restore_material(material, kreate_scene::Uid::from_raw(record.uid)));
    }
    // An index naming a material that is not in the file leaves the slot
    // empty, which is the same as an unassigned slot and is what the renderer
    // already draws.
    for (key, slots) in &scene.mesh_slots {
        let Some(mesh) = mesh_ids.get(key) else {
            continue;
        };
        let resolved: Vec<Option<kreate_scene::MaterialId>> = slots
            .iter()
            .map(|s| s.and_then(|i| material_ids.get(i).copied()))
            .collect();
        document.set_mesh_slots(*mesh, resolved);
    }

    let mut light_ids: Vec<kreate_scene::LightId> = Vec::with_capacity(scene.lights.len());
    for record in &scene.lights {
        let light = light_from_record(record);
        light_ids.push(document.restore_light(light, kreate_scene::Uid::from_raw(record.uid)));
    }

    let mut world_ids: Vec<kreate_scene::WorldId> = Vec::with_capacity(scene.worlds.len());
    for record in &scene.worlds {
        let world = world_from_record(record, &document);
        world_ids.push(document.restore_world(world, kreate_scene::Uid::from_raw(record.uid)));
    }
    // An out-of-range index leaves the scene lit by its own lights rather than
    // failing the load, which is the same judgement as an unknown projection.
    if let Some(id) = scene.active_world.and_then(|i| world_ids.get(i)) {
        document.set_active_world(Some(*id));
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
            None => match record.camera.and_then(|i| camera_ids.get(i)) {
                Some(id) => ObjectData::Camera(*id),
                None => match record.light.and_then(|i| light_ids.get(i)) {
                    Some(id) => ObjectData::Light(*id),
                    None => ObjectData::Empty,
                },
            },
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
        object.material_overrides = record
            .material_overrides
            .iter()
            .map(|s| s.and_then(|i| material_ids.get(i).copied()))
            .collect();
        object.linkage = match &record.linked {
            None => Linkage::Local,
            Some([library, name]) => Linkage::Linked {
                library: library.clone(),
                name: name.clone(),
            },
        };
        object_ids.push(match record.uid {
            0 => document.add_object(object),
            raw => document.restore_object(object, kreate_scene::Uid::from_raw(raw)),
        });
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
