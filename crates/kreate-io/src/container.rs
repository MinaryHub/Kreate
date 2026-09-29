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
//! images/<key>.ktile   binary, STORED, the resident tiles of one image
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
///
/// 4: images (`SceneFile::images`, `images/<key>.ktile`). An older reader keeps
/// the records in its `extra` map and the tile files as unknown entries, and
/// writes both back — so an HDRI survives a round trip through a build that has
/// never heard of one. What such a reader cannot do is *resolve* a world's
/// background to it, and that was already true: `WorldRecord::image_uid` has
/// been written since minor 2 against exactly this day.
pub const VERSION_MINOR: u32 = 4;

const MANIFEST: &str = "manifest.json";
const SCENE: &str = "scene.json";
const MESH_DIR: &str = "meshes/";
const IMAGE_DIR: &str = "images/";

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
    #[error("image {key}: {source}")]
    Image {
        key: String,
        source: crate::ktile::KtileError,
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
    /// Index into [`SceneFile::actions`]: the action this object plays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<usize>,
    pub visible: bool,
    pub selectable: bool,
    /// `None` for a local datablock; otherwise the library and name it came
    /// from.
    pub linked: Option<[String; 2]>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// An action: its curves, each a property of whoever plays it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionRecord {
    pub name: String,
    #[serde(default)]
    pub uid: u64,
    /// What kind of datablock plays it — `material`, `light` — or absent for
    /// an object's, which most are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animates: Option<String>,
    pub curves: Vec<CurveRecord>,
}

/// One curve: a component of a property, and its keys.
///
/// The property by the name its type declares, for the reason
/// `kreate_reflect::path` gives: a name is code and stays put, a number would
/// silently retarget every curve the day its table was reordered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveRecord {
    pub property: String,
    pub index: u8,
    pub keys: Vec<KeyRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyRecord {
    pub frame: f64,
    pub value: f64,
    /// Written only when it is not the default Bézier, which most keys are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<String>,
    /// Written only when set: what the dope sheet was showing selected.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub selected: bool,
    /// A hand-set slope, value per frame; absent for auto.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slope: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaybackRecord {
    pub frame: f64,
    pub start: i32,
    pub end: i32,
    pub fps: f64,
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
    /// Whether the exposure is metered off the scene.
    ///
    /// Defaulted to *on*, which is what a file written before this existed
    /// should get: those files were saved under a fixed exposure that suited
    /// whatever lit them, and a metered one lands in the same place for the
    /// scenes where it did while fixing the ones where it did not.
    #[serde(default = "default_auto_exposure")]
    pub auto_exposure: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// What a camera from before the exposure existed is read as.
fn default_iso() -> f64 {
    100.0
}

/// And what one from before the meter existed is read as. See the field.
fn default_auto_exposure() -> bool {
    true
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

    /// Index into [`SceneFile::actions`]: what animates this light.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<usize>,
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
    /// An identity because an image is a datablock in its own right, and it was
    /// written here for three minor versions before the archive could carry an
    /// image at all. That is the point: the pointer survived every save in
    /// between and started resolving the day `images/` arrived, with no change
    /// to this format. An index into a list that did not exist would have had
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
    /// The base colour texture's identity, if one is assigned.
    ///
    /// A uid and not an index, for the reason [`WorldRecord::image_uid`] is
    /// one: a position says where the image sits in today's file and an
    /// identity says which image it is. And absent rather than zero when there
    /// is none, so that a file written before this field existed reads as "no
    /// texture" rather than as "the image whose uid is zero".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_uid: Option<u64>,
    /// The packed metallic-roughness map's identity. Absent for none, for the
    /// reason `image_uid` is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic_roughness_uid: Option<u64>,
    /// The normal map's identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_uid: Option<u64>,
    /// The normal map's strength, written only when it is not one.
    ///
    /// Absent reads as one, which is the value every file written before this
    /// field had — so an old file and a new file that never touched the dial
    /// are byte-identical, and the golden file does not move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_strength: Option<f32>,
    /// The material's own node graph, when it uses one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphRecord>,
    /// Index into [`SceneFile::actions`]: what animates this material.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<usize>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A material's shading graph, as the file writes it.
///
/// Nodes by tag and flat numbers rather than by serde's view of the enum:
/// `NodeKind::tag` is the stable name the module docs promise, and inserting a
/// kind must not renumber every file. An image travels as an identity, for
/// the reason `MaterialRecord::image_uid` does.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphRecord {
    pub nodes: Vec<NodeRecord>,
    /// `[from_node, from_socket, to_node, to_socket]`.
    pub links: Vec<[usize; 4]>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeRecord {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub position: [f32; 2],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_uid: Option<u64>,
}

/// A graph as its record.
fn graph_record(graph: &kreate_scene::shader::ShaderGraph, document: &Document) -> GraphRecord {
    GraphRecord {
        nodes: graph
            .nodes
            .iter()
            .map(|node| NodeRecord {
                name: node.name.clone(),
                kind: node.kind.tag().to_string(),
                position: node.position,
                values: node.kind.values(),
                image_uid: match node.kind {
                    kreate_scene::shader::NodeKind::Image { image } => image
                        .and_then(|id| document.image_uid(id))
                        .map(|uid| uid.raw()),
                    _ => None,
                },
            })
            .collect(),
        links: graph
            .links
            .iter()
            .map(|l| [l.from_node, l.from_socket, l.to_node, l.to_socket])
            .collect(),
    }
}

/// A record as a graph, or `None` when it cannot be one.
///
/// **All or nothing.** A node of a kind this build does not know makes the
/// whole graph unreadable, and the material falls back to its flat fields —
/// which it kept for exactly this. A graph with the unknown node dropped would
/// load, draw, and be a different material with nothing saying so. Links are
/// checked the way the editor checks them, and one the graph refuses is
/// dropped: a wire is not a material, and the file can say anything.
fn graph_from_record(
    record: &GraphRecord,
    document: &Document,
) -> Option<kreate_scene::shader::ShaderGraph> {
    use kreate_scene::shader::{Link, Node, NodeKind, ShaderGraph};
    let mut graph = ShaderGraph {
        name: String::new(),
        linkage: Linkage::Local,
        uid: kreate_scene::Uid::NONE,
        nodes: Vec::with_capacity(record.nodes.len()),
        links: Vec::new(),
    };
    for node in &record.nodes {
        let image = node
            .image_uid
            .and_then(|raw| document.image_by_uid(kreate_scene::Uid::from_raw(raw)));
        let kind = NodeKind::from_parts(&node.kind, &node.values, image)?;
        let mut made = Node::new(node.name.clone(), kind);
        made.position = node.position;
        graph.add_node(made)?;
    }
    graph.output()?;
    for [from_node, from_socket, to_node, to_socket] in &record.links {
        let _ = graph.link(Link {
            from_node: *from_node,
            from_socket: *from_socket,
            to_node: *to_node,
            to_socket: *to_socket,
        });
    }
    Some(graph)
}

/// An image datablock, as the file describes it.
///
/// The pixels are not here: they are in `images/<key>.ktile`, where `<key>` is
/// this record's position in [`SceneFile::images`]. Everything that is *about*
/// the image rather than *in* it lives here, so that reading a document's
/// image list — what it is called, how big, what space its numbers are in —
/// does not mean opening every tile file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageRecord {
    pub name: String,
    /// The datablock's identity, which outlives this file.
    ///
    /// Load bearing here in a way it is not for a mesh: a world's background
    /// names its image by uid ([`WorldRecord::image_uid`]), and has done since
    /// before images could be written at all.
    #[serde(default)]
    pub uid: u64,
    pub width: u32,
    pub height: u32,
    /// `rgba8`, `rgba16f` or `r8`.
    pub format: String,
    /// `srgb` or `raw`. §2.1's rule: never guessed, always written down.
    pub color_space: String,
    /// The path this image was loaded from, if it was.
    ///
    /// Kept even when the pixels are also in the archive, because it is the
    /// authored thing — a user who edits `sky.hdr` outside and reopens expects
    /// the new sky, and without the path there is nothing to reload from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Whether `images/<key>.ktile` exists for this record.
    ///
    /// A statement about the file rather than a preference. Today the writer
    /// packs every image that has pixels, which is what makes a `.kreate` a
    /// single file somebody can send: an unpacked document with a texture is a
    /// document that is broken on any other machine.
    ///
    /// The thrifty version — leave a `File` image's pixels out and reload them
    /// from `path` — is a later addition and needs nothing new here, which is
    /// why this is a field and not an assumption.
    #[serde(default)]
    pub packed: bool,
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
    /// Image datablocks, in the order the document holds them. The pixels of
    /// the k-th are in `images/<k>.ktile` when [`ImageRecord::packed`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageRecord>,
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
    /// Every action, in the order objects' `action` indices count.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<ActionRecord>,
    /// The scene's clock. Absent when it is at its defaults, so a file that
    /// never touched the timeline reads as it did before there was one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub playback: Option<PlaybackRecord>,
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
        action: None,
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
        action: None,
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
/// flat colour beside it. Now an error case rather than the ordinary one: since
/// minor 4 the archive carries images, so a background that does not resolve
/// means the image was removed or its record is from somewhere else.
///
/// The fallback stays, and the reason it is grey rather than black is the same
/// as it was: a missing texture should read as a missing texture.
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
                auto_exposure: camera.auto_exposure,
                extra: BTreeMap::new(),
            })
            .collect();

        let material_index: BTreeMap<kreate_scene::MaterialId, usize> = document
            .materials()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        // Actions before the objects that name them, by position in this
        // list — the file's convention for a reference that is rewritten
        // wholesale on load.
        let action_index: std::collections::HashMap<kreate_scene::ActionId, usize> = document
            .actions()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();
        scene.actions = document
            .actions()
            .map(|(_, a)| ActionRecord {
                name: a.name.clone(),
                uid: a.uid.raw(),
                animates: (a.animates != kreate_scene::DatablockKind::Object).then(|| a.animates.id().to_string()),
                curves: a
                    .curves
                    .iter()
                    .filter_map(|c| {
                        Some(CurveRecord {
                            property: c.path.property.to_string(),
                            index: c.path.index?,
                            keys: c
                                .keys
                                .iter()
                                .map(|k| KeyRecord {
                                    frame: k.frame,
                                    value: k.value,
                                    interpolation: (k.interpolation
                                        != kreate_scene::animation::Interpolation::default())
                                    .then(|| k.interpolation.tag().to_string()),
                                    selected: k.selected,
                                    slope: k.slope,
                                })
                                .collect(),
                        })
                    })
                    .collect(),
            })
            .collect();
        scene.playback = (document.playback != kreate_scene::animation::Playback::default()).then(|| {
            let p = document.playback;
            PlaybackRecord { frame: p.frame, start: p.start, end: p.end, fps: p.fps }
        });
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
                    image_uid: m
                        .base_color_image
                        .and_then(|id| document.image_uid(id))
                        .map(|uid| uid.raw()),
                    metallic_roughness_uid: m
                        .metallic_roughness_image
                        .and_then(|id| document.image_uid(id))
                        .map(|uid| uid.raw()),
                    normal_uid: m
                        .normal_image
                        .and_then(|id| document.image_uid(id))
                        .map(|uid| uid.raw()),
                    normal_strength: (m.normal_strength != 1.0).then_some(m.normal_strength),
                    graph: m.graph.as_ref().map(|g| graph_record(g, document)),
                    action: m.action.and_then(|a| action_index.get(&a).copied()),
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
            .map(|(_, light)| {
                let mut record = light_record(light);
                record.action = light.action.and_then(|a| action_index.get(&a).copied());
                record
            })
            .collect();

        // Before the worlds, because a world's background names an image and
        // the record has to exist for the name to mean anything. The document's
        // own order is the archive's, the same rule as every other datablock.
        scene.images = document
            .images()
            .map(|(_, image)| ImageRecord {
                name: image.name.clone(),
                uid: image.uid.raw(),
                width: image.width(),
                height: image.height(),
                format: image.format.id().to_string(),
                color_space: image.color_space.id().to_string(),
                path: match &image.source {
                    kreate_scene::Source::File(path) => Some(path.to_string_lossy().into_owned()),
                    _ => None,
                },
                // Every image that holds anything. See `ImageRecord::packed`
                // for why that and not something cleverer.
                packed: image.resident_tiles() > 0,
                extra: BTreeMap::new(),
            })
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
                action: object.action.and_then(|a| action_index.get(&a).copied()),
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

        // STORED like the meshes, and for the same reason: a deflated entry
        // cannot be memory mapped, and a tile being already the shape a sampler
        // addresses is the whole of what `.ktile` buys.
        for (slot, (_, image)) in document.images().enumerate() {
            if image.resident_tiles() == 0 {
                continue;
            }
            zip.start_file(format!("{IMAGE_DIR}{}.ktile", mesh_key(slot)), binary)?;
            zip.write_all(&crate::ktile::write(image))?;
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
    let mut image_tiles: BTreeMap<String, Vec<u8>> = BTreeMap::new();
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
            let mut loaded = crate::kmesh::read(&bytes).map_err(|source| ContainerError::Mesh {
                key: key.to_string(),
                source,
            })?;
            rename_legacy_uv(&mut loaded.mesh);
            meshes.insert(key.to_string(), loaded.mesh);
        } else if let Some(key) = name
            .strip_prefix(IMAGE_DIR)
            .and_then(|k| k.strip_suffix(".ktile"))
        {
            // Kept as bytes and decoded beside its record, because the record
            // is what says whose pixels these are.
            image_tiles.insert(key.to_string(), bytes);
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
            auto_exposure: record.auto_exposure,
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

    // Before anything that can name an image: a world's background and now a
    // material's base colour both resolve a uid against the document, and a
    // resolution that runs first finds nothing and silently falls back. The
    // world's has been written since minor 2 and did not resolve until minor 4,
    // because until then there was nothing to resolve to.
    for (slot, record) in scene.images.iter().enumerate() {
        let key = mesh_key(slot);
        let mut image = match image_tiles.get(&key) {
            Some(bytes) => crate::ktile::read(bytes, &record.name)
                .map_err(|source| ContainerError::Image { key, source })?,
            // Declared and empty, which is a real state: a 16K texture nobody
            // has painted costs its dimensions and a vector of absences. Also
            // what a record whose tile file went missing comes back as — the
            // image is still in the document and still named, so what breaks is
            // the picture and not the file.
            None => kreate_scene::Image::try_new(
                record.name.clone(),
                record.width,
                record.height,
                kreate_scene::PixelFormat::from_id(&record.format).unwrap_or_default(),
            )
            .ok_or(ContainerError::Image {
                key,
                source: crate::ktile::KtileError::ImplausibleSize {
                    width: record.width,
                    height: record.height,
                },
            })?,
        };
        // From the record and not from the tile file, which holds neither: an
        // image's space and where it came from are about the image, not in it.
        // An unknown space falls back to sRGB rather than failing the load, the
        // same judgement as an unknown projection — a texture decoded wrongly is
        // worth opening, a document that will not open is not.
        image.color_space =
            kreate_scene::ColorSpace::from_id(&record.color_space).unwrap_or_default();
        image.source = match &record.path {
            Some(path) => kreate_scene::Source::File(std::path::PathBuf::from(path)),
            None => kreate_scene::Source::Generated,
        };
        document.restore_image(image, kreate_scene::Uid::from_raw(record.uid));
    }

    // Actions first: objects, materials and lights all name them by position.
    let action_ids: Vec<kreate_scene::ActionId> = scene
        .actions
        .iter()
        .map(|record| {
            let animates = record
                .animates
                .as_deref()
                .and_then(kreate_reflect_kind)
                .unwrap_or(kreate_scene::DatablockKind::Object);
            let mut action = kreate_scene::animation::Action::for_kind(record.name.clone(), animates);
            for curve in &record.curves {
                let mut made = kreate_scene::animation::FCurve::new(kreate_scene::PropertyPath::component(
                    animates,
                    kreate_scene::Uid::NONE,
                    &curve.property,
                    curve.index,
                ));
                made.keys = curve
                    .keys
                    .iter()
                    .map(|k| kreate_scene::animation::Keyframe {
                        frame: k.frame,
                        value: k.value,
                        // An interpolation this build does not know is the
                        // default: the key keeps its place and value, which
                        // are what a curve is.
                        interpolation: k
                            .interpolation
                            .as_deref()
                            .and_then(kreate_scene::animation::Interpolation::from_tag)
                            .unwrap_or_default(),
                        selected: k.selected,
                        slope: k.slope.filter(|s| s.is_finite()),
                    })
                    .collect();
                made.normalise();
                action.curves.push(made);
            }
            document.restore_action(action, kreate_scene::Uid::from_raw(record.uid))
        })
        .collect();
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
            // A uid naming an image the file does not carry leaves the texture
            // off rather than failing the load — the same judgement the world's
            // background makes, and for the same reason: a missing texture
            // should read as a missing texture.
            base_color_image: record
                .image_uid
                .and_then(|raw| document.image_by_uid(kreate_scene::Uid::from_raw(raw))),
            metallic_roughness_image: record
                .metallic_roughness_uid
                .and_then(|raw| document.image_by_uid(kreate_scene::Uid::from_raw(raw))),
            normal_image: record
                .normal_uid
                .and_then(|raw| document.image_by_uid(kreate_scene::Uid::from_raw(raw))),
            normal_strength: record.normal_strength.unwrap_or(1.0),
            graph: record
                .graph
                .as_ref()
                .and_then(|g| graph_from_record(g, &document)),
            action: record.action.and_then(|i| action_ids.get(i).copied()),
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
        let mut light = light_from_record(record);
        light.action = record.action.and_then(|i| action_ids.get(i).copied());
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

    if let Some(p) = scene.playback {
        // Refused rather than trusted: an fps of zero divides every playback
        // step by nothing, and a file can say anything.
        if p.fps.is_finite() && p.fps > 0.0 && p.frame.is_finite() {
            document.playback = kreate_scene::animation::Playback {
                frame: p.frame,
                start: p.start,
                end: p.end.max(p.start),
                fps: p.fps,
            };
        }
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
        object.action = record.action.and_then(|i| action_ids.get(i).copied());
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
/// The old name for the UV layer, from before there was one name for it.
const LEGACY_UV: &str = "uv_map";

/// Renames a mesh's UV layer to the one name it has.
///
/// **A file could carry either, and the two never met.** The importers wrote
/// `uv_map` and the UV editor and the unwrapper read `names::UV`, so a model
/// imported from glTF had coordinates the editor could not see, and coordinates
/// unwrapped in Kreate did not export. Both halves worked and neither could
/// reach the other.
///
/// Here rather than in [`crate::migrate`] because that chain rewrites
/// `scene.json`, and a UV layer lives in the binary mesh beside the positions.
/// The chain is the right place for a change to the *scene's* shape; this is a
/// change to a mesh's, and putting a JSON migration in the list that silently
/// did nothing would be worse than not having one.
///
/// Renamed and not merged: a mesh with both is one somebody built with a layer
/// actually called `uv_map`, and taking theirs away would be worse than leaving
/// it where they put it.
fn rename_legacy_uv(mesh: &mut MeshData) {
    use kreate_mesh::attribute::names;
    let attrs = mesh.attributes();
    if attrs.get(names::UV).is_some() || attrs.get(LEGACY_UV).is_none() {
        return;
    }
    if let Some(layer) = mesh.attributes_mut().remove(LEGACY_UV) {
        mesh.attributes_mut()
            .insert(names::UV, layer.domain, layer.data);
    }
}

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

/// A datablock kind from its stable name.
fn kreate_reflect_kind(id: &str) -> Option<kreate_scene::DatablockKind> {
    kreate_scene::DatablockKind::from_id(id)
}

