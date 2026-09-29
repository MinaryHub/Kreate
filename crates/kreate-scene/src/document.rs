//! The document: objects, datablocks, hierarchy and selection.
//!
//! # Typed arenas, not an ECS
//!
//! A modelling document holds thousands to a few million objects, and the
//! dominant cost lives *inside* meshes rather than across objects — so an ECS
//! optimises the wrong axis. Worse, archetype moves fight structural-sharing
//! undo: adding or removing a component would copy component storage on every
//! change, which is precisely what the copy-on-write design exists to avoid.
//! And what is needed here is a *serialisable, versioned document with stable
//! ids and referential integrity*, which is a different thing from an execution
//! layout.
//!
//! # Datablock indirection from day one
//!
//! An object references a mesh by id rather than owning it. That makes linked
//! duplicates and true instancing free, and it is effectively impossible to
//! retrofit once a couple of hundred call sites assume `object.mesh` is owned.
//!
//! # Copy-on-write
//!
//! Cloning a `Document` is a handful of refcount bumps: the arenas hold `Arc`
//! datablocks, and each `MeshData` in turn holds one `Arc` per attribute array.
//! Moving fifty vertices on a million-vertex mesh copies the position array and
//! nothing else. That is what makes undo snapshots affordable, gives the
//! renderer a lock-free immutable view, and lets autosave serialise on a worker
//! thread without stalling a frame.

use std::sync::Arc;

use kreate_math::{DAffine3, DTransform, DVec3};
use kreate_mesh::arena::Arena;
use kreate_mesh::{define_id, MeshData};
use kreate_reflect::{DatablockKind, Uid};

use crate::camera::Camera;
use crate::image::Image;
use crate::light::{Light, World};

define_id! {
    /// An object in the scene graph.
    pub struct ObjectId;
}
define_id! {
    /// A mesh datablock. Several objects may share one.
    pub struct MeshId;
}
define_id! {
    /// A camera datablock.
    ///
    /// Distinct from `ObjectId`: the object is *where* the camera is and the
    /// datablock is *what* it is. Two objects can share one lens, which is
    /// what a stereo pair or a set of matched angles wants, and what §1.4's
    /// argument for meshes says about every datablock.
    pub struct CameraId;
}

kreate_mesh::define_id! {
    /// An image datablock.
    ///
    /// Several materials share one texture routinely, which is the ordinary
    /// case for datablock indirection rather than an exotic one.
    pub struct ImageId;
}

kreate_mesh::define_id! {
    /// A material datablock.
    pub struct MaterialId;
}

define_id! {
    /// A light datablock.
    ///
    /// The same separation as a camera's, and §2.2 is explicit about it: a
    /// hundred linked copies of a lamp share one bulb, so the object is *where*
    /// the light is and the datablock is *what* it is. Changing a studio's
    /// colour is then one edit rather than a hundred.
    pub struct LightId;
}

define_id! {
    /// A world datablock: what light arrives from no object in particular.
    ///
    /// An arena and not a field on the document, because a studio setup is a
    /// thing to share between files — which is what library linking is for and
    /// what a bare field could not be reached by.
    pub struct WorldId;
}
define_id! {
    /// An action datablock: a set of curves, played by whatever is given it.
    ///
    /// A datablock rather than a field on the object for §2.4's reason: a walk
    /// cycle is shared between characters, and an animation that lived inside
    /// one object could not be.
    pub struct ActionId;
}

define_id! {
    /// A collection: a named grouping of objects, orthogonal to the parent
    /// hierarchy.
    pub struct CollectionId;
}

/// What an object *is*.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectData {
    /// A transform with no geometry. Useful as a pivot or a rig control.
    Empty,
    Mesh(MeshId),
    Camera(CameraId),
    Light(LightId),
}

/// Whether a datablock is owned by this file or linked from another.
///
/// Modelled from day one even though library linking is not yet exposed:
/// retrofitting it means auditing every mutation site for "is this allowed to be
/// written", which is the kind of change that never happens cleanly.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Linkage {
    #[default]
    Local,
    /// Read-only until explicitly made local.
    Linked { library: String, name: String },
}

impl Linkage {
    pub fn is_editable(&self) -> bool {
        matches!(self, Linkage::Local)
    }
}

/// One object.
#[derive(Clone, Debug)]
pub struct Object {
    pub name: String,
    pub parent: Option<ObjectId>,
    /// Inverse of the parent's world matrix at the moment of parenting.
    ///
    /// Without this, parenting an object teleports it: its local transform would
    /// suddenly be interpreted relative to the parent. Storing the inverse keeps
    /// the object exactly where it was.
    pub parent_inverse: DAffine3,
    pub transform: DTransform,
    pub data: ObjectData,
    pub visible: bool,
    pub selectable: bool,
    pub linkage: Linkage,
    /// Per-slot materials that win over the mesh's. Usually empty.
    ///
    /// See [`MaterialOverrides`] and [`Document::material_at`].
    pub material_overrides: MaterialOverrides,
    /// This object's identity, for as long as the document exists and across
    /// saves.
    ///
    /// Assigned by [`Document::add_object`] and never by a caller — which is
    /// what makes a duplicate a new object rather than a second handle on the
    /// same one. `Uid::NONE` until then.
    ///
    /// Not the same thing as an `ObjectId`: that is an arena slot with a
    /// generation only in debug builds, so in a release build a reused slot is
    /// indistinguishable from the object that used to be there. A curve keyed
    /// on one would start driving whatever was created next.
    pub uid: Uid,
    /// Applied in order to produce the geometry that is drawn.
    ///
    /// The document holds the cage the user edits; what appears on screen is the
    /// result of running this stack over it. See [`crate::modifier`] for why the
    /// evaluated result is cached outside the document rather than here.
    pub modifiers: Vec<crate::modifier::ModifierInstance>,
    /// The action this object plays, if it is animated.
    pub action: Option<ActionId>,
}

impl Object {
    pub fn new(name: impl Into<String>, data: ObjectData) -> Self {
        Self {
            name: name.into(),
            parent: None,
            parent_inverse: DAffine3::IDENTITY,
            transform: DTransform::IDENTITY,
            data,
            visible: true,
            selectable: true,
            linkage: Linkage::Local,
            // Empty: the mesh's slots answer unless something overrides one.
            material_overrides: MaterialOverrides::new(),
            uid: Uid::NONE,
            modifiers: Vec::new(),
            action: None,
        }
    }

    pub fn mesh(&self) -> Option<MeshId> {
        match self.data {
            ObjectData::Mesh(id) => Some(id),
            _ => None,
        }
    }
}

/// What an object overrides about its mesh's materials.
///
/// Empty for almost every object, which is why it is a `Vec` and not a map:
/// the common case costs a pointer and no allocation, and the uncommon one is
/// a handful of slots.
///
/// Modelled now rather than later because the alternative is auditing every
/// material lookup afterwards. Blender has exactly this — a slot links either
/// to the data or to the object — and the case it exists for is ordinary: two
/// linked copies of a chair, one of them upholstered differently.
pub type MaterialOverrides = Vec<Option<MaterialId>>;

/// A material, deliberately a "standard surface" and nothing more.
///
/// Its parameters map near-1:1 onto both glTF metal-rough and
/// `UsdPreviewSurface`, which were designed to match each other. Arbitrary
/// shader graphs are a separate problem and are never imported.
#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    /// The base colour: a reflectance, bounded by one.
    ///
    /// Typed rather than `[f32; 4]`, which is what this was before §2.1's
    /// colour work: a bare array says nothing about which space it is in, and
    /// the two colour bugs that discipline was introduced after were both a
    /// value in the wrong one. For a dielectric this is the diffuse albedo and
    /// for a metal the specular colour — `metallic` says which, which is the
    /// whole trick of the metal-roughness parameterisation.
    ///
    /// glTF's `baseColorFactor` is linear too, so the boundary is a copy and
    /// not a decode.
    pub base_color: kreate_math::SceneRgb,
    /// Opacity. Separate from the colour because it is not one: a transfer
    /// function applies to the three and never to this.
    pub alpha: f32,
    pub metallic: f32,
    pub roughness: f32,
    /// A texture multiplying the base colour, if one is assigned.
    ///
    /// A field beside the numbers rather than a node the user wired, for the
    /// reason v1 §5.6 gave the standard surface a fast path: the graph is the
    /// canonical form and these are a shorter route to the same model, which is
    /// what [`crate::shader::ShaderGraph::for_material`] exists to make
    /// checkable. When there is a node editor to build one in, a material will
    /// carry its own graph and this becomes the thing that graph is built from
    /// on the way in.
    pub base_color_image: Option<ImageId>,
    /// glTF's `metallicRoughnessTexture`: roughness in green, metallic in blue.
    ///
    /// Each channel *multiplies* its dial, the way the base colour map
    /// multiplies the colour — so a map assigned over a roughness of 1 means
    /// what the map says, and the dial goes on meaning something afterwards.
    /// Red is left alone: glTF packs occlusion there, and a map that serves
    /// both is the ordinary case.
    pub metallic_roughness_image: Option<ImageId>,
    /// A tangent-space normal map, +Y up (the OpenGL and glTF convention).
    pub normal_image: Option<ImageId>,
    /// How far the normal map tilts the surface. glTF's `normalTexture.scale`:
    /// it scales the map's X and Y before the vector is renormalised, so zero
    /// is a flat surface and one is the map as painted.
    pub normal_strength: f32,
    /// The material's own shading graph, once somebody has asked for one.
    ///
    /// `None` is the fast path v1 §5.6 describes: the flat fields above *are*
    /// the material, and [`crate::shader::ShaderGraph::for_material`] reaches
    /// the same model by a shorter route. `Some` is Blender's "Use Nodes": the
    /// graph is authoritative and the flat fields are what the material was
    /// before, kept so that turning nodes off gives it back.
    ///
    /// Inline rather than a datablock of its own. A graph shared between
    /// materials is Blender's node *group*, which is a different feature and
    /// wants its own identity; a material's tree belongs to the material.
    pub graph: Option<crate::shader::ShaderGraph>,
    /// The action animating this material's values, if any.
    pub action: Option<ActionId>,
    pub linkage: Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Material".into(),
            base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            alpha: 1.0,
            metallic: 0.0,
            roughness: 0.5,
            base_color_image: None,
            metallic_roughness_image: None,
            normal_image: None,
            normal_strength: 1.0,
            graph: None,
            action: None,
            linkage: Linkage::Local,
            uid: Uid::NONE,
        }
    }
}

/// Which of a material's images: the texture slots a standard surface has.
///
/// One enum rather than three fields read by name everywhere, because every
/// place that touches one touches all of them — the file, the clipboard, the
/// operator, the panel — and a map added later must not be one those places
/// forget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialMap {
    BaseColor,
    MetallicRoughness,
    Normal,
}

impl MaterialMap {
    /// Every map, in the order the panel lists them and the operator numbers
    /// them.
    pub const ALL: [MaterialMap; 3] = [
        MaterialMap::BaseColor,
        MaterialMap::MetallicRoughness,
        MaterialMap::Normal,
    ];

    /// A stable name for scripts and files. Not the discriminant, for the
    /// reason `NodeKind::tag` gives.
    pub fn tag(self) -> &'static str {
        match self {
            MaterialMap::BaseColor => "base_color",
            MaterialMap::MetallicRoughness => "metallic_roughness",
            MaterialMap::Normal => "normal",
        }
    }

    /// What the numbers in an image in this slot are.
    ///
    /// **Declared by the slot, not guessed from the file** (§2.1). A base colour
    /// is a reflectance somebody chose by eye, which is what an eight-bit file
    /// holds; a roughness or a normal is a measurement, and running it through
    /// sRGB's curve moves every value but the two ends.
    pub fn color_space(self) -> crate::image::ColorSpace {
        match self {
            MaterialMap::BaseColor => crate::image::ColorSpace::Srgb,
            MaterialMap::MetallicRoughness | MaterialMap::Normal => {
                crate::image::ColorSpace::Raw
            }
        }
    }
}

impl Material {
    /// The graph this material shades with: its own, or the one its flat
    /// fields are.
    ///
    /// The one question every consumer asks — the compiler, the exporter, the
    /// node editor — answered in one place, so none of them can quietly read
    /// the flat fields of a material whose graph says something else.
    pub fn shading_graph(&self) -> crate::shader::ShaderGraph {
        match &self.graph {
            Some(graph) => graph.clone(),
            None => crate::shader::ShaderGraph::for_material(self),
        }
    }

    /// Every image the material reaches, through its graph or its maps.
    pub fn images(&self) -> Vec<ImageId> {
        match &self.graph {
            Some(graph) => graph.images(),
            None => MaterialMap::ALL.iter().filter_map(|m| self.map(*m)).collect(),
        }
    }

    /// The image in one of the material's texture slots.
    pub fn map(&self, map: MaterialMap) -> Option<ImageId> {
        match map {
            MaterialMap::BaseColor => self.base_color_image,
            MaterialMap::MetallicRoughness => self.metallic_roughness_image,
            MaterialMap::Normal => self.normal_image,
        }
    }

    /// The slot itself, to assign or clear.
    pub fn map_mut(&mut self, map: MaterialMap) -> &mut Option<ImageId> {
        match map {
            MaterialMap::BaseColor => &mut self.base_color_image,
            MaterialMap::MetallicRoughness => &mut self.metallic_roughness_image,
            MaterialMap::Normal => &mut self.normal_image,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Collection {
    pub name: String,
    pub objects: Vec<ObjectId>,
}

/// What is currently selected.
///
/// Object-level selection lives here; *mesh element* selection lives in
/// attribute layers on the mesh itself, so it round-trips through edit mode and
/// shares structurally under undo with no extra machinery.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    pub objects: Vec<ObjectId>,
    /// The object a modal operator acts on and the properties panel shows.
    /// Always also present in `objects` when set.
    pub active: Option<ObjectId>,
}

impl Selection {
    pub fn clear(&mut self) {
        self.objects.clear();
        self.active = None;
    }

    pub fn contains(&self, id: ObjectId) -> bool {
        self.objects.contains(&id)
    }

    pub fn add(&mut self, id: ObjectId) {
        if !self.contains(id) {
            self.objects.push(id);
        }
        self.active = Some(id);
    }

    pub fn remove(&mut self, id: ObjectId) {
        self.objects.retain(|o| *o != id);
        if self.active == Some(id) {
            self.active = self.objects.last().copied();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

/// Why a reparent was refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ReparentError {
    #[error("object does not exist")]
    NoSuchObject,
    #[error("parent does not exist")]
    NoSuchParent,
    #[error("that would create a cycle in the hierarchy")]
    WouldCycle,
}

/// A whole document.
#[derive(Clone, Debug, Default)]
pub struct Document {
    objects: Arena<ObjectId, Object>,
    /// Meshes are `Arc`ed so a document clone is a refcount bump. The `MeshData`
    /// inside is itself copy-on-write per attribute array.
    meshes: Arena<MeshId, MeshEntry>,
    materials: Arena<MaterialId, Arc<Material>>,
    cameras: Arena<CameraId, Arc<Camera>>,
    /// Images are `Arc`ed like meshes, and each one is itself tiled with an
    /// `Arc` per tile — so a document clone is a refcount bump and a brush
    /// stroke copies the tiles it touched. See [`crate::image`].
    images: Arena<ImageId, Arc<Image>>,
    lights: Arena<LightId, Arc<Light>>,
    worlds: Arena<WorldId, Arc<World>>,
    actions: Arena<ActionId, Arc<crate::animation::Action>>,
    /// The scene's clock: the frame it is on, the range it plays, how fast.
    pub playback: crate::animation::Playback,
    /// Which world is lighting the scene.
    ///
    /// A pointer and not the world itself, so that swapping a lighting setup is
    /// one assignment and the setup that was there is still in the file. `None`
    /// is a scene lit by nothing but its own lights, which is a real thing to
    /// want and not a missing value to paper over.
    active_world: Option<WorldId>,
    collections: Arena<CollectionId, Collection>,
    /// Objects with no parent, in user-visible order.
    roots: Vec<ObjectId>,
    pub selection: Selection,
    /// Bumped on every mutation. A cheap "has anything changed" check for
    /// callers that cannot use `Arc::ptr_eq` on a specific array.
    version: u64,
    /// The last identity handed out for each kind of datablock.
    ///
    /// Travels with the document, which is what makes undo and redo
    /// identity-preserving: undoing a creation restores the counter, so redoing
    /// it hands out the *same* id and a curve pointing at the object is still
    /// pointing at it. A counter living outside the snapshot would give the
    /// redone object a new identity and orphan its animation.
    uids: UidCounters,
}

/// A mesh datablock: its geometry and its identity.
///
/// The identity sits beside the data rather than in a table alongside the
/// arena, because a table has to be kept in step and this cannot drift.
#[derive(Clone, Debug)]
struct MeshEntry {
    uid: Uid,
    data: Arc<MeshData>,
    /// The material slots this mesh's faces index into.
    ///
    /// A face's `material_index` attribute is a slot number, not a document
    /// id, and this is the list it numbers. Two reasons it has to be that way
    /// round: a `MeshData` has to be transferable between documents, where an
    /// arena id means nothing; and `edit::join` already offsets each part's
    /// indices as it merges, which is only coherent against a per-mesh list.
    ///
    /// On the *mesh* rather than the object because a linked copy shares the
    /// mesh (§1.4), so it shares the slots — assigning a material to one copy
    /// assigns it to all of them, which is what an artist means by "the
    /// material of this chair". [`Object::material_overrides`] is how one copy
    /// differs.
    ///
    /// `None` in a slot is a slot nothing is assigned to, which is the state
    /// of every freshly added cube.
    slots: Vec<Option<MaterialId>>,
}

/// The last identity issued for each kind.
///
/// One counter per kind, so the numbers stay small and a person reading a scene
/// file can match a curve to its target by eye. [`DatablockKind`] is what keeps
/// object 1 and mesh 1 apart.
#[derive(Clone, Debug, Default, PartialEq)]
struct UidCounters {
    object: Uid,
    mesh: Uid,
    camera: Uid,
    image: Uid,
    light: Uid,
    world: Uid,
    material: Uid,
    action: Uid,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    /// A counter that increases on every mutation.
    ///
    /// **Monotonic, including across undo.** A caching consumer — the GPU mesh
    /// cache, an evaluation cache, an autosave timer — compares it against the
    /// version it last saw, so a version that can go *backwards* is worse than no
    /// version at all: restoring a snapshot and re-running an operator lands on
    /// the same number the cache already holds, and the cache concludes nothing
    /// changed while the document underneath it is different.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Replaces this document's contents with `other`, keeping the version
    /// moving forward.
    ///
    /// This is how undo puts a snapshot back. It is not `*self = other`, because
    /// that would carry the snapshot's older version along with it — see
    /// [`Document::version`].
    pub fn adopt(&mut self, other: Document) {
        let next = self.version.max(other.version) + 1;
        *self = other;
        self.version = next;
    }

    #[inline]
    fn touch(&mut self) {
        self.version += 1;
    }

    // --- objects ------------------------------------------------------------

    pub fn add_object(&mut self, object: Object) -> ObjectId {
        let uid = self.issue(DatablockKind::Object);
        self.insert_object(object, uid)
    }

    /// Adds an object with an identity it already had.
    ///
    /// For a loader restoring a scene, and for nothing else: ordinary creation
    /// goes through [`Document::add_object`], which is what stops a duplicate
    /// silently inheriting the original's identity. The counter is advanced
    /// past `uid` so that the next new object cannot collide with anything the
    /// file already used — including ids the file skipped, which is the case a
    /// naive "count what we loaded" would get wrong.
    ///
    /// `Uid::NONE` issues a fresh identity instead of storing it. That is what
    /// a hand-edited or truncated file can present, and storing it would put a
    /// datablock with no identity into the document — which every lookup would
    /// then have to guard against, forever, for one bad file. Refusing it here
    /// means **no datablock in a document holds `Uid::NONE`**, and that is an
    /// invariant rather than a hope.
    pub fn restore_object(&mut self, object: Object, uid: Uid) -> ObjectId {
        if uid.is_none() {
            return self.add_object(object);
        }
        self.uids.object = self.uids.object.max(uid);
        self.insert_object(object, uid)
    }

    fn insert_object(&mut self, mut object: Object, uid: Uid) -> ObjectId {
        object.uid = uid;
        let has_parent = object.parent.is_some();
        let id = self.objects.insert(object);
        if !has_parent {
            self.roots.push(id);
        }
        self.touch();
        id
    }

    /// The next identity for a kind of datablock.
    fn issue(&mut self, kind: DatablockKind) -> Uid {
        let counter = match kind {
            DatablockKind::Object => &mut self.uids.object,
            DatablockKind::Mesh => &mut self.uids.mesh,
            DatablockKind::Camera => &mut self.uids.camera,
            DatablockKind::Image => &mut self.uids.image,
            DatablockKind::Material => &mut self.uids.material,
            DatablockKind::Light => &mut self.uids.light,
            DatablockKind::World => &mut self.uids.world,
            DatablockKind::Action => &mut self.uids.action,
            // Nothing else is a datablock in this document yet. Reaching here
            // means a kind was added to the enum and not to the counters, and
            // handing back the unassigned id makes that a resolvable path to
            // nothing rather than a silent collision with object 1.
            _ => return Uid::NONE,
        };
        *counter = counter.next();
        *counter
    }

    /// This object's identity, for a property path.
    pub fn object_uid(&self, id: ObjectId) -> Option<Uid> {
        self.objects.get(id).map(|o| o.uid)
    }

    /// The object with this identity, if it is still here.
    ///
    /// A linear scan. The alternative is a map to keep in step, and the callers
    /// are a curve resolving once per frame and a file loader resolving once
    /// per link — at the scale of a scene's object count that is nothing, and
    /// it becomes a map the moment a measurement says so.
    /// `Uid::NONE` finds nothing, and needs no special case to do so: no
    /// datablock in a document holds it — see [`Document::restore_object`] —
    /// so the scan simply matches nothing. A guard here would be a branch
    /// nothing can take, which is worse than none: a reader would take it for
    /// evidence that the invariant is not relied on.
    pub fn object_by_uid(&self, uid: Uid) -> Option<ObjectId> {
        self.objects
            .iter()
            .find(|(_, o)| o.uid == uid)
            .map(|(id, _)| id)
    }

    /// This mesh datablock's identity.
    pub fn mesh_uid(&self, id: MeshId) -> Option<Uid> {
        self.meshes.get(id).map(|m| m.uid)
    }

    /// The mesh datablock with this identity, if it is still here.
    pub fn mesh_by_uid(&self, uid: Uid) -> Option<MeshId> {
        self.meshes
            .iter()
            .find(|(_, m)| m.uid == uid)
            .map(|(id, _)| id)
    }

    pub fn object(&self, id: ObjectId) -> Option<&Object> {
        self.objects.get(id)
    }

    /// Mutable access, bumping the version.
    ///
    /// Deliberately not an `IndexMut`: every write should be visible as a write.
    pub fn object_mut(&mut self, id: ObjectId) -> Option<&mut Object> {
        let found = self.objects.get_mut(id).is_some();
        if found {
            self.version += 1;
        }
        self.objects.get_mut(id)
    }

    pub fn objects(&self) -> impl Iterator<Item = (ObjectId, &Object)> {
        self.objects.iter()
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    #[inline]
    pub fn roots(&self) -> &[ObjectId] {
        &self.roots
    }

    /// Removes an object and reparents its children to its own parent.
    ///
    /// Silently orphaning children would leave the hierarchy referring to a dead
    /// id; deleting them recursively is a different operation and should be the
    /// caller's explicit choice.
    pub fn remove_object(&mut self, id: ObjectId) -> Option<Object> {
        let removed = self.objects.remove(id)?;
        let children: Vec<ObjectId> = self
            .objects
            .iter()
            .filter(|(_, o)| o.parent == Some(id))
            .map(|(cid, _)| cid)
            .collect();
        for child in children {
            if let Some(c) = self.objects.get_mut(child) {
                c.parent = removed.parent;
            }
            if removed.parent.is_none() {
                self.roots.push(child);
            }
        }
        self.roots.retain(|r| *r != id);
        self.selection.remove(id);
        self.touch();
        Some(removed)
    }

    /// Reparents, keeping the object's world transform.
    ///
    /// Refuses to create a cycle. A cycle here is not a cosmetic problem: every
    /// world-transform walk would recurse forever.
    pub fn set_parent(
        &mut self,
        id: ObjectId,
        parent: Option<ObjectId>,
    ) -> Result<(), ReparentError> {
        if self.objects.get(id).is_none() {
            return Err(ReparentError::NoSuchObject);
        }
        if let Some(p) = parent {
            if self.objects.get(p).is_none() {
                return Err(ReparentError::NoSuchParent);
            }
            if p == id || self.is_ancestor(id, p) {
                return Err(ReparentError::WouldCycle);
            }
        }

        let parent_inverse = match parent {
            Some(p) => self.world_transform(p).inverse(),
            None => DAffine3::IDENTITY,
        };
        let was_root = self.objects[id].parent.is_none();
        {
            let o = &mut self.objects[id];
            o.parent = parent;
            o.parent_inverse = parent_inverse;
        }
        match (was_root, parent.is_none()) {
            (true, false) => self.roots.retain(|r| *r != id),
            (false, true) => self.roots.push(id),
            _ => {}
        }
        self.touch();
        Ok(())
    }

    /// Whether `ancestor` is somewhere above `id` in the hierarchy.
    pub fn is_ancestor(&self, ancestor: ObjectId, id: ObjectId) -> bool {
        let mut current = self.objects.get(id).and_then(|o| o.parent);
        // Bounded by the object count: a malformed document with a pre-existing
        // cycle must not hang the application.
        for _ in 0..=self.objects.slot_count() {
            match current {
                Some(c) if c == ancestor => return true,
                Some(c) => current = self.objects.get(c).and_then(|o| o.parent),
                None => return false,
            }
        }
        false
    }

    /// World matrix, walking the parent chain.
    ///
    /// `parent_world * parent_inverse * local`, so an object stays put when it is
    /// parented and follows the parent thereafter.
    ///
    /// Iterative, not recursive. The recursive form is shorter and overflows the
    /// stack on a deep chain — and a deep chain is not exotic: a rig, a spine, a
    /// length of chain links. A few thousand deep is enough, and the failure is a
    /// hard crash rather than anything a user could act on.
    ///
    /// The walk up is bounded by the arena's slot count, so a document that
    /// already contains a parent cycle returns a wrong answer rather than
    /// hanging. `set_parent` refuses to create one; a corrupt file is the case
    /// this covers.
    pub fn world_transform(&self, id: ObjectId) -> DAffine3 {
        let mut chain: Vec<ObjectId> = Vec::new();
        let mut current = Some(id);
        for _ in 0..=self.objects.slot_count() {
            let Some(next) = current else { break };
            if self.objects.get(next).is_none() {
                break;
            }
            chain.push(next);
            current = self.objects[next].parent;
        }

        // Multiplied from the root down, which is the order the recursive form
        // effectively unwound in.
        let mut world = DAffine3::IDENTITY;
        for id in chain.iter().rev() {
            let object = &self.objects[*id];
            let local = object.transform.to_affine();
            world = if object.parent.is_some() {
                world * object.parent_inverse * local
            } else {
                local
            };
        }
        world
    }

    /// Depth-first traversal from the roots, parents before children.
    ///
    /// The children are indexed in one pass first. Searching every object for
    /// the children of each node instead is the obvious way to write this and is
    /// quadratic: at a hundred thousand objects it is ten billion comparisons,
    /// and the outliner asks for this whenever the scene changes. The cost is
    /// invisible at twenty objects, which is why it needs saying rather than
    /// measuring.
    pub fn hierarchy(&self) -> Vec<(ObjectId, usize)> {
        use kreate_mesh::arena::ArenaIndex as _;

        let slots = self.objects.slot_count();
        // Flat CSR rather than a `Vec` per parent: one allocation instead of one
        // per object, and the counting pass is the same walk either way.
        let mut counts = vec![0u32; slots + 1];
        for (_, object) in self.objects.iter() {
            if let Some(parent) = object.parent {
                counts[parent.index() as usize + 1] += 1;
            }
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let mut children = vec![None; offsets[slots] as usize];
        for (id, object) in self.objects.iter() {
            if let Some(parent) = object.parent {
                let slot = &mut cursor[parent.index() as usize];
                children[*slot as usize] = Some(id);
                *slot += 1;
            }
        }
        let children_of = |parent: ObjectId| -> &[Option<ObjectId>] {
            let i = parent.index() as usize;
            &children[offsets[i] as usize..offsets[i + 1] as usize]
        };

        let mut out = Vec::with_capacity(self.objects.len());
        let mut stack: Vec<(ObjectId, usize)> =
            self.roots.iter().rev().map(|id| (*id, 0)).collect();
        while let Some((id, depth)) = stack.pop() {
            if self.objects.get(id).is_none() {
                continue;
            }
            out.push((id, depth));
            // Reversed onto the stack so they come back off in arena order,
            // which is what makes the outliner's row order reproducible.
            for child in children_of(id).iter().rev().flatten() {
                stack.push((*child, depth + 1));
            }
            // A pre-existing cycle would otherwise loop forever here.
            if out.len() > slots {
                break;
            }
        }
        out
    }

    // --- datablocks ---------------------------------------------------------

    pub fn add_mesh(&mut self, mesh: MeshData) -> MeshId {
        let uid = self.issue(DatablockKind::Mesh);
        self.touch();
        self.meshes.insert(MeshEntry {
            uid,
            data: Arc::new(mesh),
            // A mesh arrives with no material slots. A freshly added cube has
            // no materials, which is a state and not an omission.
            slots: Vec::new(),
        })
    }

    /// Adds a mesh with an identity it already had. See
    /// [`Document::restore_object`], including what `Uid::NONE` does.
    pub fn restore_mesh(&mut self, mesh: MeshData, uid: Uid) -> MeshId {
        if uid.is_none() {
            return self.add_mesh(mesh);
        }
        self.uids.mesh = self.uids.mesh.max(uid);
        self.touch();
        self.meshes.insert(MeshEntry {
            uid,
            data: Arc::new(mesh),
            // A mesh arrives with no material slots. A freshly added cube has
            // no materials, which is a state and not an omission.
            slots: Vec::new(),
        })
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshData> {
        self.meshes.get(id).map(|m| m.data.as_ref())
    }

    /// Copy-on-write access to a mesh.
    ///
    /// Clones the `MeshData` only if it is shared with a snapshot — and even then
    /// the clone is a set of refcount bumps, because `MeshData`'s own arrays are
    /// individually `Arc`ed.
    pub fn mesh_mut(&mut self, id: MeshId) -> Option<&mut MeshData> {
        self.version += 1;
        self.meshes.get_mut(id).map(|m| Arc::make_mut(&mut m.data))
    }

    /// The mesh datablock as a shared handle.
    ///
    /// A consumer that keeps a clone of this can tell whether the mesh changed
    /// with `Arc::ptr_eq` — free, and exact, with no invalidation discipline to
    /// forget. Holding the clone is what makes it work: `Arc::make_mut` only
    /// copies when the handle is shared, so a cache that does *not* hold one
    /// would watch the document mutate in place behind an unchanged pointer.
    pub fn mesh_arc(&self, id: MeshId) -> Option<&Arc<MeshData>> {
        self.meshes.get(id).map(|m| &m.data)
    }

    /// Every mesh datablock, in arena slot order.
    ///
    /// Deterministic, which is what lets the debug-build validation hook name
    /// the same datablock on every run.
    pub fn meshes(&self) -> impl Iterator<Item = (MeshId, &MeshData)> + '_ {
        self.meshes.iter().map(|(id, m)| (id, m.data.as_ref()))
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// How many objects reference this mesh.
    ///
    /// Greater than one means a linked duplicate: editing the mesh changes every
    /// user, and the UI must say so before the user is surprised by it.
    pub fn mesh_users(&self, id: MeshId) -> usize {
        self.objects
            .values()
            .filter(|o| o.mesh() == Some(id))
            .count()
    }

    /// How many objects reference each mesh, by mesh slot.
    ///
    /// One pass for the whole scene. Asking [`Document::mesh_users`] per row is
    /// quadratic, and the outliner is exactly the caller that wants it for every
    /// object at once — which is how a scene of a hundred thousand objects turns
    /// a panel into ten billion comparisons.
    pub fn mesh_user_counts(&self) -> Vec<u32> {
        use kreate_mesh::arena::ArenaIndex as _;

        let mut counts = vec![0u32; self.meshes.slot_count()];
        for object in self.objects.values() {
            if let Some(mesh) = object.mesh() {
                if let Some(slot) = counts.get_mut(mesh.index() as usize) {
                    *slot += 1;
                }
            }
        }
        counts
    }

    /// The live handle for an object slot, if that slot is occupied.
    ///
    /// How an index that came from outside — an outliner row, a GPU id buffer, a
    /// script — becomes a handle, in constant time. Scanning for a matching
    /// index is what the outliner used to do, once per key, which made resolving
    /// a large selection quadratic.
    ///
    /// A slot that has been *reused* resolves to whatever occupies it now: the
    /// index carries no generation, so that is the best available answer and the
    /// window between a delete and a click is a frame. A slot that is simply
    /// empty resolves to nothing.
    pub fn object_at_slot(&self, index: u32) -> Option<ObjectId> {
        self.objects.handle_at(index)
    }

    // --- cameras ------------------------------------------------------------

    pub fn add_camera(&mut self, camera: Camera) -> CameraId {
        let uid = self.issue(DatablockKind::Camera);
        self.touch();
        self.cameras.insert(Arc::new(Camera { uid, ..camera }))
    }

    /// Adds a camera with an identity it already had. See
    /// [`Document::restore_object`].
    pub fn restore_camera(&mut self, camera: Camera, uid: Uid) -> CameraId {
        if uid.is_none() {
            return self.add_camera(camera);
        }
        self.uids.camera = self.uids.camera.max(uid);
        self.touch();
        self.cameras.insert(Arc::new(Camera { uid, ..camera }))
    }

    pub fn camera(&self, id: CameraId) -> Option<&Camera> {
        self.cameras.get(id).map(|c| c.as_ref())
    }

    /// Copy-on-write access to a camera.
    pub fn camera_mut(&mut self, id: CameraId) -> Option<&mut Camera> {
        self.touch();
        self.cameras.get_mut(id).map(Arc::make_mut)
    }

    /// Every camera datablock, in arena slot order.
    pub fn cameras(&self) -> impl Iterator<Item = (CameraId, &Camera)> {
        self.cameras.iter().map(|(id, c)| (id, c.as_ref()))
    }

    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    pub fn camera_uid(&self, id: CameraId) -> Option<Uid> {
        self.cameras.get(id).map(|c| c.uid)
    }

    pub fn camera_by_uid(&self, uid: Uid) -> Option<CameraId> {
        self.cameras
            .iter()
            .find(|(_, c)| c.uid == uid)
            .map(|(id, _)| id)
    }

    // --- images -------------------------------------------------------------

    pub fn add_image(&mut self, image: Image) -> ImageId {
        let uid = self.issue(DatablockKind::Image);
        self.touch();
        let mut image = image;
        image.uid = uid;
        self.images.insert(Arc::new(image))
    }

    /// Adds an image with an identity it already had. See
    /// [`Document::restore_object`].
    pub fn restore_image(&mut self, image: Image, uid: Uid) -> ImageId {
        if uid.is_none() {
            return self.add_image(image);
        }
        self.uids.image = self.uids.image.max(uid);
        self.touch();
        let mut image = image;
        image.uid = uid;
        self.images.insert(Arc::new(image))
    }

    pub fn image(&self, id: ImageId) -> Option<&Image> {
        self.images.get(id).map(|i| i.as_ref())
    }

    /// Copy-on-write access to an image.
    ///
    /// The outer `make_mut` copies the image's *structure* — its dimensions and
    /// its vector of tile handles — and not its pixels: the handles are cloned,
    /// so the tiles stay shared until one is written to. Which is what makes a
    /// stroke on a shared texture cost the tiles it touched.
    pub fn image_mut(&mut self, id: ImageId) -> Option<&mut Image> {
        self.touch();
        self.images.get_mut(id).map(Arc::make_mut)
    }

    pub fn images(&self) -> impl Iterator<Item = (ImageId, &Image)> {
        self.images.iter().map(|(id, i)| (id, i.as_ref()))
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn image_uid(&self, id: ImageId) -> Option<Uid> {
        self.images.get(id).map(|i| i.uid)
    }

    pub fn image_by_uid(&self, uid: Uid) -> Option<ImageId> {
        self.images
            .iter()
            .find(|(_, i)| i.uid == uid)
            .map(|(id, _)| id)
    }

    // --- actions ------------------------------------------------------------

    pub fn add_action(&mut self, action: crate::animation::Action) -> ActionId {
        let uid = self.issue(DatablockKind::Action);
        self.touch();
        self.actions.insert(Arc::new(crate::animation::Action { uid, ..action }))
    }

    /// Adds an action with an identity it already had. See
    /// [`Document::restore_object`].
    pub fn restore_action(&mut self, action: crate::animation::Action, uid: Uid) -> ActionId {
        if uid.is_none() {
            return self.add_action(action);
        }
        self.uids.action = self.uids.action.max(uid);
        self.touch();
        self.actions.insert(Arc::new(crate::animation::Action { uid, ..action }))
    }

    pub fn action(&self, id: ActionId) -> Option<&crate::animation::Action> {
        self.actions.get(id).map(|a| a.as_ref())
    }

    /// Copy-on-write access to an action.
    pub fn action_mut(&mut self, id: ActionId) -> Option<&mut crate::animation::Action> {
        self.touch();
        self.actions.get_mut(id).map(Arc::make_mut)
    }

    pub fn actions(&self) -> impl Iterator<Item = (ActionId, &crate::animation::Action)> {
        self.actions.iter().map(|(id, a)| (id, a.as_ref()))
    }

    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    pub fn action_uid(&self, id: ActionId) -> Option<Uid> {
        self.actions.get(id).map(|a| a.uid)
    }

    pub fn action_by_uid(&self, uid: Uid) -> Option<ActionId> {
        self.actions.iter().find(|(_, a)| a.uid == uid).map(|(id, _)| id)
    }

    /// Plays every animated object's action at `frame`, and moves the clock
    /// there.
    ///
    /// Blender's model: an animated property *is* its curve's value at the
    /// current frame, written into the property so that every reader — the
    /// renderer, picking, the panels — sees the animated scene without knowing
    /// animation exists. Not an edit: stepping through the timeline leaves
    /// nothing to undo, and a change to a keyed property that is not keyed is
    /// replaced the next time the frame moves, as it is in Blender.
    ///
    /// Returns how many objects moved, so a caller can skip a redraw when a
    /// frame change animated nothing.
    pub fn animate(&mut self, frame: f64) -> usize {
        self.playback.frame = frame;
        let animated: Vec<(ObjectId, ActionId)> = self
            .objects
            .iter()
            .filter_map(|(id, o)| o.action.map(|a| (id, a)))
            .collect();
        let mut moved = 0;
        // Materials and lights, which play their own actions: the colour of a
        // warning light blinking, a lamp dimming at dusk.
        let materials: Vec<(MaterialId, ActionId)> = self
            .materials
            .iter()
            .filter_map(|(id, m)| m.action.map(|a| (id, a)))
            .collect();
        for (id, action) in materials {
            let Some(action) = self.actions.get(action).cloned() else { continue };
            if let Some(m) = self.materials.get_mut(id).map(Arc::make_mut) {
                let before = m.clone();
                crate::animation::apply_to_material(&action, frame, m);
                moved += usize::from(m.base_color != before.base_color
                    || m.metallic != before.metallic
                    || m.roughness != before.roughness
                    || m.alpha != before.alpha
                    || m.normal_strength != before.normal_strength);
            }
        }
        let lights: Vec<(LightId, ActionId)> = self
            .lights
            .iter()
            .filter_map(|(id, l)| l.action.map(|a| (id, a)))
            .collect();
        for (id, action) in lights {
            let Some(action) = self.actions.get(action).cloned() else { continue };
            if let Some(l) = self.lights.get_mut(id).map(Arc::make_mut) {
                let before = (l.color, l.intensity, l.exposure);
                crate::animation::apply_to_light(&action, frame, l);
                moved += usize::from(before != (l.color, l.intensity, l.exposure));
            }
        }
        for (object, action) in animated {
            let Some(action) = self.actions.get(action).cloned() else {
                continue;
            };
            let Some(o) = self.objects.get_mut(object) else {
                continue;
            };
            let before = o.transform;
            crate::animation::apply_to_object(&action, frame, &mut o.transform);
            if o.transform != before {
                moved += 1;
            }
        }
        if moved > 0 {
            self.touch();
        }
        moved
    }

    // --- lights -------------------------------------------------------------

    pub fn add_light(&mut self, light: Light) -> LightId {
        let uid = self.issue(DatablockKind::Light);
        self.touch();
        self.lights.insert(Arc::new(Light { uid, ..light }))
    }

    /// Adds a light with an identity it already had. See
    /// [`Document::restore_object`].
    pub fn restore_light(&mut self, light: Light, uid: Uid) -> LightId {
        if uid.is_none() {
            return self.add_light(light);
        }
        self.uids.light = self.uids.light.max(uid);
        self.touch();
        self.lights.insert(Arc::new(Light { uid, ..light }))
    }

    pub fn light(&self, id: LightId) -> Option<&Light> {
        self.lights.get(id).map(|l| l.as_ref())
    }

    /// Copy-on-write access to a light.
    pub fn light_mut(&mut self, id: LightId) -> Option<&mut Light> {
        self.touch();
        self.lights.get_mut(id).map(Arc::make_mut)
    }

    /// Every light datablock, in arena slot order.
    pub fn lights(&self) -> impl Iterator<Item = (LightId, &Light)> {
        self.lights.iter().map(|(id, l)| (id, l.as_ref()))
    }

    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    pub fn light_uid(&self, id: LightId) -> Option<Uid> {
        self.lights.get(id).map(|l| l.uid)
    }

    pub fn light_by_uid(&self, uid: Uid) -> Option<LightId> {
        self.lights
            .iter()
            .find(|(_, l)| l.uid == uid)
            .map(|(id, _)| id)
    }

    /// Every light in the scene with the transform that places it.
    ///
    /// The join a renderer actually needs, and the reason it is here rather
    /// than at each call site: a light's direction and position come from the
    /// *object*, its colour and intensity from the datablock, and a caller that
    /// walks objects itself gets the linked-copy case wrong by reading the
    /// datablock once. Invisible objects are skipped, because turning an object
    /// off has to turn its light off — anything else makes the eye icon a lie.
    pub fn lit_by(&self) -> impl Iterator<Item = (ObjectId, &Light, DAffine3)> + '_ {
        self.objects.iter().filter_map(move |(id, object)| {
            if !object.visible {
                return None;
            }
            let ObjectData::Light(light) = object.data else {
                return None;
            };
            Some((id, self.light(light)?, self.world_transform(id)))
        })
    }

    // --- worlds -------------------------------------------------------------

    pub fn add_world(&mut self, world: World) -> WorldId {
        let uid = self.issue(DatablockKind::World);
        self.touch();
        self.worlds.insert(Arc::new(World { uid, ..world }))
    }

    /// Adds a world with an identity it already had. See
    /// [`Document::restore_object`].
    pub fn restore_world(&mut self, world: World, uid: Uid) -> WorldId {
        if uid.is_none() {
            return self.add_world(world);
        }
        self.uids.world = self.uids.world.max(uid);
        self.touch();
        self.worlds.insert(Arc::new(World { uid, ..world }))
    }

    pub fn world(&self, id: WorldId) -> Option<&World> {
        self.worlds.get(id).map(|w| w.as_ref())
    }

    /// Copy-on-write access to a world.
    pub fn world_mut(&mut self, id: WorldId) -> Option<&mut World> {
        self.touch();
        self.worlds.get_mut(id).map(Arc::make_mut)
    }

    pub fn worlds(&self) -> impl Iterator<Item = (WorldId, &World)> {
        self.worlds.iter().map(|(id, w)| (id, w.as_ref()))
    }

    pub fn world_count(&self) -> usize {
        self.worlds.len()
    }

    pub fn world_uid(&self, id: WorldId) -> Option<Uid> {
        self.worlds.get(id).map(|w| w.uid)
    }

    pub fn world_by_uid(&self, uid: Uid) -> Option<WorldId> {
        self.worlds
            .iter()
            .find(|(_, w)| w.uid == uid)
            .map(|(id, _)| id)
    }

    /// Which world is lighting the scene, if any.
    pub fn active_world(&self) -> Option<WorldId> {
        self.active_world
    }

    /// Lights the scene with `world`, or with nothing.
    ///
    /// A world that is not in this document is refused rather than stored: an
    /// active pointer to nothing would have every consumer of the background
    /// carry a "and if it is missing" branch for a state no honest path
    /// produces.
    pub fn set_active_world(&mut self, world: Option<WorldId>) -> bool {
        if let Some(id) = world {
            if self.worlds.get(id).is_none() {
                return false;
            }
        }
        self.active_world = world;
        self.touch();
        true
    }

    // --- materials ----------------------------------------------------------

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        let uid = self.issue(DatablockKind::Material);
        self.touch();
        self.materials.insert(Arc::new(Material { uid, ..material }))
    }

    /// Adds a material with an identity it already had. See
    /// [`Document::restore_object`].
    pub fn restore_material(&mut self, material: Material, uid: Uid) -> MaterialId {
        if uid.is_none() {
            return self.add_material(material);
        }
        self.uids.material = self.uids.material.max(uid);
        self.touch();
        self.materials.insert(Arc::new(Material { uid, ..material }))
    }

    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id).map(|m| m.as_ref())
    }

    /// Copy-on-write access to a material.
    pub fn material_mut(&mut self, id: MaterialId) -> Option<&mut Material> {
        self.touch();
        self.materials.get_mut(id).map(Arc::make_mut)
    }

    pub fn materials(&self) -> impl Iterator<Item = (MaterialId, &Material)> {
        self.materials.iter().map(|(id, m)| (id, m.as_ref()))
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    pub fn material_uid(&self, id: MaterialId) -> Option<Uid> {
        self.materials.get(id).map(|m| m.uid)
    }

    pub fn material_by_uid(&self, uid: Uid) -> Option<MaterialId> {
        self.materials
            .iter()
            .find(|(_, m)| m.uid == uid)
            .map(|(id, _)| id)
    }

    // --- material slots -----------------------------------------------------

    /// The material slots a mesh's faces index into.
    pub fn mesh_slots(&self, mesh: MeshId) -> &[Option<MaterialId>] {
        self.meshes.get(mesh).map(|m| &m.slots[..]).unwrap_or(&[])
    }

    /// Replaces a mesh's slot list.
    pub fn set_mesh_slots(&mut self, mesh: MeshId, slots: Vec<Option<MaterialId>>) -> bool {
        let Some(entry) = self.meshes.get_mut(mesh) else {
            return false;
        };
        entry.slots = slots;
        self.touch();
        true
    }

    /// Puts `material` in slot `slot`, growing the list if it is short.
    ///
    /// Growing rather than refusing, because a mesh imported with a face
    /// pointing at slot three is exactly the case this has to cope with, and
    /// refusing would lose the assignment rather than the index.
    pub fn set_mesh_slot(
        &mut self,
        mesh: MeshId,
        slot: usize,
        material: Option<MaterialId>,
    ) -> bool {
        let Some(entry) = self.meshes.get_mut(mesh) else {
            return false;
        };
        if entry.slots.len() <= slot {
            entry.slots.resize(slot + 1, None);
        }
        entry.slots[slot] = material;
        self.touch();
        true
    }

    /// The material an object shows in `slot`.
    ///
    /// The one place the override rule lives: the object's list wins where it
    /// has an entry, the mesh's answers otherwise. A second copy of this rule
    /// anywhere is a place for the two to disagree about which chair is
    /// upholstered.
    ///
    /// `None` for a slot nothing is assigned to, for an index past the end,
    /// and for an object with no mesh. None of those is an error: a freshly
    /// added cube has no materials at all, and the renderer draws it with a
    /// default surface rather than refusing.
    pub fn material_at(&self, object: ObjectId, slot: usize) -> Option<&Material> {
        self.material(self.material_id_at(object, slot)?)
    }

    /// Which material a slot resolves to, rather than the material itself.
    ///
    /// The same rule, and the reason it is the one with the body: editing a
    /// material needs the id, and a caller that worked it out separately would
    /// be a second opinion about whether the object's override or the mesh's
    /// slot wins.
    pub fn material_id_at(&self, object: ObjectId, slot: usize) -> Option<MaterialId> {
        let object = self.objects.get(object)?;
        if let Some(Some(overridden)) = object.material_overrides.get(slot) {
            return Some(*overridden);
        }
        let mesh = object.mesh()?;
        *self.meshes.get(mesh)?.slots.get(slot)?
    }

    pub fn add_collection(&mut self, collection: Collection) -> CollectionId {
        self.touch();
        self.collections.insert(collection)
    }

    pub fn collection(&self, id: CollectionId) -> Option<&Collection> {
        self.collections.get(id)
    }

    // --- convenience --------------------------------------------------------

    /// Adds a mesh object in one step.
    pub fn add_mesh_object(&mut self, name: impl Into<String>, mesh: MeshData) -> ObjectId {
        let mesh_id = self.add_mesh(mesh);
        self.add_object(Object::new(name, ObjectData::Mesh(mesh_id)))
    }

    /// Duplicates an object so that it *shares* the original's mesh.
    ///
    /// The cheap duplicate, and the reason for datablock indirection: geometry is
    /// not copied, so a thousand instances cost a thousand transforms.
    pub fn duplicate_linked(&mut self, id: ObjectId) -> Option<ObjectId> {
        let mut copy = self.objects.get(id)?.clone();
        copy.name = format!("{}.001", copy.name);
        Some(self.add_object(copy))
    }

    /// Duplicates an object *and* its mesh, so the two can diverge.
    pub fn duplicate_full(&mut self, id: ObjectId) -> Option<ObjectId> {
        let mut copy = self.objects.get(id)?.clone();
        copy.name = format!("{}.001", copy.name);
        if let ObjectData::Mesh(mesh_id) = copy.data {
            let cloned = self.mesh(mesh_id)?.clone();
            copy.data = ObjectData::Mesh(self.add_mesh(cloned));
        }
        Some(self.add_object(copy))
    }

    /// Bounding box of the whole scene, in world space.
    ///
    /// Every visible object counts: a mesh by its vertices, and anything
    /// without a surface — a lamp, a camera, an empty — by where it is. This
    /// is what Frame All frames, and Blender's frames the lamp and the camera
    /// too; a new document whose lamp sat outside the first view read as a
    /// document with no lamp in it.
    pub fn world_bounds(&self) -> kreate_math::Aabb {
        let mut bounds = kreate_math::Aabb::EMPTY;
        for (id, object) in self.objects.iter() {
            if !object.visible {
                continue;
            }
            let world = self.world_transform(id);
            match object.mesh().and_then(|m| self.mesh(m)) {
                Some(mesh) => {
                    for p in mesh.positions() {
                        let w = world
                            .transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
                        bounds = bounds.extend(w.as_vec3());
                    }
                }
                None => {
                    bounds = bounds.extend(world.translation.as_vec3());
                }
            }
        }
        bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DQuat;

    #[test]
    fn a_deep_parent_chain_does_not_overflow_the_stack() {
        // A rig, a spine, a length of chain links. The recursive form of
        // `world_transform` is shorter and dies here — and it dies as a hard
        // crash on the first frame that asks for a transform, which is not
        // something a user can act on or a report anyone can read.
        //
        // Depth rather than breadth: it is the chain length that costs stack
        // frames, and a flat scene of the same size is fine either way.
        // Parented at construction rather than with `set_parent`, which asks for
        // the parent's world transform and so builds a chain in quadratic time.
        // The hierarchy is the same; only the cost of making it differs.
        let mut doc = Document::new();
        let mut parent = None;
        for i in 0..20_000 {
            let mut object = Object::new(format!("Link{i}"), ObjectData::Empty);
            object.parent = parent;
            object.transform.translation = DVec3::Z;
            parent = Some(doc.add_object(object));
        }

        let leaf = parent.expect("the chain was built");
        let world = doc.world_transform(leaf);
        assert!(
            world.translation.is_finite(),
            "the deepest link has a usable transform: {world:?}"
        );
    }

    fn doc_with_cube() -> (Document, ObjectId) {
        let mut d = Document::new();
        let id = d.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        (d, id)
    }

    #[test]
    fn adding_an_object_registers_it_as_a_root() {
        let (d, id) = doc_with_cube();
        assert_eq!(d.object_count(), 1);
        assert_eq!(d.roots(), &[id]);
        assert_eq!(d.object(id).unwrap().name, "Cube");
    }

    #[test]
    fn adopting_a_snapshot_moves_the_version_forward_not_back() {
        // The defect this guards: a cache keyed on the version sees the same
        // number after "restore, then re-run" as it saw before, concludes nothing
        // changed, and keeps drawing the old geometry.
        let (mut d, _) = doc_with_cube();
        let snapshot = d.clone();
        d.add_mesh_object("Second", kreate_mesh::primitives::cube(1.0));
        let after_edit = d.version();
        assert!(after_edit > snapshot.version());

        d.adopt(snapshot);
        assert_eq!(d.object_count(), 1, "the content really is the older one");
        assert!(
            d.version() > after_edit,
            "the version went backwards: {} is not past {after_edit}",
            d.version()
        );
    }

    #[test]
    fn repeated_adopt_and_edit_never_repeats_a_version() {
        let (mut d, _) = doc_with_cube();
        let mut seen = std::collections::HashSet::new();
        seen.insert(d.version());
        for _ in 0..10 {
            let snapshot = d.clone();
            d.add_mesh_object("Temp", kreate_mesh::primitives::cube(1.0));
            assert!(seen.insert(d.version()), "version {} repeated", d.version());
            d.adopt(snapshot);
            assert!(seen.insert(d.version()), "version {} repeated", d.version());
        }
    }

    #[test]
    fn version_advances_on_mutation() {
        let (mut d, id) = doc_with_cube();
        let before = d.version();
        d.object_mut(id).unwrap().transform.translation.x = 1.0;
        assert!(d.version() > before);
    }

    #[test]
    fn a_linked_duplicate_shares_geometry() {
        // The payoff of datablock indirection: a thousand instances cost a
        // thousand transforms, not a thousand meshes.
        let (mut d, id) = doc_with_cube();
        let copy = d.duplicate_linked(id).unwrap();
        let mesh_id = d.object(id).unwrap().mesh().unwrap();
        assert_eq!(d.object(copy).unwrap().mesh(), Some(mesh_id));
        assert_eq!(d.mesh_count(), 1, "geometry must not be copied");
        assert_eq!(d.mesh_users(mesh_id), 2);
    }

    #[test]
    fn a_full_duplicate_lets_the_copies_diverge() {
        let (mut d, id) = doc_with_cube();
        let copy = d.duplicate_full(id).unwrap();
        let a = d.object(id).unwrap().mesh().unwrap();
        let b = d.object(copy).unwrap().mesh().unwrap();
        assert_ne!(a, b);
        assert_eq!(d.mesh_count(), 2);

        d.mesh_mut(b).unwrap().positions_mut()[0] = kreate_math::Vec3::splat(9.0);
        assert_ne!(
            d.mesh(a).unwrap().positions()[0],
            d.mesh(b).unwrap().positions()[0]
        );
    }

    #[test]
    fn parenting_does_not_move_the_object() {
        // The bug `parent_inverse` exists to prevent: without it, parenting
        // reinterprets the local transform relative to the parent and the child
        // visibly teleports.
        let mut d = Document::new();
        let parent = d.add_object(Object::new("Parent", ObjectData::Empty));
        d.object_mut(parent).unwrap().transform =
            DTransform::from_translation(DVec3::new(10.0, 0.0, 0.0));

        let child = d.add_object(Object::new("Child", ObjectData::Empty));
        d.object_mut(child).unwrap().transform =
            DTransform::from_translation(DVec3::new(1.0, 2.0, 3.0));
        let before = d.world_transform(child).translation;

        d.set_parent(child, Some(parent)).unwrap();
        let after = d.world_transform(child).translation;
        assert!(
            (after - before).length() < 1e-12,
            "child moved from {before:?} to {after:?} on parenting"
        );
    }

    #[test]
    fn a_child_follows_its_parent_after_parenting() {
        let mut d = Document::new();
        let parent = d.add_object(Object::new("Parent", ObjectData::Empty));
        let child = d.add_object(Object::new("Child", ObjectData::Empty));
        d.set_parent(child, Some(parent)).unwrap();

        d.object_mut(parent).unwrap().transform =
            DTransform::from_translation(DVec3::new(0.0, 5.0, 0.0));
        assert!(
            (d.world_transform(child).translation - DVec3::new(0.0, 5.0, 0.0)).length() < 1e-12
        );
    }

    #[test]
    fn rotating_a_parent_carries_its_child_around() {
        let mut d = Document::new();
        let parent = d.add_object(Object::new("Parent", ObjectData::Empty));
        let child = d.add_object(Object::new("Child", ObjectData::Empty));
        d.object_mut(child).unwrap().transform =
            DTransform::from_translation(DVec3::new(1.0, 0.0, 0.0));
        d.set_parent(child, Some(parent)).unwrap();

        d.object_mut(parent).unwrap().transform =
            DTransform::from_rotation(DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2));
        let w = d.world_transform(child).translation;
        assert!((w - DVec3::new(0.0, 1.0, 0.0)).length() < 1e-9, "got {w:?}");
    }

    #[test]
    fn reparenting_refuses_to_create_a_cycle() {
        // Not cosmetic: a cycle makes every world-transform walk recurse forever.
        let mut d = Document::new();
        let a = d.add_object(Object::new("A", ObjectData::Empty));
        let b = d.add_object(Object::new("B", ObjectData::Empty));
        let c = d.add_object(Object::new("C", ObjectData::Empty));
        d.set_parent(b, Some(a)).unwrap();
        d.set_parent(c, Some(b)).unwrap();

        assert_eq!(d.set_parent(a, Some(c)), Err(ReparentError::WouldCycle));
        assert_eq!(d.set_parent(a, Some(a)), Err(ReparentError::WouldCycle));
        // The refusal must leave the hierarchy untouched.
        assert_eq!(d.object(a).unwrap().parent, None);
    }

    #[test]
    fn reparenting_updates_the_root_list() {
        let mut d = Document::new();
        let a = d.add_object(Object::new("A", ObjectData::Empty));
        let b = d.add_object(Object::new("B", ObjectData::Empty));
        assert_eq!(d.roots().len(), 2);
        d.set_parent(b, Some(a)).unwrap();
        assert_eq!(d.roots(), &[a]);
        d.set_parent(b, None).unwrap();
        assert_eq!(d.roots().len(), 2);
    }

    #[test]
    fn removing_an_object_reparents_its_children() {
        // Orphaning them would leave the hierarchy pointing at a dead id.
        let mut d = Document::new();
        let a = d.add_object(Object::new("A", ObjectData::Empty));
        let b = d.add_object(Object::new("B", ObjectData::Empty));
        let c = d.add_object(Object::new("C", ObjectData::Empty));
        d.set_parent(b, Some(a)).unwrap();
        d.set_parent(c, Some(b)).unwrap();

        d.remove_object(b).unwrap();
        assert_eq!(d.object(c).unwrap().parent, Some(a));
        assert!(d.object(b).is_none());
        assert_eq!(d.hierarchy().len(), 2);
    }

    #[test]
    fn removing_an_object_clears_it_from_the_selection() {
        let (mut d, id) = doc_with_cube();
        d.selection.add(id);
        d.remove_object(id);
        assert!(d.selection.is_empty());
        assert_eq!(d.selection.active, None);
    }

    #[test]
    fn hierarchy_lists_parents_before_children_with_depth() {
        let mut d = Document::new();
        let a = d.add_object(Object::new("A", ObjectData::Empty));
        let b = d.add_object(Object::new("B", ObjectData::Empty));
        let c = d.add_object(Object::new("C", ObjectData::Empty));
        d.set_parent(b, Some(a)).unwrap();
        d.set_parent(c, Some(b)).unwrap();

        assert_eq!(d.hierarchy(), vec![(a, 0), (b, 1), (c, 2)]);
    }

    #[test]
    fn a_held_handle_makes_a_change_detectable_by_pointer() {
        // The mechanism a GPU cache uses instead of a version stamp. It works
        // only because the cache holds a handle: with a unique handle,
        // `Arc::make_mut` mutates in place and the pointer never changes.
        let (mut d, id) = doc_with_cube();
        let mesh_id = d.object(id).unwrap().mesh().unwrap();
        let held = d.mesh_arc(mesh_id).unwrap().clone();

        assert!(Arc::ptr_eq(&held, d.mesh_arc(mesh_id).unwrap()));
        d.mesh_mut(mesh_id).unwrap().positions_mut()[0] = kreate_math::Vec3::splat(3.0);
        assert!(
            !Arc::ptr_eq(&held, d.mesh_arc(mesh_id).unwrap()),
            "the edit must be visible as a different handle"
        );
    }

    #[test]
    fn without_a_held_handle_an_edit_is_invisible_to_a_pointer_check() {
        // Stated as a test because it is the trap: a cache that stores only the
        // id would compare unchanged pointers and never rebuild.
        let (mut d, id) = doc_with_cube();
        let mesh_id = d.object(id).unwrap().mesh().unwrap();
        let before = Arc::as_ptr(d.mesh_arc(mesh_id).unwrap());

        d.mesh_mut(mesh_id).unwrap().positions_mut()[0] = kreate_math::Vec3::splat(3.0);
        assert_eq!(
            before,
            Arc::as_ptr(d.mesh_arc(mesh_id).unwrap()),
            "a uniquely-owned handle is mutated in place, so the pointer is stable"
        );
    }

    #[test]
    fn a_document_clone_shares_mesh_storage() {
        // The property the whole undo design rests on: a snapshot is refcount
        // bumps, not a deep copy.
        let (mut d, id) = doc_with_cube();
        let mesh_id = d.object(id).unwrap().mesh().unwrap();
        let snapshot = d.clone();

        d.mesh_mut(mesh_id).unwrap().positions_mut()[0] = kreate_math::Vec3::splat(5.0);

        assert_eq!(
            snapshot.mesh(mesh_id).unwrap().positions()[0],
            kreate_mesh::primitives::cube(2.0).positions()[0],
            "the snapshot must not see the edit"
        );
        // And the topology arrays are still shared, not copied.
        assert!(std::ptr::eq(
            d.mesh(mesh_id).unwrap().corner_verts().as_ptr(),
            snapshot.mesh(mesh_id).unwrap().corner_verts().as_ptr(),
        ));
    }

    #[test]
    fn selection_tracks_an_active_object() {
        let mut d = Document::new();
        let a = d.add_object(Object::new("A", ObjectData::Empty));
        let b = d.add_object(Object::new("B", ObjectData::Empty));
        d.selection.add(a);
        d.selection.add(b);
        assert_eq!(d.selection.active, Some(b));
        assert_eq!(d.selection.objects.len(), 2);

        // Adding twice must not duplicate the entry.
        d.selection.add(b);
        assert_eq!(d.selection.objects.len(), 2);

        d.selection.remove(b);
        assert_eq!(
            d.selection.active,
            Some(a),
            "active falls back to what is left"
        );
    }

    #[test]
    fn linked_datablocks_are_not_editable() {
        let m = Material {
            linkage: Linkage::Linked {
                library: "lib.kreate".into(),
                name: "Steel".into(),
            },
            ..Material::default()
        };
        assert!(!m.linkage.is_editable());
        assert!(Material::default().linkage.is_editable());
    }

    #[test]
    fn world_bounds_account_for_transforms() {
        let mut d = Document::new();
        let id = d.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        d.object_mut(id).unwrap().transform =
            DTransform::from_translation(DVec3::new(10.0, 0.0, 0.0));
        let b = d.world_bounds();
        assert!((b.center().x - 10.0).abs() < 1e-4, "got {}", b.center().x);
        assert!((b.size().x - 2.0).abs() < 1e-4);
    }

    #[test]
    fn a_lamp_extends_the_bounds_to_where_it_is() {
        // Frame All has to bring the lamp into view, or a new document reads
        // as one with no lamp in it. A lamp has no vertices, so it counts by
        // its origin.
        let mut d = Document::new();
        d.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let before = d.world_bounds();
        let light = d.add_light(crate::Light::of_kind("Lamp", crate::LightKind::BULB));
        let mut lamp = Object::new("Lamp", ObjectData::Light(light));
        lamp.transform.translation = DVec3::new(3.0, -3.0, 5.0);
        d.add_object(lamp);
        let after = d.world_bounds();
        assert!(after.max.z >= 5.0 - 1e-6, "the lamp at z=5 is outside {after:?}");
        assert!(after.max.x >= 3.0 - 1e-6 && after.min.y <= -3.0 + 1e-6);
        assert!(after.max.z > before.max.z, "the lamp did not extend the bounds");
    }

    #[test]
    fn hidden_objects_do_not_affect_world_bounds() {
        let mut d = Document::new();
        d.add_mesh_object("Visible", kreate_mesh::primitives::cube(2.0));
        let hidden = d.add_mesh_object("Hidden", kreate_mesh::primitives::cube(2.0));
        d.object_mut(hidden).unwrap().transform =
            DTransform::from_translation(DVec3::new(1000.0, 0.0, 0.0));
        d.object_mut(hidden).unwrap().visible = false;
        assert!(d.world_bounds().max.x < 2.0);
    }

    #[test]
    fn an_empty_document_is_coherent() {
        let d = Document::new();
        assert_eq!(d.object_count(), 0);
        assert!(d.roots().is_empty());
        assert!(d.hierarchy().is_empty());
        assert!(d.world_bounds().is_empty());
    }
}

#[cfg(test)]
mod uid_tests {
    use super::*;
    use kreate_mesh::primitives;

    fn doc_with(n: usize) -> (Document, Vec<ObjectId>) {
        let mut doc = Document::new();
        let mut ids = Vec::new();
        for i in 0..n {
            let mesh = doc.add_mesh(primitives::cube(1.0 + i as f32));
            ids.push(doc.add_object(Object::new(format!("Cube{i}"), ObjectData::Mesh(mesh))));
        }
        (doc, ids)
    }

    #[test]
    fn every_datablock_gets_an_identity_and_no_two_share_one() {
        let (doc, ids) = doc_with(4);
        let mut seen: Vec<Uid> = Vec::new();
        for id in &ids {
            let uid = doc.object_uid(*id).expect("an identity");
            assert!(!uid.is_none());
            assert!(!seen.contains(&uid), "{uid} was handed out twice");
            seen.push(uid);
        }
        assert_eq!(doc.object_by_uid(seen[2]), Some(ids[2]));
    }

    #[test]
    fn renaming_an_object_does_not_change_its_identity() {
        // The whole reason identities exist. Blender's name-based paths break
        // here, and renaming is the first thing anyone does after adding an
        // object.
        let (mut doc, ids) = doc_with(1);
        let before = doc.object_uid(ids[0]).expect("an identity");
        doc.object_mut(ids[0]).expect("object").name = "Something else".to_string();
        assert_eq!(doc.object_uid(ids[0]), Some(before));
        assert_eq!(doc.object_by_uid(before), Some(ids[0]));
    }

    #[test]
    fn a_duplicate_is_a_new_object_and_not_a_second_handle() {
        // `duplicate_linked` clones the whole struct, identity field included,
        // so the assignment has to happen on the way *in* to the document
        // rather than being trusted from the caller. Two objects sharing an
        // identity would have a curve drive both.
        let (mut doc, ids) = doc_with(1);
        let original = doc.object_uid(ids[0]).expect("an identity");

        let linked = doc.duplicate_linked(ids[0]).expect("duplicates");
        assert_ne!(doc.object_uid(linked), Some(original));
        assert_eq!(doc.object_by_uid(original), Some(ids[0]));

        // And a full duplicate's *mesh* is a new datablock with a new identity,
        // which is the difference between the two duplicate operations.
        let full = doc.duplicate_full(ids[0]).expect("duplicates");
        let original_mesh = doc
            .object(ids[0])
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh_uid(m))
            .expect("an identity");
        let copy_mesh = doc
            .object(full)
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh_uid(m))
            .expect("an identity");
        assert_ne!(original_mesh, copy_mesh);

        // Where a *linked* duplicate shares the datablock and therefore its
        // identity, which is what makes them linked.
        let linked_mesh = doc
            .object(linked)
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh_uid(m))
            .expect("an identity");
        assert_eq!(linked_mesh, original_mesh);
    }

    #[test]
    fn an_identity_is_not_reused_when_a_slot_is() {
        // The failure an `ObjectId` cannot avoid. Arena slots are reused, and
        // the generation that would tell them apart is only compiled in for a
        // debug build — so a curve keyed on a slot starts driving whatever is
        // created next. In a release build, silently.
        let (mut doc, ids) = doc_with(1);
        let gone = doc.object_uid(ids[0]).expect("an identity");
        doc.remove_object(ids[0]);
        assert_eq!(doc.object_by_uid(gone), None);

        let replacement = doc.add_object(Object::new("Next", ObjectData::Empty));
        assert_ne!(
            doc.object_uid(replacement),
            Some(gone),
            "the identity of a deleted object was handed out again"
        );
        assert_eq!(doc.object_by_uid(gone), None);
    }

    #[test]
    fn the_counters_travel_with_the_document_so_undo_preserves_identity() {
        // Undo restores a whole `Document`, counters included, which is what
        // makes undo and redo identity-preserving: the redone object is the
        // same object, and a curve pointing at it still points at it.
        //
        // A counter living outside the snapshot would give the redone object a
        // fresh identity and orphan its animation — and the user would see an
        // undo followed by a redo silently lose their keys.
        let (mut doc, _) = doc_with(1);
        let snapshot = doc.clone();

        let added = doc.add_object(Object::new("Added", ObjectData::Empty));
        let uid = doc.object_uid(added).expect("an identity");

        // Undo, then redo the same creation.
        doc = snapshot;
        let redone = doc.add_object(Object::new("Added", ObjectData::Empty));
        assert_eq!(
            doc.object_uid(redone),
            Some(uid),
            "redoing a creation should reach the same object"
        );
    }

    #[test]
    fn objects_and_meshes_number_separately() {
        // One counter per kind keeps the numbers small enough for a person to
        // match a curve to its target by eye in a scene file. `DatablockKind`
        // is what stops object 1 and mesh 1 being confused, and it is carried
        // in the path rather than in the id.
        let (doc, ids) = doc_with(1);
        let object = doc.object_uid(ids[0]).expect("an identity");
        let mesh = doc
            .object(ids[0])
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh_uid(m))
            .expect("an identity");
        assert_eq!(object, Uid::FIRST);
        assert_eq!(mesh, Uid::FIRST);
    }

    #[test]
    fn a_restored_identity_pushes_the_counter_past_it() {
        // What a loader needs. A file holding ids 1, 2 and 40 must not have the
        // next new object collide with 40 — including the case a naive "count
        // what we loaded" gets wrong, where the file skipped numbers because
        // objects were deleted before it was saved.
        let mut doc = Document::new();
        doc.restore_object(Object::new("A", ObjectData::Empty), Uid::from_raw(1));
        doc.restore_object(Object::new("B", ObjectData::Empty), Uid::from_raw(40));
        let next = doc.add_object(Object::new("C", ObjectData::Empty));
        assert_eq!(doc.object_uid(next), Some(Uid::from_raw(41)));

        // And restoring out of order does not walk the counter backwards.
        let mut doc = Document::new();
        doc.restore_object(Object::new("A", ObjectData::Empty), Uid::from_raw(40));
        doc.restore_object(Object::new("B", ObjectData::Empty), Uid::from_raw(1));
        let next = doc.add_object(Object::new("C", ObjectData::Empty));
        assert_eq!(doc.object_uid(next), Some(Uid::from_raw(41)));
    }

    #[test]
    fn nothing_in_a_document_holds_the_unassigned_identity() {
        // The invariant `object_by_uid` relies on instead of a guard. An object
        // holds `Uid::NONE` between `Object::new` and being added, and a
        // hand-edited file can ask for it back — so the restore path issues one
        // rather than storing nothing, and every lookup is spared a branch it
        // would otherwise carry forever for the sake of one bad file.
        let (mut doc, _) = doc_with(2);
        assert!(Object::new("Loose", ObjectData::Empty).uid.is_none());

        let restored =
            doc.restore_object(Object::new("FromABadFile", ObjectData::Empty), Uid::NONE);
        assert!(
            doc.object_uid(restored).is_some_and(|u| !u.is_none()),
            "restoring nothing stored nothing"
        );
        let mesh = doc.restore_mesh(primitives::cube(1.0), Uid::NONE);
        assert!(doc.mesh_uid(mesh).is_some_and(|u| !u.is_none()));

        for (_, object) in doc.objects() {
            assert!(!object.uid.is_none(), "{} has no identity", object.name);
        }
        for (id, _) in doc.meshes() {
            assert!(doc.mesh_uid(id).is_some_and(|u| !u.is_none()));
        }

        // And so the unassigned id finds nothing, without being asked about.
        assert_eq!(doc.object_by_uid(Uid::NONE), None);
        assert_eq!(doc.mesh_by_uid(Uid::NONE), None);
    }
}
