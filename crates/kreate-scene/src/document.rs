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

define_id! {
    /// An object in the scene graph.
    pub struct ObjectId;
}
define_id! {
    /// A mesh datablock. Several objects may share one.
    pub struct MeshId;
}
define_id! {
    /// A material datablock.
    pub struct MaterialId;
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
    Camera,
    Light,
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
    /// Applied in order to produce the geometry that is drawn.
    ///
    /// The document holds the cage the user edits; what appears on screen is the
    /// result of running this stack over it. See [`crate::modifier`] for why the
    /// evaluated result is cached outside the document rather than here.
    pub modifiers: Vec<crate::modifier::ModifierInstance>,
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
            modifiers: Vec::new(),
        }
    }

    pub fn mesh(&self) -> Option<MeshId> {
        match self.data {
            ObjectData::Mesh(id) => Some(id),
            _ => None,
        }
    }
}

/// A material, deliberately a "standard surface" and nothing more.
///
/// Its parameters map near-1:1 onto both glTF metal-rough and
/// `UsdPreviewSurface`, which were designed to match each other. Arbitrary
/// shader graphs are a separate problem and are never imported.
#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub linkage: Linkage,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Material".into(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            linkage: Linkage::Local,
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
    meshes: Arena<MeshId, Arc<MeshData>>,
    materials: Arena<MaterialId, Arc<Material>>,
    collections: Arena<CollectionId, Collection>,
    /// Objects with no parent, in user-visible order.
    roots: Vec<ObjectId>,
    pub selection: Selection,
    /// Bumped on every mutation. A cheap "has anything changed" check for
    /// callers that cannot use `Arc::ptr_eq` on a specific array.
    version: u64,
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
        let has_parent = object.parent.is_some();
        let id = self.objects.insert(object);
        if !has_parent {
            self.roots.push(id);
        }
        self.touch();
        id
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
    pub fn world_transform(&self, id: ObjectId) -> DAffine3 {
        let Some(object) = self.objects.get(id) else {
            return DAffine3::IDENTITY;
        };
        let local = object.transform.to_affine();
        match object.parent {
            None => local,
            Some(parent) => self.world_transform(parent) * object.parent_inverse * local,
        }
    }

    /// Depth-first traversal from the roots, parents before children.
    pub fn hierarchy(&self) -> Vec<(ObjectId, usize)> {
        let mut out = Vec::with_capacity(self.objects.len());
        let mut stack: Vec<(ObjectId, usize)> =
            self.roots.iter().rev().map(|id| (*id, 0)).collect();
        while let Some((id, depth)) = stack.pop() {
            if self.objects.get(id).is_none() {
                continue;
            }
            out.push((id, depth));
            let mut children: Vec<ObjectId> = self
                .objects
                .iter()
                .filter(|(_, o)| o.parent == Some(id))
                .map(|(cid, _)| cid)
                .collect();
            children.reverse();
            stack.extend(children.into_iter().map(|c| (c, depth + 1)));
            // A pre-existing cycle would otherwise loop forever here.
            if out.len() > self.objects.slot_count() {
                break;
            }
        }
        out
    }

    // --- datablocks ---------------------------------------------------------

    pub fn add_mesh(&mut self, mesh: MeshData) -> MeshId {
        self.touch();
        self.meshes.insert(Arc::new(mesh))
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshData> {
        self.meshes.get(id).map(|m| m.as_ref())
    }

    /// Copy-on-write access to a mesh.
    ///
    /// Clones the `MeshData` only if it is shared with a snapshot — and even then
    /// the clone is a set of refcount bumps, because `MeshData`'s own arrays are
    /// individually `Arc`ed.
    pub fn mesh_mut(&mut self, id: MeshId) -> Option<&mut MeshData> {
        self.version += 1;
        self.meshes.get_mut(id).map(Arc::make_mut)
    }

    /// The mesh datablock as a shared handle.
    ///
    /// A consumer that keeps a clone of this can tell whether the mesh changed
    /// with `Arc::ptr_eq` — free, and exact, with no invalidation discipline to
    /// forget. Holding the clone is what makes it work: `Arc::make_mut` only
    /// copies when the handle is shared, so a cache that does *not* hold one
    /// would watch the document mutate in place behind an unchanged pointer.
    pub fn mesh_arc(&self, id: MeshId) -> Option<&Arc<MeshData>> {
        self.meshes.get(id)
    }

    /// Every mesh datablock, in arena slot order.
    ///
    /// Deterministic, which is what lets the debug-build validation hook name
    /// the same datablock on every run.
    pub fn meshes(&self) -> impl Iterator<Item = (MeshId, &MeshData)> + '_ {
        self.meshes.iter().map(|(id, m)| (id, m.as_ref()))
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

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        self.touch();
        self.materials.insert(Arc::new(material))
    }

    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id).map(|m| m.as_ref())
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
    pub fn world_bounds(&self) -> kreate_math::Aabb {
        let mut bounds = kreate_math::Aabb::EMPTY;
        for (id, object) in self.objects.iter() {
            if !object.visible {
                continue;
            }
            let Some(mesh) = object.mesh().and_then(|m| self.mesh(m)) else {
                continue;
            };
            let world = self.world_transform(id);
            for p in mesh.positions() {
                let w = world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
                bounds = bounds.extend(w.as_vec3());
            }
        }
        bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DQuat;

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
