//! Keeping copies of one document in step, a datablock at a time.
//!
//! A shared session does not replay operators: an import, a bake or anything
//! that read the selection would come out differently on another machine. It
//! sends what an edit *left*: the datablocks that differ from the last state
//! everyone agreed on, each named by its identity. Arena ids are not shared
//! between copies, so every reference inside a patch is carried through the
//! identity of what it points at.
//!
//! A patch is itself a [`Document`], so it travels in the same container a
//! file does. What a changed datablock points at but did not change rides along
//! as a stand-in — an empty mesh, a one-pixel image, a bare empty — which the
//! receiver uses only to resolve the reference and never writes over its own.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use kreate_mesh::MeshData;
use kreate_reflect::{DatablockKind, Uid};

use super::{
    ActionId, ArmatureId, CameraId, Document, ImageId, LightId, Material, MaterialId, MeshEntry,
    MeshId, Object, ObjectData, ObjectId, WorldId,
};
use crate::image::{Image, PixelFormat};
use crate::light::Background;

/// How far a participant's range of identities is shifted: participant `n`
/// numbers from `n << SITE_SHIFT`, leaving each of them 2⁴⁸ identities.
pub const SITE_SHIFT: u32 = 48;

/// One datablock, by kind and identity.
pub type BlockKey = (DatablockKind, Uid);

/// What differs between two states of a document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Changes {
    /// Datablocks added or altered.
    pub changed: BTreeSet<BlockKey>,
    /// Objects that went. Other datablocks are left where they are: one that
    /// nothing uses any more is the purge's business, not the session's.
    pub removed: BTreeSet<Uid>,
    /// The scene's own settings: its range and rate, render, physics, units
    /// and which world lights it.
    pub settings: bool,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.removed.is_empty() && !self.settings
    }

    /// Whether `key` changed.
    pub fn has(&self, kind: DatablockKind, uid: Uid) -> bool {
        self.changed.contains(&(kind, uid))
    }
}

impl Document {
    /// Makes this copy participant `site` of a shared session, so what it
    /// adds is numbered in its own range; 0 ends that.
    ///
    /// Counts on from the highest identity already in that range, so a
    /// participant who leaves and comes back with the same number does not
    /// hand out an identity its earlier self already used.
    pub fn set_uid_site(&mut self, site: u16) {
        self.site = site;
        self.site_uids = Default::default();
        if site == 0 {
            return;
        }
        let base = u64::from(site) << SITE_SHIFT;
        let top = base + ((1u64 << SITE_SHIFT) - 1);
        for (kind, uid) in self.all_keys() {
            if (base..=top).contains(&uid.raw()) {
                let counter = self.site_uids.of(kind);
                *counter = (*counter).max(uid);
            }
        }
    }

    /// The participant this copy is, 0 outside a shared session.
    pub fn uid_site(&self) -> u16 {
        self.site
    }

    fn all_keys(&self) -> Vec<BlockKey> {
        use DatablockKind as K;
        let mut keys = Vec::new();
        keys.extend(self.objects.values().map(|o| (K::Object, o.uid)));
        keys.extend(self.meshes.values().map(|m| (K::Mesh, m.uid)));
        keys.extend(self.materials.values().map(|m| (K::Material, m.uid)));
        keys.extend(self.images.values().map(|m| (K::Image, m.uid)));
        keys.extend(self.cameras.values().map(|m| (K::Camera, m.uid)));
        keys.extend(self.lights.values().map(|m| (K::Light, m.uid)));
        keys.extend(self.worlds.values().map(|m| (K::World, m.uid)));
        keys.extend(self.actions.values().map(|m| (K::Action, m.uid)));
        keys.extend(self.armatures.values().map(|m| (K::Armature, m.uid)));
        keys
    }

    /// What this document has that `base` — an earlier state of it — had
    /// not.
    ///
    /// Datablocks behind an `Arc` compare by pointer, which is exact: an edit
    /// goes through `make_mut`, and `base` holding the old handle is what
    /// makes that copy. Objects are kept inline and compared whole.
    pub fn changes_since(&self, base: &Document) -> Changes {
        use DatablockKind as K;
        let mut changes = Changes::default();

        let before: HashMap<Uid, &Object> = base.objects.values().map(|o| (o.uid, o)).collect();
        for object in self.objects.values() {
            let same = before
                .get(&object.uid)
                .is_some_and(|b| format!("{b:?}") == format!("{object:?}"));
            if !same {
                changes.changed.insert((K::Object, object.uid));
            }
        }
        for uid in before.keys() {
            if !self.objects.values().any(|o| o.uid == *uid) {
                changes.removed.insert(*uid);
            }
        }

        let slot_uids = |doc: &Document, entry: &MeshEntry| -> Vec<Option<Uid>> {
            entry
                .slots
                .iter()
                .map(|s| s.and_then(|m| doc.material_uid(m)))
                .collect()
        };
        let meshes: HashMap<Uid, &MeshEntry> = base.meshes.values().map(|m| (m.uid, m)).collect();
        for entry in self.meshes.values() {
            let same = meshes.get(&entry.uid).is_some_and(|b| {
                Arc::ptr_eq(&b.data, &entry.data) && slot_uids(base, b) == slot_uids(self, entry)
            });
            if !same {
                changes.changed.insert((K::Mesh, entry.uid));
            }
        }

        fn arcs<T>(
            kind: DatablockKind,
            now: impl Iterator<Item = (Uid, Arc<T>)>,
            then: impl Iterator<Item = (Uid, Arc<T>)>,
            out: &mut BTreeSet<BlockKey>,
        ) {
            let then: HashMap<Uid, Arc<T>> = then.collect();
            for (uid, block) in now {
                if !then.get(&uid).is_some_and(|b| Arc::ptr_eq(b, &block)) {
                    out.insert((kind, uid));
                }
            }
        }
        let out = &mut changes.changed;
        macro_rules! compare {
            ($kind:expr, $arena:ident) => {
                arcs(
                    $kind,
                    self.$arena.values().map(|b| (b.uid, b.clone())),
                    base.$arena.values().map(|b| (b.uid, b.clone())),
                    out,
                )
            };
        }
        compare!(K::Material, materials);
        compare!(K::Image, images);
        compare!(K::Camera, cameras);
        compare!(K::Light, lights);
        compare!(K::World, worlds);
        compare!(K::Action, actions);
        compare!(K::Armature, armatures);

        changes.settings = self.settings_text() != base.settings_text();
        changes
    }

    /// The scene settings a session shares, as text to compare. Not the
    /// frame: each person scrubs their own timeline.
    fn settings_text(&self) -> String {
        let mut playback = self.playback;
        playback.frame = 0.0;
        let world = self.active_world.and_then(|w| self.world_uid(w));
        format!(
            "{playback:?}{:?}{:?}{:?}{world:?}",
            self.render, self.physics, self.design
        )
    }

    /// The datablocks `changes` names, with what they point at, as a document
    /// of their own that keeps every identity.
    pub fn patch(&self, changes: &Changes) -> Document {
        use DatablockKind as K;
        let mut carry = Carry {
            src: self,
            out: Document::new(),
            changes,
            objects: HashMap::new(),
            meshes: HashMap::new(),
            materials: HashMap::new(),
            images: HashMap::new(),
            cameras: HashMap::new(),
            lights: HashMap::new(),
            worlds: HashMap::new(),
            actions: HashMap::new(),
            armatures: HashMap::new(),
        };
        for (kind, uid) in &changes.changed {
            match kind {
                K::Object => {
                    if let Some(id) = self.object_by_uid(*uid) {
                        carry.object(id);
                    }
                }
                K::Mesh => {
                    if let Some(id) = self.mesh_by_uid(*uid) {
                        carry.mesh(id);
                    }
                }
                K::Material => {
                    if let Some(id) = find(&self.materials, *uid) {
                        carry.material(id);
                    }
                }
                K::Image => {
                    if let Some(id) = find(&self.images, *uid) {
                        carry.image(id);
                    }
                }
                K::Camera => {
                    if let Some(id) = find(&self.cameras, *uid) {
                        carry.camera(id);
                    }
                }
                K::Light => {
                    if let Some(id) = find(&self.lights, *uid) {
                        carry.light(id);
                    }
                }
                K::World => {
                    if let Some(id) = find(&self.worlds, *uid) {
                        carry.world(id);
                    }
                }
                K::Action => {
                    if let Some(id) = find(&self.actions, *uid) {
                        carry.action(id);
                    }
                }
                K::Armature => {
                    if let Some(id) = find(&self.armatures, *uid) {
                        carry.armature(id);
                    }
                }
            }
        }
        if changes.settings {
            carry.out.playback = self.playback;
            carry.out.render = self.render.clone();
            carry.out.physics = self.physics;
            carry.out.design = self.design.clone();
            let world = self.active_world.and_then(|w| carry.world(w));
            carry.out.active_world = world;
        }
        carry.out
    }

    /// Takes in a patch made by [`Document::patch`] from another copy.
    ///
    /// A datablock `changes` names replaces this copy's one of the same
    /// identity, or is added with that identity; a stand-in is only looked
    /// up. Returns the objects that arrived or changed.
    pub fn apply_patch(&mut self, patch: &Document, changes: &Changes) -> Vec<ObjectId> {
        use DatablockKind as K;
        let fresh = |kind, uid| changes.has(kind, uid);

        // Images, actions and armatures point at nothing.
        let images = merge_arc(&mut self.images, &patch.images, K::Image, &fresh, |_| {});
        let actions = merge_arc(&mut self.actions, &patch.actions, K::Action, &fresh, |_| {});
        let armatures = merge_arc(
            &mut self.armatures,
            &patch.armatures,
            K::Armature,
            &fresh,
            |_| {},
        );
        let materials = merge_arc(
            &mut self.materials,
            &patch.materials,
            K::Material,
            &fresh,
            |m: &mut Material| {
                remap_material(m, |i| images.get(&i).copied(), |a| actions.get(&a).copied())
            },
        );
        let cameras = merge_arc(&mut self.cameras, &patch.cameras, K::Camera, &fresh, |_| {});
        let lights = merge_arc(
            &mut self.lights,
            &patch.lights,
            K::Light,
            &fresh,
            |l: &mut crate::light::Light| {
                l.action = l.action.and_then(|a| actions.get(&a).copied())
            },
        );
        let worlds = merge_arc(
            &mut self.worlds,
            &patch.worlds,
            K::World,
            &fresh,
            |w: &mut crate::light::World| {
                if let Background::Image(image) = w.background {
                    match images.get(&image) {
                        Some(i) => w.background = Background::Image(*i),
                        None => w.background = Background::Flat(Default::default()),
                    }
                }
            },
        );

        let mut meshes: HashMap<MeshId, MeshId> = HashMap::new();
        for (pid, entry) in patch.meshes.iter() {
            let slots: Vec<Option<MaterialId>> = entry
                .slots
                .iter()
                .map(|s| s.and_then(|m| materials.get(&m).copied()))
                .collect();
            let id = match find_mesh(&self.meshes, entry.uid) {
                Some(id) => {
                    if fresh(K::Mesh, entry.uid) {
                        let mine = &mut self.meshes[id];
                        mine.data = entry.data.clone();
                        mine.slots = slots;
                    }
                    id
                }
                None => {
                    self.site_uid_seen(K::Mesh, entry.uid);
                    self.meshes.insert(MeshEntry {
                        uid: entry.uid,
                        data: entry.data.clone(),
                        slots,
                    })
                }
            };
            meshes.insert(pid, id);
        }

        // Objects in two passes: the parent may come after its child.
        let mut objects: HashMap<ObjectId, ObjectId> = HashMap::new();
        let mut arrived = Vec::new();
        for (pid, object) in patch.objects.iter() {
            let existing = self.object_by_uid(object.uid);
            if !fresh(K::Object, object.uid) {
                // A stand-in for a parent. One this copy no longer has was
                // deleted here, and a reference is no reason to bring it back.
                if let Some(id) = existing {
                    objects.insert(pid, id);
                }
                continue;
            }
            let mut copy = object.clone();
            copy.data = match object.data {
                ObjectData::Mesh(m) => meshes
                    .get(&m)
                    .map_or(ObjectData::Empty, |m| ObjectData::Mesh(*m)),
                ObjectData::Light(l) => lights
                    .get(&l)
                    .map_or(ObjectData::Empty, |l| ObjectData::Light(*l)),
                ObjectData::Camera(c) => cameras
                    .get(&c)
                    .map_or(ObjectData::Empty, |c| ObjectData::Camera(*c)),
                ObjectData::Armature(a) => armatures
                    .get(&a)
                    .map_or(ObjectData::Empty, |a| ObjectData::Armature(*a)),
                ObjectData::Empty => ObjectData::Empty,
            };
            copy.material_overrides = object
                .material_overrides
                .iter()
                .map(|s| s.and_then(|m| materials.get(&m).copied()))
                .collect();
            copy.action = object.action.and_then(|a| actions.get(&a).copied());
            copy.nla = object
                .nla
                .iter()
                .filter_map(|s| {
                    Some(crate::nla::NlaStrip {
                        action: *actions.get(&s.action)?,
                        ..s.clone()
                    })
                })
                .collect();
            let id = match existing {
                Some(id) => {
                    // The parent is the second pass's; keep the present one.
                    copy.parent = self.objects[id].parent;
                    self.objects[id] = copy;
                    id
                }
                None => {
                    copy.parent = None;
                    self.site_uid_seen(K::Object, object.uid);
                    let uid = object.uid;
                    self.insert_object(copy, uid)
                }
            };
            objects.insert(pid, id);
            arrived.push((pid, id));
        }
        for (pid, id) in &arrived {
            let wanted = patch.objects[*pid]
                .parent
                .and_then(|p| objects.get(&p).copied());
            // Two people parenting each to the other at once would make a
            // loop; the later one is let go.
            let wanted = wanted.filter(|p| *p != *id && !self.is_ancestor(*id, *p));
            let was_root = self.objects[*id].parent.is_none();
            self.objects[*id].parent = wanted;
            match (was_root, wanted.is_none()) {
                (true, false) => self.roots.retain(|r| r != id),
                (false, true) => self.roots.push(*id),
                _ => {}
            }
        }

        for uid in &changes.removed {
            if let Some(id) = self.object_by_uid(*uid) {
                self.remove_object(id);
            }
        }

        if changes.settings {
            let frame = self.playback.frame;
            self.playback = patch.playback;
            self.playback.frame = frame;
            self.render = patch.render.clone();
            self.physics = patch.physics;
            self.design = patch.design.clone();
            self.active_world = patch.active_world.and_then(|w| worlds.get(&w).copied());
        }
        let _ = (images, armatures);
        self.touch();
        arrived.into_iter().map(|(_, id)| id).collect()
    }

    /// Keeps this copy's own counter past an identity that arrived, in case it
    /// is in this copy's range — its own work coming back after a rejoin.
    fn site_uid_seen(&mut self, kind: DatablockKind, uid: Uid) {
        let counter = self.uids.of(kind);
        *counter = (*counter).max(uid);
        if self.site != 0 && uid.raw() >> SITE_SHIFT == u64::from(self.site) {
            let counter = self.site_uids.of(kind);
            *counter = (*counter).max(uid);
        }
    }
}

/// Points a material's images and action through `image` and `action`.
fn remap_material(
    material: &mut Material,
    mut image: impl FnMut(ImageId) -> Option<ImageId>,
    mut action: impl FnMut(ActionId) -> Option<ActionId>,
) {
    for map in super::MaterialMap::ALL {
        let slot = material.map_mut(map);
        *slot = slot.and_then(&mut image);
    }
    if let Some(graph) = &mut material.graph {
        for node in &mut graph.nodes {
            if let crate::shader::NodeKind::Image { image: i, .. } = &mut node.kind {
                *i = i.and_then(&mut image);
            }
        }
    }
    material.action = material.action.and_then(&mut action);
}

trait HasUid {
    fn uid(&self) -> Uid;
    fn set_uid(&mut self, uid: Uid);
}
macro_rules! has_uid {
    ($($t:ty),*) => {$(
        impl HasUid for $t {
            fn uid(&self) -> Uid { self.uid }
            fn set_uid(&mut self, uid: Uid) { self.uid = uid; }
        }
    )*};
}
has_uid!(
    Material,
    Image,
    crate::camera::Camera,
    crate::light::Light,
    crate::light::World,
    crate::animation::Action,
    crate::armature::Armature
);

fn find<I: kreate_mesh::arena::ArenaIndex, T: HasUid>(
    arena: &kreate_mesh::arena::Arena<I, Arc<T>>,
    uid: Uid,
) -> Option<I> {
    arena.iter().find(|(_, b)| b.uid() == uid).map(|(id, _)| id)
}

fn find_mesh(arena: &kreate_mesh::arena::Arena<MeshId, MeshEntry>, uid: Uid) -> Option<MeshId> {
    arena.iter().find(|(_, m)| m.uid == uid).map(|(id, _)| id)
}

/// Takes one kind of `Arc`ed datablock in from a patch; returns where each of
/// the patch's ended up here.
fn merge_arc<I, T>(
    mine: &mut kreate_mesh::arena::Arena<I, Arc<T>>,
    theirs: &kreate_mesh::arena::Arena<I, Arc<T>>,
    kind: DatablockKind,
    fresh: &impl Fn(DatablockKind, Uid) -> bool,
    mut remap: impl FnMut(&mut T),
) -> HashMap<I, I>
where
    I: kreate_mesh::arena::ArenaIndex + std::hash::Hash + Eq,
    T: HasUid + Clone,
{
    let mut map = HashMap::new();
    for (pid, block) in theirs.iter() {
        let uid = block.uid();
        let id = match find(mine, uid) {
            Some(id) if !fresh(kind, uid) => id,
            existing => {
                let mut copy = (**block).clone();
                remap(&mut copy);
                copy.set_uid(uid);
                match existing {
                    Some(id) => {
                        mine[id] = Arc::new(copy);
                        id
                    }
                    None => mine.insert(Arc::new(copy)),
                }
            }
        };
        map.insert(pid, id);
    }
    map
}

/// Builds a patch: each datablock carried once, a changed one whole and an
/// unchanged one as a stand-in.
struct Carry<'a> {
    src: &'a Document,
    out: Document,
    changes: &'a Changes,
    objects: HashMap<ObjectId, ObjectId>,
    meshes: HashMap<MeshId, MeshId>,
    materials: HashMap<MaterialId, MaterialId>,
    images: HashMap<ImageId, ImageId>,
    cameras: HashMap<CameraId, CameraId>,
    lights: HashMap<LightId, LightId>,
    worlds: HashMap<WorldId, WorldId>,
    actions: HashMap<ActionId, ActionId>,
    armatures: HashMap<ArmatureId, ArmatureId>,
}

impl Carry<'_> {
    fn object(&mut self, id: ObjectId) -> Option<ObjectId> {
        if let Some(done) = self.objects.get(&id) {
            return Some(*done);
        }
        let source = self.src.objects.get(id)?;
        let uid = source.uid;
        let copy = if self.changes.has(DatablockKind::Object, uid) {
            let mut copy = source.clone();
            copy.data = match source.data {
                ObjectData::Mesh(m) => self.mesh(m).map_or(ObjectData::Empty, ObjectData::Mesh),
                ObjectData::Light(l) => self.light(l).map_or(ObjectData::Empty, ObjectData::Light),
                ObjectData::Camera(c) => {
                    self.camera(c).map_or(ObjectData::Empty, ObjectData::Camera)
                }
                ObjectData::Armature(a) => self
                    .armature(a)
                    .map_or(ObjectData::Empty, ObjectData::Armature),
                ObjectData::Empty => ObjectData::Empty,
            };
            copy.material_overrides = source
                .material_overrides
                .iter()
                .map(|s| s.and_then(|m| self.material(m)))
                .collect();
            copy.action = source.action.and_then(|a| self.action(a));
            copy.nla = source
                .nla
                .iter()
                .filter_map(|s| {
                    Some(crate::nla::NlaStrip {
                        action: self.action(s.action)?,
                        ..s.clone()
                    })
                })
                .collect();
            copy.parent = source.parent.and_then(|p| self.object(p));
            copy
        } else {
            Object::new(source.name.clone(), ObjectData::Empty)
        };
        let made = self.out.restore_object(copy, uid);
        self.objects.insert(id, made);
        Some(made)
    }

    fn mesh(&mut self, id: MeshId) -> Option<MeshId> {
        if let Some(done) = self.meshes.get(&id) {
            return Some(*done);
        }
        let entry = self.src.meshes.get(id)?;
        let whole = self.changes.has(DatablockKind::Mesh, entry.uid);
        let data = if whole {
            entry.data.clone()
        } else {
            Arc::new(MeshData::empty())
        };
        let slots: Vec<Option<MaterialId>> = if whole {
            entry
                .slots
                .clone()
                .into_iter()
                .map(|s| s.and_then(|m| self.material(m)))
                .collect()
        } else {
            Vec::new()
        };
        let made = self.out.meshes.insert(MeshEntry {
            uid: entry.uid,
            data,
            slots,
        });
        self.meshes.insert(id, made);
        Some(made)
    }

    fn material(&mut self, id: MaterialId) -> Option<MaterialId> {
        if let Some(done) = self.materials.get(&id) {
            return Some(*done);
        }
        let mut copy = (**self.src.materials.get(id)?).clone();
        let images: Vec<(ImageId, Option<ImageId>)> = copy
            .images()
            .into_iter()
            .map(|i| (i, self.image(i)))
            .collect();
        let action = copy.action.and_then(|a| self.action(a));
        remap_material(
            &mut copy,
            |i| {
                images
                    .iter()
                    .find(|(from, _)| *from == i)
                    .and_then(|(_, to)| *to)
            },
            |_| action,
        );
        let uid = copy.uid;
        let made = self.out.restore_material(copy, uid);
        self.materials.insert(id, made);
        Some(made)
    }

    fn image(&mut self, id: ImageId) -> Option<ImageId> {
        if let Some(done) = self.images.get(&id) {
            return Some(*done);
        }
        let source = self.src.images.get(id)?;
        let copy = if self.changes.has(DatablockKind::Image, source.uid) {
            (**source).clone()
        } else {
            Image::new(source.name.clone(), 1, 1, PixelFormat::Rgba8)
        };
        let made = self.out.restore_image(copy, source.uid);
        self.images.insert(id, made);
        Some(made)
    }

    fn camera(&mut self, id: CameraId) -> Option<CameraId> {
        if let Some(done) = self.cameras.get(&id) {
            return Some(*done);
        }
        let copy = (**self.src.cameras.get(id)?).clone();
        let uid = copy.uid;
        let made = self.out.restore_camera(copy, uid);
        self.cameras.insert(id, made);
        Some(made)
    }

    fn light(&mut self, id: LightId) -> Option<LightId> {
        if let Some(done) = self.lights.get(&id) {
            return Some(*done);
        }
        let mut copy = (**self.src.lights.get(id)?).clone();
        copy.action = copy.action.and_then(|a| self.action(a));
        let uid = copy.uid;
        let made = self.out.restore_light(copy, uid);
        self.lights.insert(id, made);
        Some(made)
    }

    fn world(&mut self, id: WorldId) -> Option<WorldId> {
        if let Some(done) = self.worlds.get(&id) {
            return Some(*done);
        }
        let mut copy = (**self.src.worlds.get(id)?).clone();
        if let Background::Image(image) = copy.background {
            copy.background = match self.image(image) {
                Some(i) => Background::Image(i),
                None => Background::Flat(Default::default()),
            };
        }
        let uid = copy.uid;
        let made = self.out.restore_world(copy, uid);
        self.worlds.insert(id, made);
        Some(made)
    }

    fn action(&mut self, id: ActionId) -> Option<ActionId> {
        if let Some(done) = self.actions.get(&id) {
            return Some(*done);
        }
        let copy = (**self.src.actions.get(id)?).clone();
        let uid = copy.uid;
        let made = self.out.restore_action(copy, uid);
        self.actions.insert(id, made);
        Some(made)
    }

    fn armature(&mut self, id: ArmatureId) -> Option<ArmatureId> {
        if let Some(done) = self.armatures.get(&id) {
            return Some(*done);
        }
        let copy = (**self.src.armatures.get(id)?).clone();
        let uid = copy.uid;
        let made = self.out.restore_armature(copy, uid);
        self.armatures.insert(id, made);
        Some(made)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DVec3;

    fn shared() -> (Document, Document) {
        let mut a = Document::new();
        let cube = a.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let material = a.add_material(Material {
            name: "Red".into(),
            ..Default::default()
        });
        let mesh = a.object(cube).unwrap().mesh().unwrap();
        a.set_mesh_slots(mesh, vec![Some(material)]);
        // The second copy starts as the first, the way a joiner does.
        let all = Changes {
            changed: a.all_keys().into_iter().collect(),
            ..Default::default()
        };
        let mut b = Document::new();
        b.apply_patch(&a.patch(&all), &all);
        a.set_uid_site(1);
        b.set_uid_site(2);
        (a, b)
    }

    fn send(from: &Document, base: &Document, to: &mut Document) -> Changes {
        let changes = from.changes_since(base);
        to.apply_patch(&from.patch(&changes), &changes);
        changes
    }

    #[test]
    fn a_joiner_gets_the_same_scene_with_the_same_identities() {
        let (a, b) = shared();
        assert_eq!(b.object_count(), 1);
        let (_, cube) = b.objects().next().unwrap();
        assert_eq!(cube.uid, a.objects().next().unwrap().1.uid);
        let mesh = cube.mesh().unwrap();
        assert_eq!(
            b.mesh(mesh).unwrap().num_faces(),
            a.meshes().next().unwrap().1.num_faces()
        );
        let slot = b.mesh_slots(mesh)[0].expect("the slot came along");
        assert_eq!(b.material(slot).unwrap().name, "Red");
    }

    #[test]
    fn a_move_sends_the_object_and_not_its_mesh() {
        let (mut a, mut b) = shared();
        let base = a.clone();
        let id = a.objects().next().unwrap().0;
        a.object_mut(id).unwrap().transform.translation = DVec3::new(3.0, 0.0, 0.0);
        let changes = a.changes_since(&base);
        assert_eq!(changes.changed.len(), 1, "{changes:?}");
        let patch = a.patch(&changes);
        assert_eq!(
            patch.meshes().next().unwrap().1.num_faces(),
            0,
            "the mesh is only a stand-in"
        );
        b.apply_patch(&patch, &changes);
        let (_, moved) = b.objects().next().unwrap();
        assert_eq!(moved.transform.translation.x, 3.0);
        let mesh = moved.mesh().unwrap();
        assert_eq!(
            b.mesh(mesh).unwrap().num_faces(),
            6,
            "the cube is still whole"
        );
    }

    #[test]
    fn two_people_adding_at_once_make_two_objects() {
        let (mut a, mut b) = shared();
        let (base_a, base_b) = (a.clone(), b.clone());
        a.add_mesh_object("FromA", kreate_mesh::primitives::cube(1.0));
        b.add_mesh_object("FromB", kreate_mesh::primitives::cube(1.0));
        let to_b = a.changes_since(&base_a);
        let to_a = b.changes_since(&base_b);
        b.apply_patch(&a.patch(&to_b), &to_b);
        a.apply_patch(&b.patch(&to_a), &to_a);
        assert_eq!(a.object_count(), 3);
        assert_eq!(b.object_count(), 3);
        let names = |d: &Document| {
            let mut n: Vec<_> = d.objects().map(|(_, o)| (o.uid, o.name.clone())).collect();
            n.sort();
            n
        };
        assert_eq!(names(&a), names(&b));
    }

    #[test]
    fn edits_to_different_objects_both_survive() {
        let (mut a, mut b) = shared();
        let other = a.add_mesh_object("Other", kreate_mesh::primitives::cube(1.0));
        send(&a, &Document::new(), &mut b);
        let (base_a, base_b) = (a.clone(), b.clone());
        a.object_mut(other).unwrap().transform.translation.y = 5.0;
        let cube_b = b.objects().find(|(_, o)| o.name == "Cube").unwrap().0;
        b.object_mut(cube_b).unwrap().transform.translation.z = 7.0;
        let ab = a.changes_since(&base_a);
        let ba = b.changes_since(&base_b);
        b.apply_patch(&a.patch(&ab), &ab);
        a.apply_patch(&b.patch(&ba), &ba);
        for doc in [&a, &b] {
            let find = |n: &str| doc.objects().find(|(_, o)| o.name == n).unwrap().1;
            assert_eq!(find("Other").transform.translation.y, 5.0);
            assert_eq!(find("Cube").transform.translation.z, 7.0);
        }
    }

    #[test]
    fn a_deletion_and_a_reparent_arrive() {
        let (mut a, mut b) = shared();
        let child = a.add_mesh_object("Child", kreate_mesh::primitives::cube(1.0));
        let doomed = a.add_mesh_object("Doomed", kreate_mesh::primitives::cube(1.0));
        send(&a, &Document::new(), &mut b);
        let base = a.clone();
        let cube = a.objects().find(|(_, o)| o.name == "Cube").unwrap().0;
        a.set_parent(child, Some(cube)).unwrap();
        a.remove_object(doomed);
        send(&a, &base, &mut b);
        assert_eq!(b.object_count(), 2);
        let child_b = b.objects().find(|(_, o)| o.name == "Child").unwrap();
        let parent = b.object(child_b.1.parent.expect("parented")).unwrap();
        assert_eq!(parent.name, "Cube");
        assert!(!b.roots().contains(&child_b.0));
    }

    #[test]
    fn a_rejoining_site_counts_past_its_own_earlier_work() {
        let mut doc = Document::new();
        doc.set_uid_site(3);
        let first = doc.add_mesh_object("A", kreate_mesh::primitives::cube(1.0));
        let first = doc.object_uid(first).unwrap();
        assert_eq!(first.raw() >> SITE_SHIFT, 3);
        doc.set_uid_site(3);
        let second = doc.add_mesh_object("B", kreate_mesh::primitives::cube(1.0));
        assert!(doc.object_uid(second).unwrap() > first);
    }
}
