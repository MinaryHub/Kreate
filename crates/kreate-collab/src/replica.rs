//! One participant's side of keeping a scene in step.
//!
//! The rule is last writer wins, and the relay's order is what "last" means.
//! Edits are shown at once and sent; until the relay answers one, a remote
//! edit to the same datablock is older than it in the relay's order, so it is
//! passed over here exactly as the relay's copy will be overwritten by ours.
//!
//! Everything is done against `base`: the scene as last agreed. What differs
//! from it is this participant's own work, waiting to go.

use std::collections::{BTreeSet, HashMap, VecDeque};

use kreate_io::sync;
use kreate_scene::{BlockKey, Changes, DatablockKind, Document, Uid};

/// A participant's record of the session's scene.
#[derive(Debug, Default)]
pub struct Replica {
    /// The scene as last agreed: what was sent or taken in.
    base: Document,
    /// Patches sent and not yet answered, oldest first.
    in_flight: VecDeque<Sent>,
    /// Who last wrote each datablock, as far as this participant has seen.
    authors: HashMap<BlockKey, u16>,
    /// This participant's number.
    site: u16,
}

#[derive(Debug)]
struct Sent {
    changed: BTreeSet<BlockKey>,
    removed: BTreeSet<Uid>,
}

impl Replica {
    /// Starts from `doc` as participant `site`.
    pub fn new(doc: &mut Document, site: u16) -> Self {
        doc.set_uid_site(site);
        Self {
            base: doc.clone(),
            in_flight: VecDeque::new(),
            authors: HashMap::new(),
            site,
        }
    }

    pub fn site(&self) -> u16 {
        self.site
    }

    /// Replaces `doc` with the session's scene, from a snapshot's body.
    pub fn adopt_snapshot(
        &mut self,
        doc: &mut Document,
        bytes: &[u8],
    ) -> Result<(), sync::SyncError> {
        let mut fresh = Document::new();
        sync::apply(&mut fresh, bytes)?;
        doc.adopt(fresh);
        doc.set_uid_site(self.site);
        self.base = doc.clone();
        self.in_flight.clear();
        self.authors.clear();
        Ok(())
    }

    /// The whole of `doc`, to seed a session nobody else is in yet.
    pub fn seed(&mut self, doc: &Document) -> Option<Vec<u8>> {
        let all = sync::everything(doc);
        let bytes = sync::encode(doc, &all).ok()?;
        self.sent(doc, all);
        Some(bytes)
    }

    /// What this participant changed since it last sent, as a patch to send;
    /// `None` when nothing did.
    pub fn outgoing(&mut self, doc: &Document) -> Option<Vec<u8>> {
        let changes = doc.changes_since(&self.base);
        if changes.is_empty() {
            return None;
        }
        let bytes = sync::encode(doc, &changes).ok()?;
        self.sent(doc, changes);
        Some(bytes)
    }

    fn sent(&mut self, doc: &Document, changes: Changes) {
        for key in &changes.changed {
            self.authors.insert(*key, self.site);
        }
        for uid in &changes.removed {
            self.authors
                .insert((DatablockKind::Object, *uid), self.site);
        }
        self.in_flight.push_back(Sent {
            changed: changes.changed,
            removed: changes.removed,
        });
        self.base = doc.clone();
    }

    /// The relay took this participant's oldest patch.
    pub fn ack(&mut self) {
        self.in_flight.pop_front();
    }

    /// How many patches are waiting on the relay.
    pub fn waiting(&self) -> usize {
        self.in_flight.len()
    }

    /// Takes in someone else's edit. **Call [`Replica::outgoing`] first** and
    /// send what it gives: `base` and `doc` must agree before a remote edit
    /// lands on both. Returns what changed here, `None` when nothing did.
    pub fn incoming(
        &mut self,
        doc: &mut Document,
        from: u16,
        bytes: &[u8],
    ) -> Result<Option<Changes>, sync::SyncError> {
        let (patch, mut changes) = sync::decode(bytes)?;
        let mine_changed: BTreeSet<BlockKey> = self
            .in_flight
            .iter()
            .flat_map(|s| s.changed.iter().copied())
            .collect();
        let mine_removed: BTreeSet<Uid> = self
            .in_flight
            .iter()
            .flat_map(|s| s.removed.iter().copied())
            .collect();
        // Ours is later in the relay's order than anything still arriving.
        changes.changed.retain(|key| {
            !mine_changed.contains(key)
                && !(key.0 == DatablockKind::Object && mine_removed.contains(&key.1))
        });
        changes
            .removed
            .retain(|uid| !mine_changed.contains(&(DatablockKind::Object, *uid)));
        if changes.is_empty() {
            return Ok(None);
        }
        for key in &changes.changed {
            self.authors.insert(*key, from);
        }
        for uid in &changes.removed {
            self.authors.insert((DatablockKind::Object, *uid), from);
        }
        doc.apply_patch(&patch, &changes);
        self.base = doc.clone();
        Ok(Some(changes))
    }

    /// Takes what `doc` is now as agreed, without sending it: for what
    /// follows from a remote edit the same way on every participant — a
    /// rigged joint's bone put under the joint someone moved — which, sent
    /// back, would cross the next edit of the one who made it and undo it.
    /// Only right straight after [`Replica::incoming`], when nothing of this
    /// participant's own is waiting to go.
    pub fn absorb(&mut self, doc: &Document) {
        self.base = doc.clone();
    }

    /// After an undo or redo: puts back what someone else did, so stepping
    /// back through one's own history takes back only one's own edits.
    ///
    /// The history's snapshots were taken before others' edits arrived, so
    /// restoring one turns theirs back as well. Every datablock that differs
    /// from the agreed scene and was last written by someone else is set to
    /// the agreed state again.
    pub fn keep_others(&mut self, doc: &mut Document) {
        let changes = doc.changes_since(&self.base);
        let theirs = |key: &BlockKey| self.authors.get(key).is_some_and(|a| *a != self.site);
        let mut back = Changes::default();
        for key in &changes.changed {
            if !theirs(key) {
                continue;
            }
            let in_base = match key.0 {
                DatablockKind::Object => self.base.object_by_uid(key.1).is_some(),
                _ => true,
            };
            if in_base {
                back.changed.insert(*key);
            } else {
                // Brought back by the undo after its author deleted it.
                back.removed.insert(key.1);
            }
        }
        for uid in &changes.removed {
            if theirs(&(DatablockKind::Object, *uid)) {
                back.changed.insert((DatablockKind::Object, *uid));
            }
        }
        if back.is_empty() {
            return;
        }
        // Their objects' meshes and the rest come whole from the agreed scene.
        let mut whole = back.clone();
        for key in &back.changed {
            if key.0 == DatablockKind::Object {
                if let Some(id) = self.base.object_by_uid(key.1) {
                    if let Some(mesh) = self.base.object(id).and_then(|o| o.mesh()) {
                        if let Some(uid) = self.base.mesh_uid(mesh) {
                            if theirs(&(DatablockKind::Mesh, uid)) || doc.mesh_by_uid(uid).is_none()
                            {
                                whole.changed.insert((DatablockKind::Mesh, uid));
                            }
                        }
                    }
                }
            }
        }
        let patch = self.base.patch(&whole);
        doc.apply_patch(&patch, &whole);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DVec3;

    /// Two participants and the relay's order between them, by hand.
    fn pair() -> (Document, Replica, Document, Replica) {
        let mut a = Document::new();
        a.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let mut ra = Replica::new(&mut a, 1);
        let seed = ra.seed(&a).unwrap();
        ra.ack();
        let mut b = Document::new();
        let mut rb = Replica::new(&mut b, 2);
        rb.adopt_snapshot(&mut b, &seed).unwrap();
        (a, ra, b, rb)
    }

    fn cube(doc: &Document) -> kreate_scene::ObjectId {
        doc.objects().find(|(_, o)| o.name == "Cube").unwrap().0
    }

    #[test]
    fn a_remote_edit_older_than_ours_is_passed_over() {
        let (mut a, mut ra, mut b, mut rb) = pair();
        let (ca, cb) = (cube(&a), cube(&b));
        a.object_mut(ca).unwrap().transform.translation.x = 1.0;
        b.object_mut(cb).unwrap().transform.translation.x = 2.0;
        let pa = ra.outgoing(&a).unwrap();
        let pb = rb.outgoing(&b).unwrap();
        // Relay order: A, then B. B hears of A's before its own ack.
        rb.incoming(&mut b, 1, &pa).unwrap();
        rb.ack();
        ra.ack();
        ra.incoming(&mut a, 2, &pb).unwrap();
        let x = |d: &Document| d.object(cube(d)).unwrap().transform.translation.x;
        assert_eq!(x(&a), 2.0, "B's was last");
        assert_eq!(x(&b), 2.0, "and B kept it");
    }

    #[test]
    fn undo_takes_back_only_ones_own_edit() {
        let (mut a, mut ra, mut b, mut rb) = pair();
        let other = b.add_mesh_object("FromB", kreate_mesh::primitives::cube(1.0));
        let pb = rb.outgoing(&b).unwrap();
        rb.ack();
        let _ = other;

        // A snapshots for undo, then B's object arrives, then A moves the cube.
        let before = a.clone();
        ra.incoming(&mut a, 2, &pb).unwrap();
        let ca = cube(&a);
        a.object_mut(ca).unwrap().transform.translation = DVec3::new(0.0, 3.0, 0.0);
        ra.outgoing(&a).unwrap();
        ra.ack();
        assert_eq!(a.object_count(), 2);

        // Undo puts the old snapshot back — without B's object — and the
        // replica brings B's object back.
        a.adopt(before);
        ra.keep_others(&mut a);
        assert_eq!(a.object_count(), 2, "B's object is still there");
        let from_b = a.objects().find(|(_, o)| o.name == "FromB").unwrap().1;
        assert_eq!(
            a.mesh(from_b.mesh().unwrap()).unwrap().num_faces(),
            6,
            "with its mesh"
        );
        assert_eq!(a.object(cube(&a)).unwrap().transform.translation.y, 0.0);
        let p = ra.outgoing(&a).unwrap();
        let (_, changes) = sync::decode(&p).unwrap();
        assert_eq!(changes.changed.len(), 1, "only the cube's move goes back");
    }
}
