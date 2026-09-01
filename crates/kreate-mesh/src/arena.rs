//! Typed index arenas for mesh elements.
//!
//! Written rather than taken from `slotmap`, for reasons that are specific to a
//! mesh kernel rather than general dislike:
//!
//! - **Key size.** `slotmap`'s keys are 8 bytes. A `BMesh` stores roughly 4 loops
//!   and 2 edges per vertex, each holding several element references, so at
//!   10 M vertices the difference between 4- and 8-byte handles is measured in
//!   gigabytes. Here a handle is a bare `u32` in release builds.
//! - **Layout.** `slotmap` is array-of-structs. The attribute system is
//!   structure-of-arrays, and the two fight.
//! - **Compaction.** Mesh editing needs bulk compaction with a remap table so
//!   attribute layers can be reindexed in lockstep. There is no hook for that.
//!
//! # Generation checking
//!
//! Handles carry a generation counter **in debug builds only**. Debug builds pay
//! 8 bytes per handle and catch use-after-free precisely, at the moment it
//! happens; release builds pay 4 bytes and no check. Since the arenas are purely
//! in-memory — nothing here is ever serialised — the size difference between
//! profiles is invisible outside this module.

use std::marker::PhantomData;

/// A handle into an [`Arena`].
///
/// Implemented by the newtypes that [`define_id!`] generates. Not intended to be
/// implemented by hand.
pub trait ArenaIndex: Copy + Eq + std::fmt::Debug {
    fn from_parts(index: u32, generation: u32) -> Self;
    fn index(self) -> u32;
    /// The generation this handle was issued with, or `0` in release builds.
    fn generation(self) -> u32;
}

/// Defines a typed arena handle.
///
/// The types are distinct so that a face arena cannot be indexed with a vertex
/// handle. In a mesh kernel that confusion is easy to write, silent at runtime,
/// and produces corruption rather than a crash.
#[macro_export]
macro_rules! define_id {
    ($(#[$meta:meta])* $vis:vis struct $name:ident;) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        $vis struct $name {
            index: u32,
            #[cfg(debug_assertions)]
            generation: u32,
        }

        impl $crate::arena::ArenaIndex for $name {
            #[inline]
            fn from_parts(index: u32, _generation: u32) -> Self {
                Self {
                    index,
                    #[cfg(debug_assertions)]
                    generation: _generation,
                }
            }
            #[inline]
            fn index(self) -> u32 {
                self.index
            }
            #[inline]
            fn generation(self) -> u32 {
                #[cfg(debug_assertions)]
                { self.generation }
                #[cfg(not(debug_assertions))]
                { 0 }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.index)
            }
        }
    };
}

/// A slot-based arena with stable handles.
///
/// Slots freed by [`Arena::remove`] are reused by later inserts, so indices stay
/// dense enough for parallel attribute arrays to remain practical. Iteration is
/// in slot order and therefore deterministic.
#[derive(Clone, Debug)]
pub struct Arena<I: ArenaIndex, T> {
    slots: Vec<Option<T>>,
    /// Bumped every time a slot is freed, so a stale handle can be detected.
    generations: Vec<u32>,
    /// Free slots, most recently freed first.
    free: Vec<u32>,
    len: u32,
    _marker: PhantomData<fn() -> I>,
}

impl<I: ArenaIndex, T> Default for Arena<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: ArenaIndex, T> Arena<I, T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            generations: Vec::with_capacity(cap),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Number of live elements. Not the same as the slot count once anything has
    /// been removed — see [`Arena::slot_count`].
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total slots, live and free. This is the length parallel arrays must have.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Fraction of slots that are free. Above roughly 0.3 it is worth calling
    /// [`Arena::compact`] to keep parallel arrays from wasting memory.
    pub fn fragmentation(&self) -> f32 {
        if self.slots.is_empty() {
            0.0
        } else {
            self.free.len() as f32 / self.slots.len() as f32
        }
    }

    pub fn insert(&mut self, value: T) -> I {
        self.len += 1;
        match self.free.pop() {
            Some(i) => {
                self.slots[i as usize] = Some(value);
                I::from_parts(i, self.generations[i as usize])
            }
            None => {
                let i = self.slots.len() as u32;
                self.slots.push(Some(value));
                self.generations.push(0);
                I::from_parts(i, 0)
            }
        }
    }

    /// Removes and returns an element, or `None` if the handle is stale or the
    /// slot is already empty.
    pub fn remove(&mut self, id: I) -> Option<T> {
        let i = id.index() as usize;
        if i >= self.slots.len() || !self.generation_matches(id) {
            return None;
        }
        let taken = self.slots[i].take();
        if taken.is_some() {
            self.len -= 1;
            // Bump on free, not on reuse, so every outstanding handle to this
            // slot becomes detectably stale immediately.
            self.generations[i] = self.generations[i].wrapping_add(1);
            self.free.push(i as u32);
        }
        taken
    }

    #[inline]
    fn generation_matches(&self, id: I) -> bool {
        if cfg!(debug_assertions) {
            self.generations
                .get(id.index() as usize)
                .is_some_and(|&g| g == id.generation())
        } else {
            true
        }
    }

    #[inline]
    pub fn get(&self, id: I) -> Option<&T> {
        if !self.generation_matches(id) {
            return None;
        }
        self.slots.get(id.index() as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut(&mut self, id: I) -> Option<&mut T> {
        if !self.generation_matches(id) {
            return None;
        }
        self.slots.get_mut(id.index() as usize)?.as_mut()
    }

    #[inline]
    pub fn contains(&self, id: I) -> bool {
        self.get(id).is_some()
    }

    pub fn clear(&mut self) {
        // Generations are kept and bumped rather than reset, so handles taken
        // before the clear do not silently become valid again.
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.take().is_some() {
                self.generations[i] = self.generations[i].wrapping_add(1);
            }
        }
        self.free = (0..self.slots.len() as u32).rev().collect();
        self.len = 0;
    }

    /// Live elements in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> + '_ {
        self.slots.iter().enumerate().filter_map(move |(i, s)| {
            s.as_ref()
                .map(|v| (I::from_parts(i as u32, self.generations[i]), v))
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (I, &mut T)> + '_ {
        let generations = &self.generations;
        self.slots.iter_mut().enumerate().filter_map(move |(i, s)| {
            s.as_mut()
                .map(|v| (I::from_parts(i as u32, generations[i]), v))
        })
    }

    pub fn ids(&self) -> impl Iterator<Item = I> + '_ {
        self.iter().map(|(id, _)| id)
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.slots.iter().filter_map(|s| s.as_ref())
    }

    /// Removes free slots, packing live elements toward the front.
    ///
    /// Returns a remap indexed by *old slot*, giving each live element's new
    /// handle. Callers must reindex every parallel array — attribute layers,
    /// adjacency tables — with the same table, which is exactly the bulk hook a
    /// general-purpose slot map does not offer.
    ///
    /// Relative order is preserved, so compaction is deterministic and does not
    /// perturb golden-test output beyond the renumbering itself.
    pub fn compact(&mut self) -> Vec<Option<I>> {
        let mut remap = vec![None; self.slots.len()];
        let mut write = 0usize;
        for (read, entry) in remap.iter_mut().enumerate() {
            if self.slots[read].is_none() {
                continue;
            }
            if write != read {
                self.slots.swap(read, write);
                // The element moved slots, so its old handle is dead. Bump the
                // destination's generation to invalidate any handle that referred
                // to whatever used to live there.
                self.generations[write] = self.generations[write].wrapping_add(1);
            }
            *entry = Some(I::from_parts(write as u32, self.generations[write]));
            write += 1;
        }
        self.slots.truncate(write);
        self.generations.truncate(write);
        self.free.clear();
        debug_assert_eq!(write, self.len as usize);
        remap
    }
}

impl<I: ArenaIndex, T> std::ops::Index<I> for Arena<I, T> {
    type Output = T;
    #[track_caller]
    fn index(&self, id: I) -> &T {
        self.get(id)
            .unwrap_or_else(|| panic!("stale or dangling handle {id:?}"))
    }
}

impl<I: ArenaIndex, T> std::ops::IndexMut<I> for Arena<I, T> {
    #[track_caller]
    fn index_mut(&mut self, id: I) -> &mut T {
        self.get_mut(id)
            .unwrap_or_else(|| panic!("stale or dangling handle {id:?}"))
    }
}

impl<I: ArenaIndex, T> FromIterator<T> for Arena<I, T> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        let mut a = Self::new();
        for v in iter {
            a.insert(v);
        }
        a
    }
}

// The handles the `BMesh` edit structure will use. Defined here so the arena and
// its handles stay in one place.
define_id! {
    /// A vertex in edit mode.
    pub struct VertId;
}
define_id! {
    /// An edge in edit mode.
    pub struct EdgeId;
}
define_id! {
    /// A face-corner: one face's use of one vertex. The unit that carries UVs and
    /// split normals, and the element the radial and loop cycles are built from.
    pub struct LoopId;
}
define_id! {
    /// A face in edit mode.
    pub struct FaceId;
}

#[cfg(test)]
mod tests {
    use super::*;

    define_id! {
        pub struct TestId;
    }
    define_id! {
        pub struct OtherId;
    }

    type A = Arena<TestId, i32>;

    #[test]
    fn insert_and_get() {
        let mut a = A::new();
        let x = a.insert(10);
        let y = a.insert(20);
        assert_eq!(a[x], 10);
        assert_eq!(a[y], 20);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn remove_frees_the_slot_and_shrinks_len() {
        let mut a = A::new();
        let x = a.insert(1);
        a.insert(2);
        assert_eq!(a.remove(x), Some(1));
        assert_eq!(a.len(), 1);
        assert!(!a.contains(x));
        assert_eq!(a.slot_count(), 2, "the slot is retained for reuse");
    }

    #[test]
    fn removed_slots_are_reused() {
        let mut a = A::new();
        let x = a.insert(1);
        a.remove(x);
        let y = a.insert(2);
        assert_eq!(y.index(), x.index(), "the free slot should be reused");
        assert_eq!(a.slot_count(), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn stale_handles_are_detected_after_reuse() {
        // The failure this exists to catch: an operator keeps a handle across a
        // topology edit, the slot is reused by a different element, and the stale
        // handle silently reads the wrong one. Corruption, not a crash.
        let mut a = A::new();
        let old = a.insert(1);
        a.remove(old);
        let new = a.insert(999);
        assert_eq!(new.index(), old.index());
        assert_eq!(a.get(old), None, "the stale handle must not resolve");
        assert_eq!(a.get(new), Some(&999));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn removing_with_a_stale_handle_does_nothing() {
        let mut a = A::new();
        let old = a.insert(1);
        a.remove(old);
        let new = a.insert(2);
        assert_eq!(a.remove(old), None, "must not remove the new occupant");
        assert_eq!(a.get(new), Some(&2));
        assert_eq!(a.len(), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "stale or dangling handle")]
    fn indexing_with_a_stale_handle_panics() {
        let mut a = A::new();
        let x = a.insert(1);
        a.remove(x);
        a.insert(2);
        let _ = a[x];
    }

    #[test]
    fn double_remove_is_not_a_double_free() {
        let mut a = A::new();
        let x = a.insert(1);
        assert_eq!(a.remove(x), Some(1));
        assert_eq!(a.remove(x), None);
        assert_eq!(a.len(), 0);
        // A double-counted free slot would hand the same index out twice.
        let p = a.insert(7);
        let q = a.insert(8);
        assert_ne!(p.index(), q.index());
    }

    #[test]
    fn iteration_is_in_slot_order_and_skips_holes() {
        let mut a = A::new();
        let ids: Vec<_> = (0..5).map(|i| a.insert(i)).collect();
        a.remove(ids[1]);
        a.remove(ids[3]);
        assert_eq!(a.values().copied().collect::<Vec<_>>(), vec![0, 2, 4]);
        assert_eq!(
            a.iter().map(|(id, _)| id.index()).collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
    }

    #[test]
    fn iter_mut_can_write_through() {
        let mut a = A::new();
        let x = a.insert(1);
        for (_, v) in a.iter_mut() {
            *v *= 10;
        }
        assert_eq!(a[x], 10);
    }

    #[test]
    fn compact_packs_and_reports_the_remap() {
        let mut a = A::new();
        let ids: Vec<_> = (0..5).map(|i| a.insert(i * 100)).collect();
        a.remove(ids[1]);
        a.remove(ids[3]);

        let remap = a.compact();
        assert_eq!(a.slot_count(), 3, "free slots should be gone");
        assert_eq!(a.len(), 3);
        assert_eq!(a.values().copied().collect::<Vec<_>>(), vec![0, 200, 400]);

        // The remap is indexed by old slot; removed elements map to None.
        assert_eq!(remap[1], None);
        assert_eq!(remap[3], None);
        assert_eq!(remap[0].unwrap().index(), 0);
        assert_eq!(remap[2].unwrap().index(), 1);
        assert_eq!(remap[4].unwrap().index(), 2);

        // Handles from the remap must resolve to the value they were issued for.
        assert_eq!(a[remap[4].unwrap()], 400);
    }

    #[test]
    fn compact_on_a_dense_arena_is_a_no_op() {
        let mut a = A::new();
        let ids: Vec<_> = (0..3).map(|i| a.insert(i)).collect();
        let remap = a.compact();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(remap[i].unwrap().index(), id.index());
            assert_eq!(a[remap[i].unwrap()], i as i32);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn handles_into_moved_slots_are_invalidated_by_compaction() {
        let mut a = A::new();
        let ids: Vec<_> = (0..3).map(|i| a.insert(i)).collect();
        a.remove(ids[0]);
        a.compact();
        // ids[2] used to name slot 2; its element now lives in slot 1, so the old
        // handle must not silently resolve to whatever occupies slot 2 now.
        assert!(!a.contains(ids[2]));
    }

    #[test]
    fn clear_empties_without_reviving_old_handles() {
        let mut a = A::new();
        let x = a.insert(1);
        a.clear();
        assert!(a.is_empty());
        let y = a.insert(2);
        assert_eq!(a[y], 2);
        #[cfg(debug_assertions)]
        assert!(
            !a.contains(x),
            "a handle from before clear() must stay dead"
        );
        let _ = x;
    }

    #[test]
    fn fragmentation_reports_free_ratio() {
        let mut a = A::new();
        assert_eq!(a.fragmentation(), 0.0);
        let ids: Vec<_> = (0..4).map(|i| a.insert(i)).collect();
        a.remove(ids[0]);
        a.remove(ids[1]);
        assert!((a.fragmentation() - 0.5).abs() < 1e-6);
        a.compact();
        assert_eq!(a.fragmentation(), 0.0);
    }

    #[test]
    fn handles_are_typed_so_arenas_cannot_be_crossed() {
        // Compile-time property, asserted here so the intent is recorded: the
        // line below must not compile.
        //
        //   let a: Arena<TestId, i32> = Arena::new();
        //   let b: Arena<OtherId, i32> = Arena::new();
        //   let id = b.insert(1);
        //   a.get(id);            // ERROR: expected TestId, found OtherId
        let mut a: Arena<TestId, i32> = Arena::new();
        let mut b: Arena<OtherId, i32> = Arena::new();
        let x = a.insert(1);
        let y = b.insert(2);
        assert_eq!(a[x], 1);
        assert_eq!(b[y], 2);
    }

    #[test]
    fn release_handles_are_four_bytes() {
        // The whole reason for not using slotmap. In debug builds the generation
        // doubles this, which is the deliberate trade.
        let expected = if cfg!(debug_assertions) { 8 } else { 4 };
        assert_eq!(std::mem::size_of::<VertId>(), expected);
        assert_eq!(std::mem::size_of::<LoopId>(), expected);
    }

    #[test]
    fn from_iter_builds_a_dense_arena() {
        let a: Arena<TestId, i32> = (0..4).collect();
        assert_eq!(a.len(), 4);
        assert_eq!(a.slot_count(), 4);
    }
}
