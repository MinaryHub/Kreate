//! A ring buffer of recently run operator ids, for crash reports.
//!
//! # Why this is the most valuable field in a crash report
//!
//! A backtrace says where the program was when it died. For a mesh kernel that
//! is rarely enough: the operator that panics is usually not the one that broke
//! the mesh, it is the first one afterwards that could not cope with what it was
//! handed. §13 names this as the top risk in the whole plan — "a mostly working
//! operator emits subtly invalid topology, and three operators later something
//! crashes with no way to trace the cause".
//!
//! The sequence of operators that led up to the crash is what closes that gap,
//! and it is nearly free to record. §3.4 asks for the last fifty.
//!
//! # Why a global
//!
//! A panic hook cannot be handed a reference to application state: it runs while
//! the stack is unwinding, from a closure that outlives everything, and whatever
//! it reads may already be mid-mutation. So this is a small, lock-free, fixed
//! size buffer with no allocation on the write path — a `Mutex` here would risk
//! deadlocking the hook against a panic that happened while the lock was held,
//! which is exactly the case a crash reporter must survive.
//!
//! Ids are interned as `&'static str` from the operator registry, so recording
//! one is a pointer store. Nothing is allocated, nothing is dropped, and reading
//! the buffer during a panic touches only atomics.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// How many operators to remember. §3.4 asks for fifty.
const CAPACITY: usize = 50;

/// The ids, oldest overwritten first.
///
/// `AtomicPtr` to the string data with a parallel length, rather than an
/// `AtomicPtr` to a `&str` fat pointer, because a fat pointer is two words and
/// cannot be stored atomically. Both halves come from the same `&'static str`,
/// so a torn read can only ever pair a pointer with a stale length from the same
/// slot — which is why the length is written before the pointer and read after
/// it.
static IDS: [AtomicPtr<u8>; CAPACITY] = [const { AtomicPtr::new(std::ptr::null_mut()) }; CAPACITY];
static LENS: [AtomicUsize; CAPACITY] = [const { AtomicUsize::new(0) }; CAPACITY];
/// How many have ever been recorded. Modulo capacity gives the next slot.
static COUNT: AtomicUsize = AtomicUsize::new(0);

/// Records that an operator ran.
///
/// Takes a `&'static str` on purpose: the caller must already own the id for the
/// life of the program, which every operator id in the registry does. A `&str`
/// with a shorter life would leave this buffer holding a dangling pointer, and
/// the moment it would be read is a moment when nothing else can be trusted
/// either.
pub fn record(id: &'static str) {
    let slot = COUNT.fetch_add(1, Ordering::Relaxed) % CAPACITY;
    // Length first, pointer second, both released: a reader that sees the new
    // pointer is guaranteed to see the length that goes with it.
    LENS[slot].store(id.len(), Ordering::Release);
    IDS[slot].store(id.as_ptr() as *mut u8, Ordering::Release);
}

/// The recorded ids, oldest first.
///
/// Safe to call from a panic hook: it takes no locks and allocates only the
/// returned vector, which the caller needs anyway.
pub fn recent() -> Vec<&'static str> {
    let total = COUNT.load(Ordering::Relaxed);
    let count = total.min(CAPACITY);
    let start = total.saturating_sub(count);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let slot = (start + i) % CAPACITY;
        let pointer = IDS[slot].load(Ordering::Acquire);
        if pointer.is_null() {
            continue;
        }
        let len = LENS[slot].load(Ordering::Acquire);
        // SAFETY: the pointer and length came from a `&'static str` passed to
        // `record`, so the bytes are live for the program's lifetime and are
        // valid UTF-8. The release/acquire pairing above means a non-null
        // pointer is published after its own length.
        let bytes = unsafe { std::slice::from_raw_parts(pointer, len) };
        match std::str::from_utf8(bytes) {
            Ok(id) => out.push(id),
            // Impossible unless the pointer was torn from a concurrent write on
            // another thread. Skipped rather than panicking: this function's
            // whole job is to run while something else has already gone wrong.
            Err(_) => continue,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests share the global buffer, so they run against it in one test rather
    /// than fighting over it. Cargo runs tests in parallel by default and this
    /// state is process-wide.
    #[test]
    fn the_buffer_remembers_in_order_and_forgets_the_oldest() {
        assert!(recent().is_empty(), "the buffer starts empty");

        record("mesh.extrude");
        record("mesh.inset");
        record("object.delete");
        assert_eq!(
            recent(),
            vec!["mesh.extrude", "mesh.inset", "object.delete"],
            "oldest first"
        );

        // Past capacity: the earliest entries have to fall off, and what remains
        // has to be the *latest* ones in order. A ring buffer that wraps without
        // reordering is easy to get wrong, and a crash report listing the fifty
        // operators from the start of the session instead of the fifty before
        // the crash would be worse than none.
        for i in 0..CAPACITY * 2 {
            record(LEAKED[i % LEAKED.len()]);
        }
        let recent = recent();
        assert_eq!(recent.len(), CAPACITY, "the buffer does not grow");
        // The last CAPACITY writes were a repeating cycle of LEAKED, so the tail
        // is predictable.
        let expected: Vec<&str> = (0..CAPACITY)
            .map(|i| LEAKED[(CAPACITY * 2 - CAPACITY + i) % LEAKED.len()])
            .collect();
        assert_eq!(recent, expected);
    }

    /// A handful of distinct static ids to cycle through.
    const LEAKED: &[&str] = &[
        "a.one", "b.two", "c.three", "d.four", "e.five", "f.six", "g.seven",
    ];
}
