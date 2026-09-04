//! The shared test corpus and the comparisons that make round-trip claims mean
//! something.
//!
//! # Why the corpus is built in code
//!
//! Every mesh here is constructed by a function rather than committed as a file.
//! That is deliberate: a corpus of committed files is a corpus whose contents
//! nobody can read. When `mixed_winding` fails, the useful thing to look at is
//! twelve lines saying which face is wound backwards and why that is hard — not
//! a binary blob and a filename. It also means the corpus survives a change to
//! `MeshData` without a migration, and that adding a specimen is a pull request
//! rather than an asset drop.
//!
//! The exception is golden *output*: the `.kmesh` and `.kreate` files under
//! `kreate-io/tests/assets` are committed precisely because their bytes are the
//! thing under test.
//!
//! # Why comparison is its own module
//!
//! "The mesh came back the same" is not a comparison anyone can write inline
//! correctly. Indices are renumbered by every welder and every exporter that
//! splits vertices; faces come back rotated, reordered between primitives, and
//! grouped by material. A byte comparison fails on all of that while a lenient
//! one passes on a model turned inside out.
//!
//! [`compare_geometry`] is invariant under exactly the things a format is
//! allowed to change — vertex numbering, face order, and where a face's cycle
//! starts — and sensitive to exactly the things it is not, above all winding.

pub mod compare;
pub mod corpus;

pub use compare::{compare_geometry, Mismatch, Tolerance};
pub use corpus::{corpus, dense_grid, Specimen};
