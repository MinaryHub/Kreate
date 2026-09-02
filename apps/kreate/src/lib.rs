//! The parts of the application that are worth testing on their own.
//!
//! A library target beside the binary, so integration tests can drive the shell
//! — panels, the view model, the composite path — without a window. An
//! executable-only crate cannot be imported by a test, and the seam between the
//! interface and the viewport is precisely where a defect would otherwise be
//! invisible to everything except a human looking at the screen.

pub mod gpu_sync;
pub mod pick;
pub mod prefs;
pub mod shell;
