//! The parts of the application that are worth testing on their own.
//!
//! A library target beside the binary, so integration tests can drive the shell
//! — panels, the view model, the composite path — without a window. An
//! executable-only crate cannot be imported by a test, and the seam between the
//! interface and the viewport is precisely where a defect would otherwise be
//! invisible to everything except a human looking at the screen.

pub mod breadcrumbs;
pub mod crashreport;
pub mod docfile;
pub mod dragpreview;
pub mod environment;
pub mod expr;
pub mod gpu_sync;
pub mod modal;
pub mod mode;
pub mod pick;
pub mod prefs;
pub mod startup;
pub mod scenelights;
pub mod shell;
pub mod snap;
pub mod uvmodal;
pub mod uvview;

/// The recently run operator ids, for the crash report.
///
/// Re-exported at the crate root because the panic hook is the only consumer and
/// spelling out the module there adds nothing.
pub fn recent_operators() -> Vec<&'static str> {
    breadcrumbs::recent()
}
