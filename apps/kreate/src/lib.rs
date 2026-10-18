//! The parts of the application that are worth testing on their own.
//!
//! A library target beside the binary, so integration tests can drive the shell
//! — panels, the view model, the composite path — without a window. An
//! executable-only crate cannot be imported by a test, and the seam between the
//! interface and the viewport is precisely where a defect would otherwise be
//! invisible to everything except a human looking at the screen.

pub mod appcall;
pub mod assets;
pub mod attach;
pub mod breadcrumbs;
pub mod bridge;
pub mod chatlog;
pub mod claude;
pub mod connect;
pub mod contact;
pub mod crashreport;
pub mod desktop;
pub mod docfile;
pub mod dragpreview;
pub mod environment;
pub mod expr;
pub mod formats;
pub mod generate;
pub mod glyphs;
pub mod gpu_sync;
pub mod logs;
pub mod materials;
pub mod meshcheck;
pub mod modal;
pub mod mode;
pub mod objaverse;
pub mod ollama;
pub mod pick;
pub mod playback;
pub mod prefs;
pub mod proportions;
pub mod rigcheck;
pub mod scenelights;
pub mod sculptview;
pub mod shell;
pub mod silhouette;
pub mod sketchfab;
pub mod sketching;
pub mod snap;
pub mod splash;
pub mod startup;
pub mod update;
pub mod uvmodal;
pub mod uvview;
pub mod version;
pub mod video;
pub mod walk;
pub mod youtube;

/// The recently run operator ids, for the crash report.
///
/// Re-exported at the crate root because the panic hook is the only consumer and
/// spelling out the module there adds nothing.
pub fn recent_operators() -> Vec<&'static str> {
    breadcrumbs::recent()
}
