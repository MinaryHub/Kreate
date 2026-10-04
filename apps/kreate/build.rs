//! Puts the application icon into the Windows executable, so Explorer, the
//! taskbar and Alt+Tab show it before the window has said anything.
//!
//! The window's own icon is set at run time from `assets/kreate-64.rgba`; this
//! is the file's. Other platforms take an icon from the bundle they ship in.

fn main() {
    println!("cargo:rerun-if-changed=assets/kreate.ico");
    // `winresource` is a build-dependency only on a Windows host (build
    // scripts are compiled for the host), so the code that names it must be too.
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/kreate.ico");
        // An icon is decoration: a machine without the resource compiler
        // still builds a working application, just with the default icon.
        if let Err(e) = res.compile() {
            println!("cargo:warning=no icon in the executable: {e}");
        }
    }
}
