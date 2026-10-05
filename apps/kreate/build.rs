//! Puts the application icon into the Windows executable, so Explorer, the
//! taskbar and Alt+Tab show it before the window has said anything.
//!
//! The window's own icon is set at run time from `assets/kreate-64.rgba`; this
//! is the file's. Other platforms take an icon from the bundle they ship in.

fn main() {
    println!("cargo:rerun-if-changed=assets/kreate.ico");
    commit();
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

/// The commit being built, for the version line: `KREATE_COMMIT`, short, with
/// `-dirty` when the tree has changes. Empty outside a git checkout — a build
/// from a source tarball is still a build, it just cannot say which commit.
fn commit() {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let hash = git(&["rev-parse", "--short=8", "HEAD"]).unwrap_or_default();
    let dirty = !hash.is_empty()
        && git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    println!(
        "cargo:rustc-env=KREATE_COMMIT={hash}{}",
        if dirty { "-dirty" } else { "" }
    );
    // Again when HEAD moves or the index does, which is when the answer can
    // change; not on every build.
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
        if let Some(head) = git(&["symbolic-ref", "-q", "HEAD"]) {
            println!("cargo:rerun-if-changed={dir}/{head}");
        }
    }
}
