//! Handing a file to the system: opening it with what the user has for it,
//! and showing it in the file browser.

use std::path::Path;

/// Opens `path` with the system's application for it: a picture in the
/// viewer, a video in the player.
pub fn open(path: &Path) {
    let program = if cfg!(windows) {
        "explorer.exe"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = crate::claude::quiet_command(program).arg(path).spawn();
}

/// Where Kreate's development can be supported.
pub const SPONSORS_URL: &str = "https://github.com/sponsors/knoxxr";

/// Opens a web address in the system's browser.
pub fn open_url(url: &str) {
    let mut command = if cfg!(windows) {
        let mut c = crate::claude::quiet_command("cmd");
        c.args(["/C", "start", ""]);
        c
    } else if cfg!(target_os = "macos") {
        crate::claude::quiet_command("open")
    } else {
        crate::claude::quiet_command("xdg-open")
    };
    let _ = command.arg(url).spawn();
}

/// Shows `path` in the system's file browser, selected where the browser can
/// select it, or its folder opened where it cannot.
pub fn show_in_folder(path: &Path) {
    if cfg!(target_os = "macos") {
        let _ = crate::claude::quiet_command("open")
            .arg("-R")
            .arg(path)
            .spawn();
    } else if cfg!(windows) {
        let mut select = std::ffi::OsString::from("/select,");
        select.push(path);
        let _ = crate::claude::quiet_command("explorer.exe")
            .arg(select)
            .spawn();
    } else if let Some(folder) = path.parent() {
        let _ = crate::claude::quiet_command("xdg-open").arg(folder).spawn();
    }
}
