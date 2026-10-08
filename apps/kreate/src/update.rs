//! Updates from the project's GitHub releases: looked for at start, fetched
//! quietly, and put in place the next time Kreate starts.
//!
//! # The release
//!
//! A release is a tag `vX.Y.Z` with one zip per platform, named by
//! [`asset_name`] — `kreate-windows-x86_64.zip` — holding the executables at
//! its top level. Pushing the tag `scripts/release.sh` makes is what builds
//! and publishes one (`.github/workflows/release.yml`; see RELEASING.md).
//!
//! # Fetching
//!
//! With `curl`, which every Windows since 10 ships, as do macOS and the Linux
//! desktops: the transfer is HTTPS and a redirect to GitHub's storage, which
//! is all the program needs, and a TLS stack of its own would be the largest
//! dependency in the build for one request a day. The zip's CRCs are checked
//! as it is unpacked, so a cut-short download is a failed one, never a
//! half-written executable.
//!
//! # Putting it in place
//!
//! A running executable on Windows cannot be overwritten but can be renamed.
//! The next start, before any window, renames each executable the update
//! replaces to `*.old`, copies the new one in its place, and starts it — undoing
//! every rename if any copy fails, so an update either happens whole or not at
//! all. The `*.old` files are deleted the start after, when nothing runs them.
//!
//! # Not in a build tree
//!
//! An executable under a `target` folder is a developer's own build, and a
//! release put over it would replace their work with an older commit. Those
//! never update unless `KREATE_UPDATE=1`; `KREATE_NO_UPDATE=1` turns updates
//! off anywhere.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

/// Where releases come from, `owner/name` on GitHub: a public repository that
/// holds only releases, since the source's is private and a request without a
/// sign-in cannot read it. `KREATE_UPDATE_REPO` overrides it.
pub const DEFAULT_REPO: &str = "knoxxr/Kreate-releases";

/// The zip this platform's release carries: `kreate-windows-x86_64.zip`,
/// `kreate-macos-arm64.zip` — the names the release workflow builds under.
pub fn asset_name() -> String {
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        other => other,
    };
    format!("kreate-{}-{arch}.zip", std::env::consts::OS)
}

/// The repository to look in.
pub fn repo() -> String {
    std::env::var("KREATE_UPDATE_REPO")
        .ok()
        .filter(|r| r.contains('/'))
        .unwrap_or_else(|| DEFAULT_REPO.to_string())
}

/// Whether this executable updates itself. See the module notes.
pub fn enabled(exe: &Path) -> bool {
    if std::env::var_os("KREATE_NO_UPDATE").is_some_and(|v| v != "0") {
        return false;
    }
    if std::env::var_os("KREATE_UPDATE").is_some_and(|v| v == "1") {
        return true;
    }
    !exe.components().any(|c| c.as_os_str() == "target")
}

/// `%LOCALAPPDATA%\Kreate\update`, beside the autosaves: machine-local and
/// disposable, like them.
pub fn update_root() -> Option<PathBuf> {
    Some(crate::docfile::autosave_root()?.parent()?.join("update"))
}

/// `v1.2.3`, `1.2.3` or `1.2.3-beta` as three numbers.
pub fn parse_version(text: &str) -> Option<(u64, u64, u64)> {
    let core = text.trim().trim_start_matches('v');
    let core = core.split(['-', '+']).next()?;
    let mut parts = core.split('.').map(|p| p.parse::<u64>().ok());
    let v = (parts.next()??, parts.next()??, parts.next()??);
    parts.next().is_none().then_some(v)
}

/// Whether `candidate` is a later release than `current`.
pub fn is_newer(candidate: &str, current: &str) -> bool {
    match (parse_version(candidate), parse_version(current)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

/// A release worth fetching: its version and where its zip is.
#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub version: String,
    pub url: String,
}

/// The release in GitHub's `releases/latest` answer, if it is a published
/// one and carries this platform's zip.
pub fn pick_release(json: &serde_json::Value, asset: &str) -> Option<Release> {
    if json["draft"].as_bool() == Some(true) || json["prerelease"].as_bool() == Some(true) {
        return None;
    }
    let tag = json["tag_name"].as_str()?;
    parse_version(tag)?;
    let url = json["assets"]
        .as_array()?
        .iter()
        .find(|a| a["name"].as_str() == Some(asset))?["browser_download_url"]
        .as_str()?;
    Some(Release {
        version: tag.trim_start_matches('v').to_string(),
        url: url.to_string(),
    })
}

/// `curl`, without a console window flashing up on Windows.
fn curl(args: &[&str]) -> Result<Vec<u8>, String> {
    let mut command = Command::new("curl");
    command.args(["-fsSL", "--retry", "2", "-A", "Kreate-updater"]);
    command.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let out = command.output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "curl: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// The version staged for the next start, and the folder its files are in.
pub fn staged(root: &Path) -> Option<(String, PathBuf)> {
    let version = std::fs::read_to_string(root.join("staged.txt")).ok()?;
    let version = version.trim().to_string();
    let files = root.join(&version).join("files");
    files.is_dir().then_some((version, files))
}

/// Unpacks a release zip into `into`, refusing any entry that would land
/// outside it.
pub fn extract(zip_path: &Path, into: &Path) -> Result<(), String> {
    let file = std::fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let Some(name) = entry.enclosed_name() else {
            return Err(format!("unsafe path in the zip: {}", entry.name()));
        };
        let target = into.join(name);
        if entry.is_dir() {
            std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut out = std::fs::File::create(&target).map_err(|e| e.to_string())?;
        // Reading to the end is what checks the entry's CRC.
        std::io::copy(&mut entry, &mut out).map_err(|e| format!("{}: {e}", target.display()))?;
        // An executable has to stay one.
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode));
        }
    }
    Ok(())
}

/// This program's executable name: `kreate.exe` on Windows.
fn exe_file_name() -> String {
    format!("kreate{}", std::env::consts::EXE_SUFFIX)
}

/// Looks for a newer release and, if there is one, downloads and stages it.
/// The staged version, or `None` when there was nothing new.
pub fn fetch_and_stage(root: &Path, repo: &str, current: &str) -> Result<Option<String>, String> {
    let api = format!("https://api.github.com/repos/{repo}/releases/latest");
    let body = curl(&["-H", "Accept: application/vnd.github+json", &api])?;
    let json: serde_json::Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
    let Some(release) = pick_release(&json, &asset_name()) else {
        return Ok(None);
    };
    if !is_newer(&release.version, current)
        || staged(root).is_some_and(|(v, _)| v == release.version)
    {
        return Ok(None);
    }
    let dir = root.join(&release.version);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let zip_path = dir.join("download.zip");
    let zip_text = zip_path.to_string_lossy().to_string();
    curl(&["-o", &zip_text, &release.url])?;
    let files = dir.join("files");
    extract(&zip_path, &files)?;
    if !files.join(exe_file_name()).is_file() {
        return Err(format!("the release has no {}", exe_file_name()));
    }
    let _ = std::fs::remove_file(&zip_path);
    crate::prefs::write_atomic(&root.join("staged.txt"), &release.version)
        .map_err(|e| e.to_string())?;
    Ok(Some(release.version))
}

/// Starts the look for a release on its own thread. What it holds becomes
/// the staged version once one is downloaded; `wake` is called then.
pub fn check_in_background(wake: impl FnOnce() + Send + 'static) -> Arc<Mutex<Option<String>>> {
    let ready = Arc::new(Mutex::new(None));
    let exe = std::env::current_exe().unwrap_or_default();
    let Some(root) = update_root().filter(|_| enabled(&exe)) else {
        return ready;
    };
    let out = ready.clone();
    std::thread::Builder::new()
        .name("update".into())
        .spawn(
            move || match fetch_and_stage(&root, &repo(), crate::version::VERSION) {
                Ok(Some(version)) => {
                    tracing::info!(%version, "update downloaded; used at the next start");
                    if let Ok(mut slot) = out.lock() {
                        *slot = Some(version);
                    }
                    wake();
                }
                Ok(None) => tracing::debug!("no newer release"),
                Err(e) => tracing::info!(error = %e, "update check failed"),
            },
        )
        .ok();
    ready
}

/// Deletes what earlier updates renamed aside, now that nothing runs it.
fn remove_old(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("kreate") && name.contains(".old") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Puts every file of `files` into `dir`, the ones it replaces renamed aside
/// first; on any failure, every rename is undone.
pub fn install(files: &Path, dir: &Path) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(files)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    entries.sort();
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut placed: Vec<PathBuf> = Vec::new();
    let result = (|| {
        for source in &entries {
            let name = source.file_name().ok_or("a nameless file")?;
            let target = dir.join(name);
            if target.is_dir() {
                return Err(format!("{} is a folder", target.display()));
            }
            if target.exists() {
                // A `.old` still running — the connector of an earlier update
                // — cannot be deleted; a numbered name beside it can be used.
                let mut aside = dir.join(format!("{}.old", name.to_string_lossy()));
                let mut n = 1;
                while aside.exists() && std::fs::remove_file(&aside).is_err() {
                    aside = dir.join(format!("{}.old.{n}", name.to_string_lossy()));
                    n += 1;
                }
                std::fs::rename(&target, &aside)
                    .map_err(|e| format!("{}: {e}", target.display()))?;
                moved.push((target.clone(), aside));
            }
            std::fs::copy(source, &target).map_err(|e| format!("{}: {e}", target.display()))?;
            placed.push(target);
        }
        Ok::<(), String>(())
    })();
    if result.is_err() {
        for target in &placed {
            let _ = std::fs::remove_file(target);
        }
        for (target, aside) in moved.iter().rev() {
            let _ = std::fs::rename(aside, target);
        }
    }
    result
}

/// At start, before anything else: puts a staged update in place and starts
/// it. True when the new version was started and this process should end.
pub fn apply_staged(args: &[String]) -> bool {
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    let (Some(dir), Some(root)) = (exe.parent(), update_root()) else {
        return false;
    };
    if !enabled(&exe) {
        return false;
    }
    remove_old(dir);
    let Some((version, files)) = staged(&root) else {
        return false;
    };
    // Whatever happens next, this staging is spent: a failure is said once,
    // not retried on every start.
    let _ = std::fs::remove_file(root.join("staged.txt"));
    if !is_newer(&version, crate::version::VERSION) {
        let _ = std::fs::remove_dir_all(root.join(&version));
        return false;
    }
    match install(&files, dir) {
        Ok(()) => {
            let _ = std::fs::remove_dir_all(root.join(&version));
            tracing::info!(%version, "updated");
            Command::new(&exe).args(args).spawn().is_ok()
        }
        Err(e) => {
            tracing::warn!(error = %e, %version, "update not installed");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn versions_compare_as_numbers() {
        assert_eq!(parse_version("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("0.10.0-beta"), Some((0, 10, 0)));
        assert_eq!(parse_version("1.2"), None);
        assert_eq!(parse_version("1.2.3.4"), None);
        assert!(is_newer("v0.10.0", "0.9.9"), "10 is more than 9, not less");
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.0.9", "0.1.0"));
        assert!(!is_newer("nonsense", "0.1.0"));
    }

    #[test]
    fn the_release_named_is_the_platforms_zip() {
        let asset = asset_name();
        let json = serde_json::json!({
            "tag_name": "v0.3.0",
            "draft": false,
            "prerelease": false,
            "assets": [
                {"name": "kreate-other-thing.zip", "browser_download_url": "https://x/other"},
                {"name": asset, "browser_download_url": "https://x/mine"},
            ]
        });
        assert_eq!(
            pick_release(&json, &asset),
            Some(Release {
                version: "0.3.0".into(),
                url: "https://x/mine".into()
            })
        );
        let mut pre = json.clone();
        pre["prerelease"] = true.into();
        assert_eq!(
            pick_release(&pre, &asset),
            None,
            "a prerelease is not offered"
        );
        assert_eq!(
            pick_release(&json, "kreate-nowhere.zip"),
            None,
            "no zip for this platform"
        );
    }

    #[test]
    fn a_developers_build_does_not_update_itself() {
        if std::env::var_os("KREATE_UPDATE").is_some()
            || std::env::var_os("KREATE_NO_UPDATE").is_some()
        {
            return;
        }
        assert!(!enabled(Path::new("C:/src/Kreate/target/debug/kreate.exe")));
        assert!(enabled(Path::new("C:/Program Files/Kreate/kreate.exe")));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kreate-update-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The executables are replaced, the old ones put aside, and a file the
    /// release does not carry is left alone.
    #[test]
    fn install_puts_the_new_files_in_and_the_old_aside() {
        let root = scratch("install");
        let (files, app) = (root.join("files"), root.join("app"));
        std::fs::create_dir_all(&files).unwrap();
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(files.join("kreate.exe"), "new").unwrap();
        std::fs::write(files.join("kreate-mcp.exe"), "new mcp").unwrap();
        std::fs::write(app.join("kreate.exe"), "old").unwrap();
        std::fs::write(app.join("notes.txt"), "mine").unwrap();

        install(&files, &app).unwrap();
        assert_eq!(
            std::fs::read_to_string(app.join("kreate.exe")).unwrap(),
            "new"
        );
        assert_eq!(
            std::fs::read_to_string(app.join("kreate-mcp.exe")).unwrap(),
            "new mcp"
        );
        assert_eq!(
            std::fs::read_to_string(app.join("kreate.exe.old")).unwrap(),
            "old"
        );
        assert_eq!(
            std::fs::read_to_string(app.join("notes.txt")).unwrap(),
            "mine"
        );

        remove_old(&app);
        assert!(!app.join("kreate.exe.old").exists());
        assert!(app.join("notes.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// One copy failing undoes the whole update: the old executable is back
    /// where it was.
    #[test]
    fn a_failed_install_leaves_the_old_version() {
        let root = scratch("rollback");
        let (files, app) = (root.join("files"), root.join("app"));
        std::fs::create_dir_all(&files).unwrap();
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(files.join("a-kreate.exe"), "new").unwrap();
        std::fs::write(files.join("z-kreate.exe"), "new").unwrap();
        std::fs::write(app.join("a-kreate.exe"), "old").unwrap();
        // A folder where the second file would go, after the first is in.
        std::fs::create_dir_all(app.join("z-kreate.exe")).unwrap();

        assert!(install(&files, &app).is_err());
        assert_eq!(
            std::fs::read_to_string(app.join("a-kreate.exe")).unwrap(),
            "old"
        );
        assert!(!app.join("a-kreate.exe.old").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_zip_unpacks_and_one_that_climbs_out_is_refused() {
        let root = scratch("zip");
        let good = root.join("good.zip");
        {
            let mut w = zip::ZipWriter::new(std::fs::File::create(&good).unwrap());
            w.start_file("kreate.exe", zip::write::SimpleFileOptions::default())
                .unwrap();
            w.write_all(b"binary").unwrap();
            w.finish().unwrap();
        }
        extract(&good, &root.join("out")).unwrap();
        assert_eq!(
            std::fs::read(root.join("out/kreate.exe")).unwrap(),
            b"binary"
        );

        let bad = root.join("bad.zip");
        {
            let mut w = zip::ZipWriter::new(std::fs::File::create(&bad).unwrap());
            w.start_file("../escape.exe", zip::write::SimpleFileOptions::default())
                .unwrap();
            w.write_all(b"x").unwrap();
            w.finish().unwrap();
        }
        assert!(extract(&bad, &root.join("out2")).is_err());
        assert!(!root.join("escape.exe").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
