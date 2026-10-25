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
//! # Inside a macOS bundle
//!
//! The executables are in `Kreate.app/Contents/MacOS`, and the bundle is
//! signed as a whole. Replacing one breaks that seal and leaves the bundle
//! naming the old version, so after an update, and again once the files put
//! aside are gone, the bundle's version is set and it is signed again, ad
//! hoc, as `scripts/macos-app.sh` signs it.
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

/// Where releases come from, `owner/name` on GitHub: the source repository
/// itself, which is public, so a request without a sign-in can read it.
/// `KREATE_UPDATE_REPO` overrides it.
pub const DEFAULT_REPO: &str = "MinaryHub/Kreate";

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
    // Installed from an MSIX package — the Microsoft Store's — the Store
    // updates it. Its folder is read-only besides, and an update staged
    // beside it would run a second copy outside the package.
    if packaged(exe) {
        return false;
    }
    if std::env::var_os("KREATE_UPDATE").is_some_and(|v| v == "1") {
        return true;
    }
    !exe.components().any(|c| c.as_os_str() == "target")
}

/// Whether `exe` runs from an MSIX package: Windows installs every package
/// under a `WindowsApps` folder, and nothing else lives there.
///
/// Read as text split at either slash, so a Windows path is one wherever
/// it is asked about: elsewhere a backslash is not a separator, and the whole
/// path was one component that never matched.
pub fn packaged(exe: &Path) -> bool {
    exe.to_string_lossy()
        .split(['/', '\\'])
        .any(|c| c.eq_ignore_ascii_case("WindowsApps"))
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
            let name = entry
                .name()
                .map_or_else(|e| e.to_string(), |n| n.into_owned());
            return Err(format!("unsafe path in the zip: {name}"));
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
/// Whether anything was.
fn remove_old(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut removed = false;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("kreate") && name.contains(".old") {
            removed |= std::fs::remove_file(entry.path()).is_ok();
        }
    }
    removed
}

/// The `.app` whose executables are in `dir`: `Kreate.app` for
/// `Kreate.app/Contents/MacOS`.
pub fn bundle_of(dir: &Path) -> Option<&Path> {
    if dir.file_name()? != "MacOS" {
        return None;
    }
    let contents = dir.parent()?;
    if contents.file_name()? != "Contents" {
        return None;
    }
    let app = contents.parent()?;
    app.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("app"))
        .then_some(app)
}

/// Seals a bundle whose executables changed: its version set to `version`,
/// when there is one, and signed again ad hoc. See the module notes.
fn reseal(bundle: &Path, version: Option<&str>) {
    if !cfg!(target_os = "macos") {
        return;
    }
    if let Some(version) = version {
        let plist = bundle.join("Contents/Info.plist");
        for key in ["CFBundleShortVersionString", "CFBundleVersion"] {
            let _ = Command::new("plutil")
                .args(["-replace", key, "-string", version])
                .arg(&plist)
                .status();
        }
    }
    match Command::new("codesign")
        .args(["--force", "--deep", "-s", "-"])
        .arg(bundle)
        .output()
    {
        Ok(out) if out.status.success() => {}
        Ok(out) => tracing::warn!(
            error = %String::from_utf8_lossy(&out.stderr).trim(),
            "bundle not signed again"
        ),
        Err(e) => tracing::warn!(error = %e, "bundle not signed again"),
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
    let Some(root) = update_root() else {
        return false;
    };
    if !enabled(&exe) {
        return false;
    }
    apply_staged_in(&exe, &root, crate::version::VERSION, args, |exe, args| {
        Command::new(exe).args(args).spawn().is_ok()
    })
}

/// [`apply_staged`] with its places and its launcher handed in: `exe` is
/// the running program, `root` the update folder, `current` this version,
/// and `launch` starts the new executable with the arguments.
///
/// Separated so a test can walk the whole start — the files put aside
/// last time cleared, the staging found, spent, installed and started —
/// in a folder of its own, with a launcher that only takes note.
pub fn apply_staged_in(
    exe: &Path,
    root: &Path,
    current: &str,
    args: &[String],
    launch: impl FnOnce(&Path, &[String]) -> bool,
) -> bool {
    let Some(dir) = exe.parent() else {
        return false;
    };
    let bundle = bundle_of(dir);
    if remove_old(dir) {
        if let Some(bundle) = bundle {
            reseal(bundle, None);
        }
    }
    let Some((version, files)) = staged(root) else {
        return false;
    };
    // Whatever happens next, this staging is spent: a failure is said once,
    // not retried on every start.
    let _ = std::fs::remove_file(root.join("staged.txt"));
    if !is_newer(&version, current) {
        let _ = std::fs::remove_dir_all(root.join(&version));
        return false;
    }
    match install(&files, dir) {
        Ok(()) => {
            let _ = std::fs::remove_dir_all(root.join(&version));
            if let Some(bundle) = bundle {
                reseal(bundle, Some(&version));
            }
            tracing::info!(%version, "updated");
            launch(exe, args)
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

    /// The whole way, against the real releases: an older version finds the
    /// latest, downloads this platform's zip and stages the executable.
    /// Needs the network, so it runs only when asked.
    #[test]
    #[ignore]
    fn the_latest_release_is_fetched_and_staged() {
        let root = scratch("fetch");
        let staged = fetch_and_stage(&root, DEFAULT_REPO, "0.0.1").expect("fetched");
        let version = staged.expect("a newer release than 0.0.1");
        let (v, files) = super::staged(&root).expect("staged");
        assert_eq!(v, version);
        assert!(files.join(exe_file_name()).is_file());
        assert_eq!(
            fetch_and_stage(&root, DEFAULT_REPO, "0.0.1"),
            Ok(None),
            "not twice"
        );

        // The next start: the release goes in over an "old" program, and
        // what was fetched runs — the connector, asked to introduce itself,
        // answers, which a cut-short or wrongly-built download would not.
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        let exe = app.join(exe_file_name());
        std::fs::write(&exe, "old").unwrap();
        let started = apply_staged_in(&exe, &root, "0.0.1", &[], |exe, _| exe.is_file());
        assert!(started);
        assert_ne!(
            std::fs::read(&exe).unwrap(),
            b"old",
            "the release is in place"
        );
        let connector = app.join(format!("kreate-mcp{}", std::env::consts::EXE_SUFFIX));
        let mut child = Command::new(&connector)
            .env("KREATE_NO_UPDATE", "1")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("the fetched connector starts");
        {
            use std::io::Write;
            let mut stdin = child.stdin.take().unwrap();
            writeln!(
                stdin,
                r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2024-11-05","capabilities":{{}},"clientInfo":{{"name":"test","version":"0"}}}}}}"#
            )
            .unwrap();
        }
        let mut answer = String::new();
        {
            use std::io::BufRead;
            let mut out = std::io::BufReader::new(child.stdout.take().unwrap());
            out.read_line(&mut answer).unwrap();
        }
        let _ = child.kill();
        let _ = child.wait();
        assert!(
            answer.contains(r#""result""#),
            "the connector answered: {answer}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A copy installed from the Store is updated by the Store, never by
    /// itself; a copy unpacked anywhere else updates itself.
    #[test]
    fn a_packaged_copy_does_not_update_itself() {
        let store = Path::new(
            r"C:\Program Files\WindowsApps\minary.Kreate_0.3.0.0_x64__ws153ybhdqgam\kreate.exe",
        );
        assert!(packaged(store));
        assert!(!enabled(store));
        let unpacked = Path::new(r"C:\Users\me\Apps\Kreate\kreate.exe");
        assert!(!packaged(unpacked));
    }

    /// Only the folder of a bundle's executables is a bundle's.
    #[test]
    fn a_bundle_is_found_from_its_executables_folder() {
        assert_eq!(
            bundle_of(Path::new("/Applications/Kreate.app/Contents/MacOS")),
            Some(Path::new("/Applications/Kreate.app"))
        );
        assert_eq!(bundle_of(Path::new("/opt/kreate/bin")), None);
        assert_eq!(bundle_of(Path::new("/x/Kreate/Contents/MacOS")), None);
    }

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

    /// A release staged for the next start, as the whole start sees it.
    fn stage(root: &Path, version: &str, body: &str) {
        let files = root.join(version).join("files");
        std::fs::create_dir_all(&files).unwrap();
        std::fs::write(files.join(exe_file_name()), body).unwrap();
        std::fs::write(root.join("staged.txt"), version).unwrap();
    }

    /// The start after a download: the staged version goes in over the
    /// running one, the staging is spent, and the new executable is started
    /// with the arguments this one was. The start after that clears what
    /// was put aside.
    #[test]
    fn a_staged_newer_version_is_installed_and_started() {
        let scratch = scratch("apply");
        let (app, root) = (scratch.join("app"), scratch.join("update"));
        std::fs::create_dir_all(&app).unwrap();
        let exe = app.join(exe_file_name());
        std::fs::write(&exe, "old").unwrap();
        stage(&root, "9.9.9", "new");
        let args = vec!["model.kreate".to_string()];

        let mut launched = None;
        let started = apply_staged_in(&exe, &root, "0.3.1", &args, |exe, args| {
            launched = Some((exe.to_path_buf(), args.to_vec()));
            true
        });
        assert!(started, "the new version was started, so this one ends");
        assert_eq!(launched, Some((exe.clone(), args)));
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "new");
        assert!(app.join(format!("{}.old", exe_file_name())).exists());
        assert!(!root.join("staged.txt").exists(), "the staging is spent");
        assert!(!root.join("9.9.9").exists(), "and its files are gone");

        // The next start, as the new version: nothing staged, the old
        // executable swept up.
        let again = apply_staged_in(&exe, &root, "9.9.9", &[], |_, _| panic!("nothing to start"));
        assert!(!again);
        assert!(!app.join(format!("{}.old", exe_file_name())).exists());
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// A staging no newer than what is running — downloaded before a
    /// manual update, say — is thrown away, and nothing is started.
    #[test]
    fn a_staged_older_version_is_thrown_away() {
        let scratch = scratch("stale");
        let (app, root) = (scratch.join("app"), scratch.join("update"));
        std::fs::create_dir_all(&app).unwrap();
        let exe = app.join(exe_file_name());
        std::fs::write(&exe, "current").unwrap();
        stage(&root, "0.0.1", "older");

        let started = apply_staged_in(&exe, &root, "0.3.1", &[], |_, _| {
            panic!("an older version must not be started")
        });
        assert!(!started);
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "current");
        assert!(!root.join("staged.txt").exists());
        assert!(!root.join("0.0.1").exists());
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// A staging that cannot go in — here, the executable's place is a
    /// folder — leaves the running version as it was and starts nothing;
    /// and it is not tried again at the next start.
    #[test]
    fn a_staging_that_cannot_be_installed_is_spent_not_retried() {
        let scratch = scratch("refused");
        let (app, root) = (scratch.join("app"), scratch.join("update"));
        std::fs::create_dir_all(&app).unwrap();
        let exe = app.join(exe_file_name());
        std::fs::write(&exe, "current").unwrap();
        stage(&root, "9.9.9", "new");
        let files = root.join("9.9.9").join("files");
        std::fs::write(files.join("z-extra"), "x").unwrap();
        std::fs::create_dir_all(app.join("z-extra")).unwrap();

        let started = apply_staged_in(&exe, &root, "0.3.1", &[], |_, _| {
            panic!("a failed install must not start anything")
        });
        assert!(!started);
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "current");
        assert!(!app.join(format!("{}.old", exe_file_name())).exists());
        assert!(!root.join("staged.txt").exists(), "said once, not retried");
        let _ = std::fs::remove_dir_all(&scratch);
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

        assert!(remove_old(&app));
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
