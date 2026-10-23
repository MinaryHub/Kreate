//! Puts the application icon into the Windows executable, so Explorer, the
//! taskbar and Alt+Tab show it before the window has said anything.
//!
//! The window's own icon is set at run time from `assets/kreate-64.rgba`; this
//! is the file's. Other platforms take an icon from the bundle they ship in.

fn main() {
    println!("cargo:rerun-if-changed=assets/kreate.ico");
    commit();
    build_number();
    shader_compiler();
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

/// A number for each build that changed something, and when it was made:
/// `KREATE_BUILD` and `KREATE_BUILT`. The release number stays what the
/// workspace says — the updater compares it — and this tells two builds of
/// one release, or of one commit, apart.
///
/// The count is kept beside the build's output, in `target/`, so it climbs
/// for as long as that folder lives. This script runs again when any source
/// of the application or the crates changes, and only then: a build that
/// changed nothing is the same program and keeps its number.
fn build_number() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    // target/<profile>/build/<crate>-<hash>/out → target
    let counter = out
        .ancestors()
        .nth(4)
        .map(|target| target.join(".kreate-build-number"));
    let number = counter
        .as_ref()
        .and_then(|path| {
            let last: u64 = std::fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            let next = last + 1;
            std::fs::write(path, next.to_string()).ok()?;
            Some(next)
        })
        .unwrap_or(0);
    println!("cargo:rustc-env=KREATE_BUILD={number}");
    println!("cargo:rustc-env=KREATE_BUILT={}", now());
    for dir in [
        "src",
        "locales",
        "assets",
        "../../crates",
        "../../Cargo.toml",
    ] {
        println!("cargo:rerun-if-changed={dir}");
    }
}

/// DirectX's modern shader compiler, `dxcompiler.dll` and its `dxil.dll`,
/// copied beside the program being built. wgpu uses DXC when it finds it
/// there and the old FXC otherwise, and FXC took 40 seconds over this
/// renderer's shaders in a release build — the first frame, and again the
/// first render — where DXC takes three or four.
///
/// From `KREATE_DXC_DIR`, or the newest Windows SDK that has both. None
/// found, the build goes on: the program runs on FXC, slowly at first.
fn shader_compiler() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    println!("cargo:rerun-if-env-changed=KREATE_DXC_DIR");
    let files = ["dxcompiler.dll", "dxil.dll"];
    let has = |dir: &std::path::Path| files.iter().all(|f| dir.join(f).is_file());
    let from = std::env::var_os("KREATE_DXC_DIR")
        .map(std::path::PathBuf::from)
        .filter(|d| has(d))
        .or_else(|| {
            let kits = std::path::Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
            let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(kits)
                .ok()?
                .flatten()
                .map(|e| e.path().join("x64"))
                .filter(|d| has(d))
                .collect();
            found.sort();
            found.pop()
        });
    let Some(from) = from else {
        println!("cargo:warning=no dxcompiler.dll found: shaders compile with the slow FXC");
        return;
    };
    // target/<profile>/build/<crate>-<hash>/out → target/<profile>
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    let Some(profile) = out.ancestors().nth(3) else {
        return;
    };
    for f in files {
        let (source, target) = (from.join(f), profile.join(f));
        let same = |a: &std::path::Path, b: &std::path::Path| {
            let len = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).ok();
            len(a).is_some() && len(a) == len(b)
        };
        if !same(&source, &target) {
            if let Err(e) = std::fs::copy(&source, &target) {
                println!("cargo:warning=copying {f}: {e}");
            }
        }
    }
}

/// The local time to the minute, `2026-10-16 22:41`: from `date` where
/// there is one, otherwise UTC worked out from the clock.
fn now() -> String {
    let local = std::process::Command::new("date")
        .arg("+%Y-%m-%d %H:%M")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| s.len() == 16);
    if let Some(local) = local {
        return local;
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's days-to-civil.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02} UTC",
        rest / 3600,
        rest % 3600 / 60
    )
}
