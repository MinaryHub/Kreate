//! Which Kreate this is.
//!
//! The release number comes from the workspace's `Cargo.toml`, the one place
//! it is written; the commit from the build script. Everything that names the
//! version — the About box, a crash report, a message to support — asks here,
//! so they cannot disagree.

/// The release, e.g. `0.2.0`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The commit it was built from, short, `-dirty` if the tree had changes;
/// empty when built outside a git checkout.
pub const COMMIT: &str = env!("KREATE_COMMIT");

/// Which build of the release this is: one more for every build that
/// changed something. The release number is left alone — the updater
/// compares it.
pub const BUILD: &str = env!("KREATE_BUILD");

/// When it was built, to the minute.
pub const BUILT: &str = env!("KREATE_BUILT");

/// `0.3.0 build 37 (1a2b3c4d, 2026-10-16 22:41)`; the parts it does not
/// know are left out.
pub fn describe() -> String {
    let mut out = VERSION.to_string();
    if !BUILD.is_empty() && BUILD != "0" {
        out.push_str(&format!(" build {BUILD}"));
    }
    let details: Vec<&str> = [COMMIT, BUILT]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    if !details.is_empty() {
        out.push_str(&format!(" ({})", details.join(", ")));
    }
    out
}

/// The platform, for a report: `macos aarch64`.
pub fn platform() -> String {
    format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_is_the_workspaces_and_reads_as_semver() {
        let parts: Vec<_> = VERSION.split('.').collect();
        assert_eq!(parts.len(), 3, "{VERSION}");
        assert!(
            parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())),
            "{VERSION}"
        );
        assert!(describe().starts_with(VERSION));
    }
}
