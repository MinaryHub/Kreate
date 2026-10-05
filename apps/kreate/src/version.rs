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

/// `0.2.0 (1a2b3c4d)`, or just `0.2.0` without a commit.
pub fn describe() -> String {
    if COMMIT.is_empty() {
        VERSION.to_string()
    } else {
        format!("{VERSION} ({COMMIT})")
    }
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
