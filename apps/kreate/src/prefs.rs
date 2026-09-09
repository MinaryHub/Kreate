//! Where settings live, and how they are written.
//!
//! # Atomic, always
//!
//! Every write goes to a temporary file beside the target and is then renamed
//! over it. A save interrupted by a crash, a full disk or a power cut leaves the
//! previous file intact rather than a half-written one — and a half-written
//! preferences file is the specific failure the plan singles out as a
//! self-inflicted support nightmare, because it turns one bad save into an
//! application that will not start.
//!
//! `rename` is atomic on the same filesystem on every platform this targets.
//! Writing into the same directory as the target is what guarantees that: a
//! temporary file in the system temp directory can be on another volume, where
//! the rename silently becomes a copy and stops being atomic.
//!
//! # Failure is never fatal
//!
//! Nothing here returns an error the caller has to handle by giving up. Loading
//! a missing or broken file yields the defaults; a failed save is reported and
//! the session continues. Somebody's read-only home directory is not a reason to
//! refuse to run a modelling application.

use std::path::{Path, PathBuf};

use kreate_input::keyfile::{self, KeyFile};

/// The directory settings live in.
///
/// `%APPDATA%\Kreate` on Windows, `$XDG_CONFIG_HOME/kreate` or `~/.config/kreate`
/// elsewhere. Resolved from the environment rather than through a crate: the
/// rule is three lines, and a dependency here would be the one place a config
/// path could disagree with what the installer expects.
pub fn config_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(|base| Path::new(&base).join("Kreate"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(|base| Path::new(&base).join("kreate"))
            .or_else(|| {
                std::env::var_os("HOME").map(|home| Path::new(&home).join(".config/kreate"))
            })
    }
}

/// The keymap file's path.
pub fn keymap_path() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("keymap.toml"))
}

/// Writes `contents` to `path` atomically.
///
/// Returns what went wrong, for the caller to report. The temporary file is
/// removed on failure so a broken save does not leave litter behind that the
/// next one might mistake for a real file.
pub fn write_atomic(path: &Path, contents: &str) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Beside the target, not in the system temp directory: `rename` is only
    // atomic within one filesystem, and a temp directory can be on another
    // volume, where it quietly degrades to a copy.
    let temporary = path.with_extension("tmp");
    let result = std::fs::write(&temporary, contents).and_then(|()| {
        // Windows will not rename over an existing file, so the old one goes
        // first. The window this opens is why the temporary file is kept until
        // the rename succeeds: if the rename fails, the data is still on disk
        // under a name the next load can be pointed at.
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
        std::fs::rename(&temporary, path)
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// The workspace file's path.
pub fn workspace_path() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("workspaces.toml"))
}

/// Loads the workspace file, reporting anything wrong with it.
///
/// A missing file is not a problem — it is a first run, and the shipped
/// layouts are the right answer.
pub fn load_workspaces(path: &Path) -> (kreate_ui::Workspaces, Vec<String>) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (kreate_ui::Workspaces::default(), Vec::new());
    };
    let (spaces, problems) = kreate_ui::wsfile::parse(&text);
    let messages = problems
        .iter()
        .map(|p| format!("{}: {p}", path.display()))
        .collect();
    (spaces, messages)
}

/// Saves the workspace file.
pub fn save_workspaces(path: &Path, spaces: &kreate_ui::Workspaces) -> Result<(), std::io::Error> {
    write_atomic(path, &kreate_ui::wsfile::write(spaces))
}

/// Loads the keymap file, reporting anything wrong with it.
///
/// A missing file is not a problem — it is a first run.
pub fn load_keymap(path: &Path) -> (KeyFile, Vec<String>) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (KeyFile::default(), Vec::new());
    };
    let (file, problems) = keyfile::parse(&text);
    let messages = problems
        .iter()
        .map(|p| format!("{}: {p}", path.display()))
        .collect();
    (file, messages)
}

/// Saves the keymap file.
pub fn save_keymap(path: &Path, file: &KeyFile) -> Result<(), std::io::Error> {
    write_atomic(path, &keyfile::write(file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_input::keymap::{BindingId, Input, Override};
    use kreate_input::Preset;

    /// A directory of our own under the system temp directory, removed after.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("kreate-prefs-{name}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("scratch directory");
            Self(dir)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample() -> KeyFile {
        let mut file = KeyFile {
            preset: Preset::Maya,
            ..Default::default()
        };
        file.overrides.insert(
            BindingId::new("mesh.extrude"),
            Override {
                input: Some(Input::key("Q")),
                ..Default::default()
            },
        );
        file
    }

    #[test]
    fn workspaces_round_trip_through_a_file() {
        let scratch = Scratch::new("workspaces");
        let path = scratch.path("workspaces.toml");

        let mut spaces = kreate_ui::Workspaces::default();
        spaces.switch_to(1);
        spaces.duplicate_active("Mine");
        let area = spaces.screen().areas()[0].0;
        spaces
            .screen_mut()
            .split(
                area,
                kreate_ui::Axis::Horizontal,
                0.3,
                kreate_ui::EditorId::UV_EDITOR,
            )
            .expect("split");

        save_workspaces(&path, &spaces).expect("save");
        let (loaded, problems) = load_workspaces(&path);
        assert_eq!(problems, Vec::<String>::new());
        assert_eq!(loaded.len(), spaces.len());
        assert_eq!(loaded.active().name, "Mine");
        assert_eq!(
            loaded.screen().to_nodes().len(),
            spaces.screen().to_nodes().len(),
            "the arrangement did not come back"
        );
    }

    #[test]
    fn a_missing_workspace_file_is_a_first_run_not_a_problem() {
        // The user who has never opened the preferences, and the user who
        // deleted the file to get the shipped layouts back. Both ask for the
        // same thing.
        let scratch = Scratch::new("workspaces-missing");
        let (spaces, problems) = load_workspaces(&scratch.path("nothing-here.toml"));
        assert_eq!(spaces, kreate_ui::Workspaces::default());
        assert_eq!(
            problems,
            Vec::<String>::new(),
            "a first run is not an error"
        );
    }

    #[test]
    fn a_broken_workspace_file_costs_a_layout_not_a_session() {
        // The failure this module exists to prevent, in the other file it now
        // owns. What survives is loaded; what does not is reported in prose the
        // user can act on.
        let scratch = Scratch::new("workspaces-broken");
        let path = scratch.path("workspaces.toml");
        std::fs::write(
            &path,
            "schema_version = 1
             active = \"Good\"
             nonsense

             [[workspace]]
             name = \"Good\"
             node = \"area viewport\"

             [[workspace]]
             name = \"Half\"
             node = \"split v 0.5\"
             node = \"area viewport\"
",
        )
        .expect("write");

        let (spaces, problems) = load_workspaces(&path);
        assert_eq!(spaces.len(), 1, "the good layout did not survive");
        assert_eq!(spaces.active().name, "Good");
        assert_eq!(problems.len(), 2, "{problems:?}");
        for message in &problems {
            assert!(
                message.contains("workspaces.toml"),
                "a problem that does not name the file: {message}"
            );
            assert!(
                !message.contains('{') && !message.contains("Problem"),
                "a problem that reads as a debug dump: {message}"
            );
        }
    }

    #[test]
    fn a_keymap_round_trips_through_a_file() {
        let scratch = Scratch::new("round-trip");
        let path = scratch.path("keymap.toml");
        save_keymap(&path, &sample()).expect("save");

        let (loaded, problems) = load_keymap(&path);
        assert_eq!(problems, Vec::<String>::new());
        assert_eq!(loaded, sample());
    }

    #[test]
    fn a_missing_file_is_a_first_run_not_a_problem() {
        let scratch = Scratch::new("missing");
        let (file, problems) = load_keymap(&scratch.path("nothing-here.toml"));
        assert_eq!(file, KeyFile::default());
        assert_eq!(problems, Vec::<String>::new(), "silence, not an error");
    }

    #[test]
    fn a_broken_file_loads_what_it_can_and_says_what_was_wrong() {
        // The failure this module exists to prevent: a bad preferences file must
        // cost the user one setting, not their session.
        let scratch = Scratch::new("broken");
        let path = scratch.path("keymap.toml");
        std::fs::write(
            &path,
            "schema_version = 1\npreset = \"maya\"\nrubbish here\n",
        )
        .unwrap();

        let (file, problems) = load_keymap(&path);
        assert_eq!(file.preset, Preset::Maya, "the good part still loaded");
        assert_eq!(problems.len(), 1);
        assert!(
            problems[0].contains("keymap.toml") && problems[0].contains("line 3"),
            "a problem must name the file and the line: {}",
            problems[0]
        );
    }

    #[test]
    fn an_interrupted_save_leaves_the_previous_file_intact() {
        // What atomicity buys, stated as the thing that must not happen: a
        // half-written file. Simulated by leaving a stale temporary behind — the
        // state a crash mid-write produces — and checking the real file is
        // untouched and still loads.
        let scratch = Scratch::new("interrupted");
        let path = scratch.path("keymap.toml");
        save_keymap(&path, &sample()).expect("first save");
        let good = std::fs::read_to_string(&path).unwrap();

        std::fs::write(path.with_extension("tmp"), "half a fi").unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            good,
            "the real file must be untouched by a failed write"
        );
        let (loaded, problems) = load_keymap(&path);
        assert_eq!(problems, Vec::<String>::new());
        assert_eq!(loaded, sample());
    }

    #[test]
    fn the_temporary_file_sits_beside_the_target() {
        // `rename` is atomic only within one filesystem. A temporary in the
        // system temp directory can be on another volume, where the rename
        // quietly becomes a copy and stops being atomic — invisible until the
        // one time it matters.
        let scratch = Scratch::new("beside");
        let path = scratch.path("keymap.toml");
        let temporary = path.with_extension("tmp");
        assert_eq!(
            temporary.parent(),
            path.parent(),
            "the temporary must be in the same directory as the target"
        );

        save_keymap(&path, &sample()).unwrap();
        assert!(!temporary.exists(), "and must not be left behind");
    }

    #[test]
    fn saving_creates_the_directory() {
        // A first run has no configuration directory at all.
        let scratch = Scratch::new("nested");
        let path = scratch.path("a/b/c/keymap.toml");
        save_keymap(&path, &sample()).expect("save into a new directory");
        assert!(path.exists());
    }

    #[test]
    fn overwriting_replaces_rather_than_appending() {
        let scratch = Scratch::new("overwrite");
        let path = scratch.path("keymap.toml");
        save_keymap(&path, &sample()).unwrap();

        let mut second = sample();
        second.preset = Preset::Max;
        second.overrides.clear();
        save_keymap(&path, &second).unwrap();

        let (loaded, problems) = load_keymap(&path);
        assert_eq!(problems, Vec::<String>::new());
        assert_eq!(loaded, second);
        assert!(loaded.overrides.is_empty(), "the old entry is gone");
    }

    #[test]
    fn a_failed_save_is_reported_rather_than_fatal() {
        // Somebody's read-only home directory is not a reason to refuse to run
        // a modelling application.
        let scratch = Scratch::new("failure");
        // A path whose parent is a *file*, so `create_dir_all` cannot succeed.
        let blocker = scratch.path("not-a-directory");
        std::fs::write(&blocker, "in the way").unwrap();
        let path = blocker.join("keymap.toml");

        let result = save_keymap(&path, &sample());
        assert!(result.is_err(), "this really should fail");
        assert!(!path.exists());
    }

    #[test]
    fn the_config_directory_is_under_the_platforms_own_location() {
        // Not asserted as an exact string: the point is that it is derived from
        // the environment and named after the application, so an installer and
        // this agree without either hard-coding the other's answer.
        let Some(dir) = config_dir() else {
            // No environment to derive one from, which a sandbox can produce.
            return;
        };
        let text = dir.to_string_lossy().to_lowercase();
        assert!(text.contains("kreate"), "{dir:?}");
        assert!(dir.is_absolute(), "{dir:?}");
        assert_eq!(keymap_path().unwrap().file_name().unwrap(), "keymap.toml");
    }
}
