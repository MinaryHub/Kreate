//! Nothing outside the mode module names a mode by its variant.
//!
//! The rule §2.6 decided and `kreate_ops::mode` is built on: the application
//! asks a mode's card what to do, and never branches on which mode it is. A
//! rule like that erodes one convenient `== EditMesh` at a time, and each one
//! is a place a new mode silently takes the wrong branch — so it is a gate, in
//! the same way the bare-string and sRGB gates are, rather than a hope.
//!
//! Test modules are exempt: a test sets up a mode by name, which is what a test
//! is for. Everything from a file's first `#[cfg(test)]` on is skipped, and so
//! are `tests/` directories.

use std::path::{Path, PathBuf};

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if matches!(name, "tests" | "benches" | "examples" | "target") {
                continue;
            }
            sources(&path, out);
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_code_outside_the_mode_module_branches_on_a_mode_variant() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    for top in ["apps", "crates"] {
        sources(&root.join(top), &mut files);
    }
    assert!(files.len() > 50, "the gate found only {} files; it is looking in the wrong place", files.len());

    let mut offences = Vec::new();
    for file in &files {
        let shown = file.strip_prefix(&root).unwrap_or(file).to_string_lossy().replace('\\', "/");
        if shown.ends_with("kreate-ops/src/mode.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(file) else { continue };
        let live = text.split("#[cfg(test)]").next().unwrap_or("");
        for (number, line) in live.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for variant in ["Mode::EditMesh", "Mode::Object", "Mode::Pose"] {
                let mut at = 0;
                while let Some(found) = code[at..].find(variant) {
                    let end = at + found + variant.len();
                    // `Mode::ObjectFoo` is some other name.
                    let next = code[end..].chars().next();
                    if !next.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                        offences.push(format!("{shown}:{}: {}", number + 1, line.trim()));
                    }
                    at = end;
                }
            }
        }
    }
    assert!(
        offences.is_empty(),
        "these branch on a mode by name; ask its card (`mode.info()`, \
         `mode.acts_on()`, `mode.edits_elements()`) instead:\n{}",
        offences.join("\n")
    );
}
