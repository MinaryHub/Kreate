//! A scene file, whatever its format, read into a document of its own.
//!
//! What Open, Import, a drop and the asset browser all read through, and what
//! `object.import` adds: one place that knows which extension is which reader,
//! so no caller has its own list to fall behind.
//!
//! # Prepared off the main thread
//!
//! Reading a large model takes long enough to stop the interface. The
//! application reads it on a thread of its own and leaves the result here,
//! with [`prepare`]; the operator that adds it finds it with [`take`] and does
//! only the adding. A file not prepared — a script replaying the step — is
//! read where it is asked for.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use kreate_scene::Document;

use crate::build::BuildNote;

/// The extensions read as scenes, beside Kreate's own `.kreate`.
pub const SCENE_FORMATS: &[&str] = &["gltf", "glb", "fbx", "obj", "stl", "ply", "3mf", "off"];

/// Whether `path` names a file this module reads.
pub fn is_scene_file(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| e == "kreate" || SCENE_FORMATS.contains(&e.as_str()))
}

/// Reads a scene file into a fresh document, through the IR — or, for a
/// `.kreate`, the document it holds, its links refreshed.
///
/// Every format lands here, which is the point: what a file said and what the
/// document holds is one conversion in one place, and §5.7's fidelity contract
/// is written against it.
pub fn read_scene_file(path: &Path) -> Result<(Document, Vec<BuildNote>), String> {
    let lower = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if lower == "kreate" {
        let file = std::fs::File::open(path).map_err(|e| format!("cannot open: {e}"))?;
        let mut document = crate::container::read(std::io::BufReader::new(file))
            .map_err(|e| e.to_string())?
            .document;
        let directory = path.parent().unwrap_or(Path::new("."));
        let _ = crate::library::refresh_links(&mut document, directory);
        return Ok((document, Vec::new()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read: {e}"))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Import".to_string());
    let text = |bytes: Vec<u8>| String::from_utf8(bytes).map_err(|e| format!("not utf-8: {e}"));
    let scene = match lower.as_str() {
        // A texture an FBX names but does not carry is looked for beside it:
        // where it says, then by its file name alone — what a model copied
        // from another machine still has.
        "fbx" => {
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            crate::parse_fbx(&bytes, &|named: &str| {
                let named = named.replace('\\', "/");
                let bare = named.rsplit('/').next().map(|n| dir.join(n));
                std::fs::read(dir.join(&named))
                    .ok()
                    .or_else(|| bare.and_then(|b| std::fs::read(b).ok()))
            })
            .map_err(|e| e.to_string())?
        }
        "3mf" => crate::parse_3mf(&bytes).map_err(|e| e.to_string())?,
        "ply" => crate::parse_ply(&bytes, &name).map_err(|e| e.to_string())?,
        "off" => crate::parse_off(&text(bytes)?, &name).map_err(|e| e.to_string())?,
        // glTF declares its own units and orientation, so there is nothing
        // for an option to override: second-guessing the one format that
        // bothers to say would be the wrong kind of helpful.
        "gltf" | "glb" => crate::gltf_import::parse_gltf_file(path, &Default::default())
            .map_err(|e| e.to_string())?,
        "stl" => crate::stl::parse_stl(&bytes, &Default::default()).map_err(|e| e.to_string())?,
        "obj" => crate::parse_obj(&text(bytes)?, Default::default()).map_err(|e| e.to_string())?,
        other => return Err(format!("`.{other}` is not a scene file Kreate reads")),
    };
    let mut doc = Document::new();
    let (made, notes) = crate::build::build_into(&scene, &mut doc);
    if let Some(first) = made.first() {
        doc.selection.add(*first);
    }
    Ok((doc, notes))
}

/// Documents read ahead, by path and the file's modification time.
type Prepared = HashMap<PathBuf, (Option<SystemTime>, Arc<Document>)>;

fn prepared() -> &'static Mutex<Prepared> {
    static PREPARED: OnceLock<Mutex<Prepared>> = OnceLock::new();
    PREPARED.get_or_init(|| Mutex::new(HashMap::new()))
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Leaves a document read from `path` for [`take`] to find.
pub fn prepare(path: &Path, document: Document) {
    if let Ok(mut map) = prepared().lock() {
        map.insert(path.to_path_buf(), (modified(path), Arc::new(document)));
    }
}

/// The document read ahead from `path`, if it is there and the file has not
/// changed since; or the file, read now.
pub fn take(path: &Path) -> Result<Arc<Document>, String> {
    let ready = prepared().lock().ok().and_then(|mut map| map.remove(path));
    match ready {
        Some((when, document)) if when == modified(path) => Ok(document),
        _ => read_scene_file(path).map(|(document, _)| Arc::new(document)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_cube_file(dir: &Path) -> PathBuf {
        std::fs::create_dir_all(dir).expect("made");
        let mesh = kreate_mesh::primitives::cube(1.0);
        let path = dir.join("cube.obj");
        std::fs::write(
            &path,
            crate::write_obj(&[("Cube".to_string(), &mesh)], Default::default()),
        )
        .expect("written");
        path
    }

    #[test]
    fn a_file_is_read_by_its_extension_and_a_stranger_is_refused() {
        let dir = std::env::temp_dir().join(format!("kreate-scenefile-{}", std::process::id()));
        let path = a_cube_file(&dir);
        let (doc, _) = read_scene_file(&path).expect("read");
        assert_eq!(doc.objects().count(), 1);
        assert!(is_scene_file(&path) && !is_scene_file(Path::new("notes.txt")));
        let other = dir.join("notes.txt");
        std::fs::write(&other, "hello").expect("written");
        assert!(read_scene_file(&other).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_prepared_document_is_taken_once_and_a_changed_file_is_read_again() {
        let dir = std::env::temp_dir().join(format!("kreate-prepared-{}", std::process::id()));
        let path = a_cube_file(&dir);
        // What was prepared is what is taken — here, a document of two
        // objects standing for the file's one.
        let mut two = Document::new();
        two.add_mesh_object("A", kreate_mesh::primitives::cube(1.0));
        two.add_mesh_object("B", kreate_mesh::primitives::cube(1.0));
        prepare(&path, two);
        assert_eq!(take(&path).expect("taken").objects().count(), 2);
        // Taken once: the second time reads the file.
        assert_eq!(take(&path).expect("read").objects().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
