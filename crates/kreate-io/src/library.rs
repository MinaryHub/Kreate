//! Linking datablocks from other documents.
//!
//! # What linking is for
//!
//! A studio's assets do not live in the file being worked on. A prop is authored
//! once, and every shot that uses it wants *that* prop — so that fixing it fixes
//! it everywhere, rather than in one of eleven copies somebody has to find. §1.4
//! puts the object-to-datablock indirection in place from day one precisely so
//! that this is possible; this module is the part that reads the other file.
//!
//! # What is deliberately not done
//!
//! **Nothing is written back.** A linked datablock is read-only, enforced by
//! [`kreate_scene::document::Linkage::is_editable`], and this module has no way
//! to save into a library. Editing an asset through a document that merely links
//! it is how a shot file silently changes a prop for every other shot; the
//! answer is to open the library, not to reach through.
//!
//! **Links are not followed transitively.** A library that itself links to a
//! third file resolves its own links when it is opened, and this reads the result
//! — it does not chase the chain. A cycle is therefore impossible by
//! construction rather than by detection, which is a much better place to be
//! than one where cycle detection has to be correct.
//!
//! # Paths
//!
//! A library reference is stored **relative to the document that holds it**, and
//! resolved against the document's own directory. This is the difference between
//! a project folder that can be moved, copied to another machine, or checked into
//! version control, and one that only works on the machine it was authored on.
//!
//! An absolute path is accepted when that is genuinely what was meant — a shared
//! asset server mounted at the same place everywhere — but it is never *produced*
//! by [`relative_reference`].

use std::path::{Path, PathBuf};

use kreate_scene::document::{Linkage, Object, ObjectData};
use kreate_scene::Document;

#[derive(Debug, thiserror::Error)]
pub enum LibraryError {
    #[error("cannot read library {path}: {source}")]
    Unreadable {
        path: String,
        #[source]
        source: crate::container::ContainerError,
    },
    #[error("library {library} has no object named {name}")]
    NoSuchObject { library: String, name: String },
    #[error("{name} in {library} has no mesh to link")]
    NotGeometry { library: String, name: String },
}

/// What a link asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// The library, as written in the document: relative to it where possible.
    pub library: String,
    /// The name of the object in that library.
    pub name: String,
}

/// Turns a library path into the form to store in a document.
///
/// Relative to `document_dir` when the two are on the same volume and the
/// library is not above it; absolute otherwise. The "not above it" rule is the
/// interesting one: a reference of `../../../assets/prop.kreate` is technically
/// relative and practically useless, because it only resolves from a directory
/// at exactly the right depth. A path that has to climb out of the project is not
/// really part of the project.
pub fn relative_reference(document_dir: &Path, library: &Path) -> String {
    let Ok(library) = library.canonicalize() else {
        return library.to_string_lossy().to_string();
    };
    let Ok(base) = document_dir.canonicalize() else {
        return library.to_string_lossy().to_string();
    };
    match library.strip_prefix(&base) {
        // Forward slashes, always. A path written with backslashes is a path
        // that only resolves on Windows, and a document is meant to be openable
        // by whoever the file is sent to.
        Ok(relative) => relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => library.to_string_lossy().to_string(),
    }
}

/// Resolves a stored reference against the document's own directory.
///
/// An absolute reference resolves to itself, and that needs no branch here:
/// `Path::join` replaces the base when its argument is absolute. A version of
/// this with an explicit `is_absolute` check was dead code — removing the check
/// changed no behaviour and no test, which is how it was found.
pub fn resolve_path(document_dir: &Path, library: &str) -> PathBuf {
    document_dir.join(Path::new(library))
}

/// Reads one object's mesh out of a library and adds it to `document`.
///
/// The new object is marked [`Linkage::Linked`], which makes it read-only. Its
/// transform is the library object's, so a linked prop arrives where the asset
/// puts it and can then be moved — moving a linked object is allowed, because
/// the transform belongs to *this* document's object, not to the asset.
pub fn link_object(
    document: &mut Document,
    document_dir: &Path,
    reference: &Reference,
) -> Result<kreate_scene::document::ObjectId, LibraryError> {
    let path = resolve_path(document_dir, &reference.library);
    let file = std::fs::File::open(&path).map_err(|e| LibraryError::Unreadable {
        path: path.display().to_string(),
        source: crate::container::ContainerError::Io(e),
    })?;
    let loaded = crate::container::read(std::io::BufReader::new(file)).map_err(|e| {
        LibraryError::Unreadable {
            path: path.display().to_string(),
            source: e,
        }
    })?;

    let (_, source) = loaded
        .document
        .objects()
        .find(|(_, object)| object.name == reference.name)
        .ok_or_else(|| LibraryError::NoSuchObject {
            library: reference.library.clone(),
            name: reference.name.clone(),
        })?;

    let ObjectData::Mesh(mesh_id) = source.data else {
        return Err(LibraryError::NotGeometry {
            library: reference.library.clone(),
            name: reference.name.clone(),
        });
    };
    let mesh = loaded
        .document
        .mesh(mesh_id)
        .ok_or_else(|| LibraryError::NotGeometry {
            library: reference.library.clone(),
            name: reference.name.clone(),
        })?;

    // Cloned into this document rather than shared. There is no shared ownership
    // to have: the library's document is dropped at the end of this function,
    // and the mesh's `Arc`-per-array layout means the clone shares every buffer
    // with the copy that was just read — so this is a handful of pointer bumps,
    // not a copy of the geometry.
    let local_mesh = document.add_mesh(mesh.clone());
    let mut object = Object::new(source.name.clone(), ObjectData::Mesh(local_mesh));
    object.transform = source.transform;
    object.linkage = Linkage::Linked {
        library: reference.library.clone(),
        name: reference.name.clone(),
    };
    Ok(document.add_object(object))
}

/// Links every mesh object a library holds, reading it once.
///
/// The asset browser's Link: a `.kreate` of props, each arriving where the
/// library put it, read-only, and refreshed from the library on every open.
/// The library is named by `library` as it should be written in the
/// document — absolute for a document not yet saved, which has no folder to
/// be relative to.
pub fn link_all(
    document: &mut Document,
    document_dir: &Path,
    library: &str,
) -> Result<Vec<kreate_scene::document::ObjectId>, LibraryError> {
    let path = resolve_path(document_dir, library);
    let file = std::fs::File::open(&path).map_err(|e| LibraryError::Unreadable {
        path: path.display().to_string(),
        source: crate::container::ContainerError::Io(e),
    })?;
    let loaded = crate::container::read(std::io::BufReader::new(file)).map_err(|e| {
        LibraryError::Unreadable {
            path: path.display().to_string(),
            source: e,
        }
    })?;
    let mut made = Vec::new();
    for (_, source) in loaded.document.objects() {
        let ObjectData::Mesh(mesh_id) = source.data else {
            continue;
        };
        let Some(mesh) = loaded.document.mesh(mesh_id) else {
            continue;
        };
        let local_mesh = document.add_mesh(mesh.clone());
        let mut object = Object::new(source.name.clone(), ObjectData::Mesh(local_mesh));
        object.transform = source.transform;
        object.linkage = Linkage::Linked {
            library: library.to_string(),
            name: source.name.clone(),
        };
        made.push(document.add_object(object));
    }
    Ok(made)
}

/// Re-reads every linked object in `document` from its library.
///
/// This is what makes linking worth having: the asset changed, and the shot that
/// links it picks the change up. Called on open.
///
/// Failures are collected rather than propagated, and this is the important
/// decision in the function. A missing or broken library must not stop the
/// document opening — the user needs to *see* their scene in order to do anything
/// about the link, and a shot file that will not open because a prop moved on the
/// server is a shot file nobody can rescue. The stale geometry stays, and the
/// caller reports what could not be refreshed.
pub fn refresh_links(document: &mut Document, document_dir: &Path) -> Vec<LibraryError> {
    let mut problems = Vec::new();

    // Collected first, because resolving mutates the document and the iteration
    // borrows it.
    let linked: Vec<(kreate_scene::document::ObjectId, Reference)> = document
        .objects()
        .filter_map(|(id, object)| match &object.linkage {
            Linkage::Linked { library, name } => Some((
                id,
                Reference {
                    library: library.clone(),
                    name: name.clone(),
                },
            )),
            Linkage::Local => None,
        })
        .collect();

    // One library read per file, not per object: a scene linking forty props
    // from one asset file should open it once.
    let mut cache: std::collections::BTreeMap<String, Option<Document>> =
        std::collections::BTreeMap::new();
    // Libraries already reported as unreadable.
    let mut reported_libraries: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();

    for (id, reference) in linked {
        let library = cache.entry(reference.library.clone()).or_insert_with(|| {
            let path = resolve_path(document_dir, &reference.library);
            std::fs::File::open(&path)
                .ok()
                .and_then(|f| crate::container::read(std::io::BufReader::new(f)).ok())
                .map(|loaded| loaded.document)
        });
        let Some(library_document) = library else {
            // Once per library, not once per object. A scene linking forty props
            // from one missing asset file has one problem — the file — and
            // saying so forty times buries it. A *missing object* in a library
            // that does read is genuinely per-object, and is reported that way
            // below.
            if reported_libraries.insert(reference.library.clone()) {
                problems.push(LibraryError::Unreadable {
                    path: resolve_path(document_dir, &reference.library)
                        .display()
                        .to_string(),
                    source: crate::container::ContainerError::NotAKreateDocument(
                        "cannot be opened".to_string(),
                    ),
                });
            }
            continue;
        };

        let found = library_document
            .objects()
            .find(|(_, object)| object.name == reference.name)
            .and_then(|(_, object)| match object.data {
                ObjectData::Mesh(mesh) => library_document.mesh(mesh),
                _ => None,
            });
        let Some(mesh) = found else {
            problems.push(LibraryError::NoSuchObject {
                library: reference.library.clone(),
                name: reference.name.clone(),
            });
            continue;
        };

        let mesh = mesh.clone();
        let new_mesh = document.add_mesh(mesh);
        if let Some(object) = document.object_mut(id) {
            // Only the geometry. The transform, the name, the parent and the
            // visibility belong to this document's object — a refresh that reset
            // them would move every linked prop back to where the asset happens
            // to sit whenever the asset was touched.
            object.data = ObjectData::Mesh(new_mesh);
        }
    }
    problems
}
