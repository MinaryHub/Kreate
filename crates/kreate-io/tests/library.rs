//! Library linking: reading datablocks out of other documents.
//!
//! The tests here are mostly about the failure modes, because that is where
//! linking earns its keep or destroys a day's work. A link that resolves is
//! unremarkable; a link that silently stops resolving, or that writes back into
//! an asset, or that stops a shot file opening, is the reason studios have
//! opinions about this feature.

use kreate_io::container::{self, Manifest};
use kreate_io::library::{self, Reference};
use kreate_mesh::MeshData;
use kreate_scene::document::{Linkage, Object, ObjectData};
use kreate_scene::Document;
use std::path::{Path, PathBuf};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("kreate-library-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn triangle(scale: f32) -> MeshData {
    MeshData::from_faces(
        vec![
            glam::Vec3::ZERO,
            glam::Vec3::X * scale,
            glam::Vec3::Y * scale,
        ],
        &[vec![0, 1, 2]],
    )
}

/// A library holding one named prop.
fn write_library(path: &Path, name: &str, mesh: MeshData) {
    let mut document = Document::new();
    let id = document.add_mesh(mesh);
    let object = document.add_object(Object::new(name, ObjectData::Mesh(id)));
    if let Some(object) = document.object_mut(object) {
        object.transform.translation = glam::DVec3::new(5.0, 0.0, 0.0);
    }
    let bytes = container::write(&document, &Manifest::default(), &[]).expect("write");
    std::fs::write(path, bytes).expect("save");
}

#[test]
fn a_linked_object_arrives_with_its_geometry() {
    let dir = scratch("link_basic");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));

    let mut document = Document::new();
    let id = library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .expect("link");

    let object = document.object(id).expect("the linked object");
    assert_eq!(object.name, "bolt");
    let mesh = object.mesh().and_then(|m| document.mesh(m)).expect("mesh");
    assert_eq!(mesh.num_verts(), 3);
    // The asset's transform comes with it, so a prop arrives where the asset
    // puts it rather than at the origin.
    assert_eq!(object.transform.translation.x, 5.0);
}

#[test]
fn a_linked_object_is_not_editable() {
    // The guard the whole feature rests on. An editable link is a link that lets
    // one shot change a prop for every other shot.
    let dir = scratch("link_readonly");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));
    let mut document = Document::new();
    let id = library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .expect("link");
    let object = document.object(id).expect("object");
    assert!(!object.linkage.is_editable());
    assert!(matches!(object.linkage, Linkage::Linked { .. }));
}

#[test]
fn linking_a_missing_object_is_an_error_not_a_silent_empty() {
    let dir = scratch("link_missing_object");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));
    let mut document = Document::new();
    let result = library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "nut".to_string(),
        },
    );
    assert!(result.is_err());
    assert_eq!(
        document.object_count(),
        0,
        "a failed link left an object behind"
    );
}

#[test]
fn linking_a_missing_library_is_an_error() {
    let dir = scratch("link_missing_library");
    let mut document = Document::new();
    assert!(library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "nowhere.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .is_err());
}

#[test]
fn linking_something_with_no_geometry_is_an_error() {
    let dir = scratch("link_no_geometry");
    let mut library_document = Document::new();
    library_document.add_object(Object::new("pivot", ObjectData::Empty));
    let bytes = container::write(&library_document, &Manifest::default(), &[]).expect("write");
    std::fs::write(dir.join("prop.kreate"), bytes).expect("save");

    let mut document = Document::new();
    assert!(library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "pivot".to_string(),
        },
    )
    .is_err());
}

#[test]
fn a_refresh_picks_up_a_change_to_the_asset() {
    // The point of the feature. Fixing the asset fixes it in every scene that
    // links it, and the scene finds out when it next opens.
    let dir = scratch("refresh");
    let library_path = dir.join("prop.kreate");
    write_library(&library_path, "bolt", triangle(1.0));

    let mut document = Document::new();
    let id = library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .expect("link");
    assert_eq!(vertex_count(&document, id), 3);

    // The asset gains geometry.
    write_library(
        &library_path,
        "bolt",
        MeshData::from_faces(
            vec![
                glam::Vec3::ZERO,
                glam::Vec3::X,
                glam::Vec3::Y,
                glam::Vec3::ONE,
            ],
            &[vec![0, 1, 2, 3]],
        ),
    );

    let problems = library::refresh_links(&mut document, &dir);
    assert!(problems.is_empty(), "{problems:?}");
    assert_eq!(
        vertex_count(&document, id),
        4,
        "the refresh did not pick up the new asset"
    );
}

fn vertex_count(document: &Document, id: kreate_scene::document::ObjectId) -> u32 {
    document
        .object(id)
        .and_then(|o| o.mesh())
        .and_then(|m| document.mesh(m))
        .map(|m| m.num_verts())
        .unwrap_or(0)
}

#[test]
fn a_refresh_keeps_this_documents_own_transform() {
    // The asset's transform arrives with the link and then belongs to this
    // document. A refresh that reset it would move every linked prop back to
    // wherever the asset sits, every time the asset was touched — which is the
    // behaviour that makes people stop using linking.
    let dir = scratch("refresh_transform");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));
    let mut document = Document::new();
    let id = library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .expect("link");

    // The user places it.
    if let Some(object) = document.object_mut(id) {
        object.transform.translation = glam::DVec3::new(-11.0, 22.0, 33.0);
        object.visible = false;
    }

    write_library(&dir.join("prop.kreate"), "bolt", triangle(2.0));
    library::refresh_links(&mut document, &dir);

    let object = document.object(id).expect("object");
    assert_eq!(
        object.transform.translation,
        glam::DVec3::new(-11.0, 22.0, 33.0),
        "the refresh moved the object"
    );
    assert!(!object.visible, "the refresh changed visibility");
}

#[test]
fn a_broken_link_does_not_stop_the_rest_from_refreshing() {
    // A shot file that will not open because one prop moved on the server is a
    // shot file nobody can rescue. Failures are collected and reported; the good
    // links still update.
    let dir = scratch("refresh_partial");
    write_library(&dir.join("good.kreate"), "bolt", triangle(1.0));

    let mut document = Document::new();
    let good = library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "good.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .expect("link");
    // A link to something that was never there.
    let mut orphan = Object::new("ghost", ObjectData::Empty);
    orphan.linkage = Linkage::Linked {
        library: "gone.kreate".to_string(),
        name: "ghost".to_string(),
    };
    document.add_object(orphan);

    write_library(
        &dir.join("good.kreate"),
        "bolt",
        MeshData::from_faces(
            vec![
                glam::Vec3::ZERO,
                glam::Vec3::X,
                glam::Vec3::Y,
                glam::Vec3::ONE,
            ],
            &[vec![0, 1, 2, 3]],
        ),
    );

    let problems = library::refresh_links(&mut document, &dir);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert_eq!(
        vertex_count(&document, good),
        4,
        "a broken link stopped a good one from refreshing"
    );
}

#[test]
fn a_refresh_leaves_local_objects_alone() {
    let dir = scratch("refresh_local");
    let mut document = Document::new();
    let mesh = document.add_mesh(triangle(1.0));
    let local = document.add_object(Object::new("mine", ObjectData::Mesh(mesh)));
    let problems = library::refresh_links(&mut document, &dir);
    assert!(problems.is_empty());
    assert_eq!(vertex_count(&document, local), 3);
}

#[test]
fn a_missing_library_is_reported_once_not_once_per_object() {
    // A scene linking forty props from one missing asset file has one problem —
    // the file — and saying so forty times buries it.
    //
    // Named for what it checks. It started out claiming to prove the library is
    // read once per refresh rather than once per object, using the number of
    // reported problems as a proxy; the proxy measured the reporting, not the
    // reads. Whether the read is cached is a performance property with no
    // observable behaviour, so there is no honest test for it here and the
    // comment in `refresh_links` has to carry it instead.
    let dir = scratch("refresh_cache");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));
    let mut document = Document::new();
    for _ in 0..3 {
        library::link_object(
            &mut document,
            &dir,
            &Reference {
                library: "prop.kreate".to_string(),
                name: "bolt".to_string(),
            },
        )
        .expect("link");
    }
    // Now make it unreadable. Three objects, one library, so exactly one problem.
    std::fs::remove_file(dir.join("prop.kreate")).expect("remove");
    let problems = library::refresh_links(&mut document, &dir);
    assert_eq!(
        problems.len(),
        1,
        "one missing library produced {} messages: {problems:?}",
        problems.len()
    );
}

#[test]
fn two_missing_libraries_are_reported_separately() {
    // The other side of the de-duplication: collapsing by library must not
    // collapse two different libraries into one message, which would leave the
    // user fixing one path and wondering why it still does not work.
    let dir = scratch("refresh_two_missing");
    let mut document = Document::new();
    for library in ["one.kreate", "two.kreate"] {
        let mut object = Object::new(format!("from-{library}"), ObjectData::Empty);
        object.linkage = Linkage::Linked {
            library: library.to_string(),
            name: "bolt".to_string(),
        };
        document.add_object(object);
    }
    let problems = library::refresh_links(&mut document, &dir);
    assert_eq!(problems.len(), 2, "{problems:?}");
}

#[test]
fn two_missing_objects_in_one_library_are_reported_separately() {
    // A missing *object* is per-object: each one is a different thing to fix,
    // and the library itself is fine.
    let dir = scratch("refresh_two_objects");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));
    let mut document = Document::new();
    for name in ["nut", "washer"] {
        let mut object = Object::new(name, ObjectData::Empty);
        object.linkage = Linkage::Linked {
            library: "prop.kreate".to_string(),
            name: name.to_string(),
        };
        document.add_object(object);
    }
    let problems = library::refresh_links(&mut document, &dir);
    assert_eq!(problems.len(), 2, "{problems:?}");
}

#[test]
fn a_reference_inside_the_project_is_stored_relative() {
    // The difference between a project folder that can be moved to another
    // machine and one that only works where it was authored.
    let dir = scratch("relative");
    let assets = dir.join("assets");
    std::fs::create_dir_all(&assets).expect("create");
    let library = assets.join("prop.kreate");
    write_library(&library, "bolt", triangle(1.0));

    let reference = library::relative_reference(&dir, &library);
    assert_eq!(
        reference, "assets/prop.kreate",
        "stored as {reference:?}, which will not resolve elsewhere"
    );
    // Forward slashes, so the document opens on a machine that is not Windows.
    assert!(!reference.contains('\\'), "{reference:?}");
}

#[test]
fn a_reference_outside_the_project_is_stored_absolute() {
    // `../../../assets/prop.kreate` is technically relative and practically
    // useless: it only resolves from a directory at exactly the right depth. A
    // path that has to climb out of the project is not part of the project.
    let dir = scratch("absolute");
    let inside = dir.join("scene");
    std::fs::create_dir_all(&inside).expect("create");
    let library = dir.join("prop.kreate");
    write_library(&library, "bolt", triangle(1.0));

    let reference = library::relative_reference(&inside, &library);
    assert!(
        Path::new(&reference).is_absolute(),
        "stored as {reference:?}, which climbs out of the project"
    );
}

#[test]
fn a_relative_reference_resolves_against_the_documents_directory() {
    let dir = scratch("resolve");
    let resolved = library::resolve_path(&dir, "assets/prop.kreate");
    assert_eq!(resolved, dir.join("assets").join("prop.kreate"));
}

#[test]
fn an_absolute_reference_resolves_to_itself() {
    // This documents `Path::join`'s behaviour rather than ours: joining an
    // absolute path onto a base replaces the base. It is asserted because
    // `resolve_path` relies on it and has no code of its own for the case — if
    // that ever stopped being true, every absolute library reference would
    // resolve to nonsense and nothing else would notice.
    let dir = scratch("resolve_absolute");
    let elsewhere = std::env::temp_dir().join("shared").join("prop.kreate");
    let resolved = library::resolve_path(&dir, &elsewhere.to_string_lossy());
    assert_eq!(resolved, elsewhere);
}

#[test]
fn a_link_survives_a_save_and_reopen() {
    // The linkage has to be in the file, or every open produces a document full
    // of editable copies and the asset connection is gone.
    let dir = scratch("link_persists");
    write_library(&dir.join("prop.kreate"), "bolt", triangle(1.0));
    let mut document = Document::new();
    library::link_object(
        &mut document,
        &dir,
        &Reference {
            library: "prop.kreate".to_string(),
            name: "bolt".to_string(),
        },
    )
    .expect("link");

    let bytes = container::write(&document, &Manifest::default(), &[]).expect("write");
    let loaded = container::read(std::io::Cursor::new(bytes)).expect("read");
    let (_, object) = loaded
        .document
        .objects()
        .find(|(_, o)| o.name == "bolt")
        .expect("the linked object");
    match &object.linkage {
        Linkage::Linked { library, name } => {
            assert_eq!(library, "prop.kreate");
            assert_eq!(name, "bolt");
        }
        Linkage::Local => panic!("the link became a local copy"),
    }
}
