//! Committed files that this build must keep reading, byte for byte.
//!
//! The plan's rule: every format change is deliberate, and a golden file is what
//! makes it so. Both formats are round-trip tested elsewhere, and a round-trip
//! test is satisfied by *any* self-consistent format — including one that
//! changed this morning and can no longer open a file saved yesterday.
//!
//! So these tests do two different things, and the difference is the point:
//!
//! - **The file still reads**, and produces the document it always did. This is
//!   the compatibility promise to the user.
//! - **The writer still produces exactly these bytes.** This is not a promise to
//!   anyone; it is a tripwire. A change to a field name, a default, an entry
//!   order or a padding rule fails here, and the author has to look at the diff
//!   and decide whether it was intended.
//!
//! # Regenerating
//!
//! `UPDATE_GOLDENS=1 cargo test -p kreate-io --test golden`
//!
//! Deliberately awkward, and the regenerated files must be read in the diff
//! before being committed. A golden file regenerated reflexively is a golden
//! file that tests nothing — it records whatever the code does, which is what
//! the code already does.
//!
//! # What these files are
//!
//! Written by version 1.0, the first released format. When a migration is added,
//! this file becomes the input its test uses — see [`kreate_io::migrate`], which
//! is explicit that a migration tested against a file synthesised by today's
//! code tests today's *idea* of what the old version wrote.

use kreate_io::container::{self, Manifest};
use kreate_io::kmesh;
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_scene::document::{Document, Object, ObjectData};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets")
}

/// Compares against a committed file, or rewrites it when asked.
fn check_golden(name: &str, produced: &[u8]) {
    let path = assets().join(name);
    if std::env::var("UPDATE_GOLDENS").is_ok() {
        std::fs::create_dir_all(assets()).expect("assets directory");
        std::fs::write(&path, produced).expect("write golden");
        eprintln!("updated {}", path.display());
        return;
    }
    let expected = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\nrun with UPDATE_GOLDENS=1 to create it",
            path.display()
        )
    });
    assert_eq!(
        produced.len(),
        expected.len(),
        "{name} changed size: {} bytes now against {} committed. \
         If the change is intended, regenerate with UPDATE_GOLDENS=1 and read the diff.",
        produced.len(),
        expected.len()
    );
    if produced != expected {
        let at = produced
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "{name} first differs at byte {at}: {} now against {} committed. \
             If the change is intended, regenerate with UPDATE_GOLDENS=1 and read the diff.",
            produced[at], expected[at]
        );
    }
}

/// The mesh the golden `.kmesh` holds.
///
/// Fixed by construction rather than by a primitive's current output: a change
/// to `primitives::cube` would otherwise change the golden file and the failure
/// would point at the format instead of at the primitive.
fn golden_mesh() -> kreate_mesh::MeshData {
    let positions = vec![
        glam::Vec3::new(-1.0, -1.0, 0.0),
        glam::Vec3::new(1.0, -1.0, 0.0),
        glam::Vec3::new(1.0, 1.0, 0.0),
        glam::Vec3::new(-1.0, 1.0, 0.0),
        glam::Vec3::new(0.0, 0.0, 1.5),
    ];
    // A quad and a triangle: an n-gon and a tri, so the CSR offsets are not all
    // the same stride and a reader that assumed a fixed face size would fail.
    let mut mesh = kreate_mesh::MeshData::from_faces(
        positions,
        &[vec![0, 1, 2, 3], vec![0, 1, 4], vec![1, 2, 4]],
    );
    let verts = mesh.num_verts() as usize;
    let corners = mesh.corner_verts().len();
    let faces = mesh.num_faces() as usize;
    let edges = mesh.num_edges() as usize;

    mesh.attributes_mut().insert(
        "uv0",
        Domain::Corner,
        AttrData::F32x2(Arc::new(
            (0..corners)
                .map(|i| glam::Vec2::new(i as f32 * 0.125, 1.0 - i as f32 * 0.0625))
                .collect(),
        )),
    );
    mesh.attributes_mut().insert(
        "material_index",
        Domain::Face,
        AttrData::I32(Arc::new((0..faces).map(|i| i as i32).collect())),
    );
    mesh.attributes_mut().insert(
        "crease",
        Domain::Edge,
        AttrData::F32(Arc::new((0..edges).map(|i| i as f32 * 0.25).collect())),
    );
    mesh.attributes_mut().insert(
        names::SELECT_VERT,
        Domain::Point,
        AttrData::Bool(Arc::new((0..verts).map(|i| i % 2 == 0).collect())),
    );
    mesh
}

/// The document the golden `.kreate` holds.
fn golden_document() -> Document {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(golden_mesh());
    let base = doc.add_object(Object::new("Base", ObjectData::Mesh(mesh)));
    // A linked duplicate and an empty parent, so the file exercises datablock
    // sharing and the hierarchy rather than one lone object.
    let twin = doc.add_object(Object::new("Twin", ObjectData::Mesh(mesh)));
    let pivot = doc.add_object(Object::new("Pivot", ObjectData::Empty));
    doc.set_parent(twin, Some(pivot)).expect("parent");

    if let Some(o) = doc.object_mut(base) {
        o.transform.translation = glam::DVec3::new(0.5, -1.25, 2.0);
    }
    if let Some(o) = doc.object_mut(twin) {
        o.transform.rotation = glam::DQuat::from_rotation_z(0.75);
        o.transform.scale = glam::DVec3::splat(1.5);
        o.visible = false;
    }
    doc
}

/// The manifest the golden file was written with.
///
/// Pinned rather than taken from `Manifest::default()`, whose `generator` field
/// carries the crate version — the golden would then change on every release for
/// no reason connected to the format.
fn golden_manifest() -> Manifest {
    Manifest {
        generator: "kreate golden".to_string(),
        ..Manifest::default()
    }
}

#[test]
fn the_golden_kmesh_is_still_what_this_build_writes() {
    check_golden("golden-v1.kmesh", &kmesh::write(&golden_mesh(), &[]));
}

#[test]
fn the_golden_kreate_is_still_what_this_build_writes() {
    let bytes = container::write(&golden_document(), &golden_manifest(), &[]).expect("writes");
    check_golden("golden-v1.kreate", &bytes);
}

#[test]
fn the_golden_kmesh_still_reads_correctly() {
    // The compatibility promise, checked against the committed bytes rather than
    // against something this build just wrote. If the two tests ever disagree —
    // this one passing while the byte comparison fails — the format changed in a
    // way that is still backwards compatible, which is exactly the situation the
    // author needs to be told about rather than left to discover.
    let path = assets().join("golden-v1.kmesh");
    let Ok(bytes) = std::fs::read(&path) else {
        panic!("missing {}; run with UPDATE_GOLDENS=1", path.display());
    };
    let loaded = kmesh::read(&bytes).expect("the committed file must still open");
    let expected = golden_mesh();

    assert_eq!(loaded.mesh.num_verts(), expected.num_verts());
    assert_eq!(loaded.mesh.face_offsets(), expected.face_offsets());
    assert_eq!(loaded.mesh.corner_verts(), expected.corner_verts());
    assert_eq!(loaded.mesh.edges(), expected.edges());
    assert_eq!(
        loaded.mesh.attributes().iter().count(),
        expected.attributes().iter().count(),
        "an attribute was lost"
    );
    for layer in expected.attributes().iter() {
        let got = loaded
            .mesh
            .attributes()
            .iter()
            .find(|l| l.name == layer.name)
            .unwrap_or_else(|| panic!("{} is missing", layer.name));
        assert_eq!(got.domain, layer.domain, "{} domain", layer.name);
        assert_eq!(
            format!("{:?}", got.data),
            format!("{:?}", layer.data),
            "{} values",
            layer.name
        );
    }
}

#[test]
fn the_golden_kreate_still_reads_correctly() {
    let path = assets().join("golden-v1.kreate");
    let Ok(bytes) = std::fs::read(&path) else {
        panic!("missing {}; run with UPDATE_GOLDENS=1", path.display());
    };
    let loaded = container::read(Cursor::new(bytes)).expect("the committed file must still open");
    let expected = golden_document();

    assert_eq!(loaded.manifest.version_major, 1);
    assert_eq!(loaded.manifest.version_minor, 0);
    assert_eq!(
        loaded.document.objects().count(),
        expected.objects().count(),
        "object count"
    );
    assert_eq!(
        loaded.document.meshes().count(),
        1,
        "the linked duplicate should still share one datablock"
    );

    let names: Vec<_> = loaded
        .document
        .objects()
        .map(|(_, o)| o.name.clone())
        .collect();
    assert_eq!(names, vec!["Base", "Twin", "Pivot"]);

    for ((_, a), (_, b)) in expected.objects().zip(loaded.document.objects()) {
        assert_eq!(
            a.transform.translation, b.transform.translation,
            "{}",
            a.name
        );
        assert_eq!(a.transform.scale, b.transform.scale, "{}", a.name);
        assert_eq!(a.transform.rotation, b.transform.rotation, "{}", a.name);
        assert_eq!(a.visible, b.visible, "{} visibility", a.name);
    }
}

#[test]
fn the_golden_files_are_committed() {
    // A golden test that silently creates its own input the first time it runs
    // is a golden test that never fails. This says so out loud.
    for name in ["golden-v1.kmesh", "golden-v1.kreate"] {
        let path = assets().join(name);
        assert!(
            path.exists(),
            "{} is missing. Golden files are committed on purpose: without them \
             these tests compare this build against itself.",
            path.display()
        );
    }
}
