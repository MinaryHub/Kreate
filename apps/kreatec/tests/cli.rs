//! `kreatec` end to end, by running the binary.
//!
//! Run rather than linked. The unit under test is a command line tool, so what
//! matters is what it prints and what exit code it returns — and those depend on
//! argument parsing and output formatting, which is most of what a CLI *is*. A
//! test that called the functions directly would pass while the command spelled
//! its flag wrong.
//!
//! `CARGO_BIN_EXE_kreatec` is set by cargo for integration tests of a package
//! with a binary, so there is no path guessing and no build step to arrange.

use kreate_io::container::{self, Manifest, UnknownEntry};
use kreate_mesh::MeshData;
use kreate_scene::document::{Document, Object, ObjectData};
use std::path::{Path, PathBuf};
use std::process::Command;

fn exe() -> &'static str {
    env!("CARGO_BIN_EXE_kreatec")
}

struct Run {
    stdout: String,
    ok: bool,
}

fn run(args: &[&str]) -> Run {
    let out = Command::new(exe())
        .args(args)
        .output()
        .expect("kreatec should be runnable");
    Run {
        // Lossy rather than strict: a mangled byte in the output should show up
        // as a failed assertion about content, not as a panic in the harness.
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        ok: out.status.success(),
    }
}

/// A directory of this test's own, named after the test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("kreatec-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn cube() -> MeshData {
    let p = glam::Vec3::new;
    MeshData::from_faces(
        vec![
            p(-1.0, -1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(-1.0, 1.0, -1.0),
            p(-1.0, -1.0, 1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
        ],
        &[
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
    )
}

fn write_document(path: &Path, document: &Document, unknown: &[UnknownEntry]) {
    let bytes = container::write(document, &Manifest::default(), unknown).expect("write");
    std::fs::write(path, bytes).expect("save");
}

#[test]
fn textconv_is_stable_across_runs() {
    // The property git needs. If the rendering varied between invocations, every
    // diff would be full of changes nobody made.
    let dir = scratch("textconv_stable");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    write_document(&path, &document, &[]);

    let a = run(&["textconv", path.to_str().unwrap()]);
    let b = run(&["textconv", path.to_str().unwrap()]);
    assert!(a.ok && b.ok);
    assert_eq!(a.stdout, b.stdout);
    assert!(a.stdout.contains("object cube"), "{}", a.stdout);
}

#[test]
fn textconv_omits_the_generator_version() {
    // The generator string carries a version number. On the first line of every
    // file, that would make every document rebuilt by a newer build show a diff
    // whether or not anything in it moved — a change that is real, uninteresting,
    // and in the way of the ones that matter.
    let dir = scratch("textconv_generator");
    let path = dir.join("a.kreate");
    write_document(&path, &Document::new(), &[]);
    let out = run(&["textconv", path.to_str().unwrap()]).stdout;
    assert!(
        !out.contains(env!("CARGO_PKG_VERSION")),
        "the version leaked into textconv output: {out}"
    );
    assert!(out.contains("format kreate"), "{out}");
}

#[test]
fn textconv_gives_one_line_per_object() {
    // Locality: changing one object must change a few lines, not reflow the file.
    // A summary with running totals or a single JSON blob would fail this and
    // would make every diff look like a rewrite.
    let dir = scratch("textconv_lines");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    for i in 0..5 {
        let mesh = document.add_mesh(cube());
        document.add_object(Object::new(format!("cube{i}"), ObjectData::Mesh(mesh)));
    }
    write_document(&path, &document, &[]);

    let out = run(&["textconv", path.to_str().unwrap()]).stdout;
    let objects = out
        .lines()
        .filter(|l| l.trim_start().starts_with("object "))
        .count();
    assert_eq!(objects, 5, "{out}");
}

#[test]
fn textconv_indents_the_hierarchy() {
    let dir = scratch("textconv_hierarchy");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    let parent = document.add_object(Object::new("parent", ObjectData::Empty));
    let child = document.add_object(Object::new("child", ObjectData::Empty));
    document
        .set_parent(child, Some(parent))
        .expect("parenting a fresh object should work");
    write_document(&path, &document, &[]);

    let out = run(&["textconv", path.to_str().unwrap()]).stdout;
    let child_line = out
        .lines()
        .find(|l| l.contains("object child"))
        .unwrap_or_else(|| panic!("{out}"));
    assert!(
        child_line.starts_with("  "),
        "the child was not indented: {child_line:?}"
    );
}

#[test]
fn diff_of_a_file_with_itself_is_empty() {
    let dir = scratch("diff_identical");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    write_document(&path, &document, &[]);

    let out = run(&["diff", path.to_str().unwrap(), path.to_str().unwrap()]).stdout;
    assert!(out.contains("no semantic difference"), "{out}");
}

#[test]
fn diff_of_two_saves_of_one_document_is_empty() {
    // Stronger than the previous test: two separate files, written by two
    // separate calls. This is what catches a float that does not survive the
    // decimal round trip, which would otherwise make every save look like a
    // move of every object.
    let dir = scratch("diff_two_saves");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    let id = document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    // An awkward transform, not the identity: zeroes round-trip through any
    // encoding, so the identity would prove nothing.
    if let Some(object) = document.object_mut(id) {
        object.transform.translation = glam::DVec3::new(0.1, -1.0 / 3.0, 1e-7);
        object.transform.scale = glam::DVec3::new(1.0 / 7.0, 2.5, 0.333333333333);
    }
    let a = dir.join("a.kreate");
    let b = dir.join("b.kreate");
    write_document(&a, &document, &[]);
    write_document(&b, &document, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(out.contains("no semantic difference"), "{out}");
}

#[test]
fn diff_survives_a_load_and_save_round_trip() {
    // The comparison is exact, which is only defensible if the format really is
    // exact. This goes through the parser — write, convert, diff — rather than
    // writing the same in-memory document twice, because it is the *read* path
    // that would introduce a last-bit difference.
    let dir = scratch("diff_roundtrip");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    let id = document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    if let Some(object) = document.object_mut(id) {
        // Values chosen to be hostile to a decimal encoding: none of them is
        // representable as a short decimal, and one is denormal-adjacent.
        object.transform.translation = glam::DVec3::new(1.0 / 3.0, -2.0 / 7.0, 1e-300);
        object.transform.scale = glam::DVec3::new(0.1, 1.0 / 9.0, 12345.6789012345);
        object.transform.rotation =
            glam::DQuat::from_axis_angle(glam::DVec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    }
    let a = dir.join("a.kreate");
    let b = dir.join("b.kreate");
    write_document(&a, &document, &[]);
    assert!(run(&["convert", a.to_str().unwrap(), b.to_str().unwrap()]).ok);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(
        out.contains("no semantic difference"),
        "the format does not reproduce f64 exactly, so an exact diff cannot \
         stand: {out}"
    );
}

#[test]
fn diff_reports_a_move_too_small_to_see() {
    // The other half. A tolerance would have swallowed this, and a user who
    // nudged something by a nanometre and was told nothing changed has been
    // misinformed by the tool they asked.
    let dir = scratch("diff_tiny_move");
    let mut document = Document::new();
    let id = document.add_object(Object::new("pivot", ObjectData::Empty));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);
    if let Some(object) = document.object_mut(id) {
        object.transform.translation = glam::DVec3::new(1e-12, 0.0, 0.0);
    }
    let b = dir.join("b.kreate");
    write_document(&b, &document, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(
        out.contains("pivot") && out.contains("moved"),
        "a real move went unreported: {out}"
    );
}

#[test]
fn diff_reports_a_moved_object() {
    let dir = scratch("diff_moved");
    let mut document = Document::new();
    let id = document.add_object(Object::new("pivot", ObjectData::Empty));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);
    if let Some(object) = document.object_mut(id) {
        object.transform.translation = glam::DVec3::new(1.0, 2.0, 3.0);
    }
    let b = dir.join("b.kreate");
    write_document(&b, &document, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(out.contains("pivot") && out.contains("moved"), "{out}");
}

#[test]
fn diff_reports_a_geometry_change_under_the_same_name() {
    // The case name matching exists for: the object is the same object, and what
    // changed is inside its mesh.
    let dir = scratch("diff_geometry");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("body", ObjectData::Mesh(mesh)));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);

    let mut changed = Document::new();
    let mesh = changed.add_mesh(MeshData::from_faces(
        vec![glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y],
        &[vec![0, 1, 2]],
    ));
    changed.add_object(Object::new("body", ObjectData::Mesh(mesh)));
    let b = dir.join("b.kreate");
    write_document(&b, &changed, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(
        out.contains("body") && out.contains("geometry") && out.contains("8 -> 3"),
        "{out}"
    );
}

#[test]
fn diff_reports_rotation_and_scale_separately() {
    // Translation is the one everybody remembers to check. A diff that reported
    // only moves would pass every other test here while saying nothing about an
    // object that was rotated or resized.
    let dir = scratch("diff_rotate_scale");
    let mut document = Document::new();
    let id = document.add_object(Object::new("thing", ObjectData::Empty));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);
    if let Some(object) = document.object_mut(id) {
        object.transform.rotation = glam::DQuat::from_rotation_z(0.5);
        object.transform.scale = glam::DVec3::new(2.0, 2.0, 2.0);
    }
    let b = dir.join("b.kreate");
    write_document(&b, &document, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(out.contains("rotated"), "rotation went unreported: {out}");
    assert!(out.contains("scaled"), "scale went unreported: {out}");
}

#[test]
fn diff_reports_visibility() {
    let dir = scratch("diff_visibility");
    let mut document = Document::new();
    let id = document.add_object(Object::new("thing", ObjectData::Empty));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);
    if let Some(object) = document.object_mut(id) {
        object.visible = false;
    }
    let b = dir.join("b.kreate");
    write_document(&b, &document, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(out.contains("hidden"), "{out}");
}

#[test]
fn diff_reports_reparenting() {
    let dir = scratch("diff_reparent");
    let mut document = Document::new();
    let parent = document.add_object(Object::new("parent", ObjectData::Empty));
    let child = document.add_object(Object::new("child", ObjectData::Empty));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);
    document.set_parent(child, Some(parent)).expect("parent");
    let b = dir.join("b.kreate");
    write_document(&b, &document, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(out.contains("reparented"), "{out}");
}

#[test]
fn diff_reports_additions_and_removals() {
    let dir = scratch("diff_addremove");
    let mut document = Document::new();
    document.add_object(Object::new("kept", ObjectData::Empty));
    document.add_object(Object::new("gone", ObjectData::Empty));
    let a = dir.join("a.kreate");
    write_document(&a, &document, &[]);

    let mut after = Document::new();
    after.add_object(Object::new("kept", ObjectData::Empty));
    after.add_object(Object::new("added", ObjectData::Empty));
    let b = dir.join("b.kreate");
    write_document(&b, &after, &[]);

    let out = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]).stdout;
    assert!(out.contains("- gone"), "{out}");
    assert!(out.contains("+ added"), "{out}");
    assert!(
        !out.contains("kept"),
        "an unchanged object was reported: {out}"
    );
}

#[test]
fn diff_works_across_formats() {
    // A .kreate against the .obj it was exported to. This is the question a
    // pipeline actually asks — "did the export lose anything" — and it only
    // works because both sides are read as documents.
    let dir = scratch("diff_formats");
    let native = dir.join("a.kreate");
    let obj = dir.join("a.obj");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    write_document(&native, &document, &[]);
    assert!(run(&["convert", native.to_str().unwrap(), obj.to_str().unwrap()]).ok);

    let out = run(&["diff", native.to_str().unwrap(), obj.to_str().unwrap()]).stdout;
    assert!(out.contains("no semantic difference"), "{out}");
}

#[test]
fn converting_native_to_native_keeps_an_unrecognised_entry() {
    // §5.2's requirement, at the level a user meets it: a converter that dropped
    // what it did not understand would be the tool that destroys a colleague's
    // file by opening and saving it.
    let dir = scratch("convert_unknown");
    let a = dir.join("a.kreate");
    let b = dir.join("b.kreate");
    let mut document = Document::new();
    document.add_object(Object::new("thing", ObjectData::Empty));
    write_document(
        &a,
        &document,
        &[UnknownEntry {
            name: "future/thing.json".to_string(),
            bytes: b"{\"from\":\"a newer build\"}".to_vec(),
        }],
    );

    let result = run(&["convert", a.to_str().unwrap(), b.to_str().unwrap()]);
    assert!(result.ok, "{}", result.stdout);
    assert!(
        result.stdout.contains("future/thing.json"),
        "the entry was kept silently: {}",
        result.stdout
    );

    let reopened = container::read(std::io::Cursor::new(std::fs::read(&b).expect("read")))
        .expect("the converted file should open");
    assert_eq!(reopened.unknown.len(), 1, "the entry did not survive");
    assert_eq!(reopened.unknown[0].name, "future/thing.json");
}

#[test]
fn converting_native_to_obj_says_what_it_leaves_behind() {
    // An empty has no geometry, so it cannot appear in OBJ. The user finds out
    // from the tool or from the next program to open the file; the tool is
    // better.
    let dir = scratch("convert_notes");
    let native = dir.join("a.kreate");
    let obj = dir.join("a.obj");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    document.add_object(Object::new("pivot", ObjectData::Empty));
    write_document(&native, &document, &[]);

    let out = run(&["convert", native.to_str().unwrap(), obj.to_str().unwrap()]).stdout;
    assert!(out.contains("carry no mesh"), "{out}");
}

#[test]
fn inspect_shows_the_hierarchy_of_a_native_file() {
    let dir = scratch("inspect_hierarchy");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    let parent = document.add_object(Object::new("parent", ObjectData::Empty));
    let mesh = document.add_mesh(cube());
    let child = document.add_object(Object::new("child", ObjectData::Mesh(mesh)));
    document.set_parent(child, Some(parent)).expect("parent");
    write_document(&path, &document, &[]);

    let out = run(&["inspect", path.to_str().unwrap()]).stdout;
    assert!(out.contains("parent [empty]"), "{out}");
    assert!(out.contains("child [mesh]"), "{out}");
    assert!(out.contains("Z up"), "{out}");
}

#[test]
fn validate_passes_a_sound_document_and_reports_its_meshes() {
    let dir = scratch("validate_ok");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    write_document(&path, &document, &[]);

    let result = run(&["validate", path.to_str().unwrap()]);
    assert!(result.ok, "{}", result.stdout);
    assert!(result.stdout.contains("cube: ok"), "{}", result.stdout);
}

#[test]
fn a_shared_mesh_datablock_is_listed_once() {
    // Two objects, one mesh. Reporting it twice would misstate what the file
    // holds and would make a linked duplicate look like a copy — which is the
    // whole point of the datablock indirection.
    let dir = scratch("shared_mesh");
    let path = dir.join("a.kreate");
    let mut document = Document::new();
    let mesh = document.add_mesh(cube());
    document.add_object(Object::new("first", ObjectData::Mesh(mesh)));
    document.add_object(Object::new("second", ObjectData::Mesh(mesh)));
    write_document(&path, &document, &[]);

    let out = run(&["inspect", path.to_str().unwrap()]).stdout;
    assert!(out.contains("1 mesh(es)"), "{out}");
    assert!(
        out.contains("first [mesh]") && out.contains("second [mesh]"),
        "{out}"
    );
}

#[test]
fn a_missing_file_fails_rather_than_printing_nothing() {
    let result = run(&["inspect", "no-such-file.kreate"]);
    assert!(!result.ok, "a missing file should be an error");
}

#[test]
fn a_command_with_the_wrong_argument_count_fails() {
    assert!(!run(&["diff", "only-one.kreate"]).ok);
    assert!(!run(&["textconv"]).ok);
}

#[test]
fn help_succeeds_and_documents_the_git_integration() {
    let result = run(&["--help"]);
    assert!(result.ok);
    // The commands are only useful if the user can find out how to hook them up,
    // and the two-line git recipe is the part nobody guesses.
    assert!(result.stdout.contains("textconv"), "{}", result.stdout);
    assert!(
        result.stdout.contains(".gitattributes"),
        "{}",
        result.stdout
    );
}
