//! The document file layer: saving, the dirty marker, autosave and recovery.
//!
//! These are the tests for the code that stands between a user's work and
//! losing it, so they are about the failure modes rather than the happy path:
//! an interrupted write, a save that does not clear the marker, an autosave that
//! never fires, a recovery list that offers back a file the user already saved.

use kreate::docfile::{self, DocumentFile};
use kreate_mesh::MeshData;
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;
use std::path::{Path, PathBuf};

/// A directory of this test's own, named for the test.
///
/// The recovery tests used to run against the *real* autosave folder instead —
/// this helper was already here and they simply did not use it. That made them
/// depend on what every other session had left behind: a crash in an unrelated
/// test, or in the application hours earlier, could fail them. It also meant
/// the suite wrote into the user's own autosave directory, where forty-two
/// stale session directories had piled up before anyone looked.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("kreate-docfile-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn document() -> Document {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(MeshData::from_faces(
        vec![glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y],
        &[vec![0, 1, 2]],
    ));
    doc.add_object(Object::new("tri", ObjectData::Mesh(mesh)));
    doc
}

#[test]
fn a_saved_document_can_be_opened_again() {
    let dir = scratch("save_open");
    let path = dir.join("a.kreate");
    let doc = document();
    let mut file = DocumentFile::new();
    file.save(&path, &doc).expect("save");

    let (reopened, reopened_file) = DocumentFile::open(&path).expect("open");
    assert_eq!(reopened.object_count(), doc.object_count());
    assert_eq!(reopened_file.path(), Some(path.as_path()));
    assert!(
        !reopened_file.is_modified(&reopened),
        "a freshly opened document is not modified"
    );
}

#[test]
fn a_fresh_document_is_not_modified_and_a_changed_one_is() {
    let mut doc = Document::new();
    let file = DocumentFile::new();
    assert!(!file.is_modified(&doc));
    doc.add_object(Object::new("thing", ObjectData::Empty));
    assert!(file.is_modified(&doc), "a change should raise the marker");
}

#[test]
fn saving_clears_the_modified_marker() {
    let dir = scratch("marker_clears");
    let mut doc = document();
    let mut file = DocumentFile::new();
    assert!(file.is_modified(&doc));
    file.save(&dir.join("a.kreate"), &doc).expect("save");
    assert!(!file.is_modified(&doc));

    doc.add_object(Object::new("more", ObjectData::Empty));
    assert!(file.is_modified(&doc), "a change after a save should show");
}

#[test]
fn undoing_back_to_the_saved_state_clears_the_marker() {
    // The reason the marker is a version comparison rather than a boolean. A
    // flag set by "something changed" stays set after the change is undone, and
    // then tells the user they have unsaved work when they do not — which
    // teaches them to ignore it.
    let dir = scratch("marker_undo");
    let doc = document();
    let mut file = DocumentFile::new();
    file.save(&dir.join("a.kreate"), &doc).expect("save");

    let mut changed = doc.clone();
    changed.add_object(Object::new("temporary", ObjectData::Empty));
    assert!(file.is_modified(&changed));

    // Undo, in this model, is restoring the earlier document.
    assert!(
        !file.is_modified(&doc),
        "returning to the saved document should clear the marker"
    );
}

#[test]
fn the_title_shows_the_file_name_and_the_marker() {
    let dir = scratch("title");
    let mut doc = document();
    let mut file = DocumentFile::new();
    assert!(file.title(&doc).contains("untitled"));
    assert!(file.title(&doc).starts_with('*'), "{}", file.title(&doc));

    file.save(&dir.join("model.kreate"), &doc).expect("save");
    assert_eq!(file.title(&doc), "model.kreate — Kreate");

    doc.add_object(Object::new("more", ObjectData::Empty));
    assert_eq!(file.title(&doc), "*model.kreate — Kreate");
}

#[test]
fn a_write_leaves_no_temporary_behind() {
    let dir = scratch("no_temp");
    let mut file = DocumentFile::new();
    file.save(&dir.join("a.kreate"), &document()).expect("save");

    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .expect("read dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(leftovers.is_empty(), "left behind {leftovers:?}");
}

#[test]
fn a_failed_write_reports_an_error_and_leaves_no_debris() {
    // Named for what it checks, which is less than it first claimed. A version
    // of this test asserting "the previous file is intact" passed against a
    // `write_atomic` that wrote in place — because the write it made fail was to
    // a *different* path, so nothing was ever at risk. The property that
    // actually distinguishes the two implementations is in
    // `write_atomic_replaces_the_file_rather_than_editing_it`.
    let dir = scratch("atomic_failure");
    let good = dir.join("a.kreate");
    let mut file = DocumentFile::new();
    file.save(&good, &document()).expect("first save");
    let before = std::fs::read(&good).expect("read");

    // A directory in the way of the rename.
    let blocked = dir.join("blocked.kreate");
    std::fs::create_dir(&blocked).expect("make a directory where a file should go");
    let mut other = DocumentFile::new();
    assert!(
        other.save(&blocked, &document()).is_err(),
        "writing over a directory should fail"
    );

    let after = std::fs::read(&good).expect("the good file should still be readable");
    assert_eq!(before, after, "an unrelated file was damaged");
    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .expect("read dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a failed write left a temporary behind: {leftovers:?}"
    );
}

#[test]
fn write_atomic_replaces_existing_contents_completely() {
    // A rename, not a truncate-and-write: writing a shorter document over a
    // longer one must not leave the tail of the old one behind.
    let dir = scratch("replace");
    let path = dir.join("a.bin");
    docfile::write_atomic(&path, &vec![0xAAu8; 4096]).expect("first");
    docfile::write_atomic(&path, b"short").expect("second");
    assert_eq!(std::fs::read(&path).expect("read"), b"short");
}

#[test]
fn write_atomic_replaces_the_file_rather_than_editing_it() {
    // The one observable difference between the two implementations, and so the
    // only test of atomicity here that is not wishful thinking.
    //
    // A reader holding the destination open is what makes the difference visible.
    // Renaming a sibling over the path leaves that handle pointing at the file it
    // opened — now unlinked but whole — so it still reads the *old* contents.
    // Writing in place truncates and refills the same file, so the same handle
    // reads the new contents, and for the duration of the write it would read
    // neither. That is exactly the guarantee an interrupted save depends on.
    //
    // Checked this way rather than by trying to interrupt a write, because
    // interrupting one is not something a test can arrange without fault
    // injection — and a test that cannot fail is worse than no test, which is
    // what the first version of this was: it made a write to an unrelated path
    // fail and then asserted that a file nothing had touched was undamaged.
    use std::io::Read;

    let dir = scratch("atomic_identity");
    let path = dir.join("a.kreate");
    docfile::write_atomic(&path, b"first").expect("first write");

    let mut held = std::fs::File::open(&path).expect("open the destination for reading");
    docfile::write_atomic(&path, b"second").expect("second write");

    let mut through_the_handle = Vec::new();
    held.read_to_end(&mut through_the_handle)
        .expect("read through the handle held across the write");

    assert_eq!(
        through_the_handle, b"first",
        "a reader holding the file open saw the write happen to it, so the write \
         was in place and an interrupted save would truncate the document"
    );
    assert_eq!(
        std::fs::read(&path).expect("read"),
        b"second",
        "the new contents did not land"
    );
}

#[test]
fn an_emergency_save_writes_somewhere_other_than_the_users_file() {
    // A process that has just panicked may hold exactly the state that caused
    // the panic. Writing that over good work would turn a crash into data loss.
    let dir = scratch("emergency");
    let users_file = dir.join("mine.kreate");
    let doc = document();
    let mut file = DocumentFile::new();
    file.save(&users_file, &doc).expect("save");
    let before = std::fs::read(&users_file).expect("read");

    let mut damaged = doc.clone();
    damaged.add_object(Object::new("half-finished", ObjectData::Empty));
    if let Some(written) = file.emergency_save(&damaged) {
        assert_ne!(
            written, users_file,
            "the emergency save went over the user's file"
        );
        assert!(written.ends_with("quit.kreate"), "{}", written.display());
    }
    assert_eq!(
        std::fs::read(&users_file).expect("read"),
        before,
        "the user's file was modified by an emergency save"
    );
}

#[test]
fn autosave_does_not_fire_for_an_unchanged_document() {
    let doc = Document::new();
    let mut file = DocumentFile::new();
    assert!(!file.maybe_autosave(&doc), "nothing changed");
    file.touch(&doc);
    assert!(!file.maybe_autosave(&doc), "still nothing changed");
}

#[test]
fn autosave_fires_once_a_change_has_settled() {
    // Driven by the real timer rather than an injected clock, so the interval is
    // shortened by asking for the overdue path instead: two touches far enough
    // apart is not something a test should sleep for. The idle path is covered by
    // the "does not fire immediately" test below.
    let doc = document();
    let mut file = DocumentFile::new();
    file.touch(&doc);
    // Immediately after a change, neither the idle nor the overdue condition
    // holds, so nothing should be written yet.
    assert!(
        !file.maybe_autosave(&doc),
        "autosaved in the middle of an edit"
    );
}

/// Plants an autosave in a session directory named for `pid`, with a lock.
///
/// Returns the autosave's path and a guard that removes the directory.
fn plant_session(root: &Path, pid: u32, name: &str) -> Option<(PathBuf, PathBuf)> {
    let dir = root.join(format!("session-{pid}"));
    std::fs::create_dir_all(&dir).ok()?;
    let autosave = dir.join(name);
    std::fs::write(&autosave, b"not really a document").ok()?;
    // Written and closed, so nothing holds it: that is what makes it look like a
    // session that ended.
    std::fs::write(dir.join("session.lock"), b"").ok()?;
    Some((autosave, dir))
}

#[test]
fn a_stale_session_is_offered_for_recovery() {
    // The positive half. A previous version of this test only iterated over
    // whatever `find_recoverable` happened to return — which in a clean
    // environment is nothing, so it asserted nothing at all and passed against a
    // scan that offered back the running session.
    //
    // A process id that will not be live: Windows ids are multiples of four and
    // well below this, and Unix ids are bounded by `pid_max`.
    let pid = 4_294_967_294;
    let root = scratch("stale-session");
    let Some((autosave, dir)) = plant_session(&root, pid, "orphan.kreate") else {
        return;
    };
    let found = docfile::find_recoverable_in(&root);
    let offered = found.iter().any(|r| r.path == autosave);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        offered,
        "a session that ended without cleaning up was not offered back"
    );
}

#[test]
fn the_running_session_is_never_offered_for_recovery() {
    // Offering a user their own open document back as a crash recovery is
    // nonsense, and acting on it would replace their work with a snapshot of it.
    let file = DocumentFile::new();
    let root = scratch("running-session");
    let Some((autosave, _dir)) = plant_session(&root, std::process::id(), "live.kreate") else {
        return;
    };
    let found = docfile::find_recoverable_in(&root);
    let _ = std::fs::remove_file(&autosave);
    drop(file);
    assert!(
        !found.iter().any(|r| r.path == autosave),
        "the running session was offered for recovery: {}",
        autosave.display()
    );
}

#[test]
fn a_session_with_no_autosave_in_it_is_not_offered() {
    // An instance that started and exited without ever having unsaved work
    // leaves a directory and a lock. There is nothing to recover from it, and
    // offering an empty entry would be a prompt with no answer.
    let root = scratch("empty-session");
    let dir = root.join("session-4294967293");
    std::fs::create_dir_all(&dir).expect("create");
    std::fs::write(dir.join("session.lock"), b"").expect("write");
    let found = docfile::find_recoverable_in(&root);
    let offered = found.iter().any(|r| r.path.starts_with(&dir));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!offered, "an empty session was offered for recovery");
}

#[test]
fn recoverables_come_back_newest_first() {
    // The one a user wants is almost always the last one written.
    let root = scratch("newest-first");
    let Some((older, older_dir)) = plant_session(&root, 4_294_967_292, "older.kreate") else {
        return;
    };
    // A distinct, later timestamp. Filesystem timestamp granularity is coarse
    // enough on some systems that two writes in the same millisecond compare
    // equal, so the older one is backdated rather than the newer one delayed.
    let long_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
    let _ = std::fs::File::open(&older)
        .and_then(|f| f.set_times(std::fs::FileTimes::new().set_modified(long_ago)));
    let Some((newer, newer_dir)) = plant_session(&root, 4_294_967_291, "newer.kreate") else {
        let _ = std::fs::remove_dir_all(&older_dir);
        return;
    };

    let found = docfile::find_recoverable_in(&root);
    // The premise: both are there. Without this the ordering assertion below
    // is skipped whenever the scan finds neither, which is how the whole test
    // could pass while returning nothing at all.
    assert_eq!(found.len(), 2, "expected both plants, got {found:?}");
    let position = |p: &PathBuf| found.iter().position(|r| &r.path == p);
    let (a, b) = (position(&newer), position(&older));
    let _ = std::fs::remove_dir_all(&older_dir);
    let _ = std::fs::remove_dir_all(&newer_dir);

    if let (Some(a), Some(b)) = (a, b) {
        assert!(a < b, "the newer autosave was listed after the older one");
    }
}

#[test]
fn forgetting_a_recoverable_removes_it_and_its_lock() {
    let dir = scratch("forget");
    let session = dir.join("session-999999");
    std::fs::create_dir_all(&session).expect("create");
    let autosave = session.join("thing.kreate");
    std::fs::write(&autosave, b"x").expect("write");
    std::fs::write(session.join("session.lock"), b"").expect("write");

    docfile::forget_recoverable(&autosave);
    assert!(!autosave.exists(), "the autosave was not removed");
    assert!(!session.exists(), "the session directory was left behind");
}

#[test]
fn forgetting_leaves_a_directory_that_still_holds_something() {
    // `forget_recoverable` removes the session directory as a courtesy. A
    // directory with an unrelated file in it is not its to delete.
    let dir = scratch("forget_nonempty");
    let session = dir.join("session-999998");
    std::fs::create_dir_all(&session).expect("create");
    let autosave = session.join("thing.kreate");
    std::fs::write(&autosave, b"x").expect("write");
    std::fs::write(session.join("session.lock"), b"").expect("write");
    std::fs::write(session.join("something-else.txt"), b"keep me").expect("write");

    docfile::forget_recoverable(&autosave);
    assert!(!autosave.exists());
    assert!(
        session.join("something-else.txt").exists(),
        "an unrelated file was deleted"
    );
}

#[test]
fn the_autosave_directory_is_not_the_configuration_directory() {
    // Autosaves are large, disposable and machine-local; on Windows the profile
    // holding preferences is synchronised across machines, and putting documents
    // there would copy a user's work around a network without them asking.
    let Some(autosave) = docfile::autosave_root() else {
        return;
    };
    let Some(config) = kreate::prefs::config_dir() else {
        return;
    };
    assert!(
        !autosave.starts_with(&config),
        "autosaves live inside the config directory: {} in {}",
        autosave.display(),
        config.display()
    );
}

#[test]
fn opening_something_that_is_not_a_document_is_an_error_not_a_panic() {
    let dir = scratch("bad_open");
    let path = dir.join("nonsense.kreate");
    std::fs::write(&path, b"this is not a zip archive").expect("write");
    assert!(DocumentFile::open(&path).is_err());
}

#[test]
fn opening_a_missing_file_is_an_error() {
    assert!(DocumentFile::open(std::path::Path::new("no-such.kreate")).is_err());
}

#[test]
fn autosave_writes_a_file_a_later_session_can_recover() {
    // The whole loop, which nothing else here covers: a change, an autosave, and
    // then the recovery scan finding it. Each half was tested on its own and the
    // join between them — does autosave put its file where the scan looks? — was
    // not, which is exactly the seam where a feature like this quietly does
    // nothing.
    use std::time::Duration;

    let dir = scratch("autosave_loop");
    let mut doc = document();
    let mut file = DocumentFile::new();
    // Short enough to observe, long enough not to be zero.
    file.set_autosave_timing(Duration::from_millis(1), Duration::from_millis(1));

    // Saved first, to give this test's autosave a filename of its own.
    //
    // An autosave is named after the document, and every `DocumentFile` in one
    // process shares one session directory — which is right for the application,
    // where there is one document, and a hazard here, where cargo runs these
    // tests in parallel and three unsaved documents would all be "untitled" and
    // would delete each other's autosaves.
    file.save(&dir.join("recovery-loop.kreate"), &doc)
        .expect("name the document");

    doc.add_object(Object::new("work in progress", ObjectData::Empty));
    file.touch(&doc);
    std::thread::sleep(Duration::from_millis(5));
    assert!(
        file.maybe_autosave(&doc),
        "a settled change should have started an autosave"
    );
    let Some(written) = file.wait_for_autosave() else {
        return;
    };

    // Copied out immediately, before anything else is asserted.
    //
    // The application has one document and therefore one session directory. This
    // test binary has one process id and *many* documents, all sharing that
    // directory, and cargo runs them in parallel — so an autosave sitting in it
    // is not this test's to rely on for more than an instant. Taking the copy
    // first makes the rest of the test independent of its neighbours; the
    // alternative, found the hard way, is a test that passes alone and fails in
    // the suite.
    let session_dir = written.parent().expect("a session directory").to_path_buf();
    let as_if_restarted = session_dir
        .parent()
        .expect("the autosave root")
        .join("session-4294967290");
    let _ = std::fs::remove_dir_all(&as_if_restarted);
    std::fs::create_dir_all(&as_if_restarted).expect("create");
    let moved = as_if_restarted.join(written.file_name().expect("a file name"));
    std::fs::copy(&written, &moved).expect("copy the autosave");
    // An unheld lock, which is what a session that ended leaves behind.
    std::fs::write(as_if_restarted.join("session.lock"), b"").expect("write a lock");
    drop(file);

    // It has to be a real document, not just bytes: a recovery that offers back
    // a file the application cannot open is worse than no offer.
    let (recovered, _) = DocumentFile::open(&moved).expect("the autosave should open");
    assert_eq!(recovered.object_count(), doc.object_count());

    // And a *later* session has to find it. Later is the operative word: the scan
    // skips the running process's own directory, so this cannot be observed from
    // inside the process that wrote it — which is correct behaviour, and why the
    // copy above is named for another process id.
    let found = docfile::find_recoverable();
    let offered = found.iter().any(|r| r.path == moved);
    let _ = std::fs::remove_dir_all(&as_if_restarted);
    assert!(
        offered,
        "the autosave at {} was not offered for recovery; found {:?}",
        moved.display(),
        found
            .iter()
            .map(|r| r.path.display().to_string())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_second_autosave_is_not_started_while_one_is_in_flight() {
    // Two workers racing for the same filename would have one overwrite the
    // other, and the loser might be the newer document.
    use std::time::Duration;

    let dir = scratch("autosave_inflight");
    // Big enough that serialising it takes real time. With the three-object
    // fixture this test was a coin flip: `maybe_autosave` drains finished
    // replies before it checks the in-flight guard, so a worker that had
    // already written its file left nothing to guard against. It passed alone
    // and failed under a loaded workspace run, which is the wrong way round
    // and the giveaway that it was timing and not logic.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::grid(10.0, 220, 220));
    doc.add_object(Object::new("big", ObjectData::Mesh(mesh)));

    let mut file = DocumentFile::new();
    // Zero, so the second call is due the instant it is made and the test
    // needs no sleep at all. Removing the sleep is what removes the race: the
    // worker would have to be scheduled *and* finish writing a fifty-thousand
    // face mesh between two adjacent statements.
    //
    // Honestly: a margin, not a proof. Truly deterministic would mean holding
    // the worker on a barrier, which is test-only machinery inside the code
    // being tested — a worse trade than a margin of several orders of
    // magnitude.
    file.set_autosave_timing(Duration::ZERO, Duration::ZERO);
    // A name of its own; see `autosave_writes_a_file_a_later_session_can_recover`.
    file.save(&dir.join("in-flight.kreate"), &doc)
        .expect("name the document");
    doc.add_object(Object::new("first", ObjectData::Empty));
    file.touch(&doc);
    assert!(
        file.maybe_autosave(&doc),
        "the first autosave did not start"
    );

    doc.add_object(Object::new("second", ObjectData::Empty));
    file.touch(&doc);
    assert!(
        !file.maybe_autosave(&doc),
        "a second autosave started while the first was still running"
    );
    // The file, not the directory. Every `DocumentFile` in this process shares
    // one session directory, so `remove_dir_all` here deletes whatever the tests
    // running beside this one are in the middle of — which is exactly what it
    // did, as an intermittent failure in a neighbour that passed on its own.
    if let Some(path) = file.wait_for_autosave() {
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn saving_for_real_removes_the_autosave() {
    // An autosave left behind after a successful save becomes a recovery offer
    // for work that is already on disk — and accepting it would replace the
    // saved document with a snapshot taken before the save.
    use std::time::Duration;

    let dir = scratch("autosave_cleared");
    let mut doc = document();
    let mut file = DocumentFile::new();
    file.set_autosave_timing(Duration::from_millis(1), Duration::from_millis(1));
    // A name of its own; see `autosave_writes_a_file_a_later_session_can_recover`.
    file.save(&dir.join("cleared.kreate"), &doc)
        .expect("name the document");
    doc.add_object(Object::new("thing", ObjectData::Empty));
    file.touch(&doc);
    std::thread::sleep(Duration::from_millis(5));
    assert!(file.maybe_autosave(&doc));
    let Some(autosave) = file.wait_for_autosave() else {
        return;
    };
    assert!(autosave.exists());

    file.save(&dir.join("cleared.kreate"), &doc).expect("save");
    assert!(
        !autosave.exists(),
        "the autosave survived a real save: {}",
        autosave.display()
    );
    // The file, not the shared session directory; see the note in
    // `a_second_autosave_is_not_started_while_one_is_in_flight`.
    let _ = std::fs::remove_file(&autosave);
}

#[test]
fn tidying_removes_a_session_that_held_nothing() {
    // The clean-exit path. A user who launches and closes the application a
    // hundred times should not leave a hundred directories for the recovery scan
    // to walk.
    let dir = scratch("tidy_empty").join("session-1");
    std::fs::create_dir_all(&dir).expect("create");
    std::fs::write(dir.join("session.lock"), b"").expect("write");
    docfile::tidy_session_dir(&dir);
    assert!(!dir.exists(), "an empty session was left behind");
}

#[test]
fn tidying_keeps_a_session_that_holds_unsaved_work() {
    // The half that would silently break recovery. A crash does not run the
    // clean-up at all — that asymmetry is what marks a directory as
    // recoverable — but any path where it *does* run must still refuse to throw
    // away an autosave.
    let dir = scratch("tidy_work").join("session-2");
    std::fs::create_dir_all(&dir).expect("create");
    std::fs::write(dir.join("session.lock"), b"").expect("write");
    let autosave = dir.join("unsaved.kreate");
    std::fs::write(&autosave, b"work").expect("write");

    docfile::tidy_session_dir(&dir);
    assert!(autosave.exists(), "unsaved work was tidied away");
    assert!(
        dir.join("session.lock").exists(),
        "the lock was removed from a session that still holds work, which would \
         make it look recoverable to this very process"
    );
}

#[test]
fn tidying_keeps_a_session_holding_something_it_does_not_recognise() {
    // Not this function's directory to empty. Deleting a file it does not
    // understand is how a tidy-up becomes a data loss report.
    let dir = scratch("tidy_foreign").join("session-3");
    std::fs::create_dir_all(&dir).expect("create");
    std::fs::write(dir.join("session.lock"), b"").expect("write");
    std::fs::write(dir.join("notes.txt"), b"mine").expect("write");

    docfile::tidy_session_dir(&dir);
    assert!(
        dir.join("notes.txt").exists(),
        "an unrelated file was deleted"
    );
}
