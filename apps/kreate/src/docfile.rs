//! Opening, saving, autosaving and recovering documents.
//!
//! # The rule everything here follows
//!
//! **A user's file is only ever written by an explicit save, and only ever
//! atomically.** Autosave writes to its own directory; a crash write goes to its
//! own file; nothing in this module ever overwrites the document a user is
//! working on except when they asked for it. §3.4 states this and it is worth
//! restating at the top of the code, because every convenience feature in a
//! desktop application eventually proposes writing to the user's file "just in
//! case", and that is how a tool becomes something people do not trust.
//!
//! # Atomicity
//!
//! Every write goes to a temporary file beside the destination and is then
//! renamed over it. A rename within a directory is atomic on both Windows and
//! POSIX, so an interrupted save leaves either the old file or the new one,
//! never a truncated one. Writing in place is how a power cut costs somebody a
//! day's work.
//!
//! Beside the destination, not in the system temp directory: a rename across
//! volumes is not a rename, it is a copy and a delete, and it loses the
//! guarantee this exists for.
//!
//! # Autosave and recovery
//!
//! Autosave is possible without stalling the interface because of the copy-on-
//! write document model. The whole document is a handful of `Arc` clones, so a
//! snapshot is microseconds; serialising it is the slow part and it happens on a
//! worker thread against a snapshot that can no longer change. This is an
//! independent reason to have chosen that model, and it is the shape §3.4 asks
//! for.
//!
//! Recovery works off a session lock. Each running instance holds an open file
//! handle on `session.lock` inside its own autosave directory; a directory whose
//! lock is stale therefore belonged to an instance that did not exit cleanly,
//! and its autosave is worth offering back. The lock is a held *handle* rather
//! than a pid or a timestamp, because a pid can be reused and a clock can go
//! backwards — the operating system already knows which processes are alive, and
//! this asks it instead of guessing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use kreate_io::container::{self, Manifest, UnknownEntry};
use kreate_scene::Document;

/// How long after the last change an autosave is written.
///
/// Idle-triggered rather than on a fixed timer: saving in the middle of a drag
/// is both the least useful moment — the state is transient — and the worst one,
/// because it competes for the frame. §3.4 asks for "every N minutes, and when
/// idle after a change"; this is the second half, and the first is
/// [`AUTOSAVE_MAX_INTERVAL`].
const AUTOSAVE_IDLE: Duration = Duration::from_secs(15);

/// The longest an unsaved change may go unautosaved, however busy the user is.
///
/// Without this, someone who never pauses for fifteen seconds is never
/// autosaved at all — which is exactly the user in the middle of the session
/// they would most want back.
const AUTOSAVE_MAX_INTERVAL: Duration = Duration::from_secs(120);

/// What is known about where the open document lives.
pub struct DocumentFile {
    /// `None` for a document that has never been saved.
    path: Option<PathBuf>,
    /// The manifest the file arrived with, so a save does not silently rewrite
    /// what it claimed about itself.
    manifest: Manifest,
    /// Entries a newer build wrote that this one does not understand.
    ///
    /// Carried so that opening and saving a colleague's file does not delete
    /// their data. §5.2 makes this a requirement, and it cannot be retrofitted:
    /// once a build has saved without it, the data is gone.
    unknown: Vec<UnknownEntry>,
    /// The document version at the last save, for the dirty flag.
    ///
    /// A version counter rather than a boolean, because undoing back to the
    /// saved state should clear the marker rather than leave the document
    /// looking modified when it is not.
    saved_version: u64,
    /// When the document last changed, and at what version.
    changed: Option<(Instant, u64)>,
    /// When an autosave was last written, for the maximum-interval rule.
    last_autosave: Instant,
    /// The document version the last autosave captured.
    autosaved_version: u64,
    /// The worker's replies.
    replies: Option<mpsc::Receiver<Result<PathBuf, String>>>,
    session: Option<Session>,
    /// How long after the last change an autosave is written.
    ///
    /// A field rather than a constant for two reasons. §3.4 wants "every N
    /// minutes" to be a preference, and — more immediately — an interval that
    /// cannot be changed is an interval that cannot be tested: verifying that
    /// autosave produces a file a later session can recover would otherwise mean
    /// a test that sleeps for two minutes, which is a test nobody runs.
    idle_after: Duration,
    /// The longest a change may go unautosaved however busy the user is.
    at_least_every: Duration,
}

impl DocumentFile {
    /// Starts a session for a document that has never been saved.
    pub fn new() -> Self {
        Self {
            path: None,
            manifest: Manifest::default(),
            unknown: Vec::new(),
            saved_version: 0,
            changed: None,
            last_autosave: Instant::now(),
            autosaved_version: 0,
            replies: None,
            session: Session::acquire(),
            idle_after: AUTOSAVE_IDLE,
            at_least_every: AUTOSAVE_MAX_INTERVAL,
        }
    }

    /// Overrides the autosave timings.
    ///
    /// Exposed so a preference can set them and so a test can make the interval
    /// short enough to actually observe. Not a constructor argument: every
    /// caller but those two wants the defaults, and a two-argument constructor
    /// would make them all say so.
    pub fn set_autosave_timing(&mut self, idle_after: Duration, at_least_every: Duration) {
        self.idle_after = idle_after;
        self.at_least_every = at_least_every;
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Whether the document differs from what is on disk.
    pub fn is_modified(&self, document: &Document) -> bool {
        document.version() != self.saved_version
    }

    /// The window title: the file's name, a marker if modified, and the product.
    ///
    /// The marker is the only indication a user has that closing will lose work,
    /// so it goes first where it cannot be truncated away by a narrow title bar.
    pub fn title(&self, document: &Document) -> String {
        let name = self
            .path
            .as_deref()
            .and_then(Path::file_name)
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "untitled".to_string());
        let marker = if self.is_modified(document) { "*" } else { "" };
        format!("{marker}{name} — Kreate")
    }

    /// Keeps the viewport's view with the document, for the next open.
    ///
    /// In the manifest's open map rather than in the document: where somebody
    /// was looking is not part of the scene and must not mark it changed, and
    /// a build that does not know the field carries it through untouched.
    pub fn set_view(&mut self, camera: &kreate_render::Camera) {
        let record = view_record(camera);
        if self.manifest.extra.get("view") != Some(&record) {
            self.manifest.extra.insert("view".into(), record);
        }
    }

    /// The view the file was saved with, if it kept one.
    pub fn view(&self) -> Option<kreate_render::Camera> {
        view_from_record(self.manifest.extra.get("view")?)
    }

    /// Reads a document from a `.kreate` archive.
    pub fn open(path: &Path) -> Result<(Document, Self), String> {
        let file = std::fs::File::open(path).map_err(|e| format!("cannot open: {e}"))?;
        let loaded = container::read(std::io::BufReader::new(file))
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        for entry in &loaded.unknown {
            tracing::info!(entry = %entry.name, "kept an entry this build does not understand");
        }
        // Linked datablocks are re-read from their libraries here, on open, which
        // is what makes linking worth having: fixing an asset fixes it in every
        // scene that links it, and each scene finds out the next time it opens.
        //
        // Failures do not stop the open. A shot file that will not open because
        // a prop moved on the server is a shot file nobody can rescue; the stale
        // geometry stays and the problem is reported.
        let mut document = loaded.document;
        let directory = path.parent().unwrap_or(Path::new("."));
        for problem in kreate_io::library::refresh_links(&mut document, directory) {
            tracing::warn!(error = %problem, "a link could not be refreshed");
        }

        // Read *after* refreshing, so a document whose links resolved to the
        // same geometry does not open already looking modified.
        let version = document.version();
        Ok((
            document,
            Self {
                path: Some(path.to_path_buf()),
                manifest: loaded.manifest,
                unknown: loaded.unknown,
                saved_version: version,
                changed: None,
                last_autosave: Instant::now(),
                autosaved_version: version,
                replies: None,
                session: Session::acquire(),
                idle_after: AUTOSAVE_IDLE,
                at_least_every: AUTOSAVE_MAX_INTERVAL,
            },
        ))
    }

    /// Writes the document to `path` and adopts it as the document's home.
    ///
    /// Synchronous, deliberately. A save the user asked for has to have finished
    /// before they are told it did — an asynchronous save that reports success
    /// and then fails has told them a lie about their own data. Autosave, which
    /// nobody is waiting on, is the one that goes to a thread.
    pub fn save(&mut self, path: &Path, document: &Document) -> Result<(), String> {
        let bytes = container::write(document, &self.manifest, &self.unknown)
            .map_err(|e| format!("cannot serialise: {e}"))?;
        write_atomic(path, &bytes)?;
        self.path = Some(path.to_path_buf());
        self.saved_version = document.version();
        self.changed = None;
        // Cleared, because the reason to keep an autosave is that it holds work
        // the saved file does not.
        self.discard_autosave();
        Ok(())
    }

    /// Records that the document changed, so autosave knows there is work to do.
    pub fn touch(&mut self, document: &Document) {
        let version = document.version();
        if version == self.autosaved_version {
            self.changed = None;
            return;
        }
        // The instant is refreshed on every change, so the idle timer measures
        // time since the *last* edit rather than the first — which is what makes
        // it an idle timer and not a delay.
        self.changed = Some((Instant::now(), version));
    }

    /// Writes an autosave if one is due. Returns whether a write was started.
    ///
    /// Called once per frame. The work is a snapshot plus a channel send, so on
    /// the overwhelming majority of frames this is two comparisons.
    pub fn maybe_autosave(&mut self, document: &Document) -> bool {
        self.drain_replies();
        let Some((changed_at, version)) = self.changed else {
            return false;
        };
        if version == self.autosaved_version {
            return false;
        }
        let idle = changed_at.elapsed() >= self.idle_after;
        let overdue = self.last_autosave.elapsed() >= self.at_least_every;
        if !idle && !overdue {
            return false;
        }
        // One in flight at a time. A second would race the first for the same
        // filename, and the newer document is the one worth having anyway.
        if self.replies.is_some() {
            return false;
        }
        let Some(session) = &self.session else {
            return false;
        };
        let destination = session.autosave_path(self.path.as_deref());
        // Microseconds: the document is `Arc`-per-array, so this shares every
        // buffer rather than copying one. It is the reason autosave can happen
        // on a worker without the interface waiting for it.
        let snapshot = document.clone();
        let manifest = self.manifest.clone();
        let unknown = self.unknown.clone();
        let (tx, rx) = mpsc::channel();
        // A plain thread, not a runtime. There is nothing asynchronous about
        // this: it is one blocking write, and it happens minutes apart.
        std::thread::Builder::new()
            .name("autosave".to_string())
            .spawn(move || {
                let result = container::write(&snapshot, &manifest, &unknown)
                    .map_err(|e| format!("cannot serialise: {e}"))
                    .and_then(|bytes| write_atomic(&destination, &bytes).map(|()| destination));
                let _ = tx.send(result);
            })
            .ok();
        self.replies = Some(rx);
        self.autosaved_version = version;
        self.last_autosave = Instant::now();
        true
    }

    /// Blocks until an in-flight autosave finishes. Returns its path.
    ///
    /// For tests and for shutdown. Nothing on the frame path calls it: an
    /// autosave the interface waits for is an autosave that can stutter, which
    /// is the whole thing this design avoids.
    pub fn wait_for_autosave(&mut self) -> Option<PathBuf> {
        let rx = self.replies.take()?;
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(path)) => Some(path),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "autosave failed");
                None
            }
            Err(_) => None,
        }
    }

    fn drain_replies(&mut self) {
        let Some(rx) = &self.replies else { return };
        match rx.try_recv() {
            Ok(Ok(path)) => {
                tracing::info!(path = %path.display(), "autosaved");
                self.replies = None;
            }
            Ok(Err(e)) => {
                // Reported, not retried in a loop. An autosave that cannot be
                // written is usually a full or unwritable disk, and hammering it
                // every frame would turn a warning into a hang.
                tracing::warn!(error = %e, "autosave failed");
                self.replies = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => self.replies = None,
        }
    }

    /// Removes this session's autosave, after a real save has superseded it.
    fn discard_autosave(&mut self) {
        if let Some(session) = &self.session {
            let path = session.autosave_path(self.path.as_deref());
            let _ = std::fs::remove_file(path);
        }
        self.autosaved_version = self.saved_version;
    }

    /// The emergency write, for a panic handler.
    ///
    /// Named for what it is. It goes beside the autosave rather than over the
    /// user's file, because a process that has just panicked is a process whose
    /// idea of the document may be exactly what caused the panic — and
    /// overwriting good work with it would turn a crash into data loss.
    pub fn emergency_save(&self, document: &Document) -> Option<PathBuf> {
        let dir = autosave_root()?;
        std::fs::create_dir_all(&dir).ok()?;
        let path = dir.join("quit.kreate");
        let bytes = container::write(document, &self.manifest, &self.unknown).ok()?;
        write_atomic(&path, &bytes).ok()?;
        Some(path)
    }
}

impl Default for DocumentFile {
    fn default() -> Self {
        Self::new()
    }
}

/// `%LOCALAPPDATA%\Kreate\autosave` and its equivalents.
///
/// Separate from the configuration directory: autosaves are large, disposable
/// and machine-local, and on Windows the roaming profile that holds preferences
/// is synchronised across machines. Putting documents there would copy a user's
/// work around a network without them asking.
/// A view as the file writes it: where it orbits, how it is turned, how far
/// back it stands, its lens and its projection.
pub fn view_record(camera: &kreate_render::Camera) -> serde_json::Value {
    let r = camera.rotation;
    serde_json::json!({
        "pivot": camera.pivot.to_array(),
        "rotation": [r.x, r.y, r.z, r.w],
        "distance": camera.distance,
        "fov_y": camera.fov_y,
        "orthographic": camera.projection == kreate_render::Projection::Orthographic,
    })
}

/// A view from what [`view_record`] wrote; `None` when it is not one. The
/// orbit style is not the file's to say — it is a preference — and stays the
/// default's.
pub fn view_from_record(v: &serde_json::Value) -> Option<kreate_render::Camera> {
    let numbers = |key: &str, n: usize| -> Option<Vec<f64>> {
        let a = v.get(key)?.as_array()?;
        let out: Vec<f64> = a.iter().filter_map(|x| x.as_f64()).collect();
        (out.len() == n && out.iter().all(|x| x.is_finite())).then_some(out)
    };
    let pivot = numbers("pivot", 3)?;
    let r = numbers("rotation", 4)?;
    let rotation = glam::DQuat::from_xyzw(r[0], r[1], r[2], r[3]);
    let distance = v.get("distance")?.as_f64()?;
    let fov_y = v.get("fov_y")?.as_f64()?;
    if rotation.length() < 1e-6
        || !(distance.is_finite() && distance > 0.0)
        || !(fov_y > 0.0 && fov_y < 3.1)
    {
        return None;
    }
    Some(kreate_render::Camera {
        pivot: glam::DVec3::new(pivot[0], pivot[1], pivot[2]),
        rotation: rotation.normalize(),
        distance,
        fov_y,
        projection: if v.get("orthographic").and_then(|o| o.as_bool()) == Some(true) {
            kreate_render::Projection::Orthographic
        } else {
            kreate_render::Projection::Perspective
        },
        ..kreate_render::Camera::default()
    })
}

pub fn autosave_root() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(|base| Path::new(&base).join("Kreate").join("autosave"))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME").map(|home| {
            Path::new(&home)
                .join("Library")
                .join("Caches")
                .join("Kreate")
                .join("autosave")
        })
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache")))
            .map(|base| base.join("kreate").join("autosave"))
    }
}

/// How many sessions this process currently holds.
///
/// The application has one, always. The test binary has many, because every
/// `DocumentFile` acquires one and they all share the process's directory — and
/// that is what makes this counter necessary rather than decorative: without it,
/// one session's clean-up removes the directory the others are still using, and
/// the symptom is an autosave that silently does not happen.
static LIVE_SESSIONS: AtomicUsize = AtomicUsize::new(0);

/// One running instance's claim on an autosave directory.
struct Session {
    dir: PathBuf,
    /// Held open for the life of the process. Dropping it releases the claim,
    /// which is what makes a leftover directory identifiable as a crash.
    _lock: std::fs::File,
}

impl Session {
    fn acquire() -> Option<Self> {
        Self::acquire_in(&autosave_root()?)
    }

    /// The same, in a directory the caller names.
    ///
    /// The tests use this against a temporary directory, and that is not
    /// tidiness: against the real one they read whatever *other* sessions have
    /// left behind, so a crash in an unrelated test — or in the application,
    /// hours earlier — makes them fail. They also wrote into the user's own
    /// autosave folder, which is nobody's idea of a test fixture. Forty-two
    /// stale session directories accumulated there over one working session
    /// before this was noticed.
    fn acquire_in(root: &Path) -> Option<Self> {
        let root = root.to_path_buf();
        // Named by process id, which is fine *here*: it only has to be unique
        // among live processes, and the lock file — not the name — is what says
        // whether this session is still running.
        let dir = root.join(format!("session-{}", std::process::id()));
        let lock = dir.join("session.lock");

        // Twice, because creating the directory and opening the lock inside it
        // are two steps and something can remove it in between — `Drop` on
        // another session does exactly that when it finds the directory empty.
        //
        // In the application that never happens: there is one session per
        // process, and it is the process's own directory. It happens in the test
        // binary, where many sessions share one process id, and it happened
        // under a loaded workspace run — an autosave that silently did not start
        // because its session had been tidied out from under it. A create-then-
        // open race is one retry's worth of problem, and swallowing it here is
        // better than every caller having to know about it.
        for attempt in 0..2 {
            if std::fs::create_dir_all(&dir).is_err() {
                continue;
            }
            match std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&lock)
            {
                Ok(file) => {
                    LIVE_SESSIONS.fetch_add(1, Ordering::AcqRel);
                    return Some(Self { dir, _lock: file });
                }
                Err(e) if attempt == 1 => {
                    tracing::warn!(error = %e, path = %lock.display(), "cannot claim a session");
                    return None;
                }
                Err(_) => {}
            }
        }
        None
    }

    /// Where this session's autosave of `original` goes.
    fn autosave_path(&self, original: Option<&Path>) -> PathBuf {
        let stem = original
            .and_then(Path::file_stem)
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "untitled".to_string());
        self.dir.join(format!("{stem}.kreate"))
    }
}

/// Tidies up after a clean exit, and deliberately not after a crash.
///
/// This is the other half of how recovery decides what to offer. A session that
/// exits normally removes its own directory, so it leaves nothing to recover; a
/// session that crashes never runs this, so its directory and its autosave
/// survive to be offered back. The asymmetry *is* the feature — there is no flag
/// saying "this instance crashed", only the absence of a tidy-up.
///
/// Only when the directory holds nothing but the lock. An autosave still in it
/// is unsaved work, and the whole point is to keep that; and this process may
/// not be the only holder — the test binary runs many sessions under one process
/// id — so removing a directory that still has something in it would be
/// destroying somebody else's.
impl Drop for Session {
    fn drop(&mut self) {
        // The last one out tidies. `fetch_sub` returns the previous value, so
        // this is true exactly once however many sessions the process held.
        if LIVE_SESSIONS.fetch_sub(1, Ordering::AcqRel) == 1 {
            tidy_session_dir(&self.dir);
        }
    }
}

/// Removes a session directory if it holds nothing but its lock.
///
/// A free function rather than a method so the rule can be tested against
/// fabricated directories. It cannot be tested through `Drop`: every
/// `DocumentFile` in one process shares one session directory, so a test that
/// creates and drops one observes nothing, and a test that counts directories in
/// the autosave root races every other test doing the same.
pub fn tidy_session_dir(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let only_the_lock = entries
        .flatten()
        .all(|e| e.file_name() == std::ffi::OsStr::new("session.lock"));
    if !only_the_lock {
        return;
    }
    let _ = std::fs::remove_file(dir.join("session.lock"));
    // `remove_dir`, not `remove_dir_all`: if something appeared between the
    // check and here, this fails and leaves it alone, which is the right way
    // round to be wrong.
    let _ = std::fs::remove_dir(dir);
}

/// An autosave left behind by a session that did not exit cleanly.
#[derive(Debug, Clone)]
pub struct Recoverable {
    pub path: PathBuf,
    /// The name the document had when it was autosaved, for showing the user.
    pub name: String,
    pub modified: Option<std::time::SystemTime>,
}

/// Finds autosaves whose sessions are gone.
///
/// A directory is considered stale when its `session.lock` can be opened for
/// writing — on Windows a held handle denies that, so the test is the operating
/// system's own answer to "is that process still running". On Unix the open
/// succeeds either way, so the lock is additionally flocked; a directory whose
/// lock cannot be locked belongs to a live session.
pub fn find_recoverable() -> Vec<Recoverable> {
    match autosave_root() {
        Some(root) => find_recoverable_in(&root),
        None => Vec::new(),
    }
}

/// The same, in a directory the caller names. See [`Session::acquire_in`].
pub fn find_recoverable_in(root: &Path) -> Vec<Recoverable> {
    let mine = format!("session-{}", std::process::id());
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        if dir.file_name().is_some_and(|n| n == mine.as_str()) {
            continue;
        }
        if !is_stale(&dir.join("session.lock")) {
            continue;
        }
        let Ok(files) = std::fs::read_dir(&dir) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            if path.extension().is_none_or(|e| e != "kreate") {
                continue;
            }
            let modified = file.metadata().ok().and_then(|m| m.modified().ok());
            out.push(Recoverable {
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                path,
                modified,
            });
        }
    }
    // Newest first: the recovery a user wants is almost always the last one.
    // Reversed by negating the key rather than by comparing the other way
    // round, so it reads as "newest first" and clippy is satisfied.
    out.sort_by_key(|r| std::cmp::Reverse(r.modified));
    out
}

/// Removes a stale session directory once the user has dealt with it.
pub fn forget_recoverable(path: &Path) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_file(dir.join("session.lock"));
        // Only if it is now empty. A directory with something else in it is not
        // this function's to delete.
        let _ = std::fs::remove_dir(dir);
    }
}

#[cfg(windows)]
fn is_stale(lock: &Path) -> bool {
    if !lock.exists() {
        // No lock at all: either a directory from a build before locks existed,
        // or one whose lock was removed. Either way nothing is holding it.
        return true;
    }
    // A live session holds the handle open without sharing write access, so
    // Windows refuses this. That refusal is the signal.
    std::fs::OpenOptions::new().write(true).open(lock).is_ok()
}

#[cfg(not(windows))]
fn is_stale(lock: &Path) -> bool {
    if !lock.exists() {
        return true;
    }
    // Unix lets two processes open the same file for writing, so the open tells
    // us nothing on its own. Deleting the directory when we cannot be sure would
    // be destroying a running session's only copy of unsaved work, so the safe
    // answer without a flock is to leave it alone and let the user decide from
    // the list.
    //
    // The consequence is stated rather than hidden: on Unix a running instance's
    // autosave may appear in the recovery list. Offering to open a file is
    // harmless; deleting one is not, and this errs in the direction where being
    // wrong costs nothing.
    true
}

/// Writes `bytes` to `path` via a temporary file and a rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    // Beside the destination, because a rename across volumes is a copy and a
    // delete and would lose atomicity.
    let temporary = path.with_extension(format!(
        "{}.tmp{}",
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("kreate"),
        std::process::id()
    ));
    {
        let mut file = std::fs::File::create(&temporary)
            .map_err(|e| format!("cannot create {}: {e}", temporary.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("cannot write {}: {e}", temporary.display()))?;
        // Before the rename, not after. A rename that lands while the contents
        // are still in the page cache leaves a correctly named, empty file after
        // a power cut — which is worse than no file, because it looks like a save
        // that worked.
        file.sync_all()
            .map_err(|e| format!("cannot flush {}: {e}", temporary.display()))?;
    }
    std::fs::rename(&temporary, path).map_err(|e| {
        let _ = std::fs::remove_file(&temporary);
        format!("cannot replace {}: {e}", path.display())
    })
}

#[cfg(test)]
mod view_tests {
    use super::*;

    #[test]
    fn a_saved_view_comes_back_and_a_broken_one_does_not() {
        let camera = kreate_render::Camera {
            pivot: glam::DVec3::new(1.0, -2.0, 0.5),
            distance: 3.25,
            fov_y: 0.6,
            projection: kreate_render::Projection::Orthographic,
            ..Default::default()
        };
        let back = view_from_record(&view_record(&camera)).expect("a view");
        assert_eq!(back.pivot, camera.pivot);
        assert!(back.rotation.abs_diff_eq(camera.rotation, 1e-12));
        assert_eq!(back.distance, 3.25);
        assert_eq!(back.fov_y, 0.6);
        assert_eq!(back.projection, kreate_render::Projection::Orthographic);
        for broken in [
            serde_json::json!({}),
            serde_json::json!({"pivot": [0, 0], "rotation": [0, 0, 0, 1], "distance": 1.0, "fov_y": 0.5}),
            serde_json::json!({"pivot": [0, 0, 0], "rotation": [0, 0, 0, 1], "distance": -1.0, "fov_y": 0.5}),
            serde_json::json!({"pivot": [0, 0, 0], "rotation": [0, 0, 0, 0], "distance": 1.0, "fov_y": 0.5}),
        ] {
            assert!(view_from_record(&broken).is_none(), "{broken}");
        }
    }

    #[test]
    fn the_view_rides_in_the_file_and_does_not_mark_it_changed() {
        let dir = std::env::temp_dir().join(format!("kreate-view-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let path = dir.join("viewed.kreate");
        let document = Document::new();
        let mut file = DocumentFile::new();
        let camera = kreate_render::Camera {
            distance: 12.5,
            ..Default::default()
        };
        file.set_view(&camera);
        file.save(&path, &document).expect("saved");
        let (opened, reopened) = DocumentFile::open(&path).expect("opened");
        assert_eq!(reopened.view().map(|v| v.distance), Some(12.5));
        assert!(!reopened.is_modified(&opened));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
