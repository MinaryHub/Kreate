//! What Android has no counterpart for in the desktop crates, stood in for.
//!
//! The shell asks for files through `rfd`'s dialogs and reads the clipboard
//! through `arboard`, and neither crate builds for Android. These modules
//! take the same calls. Files live in the app's own `Documents` folder
//! (`$KREATE_DOCUMENTS`, set at start-up), and the system's picker brings
//! them in and takes them out:
//!
//! - **Open** opens the system's picker and returns nothing at once: the
//!   app goes on while the person chooses. The file chosen is copied into
//!   `Documents`, the app is woken, and what asked for a file is asked
//!   again — this time [`rfd::FileDialog::pick_file`] has the file to give.
//!   The shell keeps what asked (see `pending_pick` in the shell).
//! - **Save** writes into `Documents` under the name the shell suggests,
//!   and then the system's save screen offers to put a copy where the
//!   person likes.
//! - With no picker (a test, or before start-up has set one), Open takes
//!   the newest file in `Documents` of a kind asked for.
//! - A question — save before closing? — is answered yes, so nothing is
//!   lost to a dialog that cannot be shown.
//! - The clipboard calls into the activity via [`set_clipboard`] (falling back
//!   to an in-process buffer when running headless or in tests).
//! - CJK/IME text composition arrives via [`push_ime_commit`] /
//!   [`set_ime_preedit`] from Android's `InputConnection`.
//!
//! The picker itself is the activity's (apps/kreate-android), reached
//! through [`set_picker`]; the soft keyboard through [`set_keyboard`].

use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Mobile undo memory budget (64 MiB) to stay well within Android RAM limits.
pub const MOBILE_UNDO_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// What the shell asks the system's picker for.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// A file to open, of these extensions (any, when there are none).
    Open { extensions: Vec<String> },
    /// A copy of `path`, just saved, put where the person chooses, offered
    /// under `name`.
    Save { path: PathBuf, name: String },
}

type Hook<T> = Box<dyn Fn(T) + Send + Sync>;
type ClipboardGet = Box<dyn Fn() -> Option<String> + Send + Sync>;
type ClipboardSet = Box<dyn Fn(&str) + Send + Sync>;

#[derive(Default)]
struct Picker {
    open: Option<Hook<Request>>,
    wake: Option<Hook<()>>,
    keyboard: Option<Hook<bool>>,
    clipboard_get: Option<ClipboardGet>,
    clipboard_set: Option<ClipboardSet>,
    clipboard_fallback: Option<String>,
    /// Committed CJK/IME strings queued from Android's `InputConnection`.
    ime_commits: Vec<String>,
    /// Active IME composing preedit string (empty when not composing).
    ime_preedit: String,
    /// A file the picker brought in, for the next `pick_file`.
    ready: Option<PathBuf>,
    /// A `pick_file` opened the picker since the shell last asked.
    asked: bool,
}

static PICKER: Mutex<Option<Picker>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Picker) -> R) -> R {
    let mut guard = PICKER.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Picker::default))
}

fn wake_shell() {
    let has_waker = with(|p| p.wake.is_some());
    if has_waker {
        with(|p| {
            if let Some(w) = &p.wake {
                w(())
            }
        });
    }
}

/// How the system's picker is opened: set once at start-up on Android.
pub fn set_picker(open: impl Fn(Request) + Send + Sync + 'static) {
    with(|p| p.open = Some(Box::new(open)));
}

/// How the shell is woken when a file comes in.
pub fn set_waker(wake: impl Fn(()) + Send + Sync + 'static) {
    with(|p| p.wake = Some(Box::new(wake)));
}

/// How the soft keyboard is shown (`true`) and put away.
pub fn set_keyboard(show: impl Fn(bool) + Send + Sync + 'static) {
    with(|p| p.keyboard = Some(Box::new(show)));
}

/// How Android's system clipboard is read and written: set once at start-up.
pub fn set_clipboard(
    get: impl Fn() -> Option<String> + Send + Sync + 'static,
    set: impl Fn(&str) + Send + Sync + 'static,
) {
    with(|p| {
        p.clipboard_get = Some(Box::new(get));
        p.clipboard_set = Some(Box::new(set));
    });
}

/// Enqueues committed text from Android's `InputConnection` (e.g. Korean/CJK IME).
pub fn push_ime_commit(text: String) {
    if text.is_empty() {
        return;
    }
    with(|p| {
        p.ime_preedit.clear();
        p.ime_commits.push(text);
    });
    wake_shell();
}

/// Updates the active IME composing preedit text from Android's `InputConnection`.
pub fn set_ime_preedit(text: String) {
    with(|p| p.ime_preedit = text);
    wake_shell();
}

/// Drains all committed IME text chunks since the last frame.
pub fn take_ime_commits() -> Vec<String> {
    with(|p| std::mem::take(&mut p.ime_commits))
}

/// Returns the current IME composing preedit string (empty when idle).
pub fn ime_preedit() -> String {
    with(|p| p.ime_preedit.clone())
}

/// The soft keyboard shown or put away, where there is one.
pub fn keyboard(show: bool) {
    with(|p| {
        if let Some(k) = &p.keyboard {
            k(show)
        }
    });
}

/// A file the picker brought in: kept for the next `pick_file`, and the
/// shell woken to ask for it.
pub fn file_arrived(path: PathBuf) {
    with(|p| p.ready = Some(path));
    wake_shell();
}

/// Whether a file has come in that no `pick_file` has taken yet.
pub fn file_waiting() -> bool {
    with(|p| p.ready.is_some())
}

/// Whether a `pick_file` opened the picker since this was last asked:
/// what the shell was doing then is to be done again when the file comes.
pub fn take_asked() -> bool {
    with(|p| std::mem::take(&mut p.asked))
}

/// The file waiting, taken; or the picker opened for one (`None` now).
fn pick(extensions: &[String]) -> Option<PathBuf> {
    enum Next {
        Have(PathBuf),
        Asked,
        Nothing,
    }
    let next = with(|p| {
        if let Some(path) = p.ready.take() {
            return Next::Have(path);
        }
        match &p.open {
            Some(open) => {
                open(Request::Open {
                    extensions: extensions.to_vec(),
                });
                p.asked = true;
                Next::Asked
            }
            None => Next::Nothing,
        }
    });
    match next {
        Next::Have(path) => Some(path),
        Next::Asked => None,
        Next::Nothing => newest(&documents(), extensions),
    }
}

/// `path`, just saved, offered to the picker to copy where the person
/// likes.
fn offer(path: &Path) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    with(|p| {
        if let Some(open) = &p.open {
            open(Request::Save {
                path: path.to_path_buf(),
                name,
            });
        }
    });
}

/// The folder files are saved to and opened from.
pub fn documents() -> PathBuf {
    std::env::var_os("KREATE_DOCUMENTS")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Documents")))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The newest file directly in `dir` whose extension is one of `exts` (any,
/// when there are none).
fn newest(dir: &Path, exts: &[String]) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter(|e| {
            exts.is_empty()
                || e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| exts.iter().any(|k| k.eq_ignore_ascii_case(x)))
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// `rfd`'s calls, answered from [`documents`] and the system's picker.
pub mod rfd {
    use super::*;

    #[derive(Default)]
    pub struct FileDialog {
        name: Option<String>,
        exts: Vec<String>,
    }

    impl FileDialog {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn set_title(self, _title: impl Into<String>) -> Self {
            self
        }
        pub fn add_filter(
            mut self,
            _name: impl Into<String>,
            extensions: &[impl ToString],
        ) -> Self {
            self.exts.extend(extensions.iter().map(|e| e.to_string()));
            self
        }
        pub fn set_file_name(mut self, name: impl Into<String>) -> Self {
            self.name = Some(name.into());
            self
        }
        pub fn set_directory(self, _dir: impl AsRef<Path>) -> Self {
            self
        }
        pub fn set_parent<W: ?Sized>(self, _parent: &W) -> Self {
            self
        }
        pub fn pick_file(self) -> Option<PathBuf> {
            pick(&self.exts)
        }
        pub fn pick_files(self) -> Option<Vec<PathBuf>> {
            self.pick_file().map(|p| vec![p])
        }
        pub fn pick_folder(self) -> Option<PathBuf> {
            Some(documents())
        }
        pub fn save_file(self) -> Option<PathBuf> {
            let dir = documents();
            std::fs::create_dir_all(&dir).ok()?;
            let name = self.name.unwrap_or_else(|| {
                format!(
                    "untitled.{}",
                    self.exts.first().map_or("kreate", |e| e.as_str())
                )
            });
            // Only the name: a suggestion carrying folders stays in ours.
            let name = Path::new(&name).file_name()?.to_owned();
            let path = dir.join(name);
            // The shell writes it as this returns; the person is still
            // choosing where the copy goes when it has.
            offer(&path);
            Some(path)
        }
    }

    pub enum MessageLevel {
        Info,
        Warning,
        Error,
    }

    pub enum MessageButtons {
        Ok,
        OkCancel,
        YesNo,
        YesNoCancel,
        OkCustom(String),
        OkCancelCustom(String, String),
        YesNoCancelCustom(String, String, String),
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum MessageDialogResult {
        Yes,
        No,
        Ok,
        Cancel,
        Custom(String),
    }

    #[derive(Default)]
    pub struct MessageDialog;

    impl MessageDialog {
        pub fn new() -> Self {
            Self
        }
        pub fn set_title(self, _title: impl Into<String>) -> Self {
            self
        }
        pub fn set_description(self, _text: impl Into<String>) -> Self {
            self
        }
        pub fn set_level(self, _level: MessageLevel) -> Self {
            self
        }
        pub fn set_buttons(self, _buttons: MessageButtons) -> Self {
            self
        }
        pub fn set_parent<W: ?Sized>(self, _parent: &W) -> Self {
            self
        }
        /// Yes: save, go on, do what was asked.
        pub fn show(self) -> MessageDialogResult {
            MessageDialogResult::Yes
        }
    }
}

/// `arboard`-compatible clipboard backed by Android's `ClipboardManager` hooks
/// (with an in-process fallback for headless/test use).
pub mod arboard {
    use super::with;

    #[derive(Debug)]
    pub struct Error;

    pub struct Clipboard;

    pub struct Get;

    impl Clipboard {
        pub fn new() -> Result<Self, Error> {
            Ok(Self)
        }
        pub fn get(&mut self) -> Get {
            Get
        }
        pub fn get_image(&mut self) -> Result<ImageData<'static>, Error> {
            Err(Error)
        }
        pub fn get_text(&mut self) -> Result<String, Error> {
            with(|p| {
                if let Some(getter) = &p.clipboard_get {
                    getter().or_else(|| p.clipboard_fallback.clone())
                } else {
                    p.clipboard_fallback.clone()
                }
            })
            .ok_or(Error)
        }
        pub fn set_text(&mut self, text: impl Into<String>) -> Result<(), Error> {
            let s = text.into();
            with(|p| {
                if let Some(setter) = &p.clipboard_set {
                    setter(&s);
                }
                p.clipboard_fallback = Some(s);
            });
            Ok(())
        }
    }

    impl Get {
        pub fn file_list(self) -> Result<Vec<std::path::PathBuf>, Error> {
            Err(Error)
        }
    }

    pub struct ImageData<'a> {
        pub width: usize,
        pub height: usize,
        pub bytes: std::borrow::Cow<'a, [u8]>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Open brings in the newest file of a kind asked for; Save keeps
    /// only the suggested name, in the folder.
    #[test]
    fn open_takes_the_newest_file_of_its_kind() {
        let dir = std::env::temp_dir().join(format!("kreate-android-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.kreate"), "a").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join("b.kreate"), "b").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join("c.obj"), "c").unwrap();
        let kreate = vec!["kreate".to_string()];
        assert_eq!(newest(&dir, &kreate), Some(dir.join("b.kreate")));
        assert_eq!(newest(&dir, &[]), Some(dir.join("c.obj")));
        assert_eq!(newest(&dir, &["dxf".to_string()]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With a picker: Open asks it and gives nothing yet; the file it
    /// brings wakes the shell and is what the next Open gives. Save offers
    /// the file saved to it.
    #[test]
    fn the_picker_is_asked_and_its_file_given_on_the_next_open() {
        let asked = std::sync::Arc::new(Mutex::new(Vec::new()));
        let woken = std::sync::Arc::new(Mutex::new(0));
        {
            let asked = asked.clone();
            set_picker(move |r| asked.lock().unwrap().push(r));
            let woken = woken.clone();
            set_waker(move |()| *woken.lock().unwrap() += 1);
        }
        assert_eq!(
            rfd::FileDialog::new()
                .add_filter("Kreate", &["kreate"])
                .pick_file(),
            None
        );
        assert!(take_asked());
        assert!(!take_asked());
        assert_eq!(
            asked.lock().unwrap().last(),
            Some(&Request::Open {
                extensions: vec!["kreate".into()]
            })
        );
        file_arrived(PathBuf::from("/docs/part.kreate"));
        assert_eq!(*woken.lock().unwrap(), 1);
        assert!(file_waiting());
        assert_eq!(
            rfd::FileDialog::new().pick_file(),
            Some(PathBuf::from("/docs/part.kreate"))
        );
        assert!(!file_waiting() && !take_asked());

        let docs = std::env::temp_dir().join(format!("kreate-docs-{}", std::process::id()));
        std::env::set_var("KREATE_DOCUMENTS", &docs);
        let saved = rfd::FileDialog::new()
            .set_file_name("box.kreate")
            .save_file()
            .unwrap();
        assert_eq!(saved.file_name().unwrap(), "box.kreate");
        assert!(matches!(
            asked.lock().unwrap().last(),
            Some(Request::Save { name, .. }) if name == "box.kreate"
        ));
        *PICKER.lock().unwrap() = None;
        let _ = std::fs::remove_dir_all(&docs);
    }

    #[test]
    fn clipboard_and_ime_hooks_round_trip() {
        *PICKER.lock().unwrap() = None;
        // Fallback buffer works before JNI hooks are registered.
        let mut cb = arboard::Clipboard::new().expect("clipboard available");
        cb.set_text("hello kreate").unwrap();
        assert_eq!(cb.get_text().unwrap(), "hello kreate");

        // JNI-style hook overrides clipboard read/write.
        let backing = std::sync::Arc::new(Mutex::new(String::from("from-android")));
        let b_get = backing.clone();
        let b_set = backing.clone();
        set_clipboard(
            move || Some(b_get.lock().unwrap().clone()),
            move |s| *b_set.lock().unwrap() = s.to_string(),
        );
        assert_eq!(cb.get_text().unwrap(), "from-android");
        cb.set_text("한글 복사").unwrap();
        assert_eq!(*backing.lock().unwrap(), "한글 복사");
        assert_eq!(cb.get_text().unwrap(), "한글 복사");

        // CJK IME preedit and commit queue.
        set_ime_preedit("하".into());
        assert_eq!(ime_preedit(), "하");
        push_ime_commit("한글".into());
        assert_eq!(ime_preedit(), "");
        assert_eq!(take_ime_commits(), vec!["한글".to_string()]);
        assert!(take_ime_commits().is_empty());

        *PICKER.lock().unwrap() = None;
    }
}
