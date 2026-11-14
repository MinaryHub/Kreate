//! What Android has no counterpart for in the desktop crates, stood in for.
//!
//! The shell asks for files through `rfd`'s dialogs and reads the clipboard
//! through `arboard`, and neither crate builds for Android: there is no
//! file dialog there that a program can open and wait on, only the
//! system's picker, answered later through an activity. Until that picker
//! is wired in, these modules take the same calls and answer them from the
//! app's own `Documents` folder (`$KREATE_DOCUMENTS`, set at start-up to
//! the folder a computer sees over USB and the Files app shows):
//!
//! - **Save** writes there, under the name the shell suggests.
//! - **Open** opens the newest file there of a kind asked for; put a file
//!   in the folder and it is the one Open brings in.
//! - A question — save before closing? — is answered yes, so nothing is
//!   lost to a dialog that cannot be shown.
//! - The clipboard is empty.

use std::path::{Path, PathBuf};

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

/// `rfd`'s calls, answered from [`documents`].
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
            newest(&documents(), &self.exts)
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
            Some(dir.join(name))
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

/// `arboard`'s clipboard, always empty.
pub mod arboard {
    #[derive(Debug)]
    pub struct Error;

    pub struct Clipboard;

    pub struct Get;

    impl Clipboard {
        pub fn new() -> Result<Self, Error> {
            Err(Error)
        }
        pub fn get(&mut self) -> Get {
            Get
        }
        pub fn get_image(&mut self) -> Result<ImageData<'static>, Error> {
            Err(Error)
        }
        pub fn get_text(&mut self) -> Result<String, Error> {
            Err(Error)
        }
        pub fn set_text(&mut self, _text: impl Into<String>) -> Result<(), Error> {
            Err(Error)
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
}
