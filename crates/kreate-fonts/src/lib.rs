//! The fonts text can be set in: the two the program carries, and the
//! computer's own, listed by name.
//!
//! A leaf, beside the font parser: the mesh kernel sets text in these, and
//! the interface's font picker lists them, and neither needs to reach the
//! other to do so.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// The font every new text starts with: Pretendard, the interface's own,
/// which holds Latin and every modern Hangul syllable.
pub const DEFAULT_FONT: &[u8] =
    include_bytes!("../../../assets/fonts/Pretendard-Regular.subset.ttf");

/// The interface's monospaced face, IBM Plex Mono: offered for text too.
pub const MONO_FONT: &[u8] = include_bytes!("../../../assets/fonts/IBMPlexMono-Regular.ttf");

/// The name [`DEFAULT_FONT`] goes by in the font list.
pub const DEFAULT_FONT_NAME: &str = "Pretendard";

/// The name [`MONO_FONT`] goes by in the font list.
pub const MONO_FONT_NAME: &str = "IBM Plex Mono";

// --- fonts: the bundled ones and the computer's ------------------------------

/// A font text can be set in: by name, from the program or from a file.
#[derive(Clone, Debug)]
pub struct FontEntry {
    /// Its full name, as the font names itself: "Malgun Gothic Bold".
    pub name: String,
    pub source: FontSource,
}

/// Where a font's bytes come from.
#[derive(Clone, Debug)]
pub enum FontSource {
    Bundled(&'static [u8]),
    /// A file, and the face in it (a `.ttc` holds several).
    File(PathBuf, u32),
}

/// Where this computer keeps its fonts: the system's and this user's.
fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
    if cfg!(windows) {
        if let Some(w) = env("WINDIR").or_else(|| env("SystemRoot")) {
            dirs.push(w.join("Fonts"));
        }
        if let Some(l) = env("LOCALAPPDATA") {
            dirs.push(l.join("Microsoft").join("Windows").join("Fonts"));
        }
    } else if cfg!(target_os = "macos") {
        dirs.push("/System/Library/Fonts".into());
        dirs.push("/Library/Fonts".into());
        if let Some(h) = env("HOME") {
            dirs.push(h.join("Library").join("Fonts"));
        }
    } else {
        dirs.push("/usr/share/fonts".into());
        dirs.push("/usr/local/share/fonts".into());
        if let Some(h) = env("HOME") {
            dirs.push(h.join(".local").join("share").join("fonts"));
            dirs.push(h.join(".fonts"));
        }
    }
    dirs
}

/// Every font file under `dir`, a few levels down (Linux keeps them in
/// family folders).
fn font_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for e in read.flatten() {
        let p = e.path();
        if p.is_dir() {
            if depth > 0 {
                font_files(&p, depth - 1, out);
            }
            continue;
        }
        let ext = p
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| x.to_ascii_lowercase());
        if matches!(ext.as_deref(), Some("ttf" | "otf" | "ttc" | "otc")) {
            out.push(p);
        }
    }
}

/// A face's full name, in English where it has one.
fn face_name(names: ttf_parser::name::Names<'_>) -> Option<String> {
    let pick = |id: u16| {
        let mut any = None;
        for n in names.into_iter().filter(|n| n.name_id == id) {
            if let Some(s) = n.to_string().filter(|s| !s.trim().is_empty()) {
                // English (US) on Windows' platform, or the Mac's English.
                if n.language_id == 0x0409 || n.language_id == 0 {
                    return Some(s);
                }
                any.get_or_insert(s);
            }
        }
        any
    };
    pick(ttf_parser::name_id::FULL_NAME).or_else(|| {
        let family = pick(ttf_parser::name_id::FAMILY)?;
        Some(match pick(ttf_parser::name_id::SUBFAMILY) {
            Some(style) if style != "Regular" => format!("{family} {style}"),
            _ => family,
        })
    })
}

fn read_at(file: &mut std::fs::File, at: u64, len: usize) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    // A table bigger than this is a broken directory, not a name table.
    if len > 4 << 20 {
        return None;
    }
    file.seek(SeekFrom::Start(at)).ok()?;
    let mut buf = vec![0; len];
    file.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

/// The name of the face whose table directory starts at `at`, if it has
/// outlines to set text in.
fn face_at(file: &mut std::fs::File, at: u64) -> Option<String> {
    let head = read_at(file, at, 12)?;
    let count = be16(&head, 4)? as usize;
    let dir = read_at(file, at + 12, count.min(256) * 16)?;
    let mut name = None;
    let mut outlines = false;
    for i in 0..count.min(256) {
        let rec = &dir[i * 16..i * 16 + 16];
        match &rec[0..4] {
            b"name" => name = Some((be32(rec, 8)? as u64, be32(rec, 12)? as usize)),
            b"glyf" | b"CFF " | b"CFF2" => outlines = true,
            _ => {}
        }
    }
    if !outlines {
        return None;
    }
    let (offset, len) = name?;
    let bytes = read_at(file, offset, len)?;
    let table = ttf_parser::name::Table::parse(&bytes)?;
    face_name(table.names)
}

/// The faces in a file, with their names. Only each face's table directory
/// and name table are read, so the whole of a system's fonts is listed in
/// a moment, however large its CJK fonts are.
fn faces_in(path: &Path) -> Vec<(String, u32)> {
    let Ok(mut file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let Some(head) = read_at(&mut file, 0, 12) else {
        return Vec::new();
    };
    if &head[0..4] == b"ttcf" {
        // A collection: its faces' directories, by offset.
        let count = be32(&head, 8).unwrap_or(0).min(16) as usize;
        let Some(offsets) = read_at(&mut file, 12, count * 4) else {
            return Vec::new();
        };
        (0..count)
            .filter_map(|i| {
                let at = be32(&offsets, i * 4)? as u64;
                Some((face_at(&mut file, at)?, i as u32))
            })
            .collect()
    } else {
        face_at(&mut file, 0)
            .map(|n| vec![(n, 0)])
            .unwrap_or_default()
    }
}

/// Every font text can be set in: the two the program carries first, then
/// the computer's, by name, each name once. Listed the first time it is
/// asked for, and kept.
pub fn fonts() -> &'static [FontEntry] {
    static LIST: OnceLock<Vec<FontEntry>> = OnceLock::new();
    LIST.get_or_init(|| {
        let mut files = Vec::new();
        for dir in font_dirs() {
            font_files(&dir, 3, &mut files);
        }
        let mut system: Vec<FontEntry> = Vec::new();
        for path in files {
            for (name, index) in faces_in(&path) {
                system.push(FontEntry {
                    name,
                    source: FontSource::File(path.clone(), index),
                });
            }
        }
        system.sort_by_key(|f| f.name.to_lowercase());
        system.dedup_by(|a, b| a.name == b.name);
        system.retain(|f| f.name != DEFAULT_FONT_NAME && f.name != MONO_FONT_NAME);
        // A name beginning with a dot is a font macOS keeps for its own
        // interface — ".ADT Slab Numeric Light", ".SF NS" — and hides from
        // every font menu: some have numerals only, and the system may not
        // keep them from one version to the next.
        system.retain(|f| !f.name.starts_with('.'));
        let mut list = vec![
            FontEntry {
                name: DEFAULT_FONT_NAME.into(),
                source: FontSource::Bundled(DEFAULT_FONT),
            },
            FontEntry {
                name: MONO_FONT_NAME.into(),
                source: FontSource::Bundled(MONO_FONT),
            },
        ];
        list.extend(system);
        list
    })
}

/// The names [`fonts`] offers, in its order.
pub fn font_names() -> Vec<String> {
    fonts().iter().map(|f| f.name.clone()).collect()
}

/// A font's bytes and face index, by name — empty for the default; `None`
/// when this computer has no font of that name. A file is read once and
/// kept.
pub fn font_data(name: &str) -> Option<(Arc<[u8]>, u32)> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<[u8]>>>> = OnceLock::new();
    if name.trim().is_empty() || name == DEFAULT_FONT_NAME {
        return Some((Arc::from(DEFAULT_FONT), 0));
    }
    if name == MONO_FONT_NAME {
        return Some((Arc::from(MONO_FONT), 0));
    }
    let entry = fonts().iter().find(|f| f.name == name)?;
    match &entry.source {
        FontSource::Bundled(bytes) => Some((Arc::from(*bytes), 0)),
        FontSource::File(path, index) => {
            let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
            let mut cache = cache.lock().ok()?;
            if let Some(bytes) = cache.get(path) {
                return Some((bytes.clone(), *index));
            }
            let bytes: Arc<[u8]> = std::fs::read(path).ok()?.into();
            cache.insert(path.clone(), bytes.clone());
            Some((bytes, *index))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two bundled fonts come first, the system's after, by name; a
    /// font is found by its name and lays out text.
    #[test]
    fn fonts_are_listed_by_name_and_found_by_it() {
        let names = font_names();
        assert_eq!(names[0], DEFAULT_FONT_NAME);
        assert_eq!(names[1], MONO_FONT_NAME);
        let mut seen = std::collections::HashSet::new();
        assert!(
            names.iter().all(|n| seen.insert(n.clone())),
            "each name once"
        );
        // The system's first few are read and parsed; whether "Ab" draws
        // anything is the font's business — on a Mac the list opens with
        // Arabic faces, which have no Latin letters to draw.
        for (i, name) in names.iter().take(5).enumerate() {
            let (bytes, index) = font_data(name).expect(name);
            let face = ttf_parser::Face::parse(&bytes, index);
            assert!(face.is_ok(), "{name}");
            if i < 2 {
                let glyph = face.ok().and_then(|f| f.glyph_index('A'));
                assert!(glyph.is_some(), "{name} has an A");
            }
        }
        assert!(font_data("No Such Font 12345").is_none());
        assert!(font_data("").is_some(), "empty is the default");
    }
}
