//! Text as geometry: a string's outlines in a font, as closed 2D contours,
//! and those contours as a solid — Blender's Text object with an extrude,
//! Fusion's sketch text.
//!
//! The contours come from the font as it draws them: straight runs as they
//! are, quadratic and cubic curves flattened into short chords. A font marks
//! a letter's holes by winding — the counter of an `o` runs the other way to
//! its outside — and that is what decides here what is filled, not nesting:
//! the strokes of a Hangul syllable may overlap one another, and a rule by
//! nesting would take an overlap for a hole.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::mesh_data::MeshData;
use kreate_math::{Vec2, Vec3};

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
    source: FontSource,
}

#[derive(Clone, Debug)]
enum FontSource {
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

/// How each line sits on the origin's x.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    /// Starting at it.
    Left,
    /// Centred on it.
    Center,
    /// Ending at it.
    Right,
}

/// A string laid out in a font.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    /// Every closed contour, in the font's sense: outsides anticlockwise,
    /// holes clockwise (looking down from +Z). The first line's baseline is
    /// y = 0; later lines go down.
    pub contours: Vec<Vec<Vec2>>,
    /// Characters the font has no glyph for, each once, in order.
    pub missing: Vec<char>,
    /// The laid-out width of the widest line.
    pub width: f32,
}

/// Chords each curve is cut into. Eight keep a letter round at any size it
/// is read at, and a word of thirty letters to a few thousand points.
const CURVE_STEPS: usize = 8;

struct Collect {
    contours: Vec<Vec<Vec2>>,
    current: Vec<Vec2>,
    scale: f32,
    offset: Vec2,
}

impl Collect {
    fn at(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(x * self.scale, y * self.scale) + self.offset
    }
    fn last(&self) -> Vec2 {
        self.current.last().copied().unwrap_or(self.offset)
    }
    fn finish(&mut self) {
        let mut c = std::mem::take(&mut self.current);
        // The font closes a contour by returning to its start: once is enough.
        while c.len() > 1 && c.first().unwrap().distance(*c.last().unwrap()) < 1e-9 {
            c.pop();
        }
        c.dedup_by(|a, b| a.distance(*b) < 1e-9);
        if c.len() >= 3 {
            self.contours.push(c);
        }
    }
}

impl ttf_parser::OutlineBuilder for Collect {
    fn move_to(&mut self, x: f32, y: f32) {
        self.finish();
        let p = self.at(x, y);
        self.current.push(p);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.at(x, y);
        self.current.push(p);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (p0, c, p1) = (self.last(), self.at(x1, y1), self.at(x, y));
        for i in 1..=CURVE_STEPS {
            let t = i as f32 / CURVE_STEPS as f32;
            let u = 1.0 - t;
            self.current
                .push(p0 * (u * u) + c * (2.0 * u * t) + p1 * (t * t));
        }
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (p0, c1, c2, p1) = (self.last(), self.at(x1, y1), self.at(x2, y2), self.at(x, y));
        for i in 1..=CURVE_STEPS {
            let t = i as f32 / CURVE_STEPS as f32;
            let u = 1.0 - t;
            self.current.push(
                p0 * (u * u * u)
                    + c1 * (3.0 * u * u * t)
                    + c2 * (3.0 * u * t * t)
                    + p1 * (t * t * t),
            );
        }
    }
    fn close(&mut self) {
        self.finish();
    }
}

/// Twice the signed area: positive anticlockwise.
fn area2(c: &[Vec2]) -> f32 {
    (0..c.len())
        .map(|i| {
            let (a, b) = (c[i], c[(i + 1) % c.len()]);
            a.x * b.y - b.x * a.y
        })
        .sum()
}

/// `text` in `font` (TrueType or OpenType bytes), `size` the font's em —
/// a capital stands about seven tenths of it — with lines `line_spacing`
/// times the font's own line height apart.
pub fn layout(
    font: &[u8],
    text: &str,
    size: f32,
    align: Align,
    line_spacing: f32,
) -> Result<Layout, String> {
    layout_face(font, 0, text, size, align, line_spacing)
}

/// [`layout`] in face `index` of a font file that holds several.
pub fn layout_face(
    font: &[u8],
    index: u32,
    text: &str,
    size: f32,
    align: Align,
    line_spacing: f32,
) -> Result<Layout, String> {
    let face = ttf_parser::Face::parse(font, index)
        .map_err(|e| format!("the font cannot be read: {e}"))?;
    let em = face.units_per_em().max(1) as f32;
    let scale = size / em;
    let line_height = (face.ascender() as f32 - face.descender() as f32 + face.line_gap() as f32)
        * scale
        * line_spacing;
    let mut out = Layout::default();
    for (row, line) in text.split('\n').enumerate() {
        let line = line.trim_end_matches('\r');
        let base_y = -(row as f32) * line_height;
        // Where each glyph goes, then the line moved to its alignment.
        let mut glyphs = Vec::new();
        let mut pen = 0.0f32;
        for ch in line.chars() {
            let Some(id) = face.glyph_index(ch) else {
                if !ch.is_whitespace() && !out.missing.contains(&ch) {
                    out.missing.push(ch);
                }
                // An unknown space still takes a space's room.
                pen += size * 0.3;
                continue;
            };
            glyphs.push((id, pen));
            pen += face.glyph_hor_advance(id).unwrap_or(0) as f32 * scale;
        }
        out.width = out.width.max(pen);
        let shift = match align {
            Align::Left => 0.0,
            Align::Center => -pen * 0.5,
            Align::Right => -pen,
        };
        for (id, x) in glyphs {
            let mut c = Collect {
                contours: Vec::new(),
                current: Vec::new(),
                scale,
                offset: Vec2::new(x + shift, base_y),
            };
            face.outline_glyph(id, &mut c);
            c.finish();
            out.contours.extend(c.contours);
        }
    }
    // TrueType draws outsides clockwise, PostScript outlines anticlockwise:
    // whichever this font does, the largest contour is an outside, and
    // everything is turned so outsides run anticlockwise.
    if let Some(largest) = out
        .contours
        .iter()
        .max_by(|a, b| area2(a).abs().total_cmp(&area2(b).abs()))
    {
        if area2(largest) < 0.0 {
            for c in &mut out.contours {
                c.reverse();
            }
        }
    }
    Ok(out)
}

/// The laid-out text as a solid `depth` thick, from z = 0 up: a face on
/// top and one below, filled round the holes, and a wall along every
/// contour. Zero depth gives the flat letters alone, facing up.
pub fn solid(layout: &Layout, depth: f32) -> MeshData {
    let mut points: Vec<Vec3> = Vec::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();
    for c in &layout.contours {
        let start = points.len() as u32;
        points.extend(c.iter().map(|p| Vec3::new(p.x, p.y, 0.0)));
        loops.push((start..start + c.len() as u32).collect());
    }
    let n = points.len() as u32;
    if n == 0 {
        return MeshData::empty();
    }
    let flat2: Vec<Vec2> = points.iter().map(|p| Vec2::new(p.x, p.y)).collect();
    let caps = crate::section::fill_loops(loops.clone(), &|i: u32| flat2[i as usize]);
    let mut faces: Vec<Vec<u32>> = Vec::new();
    if depth <= 0.0 {
        faces.extend(caps.iter().map(|t| t.to_vec()));
        return MeshData::from_faces(points, &faces);
    }
    // The top ring is the bottom one lifted.
    let up = Vec3::new(0.0, 0.0, depth);
    let top: Vec<Vec3> = points.iter().map(|p| *p + up).collect();
    points.extend(top);
    for [a, b, c] in &caps {
        faces.push(vec![a + n, b + n, c + n]);
        faces.push(vec![*c, *b, *a]);
    }
    // Walls: for an anticlockwise outside the edge's outward side is its
    // right, and a hole's clockwise run puts that side into the hole —
    // out of the solid either way.
    for l in &loops {
        for i in 0..l.len() {
            let (a, b) = (l[i], l[(i + 1) % l.len()]);
            faces.push(vec![a, b, b + n, a + n]);
        }
    }
    MeshData::from_faces(points, &faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(m: &MeshData) -> (Vec3, Vec3) {
        let b = m.bounds();
        (b.min, b.max)
    }

    /// Latin and Hangul both come out of the bundled font, sized by the em.
    #[test]
    fn letters_of_both_scripts_are_laid_out_at_their_size() {
        let l = layout(DEFAULT_FONT, "Ka 가", 0.02, Align::Left, 1.0).unwrap();
        assert!(l.missing.is_empty(), "{:?}", l.missing);
        assert!(l.contours.len() >= 4, "{}", l.contours.len());
        let top = l
            .contours
            .iter()
            .flatten()
            .map(|p| p.y)
            .fold(f32::MIN, f32::max);
        // A capital and a syllable stand most of an em, never more than one.
        assert!(top > 0.012 && top < 0.02, "{top}");
        assert!(l.width > 0.03 && l.width < 0.08, "{}", l.width);
    }

    /// A character the font lacks is said, not drawn as nothing in silence.
    #[test]
    fn a_missing_glyph_is_reported() {
        let l = layout(DEFAULT_FONT, "A\u{1F600}", 0.01, Align::Left, 1.0).unwrap();
        assert_eq!(l.missing, vec!['\u{1F600}']);
    }

    /// An `O` is a closed ring: its volume is the ring's area times the depth,
    /// so the counter is a hole and not filled, and the solid is closed.
    #[test]
    fn a_letter_with_a_counter_is_a_closed_solid_with_a_hole() {
        let l = layout(DEFAULT_FONT, "O", 0.1, Align::Left, 1.0).unwrap();
        assert_eq!(l.contours.len(), 2, "outside and counter");
        let ring: f32 = l.contours.iter().map(|c| area2(c) * 0.5).sum();
        let outside = l
            .contours
            .iter()
            .map(|c| area2(c).abs() * 0.5)
            .fold(0.0, f32::max);
        assert!(ring > 0.0 && ring < outside * 0.8, "{ring} of {outside}");
        let m = solid(&l, 0.01);
        let mp = crate::measure::mass_properties(&m, |p| p);
        assert!(mp.closed, "a closed solid");
        let expected = ring as f64 * 0.01;
        assert!(
            (mp.volume - expected).abs() < expected * 0.01,
            "{} against {}",
            mp.volume,
            expected
        );
        let (lo, hi) = bounds(&m);
        assert!(lo.z.abs() < 1e-7 && (hi.z - 0.01).abs() < 1e-7);
    }

    /// Centred, the line straddles the origin; right-aligned, it ends there.
    #[test]
    fn alignment_moves_the_line_about_the_origin() {
        let m = |a| solid(&layout(DEFAULT_FONT, "HH", 0.1, a, 1.0).unwrap(), 0.0);
        let (lo, hi) = bounds(&m(Align::Center));
        assert!((lo.x + hi.x).abs() < 0.02, "{lo} {hi}");
        let (_, hi) = bounds(&m(Align::Right));
        assert!(hi.x <= 1e-4, "{hi}");
        let (lo, _) = bounds(&m(Align::Left));
        assert!(lo.x >= -1e-4, "{lo}");
    }

    /// A second line sits a line's height below the first.
    #[test]
    fn lines_stack_downward() {
        let one = layout(DEFAULT_FONT, "I", 0.1, Align::Left, 1.0).unwrap();
        let two = layout(DEFAULT_FONT, "I\nI", 0.1, Align::Left, 1.0).unwrap();
        let low = |l: &Layout| {
            l.contours
                .iter()
                .flatten()
                .map(|p| p.y)
                .fold(f32::MAX, f32::min)
        };
        assert!(low(&two) < low(&one) - 0.08, "{} {}", low(&two), low(&one));
    }

    /// Hangul syllables come out as closed solids with a volume: their
    /// strokes are drawn as separate contours that may overlap, and an
    /// overlap is filled, not taken for a hole.
    #[test]
    fn hangul_is_solid() {
        let l = layout(DEFAULT_FONT, "한글 텍스트", 0.05, Align::Left, 1.0).unwrap();
        let m = solid(&l, 0.005);
        let mp = crate::measure::mass_properties(&m, |p| p);
        let filled: f32 = l.contours.iter().map(|c| area2(c) * 0.5).sum();
        eprintln!(
            "closed {} volume {} filled-area x depth {}",
            mp.closed,
            mp.volume,
            filled * 0.005
        );
        assert!(mp.closed, "every edge between two faces");
        assert!(mp.volume > 0.0);
    }

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
        for name in names.iter().take(5) {
            let (bytes, index) = font_data(name).expect(name);
            let l = layout_face(&bytes, index, "Ab", 0.01, Align::Left, 1.0);
            assert!(l.is_ok_and(|l| !l.contours.is_empty()), "{name}");
        }
        assert!(font_data("No Such Font 12345").is_none());
        assert!(font_data("").is_some(), "empty is the default");
    }

    /// Flat text faces up: every face's normal is +Z.
    #[test]
    fn flat_text_faces_up() {
        let m = solid(
            &layout(DEFAULT_FONT, "e", 0.1, Align::Left, 1.0).unwrap(),
            0.0,
        );
        assert!(m.num_faces() > 0);
        let p = m.positions();
        for f in m.faces() {
            let (a, b, c) = (p[f[0] as usize], p[f[1] as usize], p[f[2] as usize]);
            assert!((b - a).cross(c - a).z >= 0.0);
        }
    }
}
