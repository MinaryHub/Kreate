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

use crate::mesh_data::MeshData;
use kreate_math::{Vec2, Vec3};

// The fonts themselves — the two bundled ones and the computer's — live in
// `kreate-fonts`, which the interface's font picker reads too without
// reaching this crate; they are offered here under their old names.
pub use kreate_fonts::{
    font_data, font_names, fonts, FontEntry, FontSource, DEFAULT_FONT, DEFAULT_FONT_NAME,
    MONO_FONT, MONO_FONT_NAME,
};

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
