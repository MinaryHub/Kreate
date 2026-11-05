//! PDF write of a drawing: vector lines on a page of a given size, for
//! printing and for sending to whoever makes the part. See
//! design/2d-drafting.md §9.
//!
//! One page, its contents one stream. The drawing's metres are the page's
//! (a sheet from [`crate::sheet`] is drawn a millimetre to a millimetre).
//! Each curve is stroked in its layer's or its own colour, weight and
//! linetype; text is drawn as the outlines of its letters in the program's
//! own font, Pretendard, filled — so Hangul reads as it does on screen,
//! with no font for the reader to find, and the page looks the same in
//! every viewer. Dimensions are drawn out as the DXF writer draws them.
//!
//! Colours on paper: AutoCAD's colour 7, white on a dark screen, prints
//! black, as every plotting program prints it.

use std::fmt::Write as _;
use std::io::Write as _;

use kreate_geom2d::{Curve2D, DVec2};
use kreate_scene::drawing::{
    ColorRef, Drawing, EntityKind, LinetypeRef, Lineweight, Text, TextAlign,
};

/// Points to a metre: 72 a 25.4 mm inch.
const PT: f64 = 72.0 / 0.0254;

/// How far a curve's strokes stray from it on paper: a hundredth of a
/// millimetre, under what a printer resolves.
const CHORD: f64 = 1e-5;

/// The drawing as a one-page PDF, `width` by `height` metres, the
/// drawing's origin at the page's bottom left.
pub fn write(drawing: &Drawing, width: f64, height: f64) -> Vec<u8> {
    let mut page = String::new();
    // Round caps and joins, as a plotter's pen draws.
    page.push_str("1 J 1 j\n");
    for e in &drawing.entities {
        if e.construction {
            continue;
        }
        let layer = drawing.layer(e.layer);
        if layer.is_some_and(|l| !l.on || l.frozen || !l.plot) {
            continue;
        }
        let colour = rgb(match (&e.color, layer) {
            (ColorRef::ByLayer, Some(l)) => &l.color,
            (c, _) => c,
        });
        let weight = match (&e.lineweight, layer) {
            (Lineweight::Mm(w), _) => *w as f64,
            (_, Some(l)) => match l.lineweight {
                Lineweight::Mm(w) => w as f64,
                _ => 0.25,
            },
            _ => 0.25,
        };
        let linetype = match (&e.linetype, layer) {
            (LinetypeRef::Named(n), _) => n.to_uppercase(),
            (_, Some(l)) => l.linetype.to_uppercase(),
            _ => String::new(),
        };
        let _ = writeln!(
            page,
            "{} {} {} RG {} {} {} rg",
            colour[0], colour[1], colour[2], colour[0], colour[1], colour[2]
        );
        let _ = writeln!(page, "{} w", num(weight * 1e-3 * PT));
        let (_, pattern) = crate::dxf_write::linetype_pattern(&linetype);
        if pattern.is_empty() {
            page.push_str("[] 0 d\n");
        } else {
            // A dot is a dash as long as the pen is wide.
            let dashes: Vec<String> = pattern
                .iter()
                .map(|v| num((v.abs() * 1e-3 * PT).max(weight * 1e-3 * PT)))
                .collect();
            let _ = writeln!(page, "[{}] 0 d", dashes.join(" "));
        }
        match &e.kind {
            EntityKind::Point(p) => {
                // A point as a dot the pen's width.
                let _ = writeln!(page, "{} {} m {} {} l S", x(p), y(p), x(p), y(p));
            }
            EntityKind::Curve(c) => stroke(&mut page, c),
            EntityKind::Text(t) => text(&mut page, t),
            EntityKind::Dimension(d) => {
                let Some(shape) = crate::dxf_write::dimension_shape(drawing, d) else {
                    continue;
                };
                page.push_str("[] 0 d\n");
                for [a, b] in &shape.lines {
                    stroke(&mut page, &Curve2D::Line { a: *a, b: *b });
                }
                if let Some((c, r, from, to)) = shape.arc {
                    stroke(
                        &mut page,
                        &Curve2D::Arc {
                            c,
                            r,
                            start: from,
                            sweep: to - from,
                        },
                    );
                }
                let words = plain(&crate::dxf_write::dimension_text(d, 1e-3));
                text(
                    &mut page,
                    &Text {
                        at: shape.at,
                        height: shape.height,
                        rotation: shape.angle,
                        content: words,
                        align: TextAlign::Center,
                        font: String::new(),
                    },
                );
            }
        }
    }
    document(&page, width * PT, height * PT)
}

fn x(p: &DVec2) -> String {
    num(p.x * PT)
}

fn y(p: &DVec2) -> String {
    num(p.y * PT)
}

fn num(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() {
        "0".into()
    } else {
        s.into()
    }
}

/// A colour on white paper, each part from 0 to 1.
fn rgb(c: &ColorRef) -> [String; 3] {
    let v: [u8; 3] = match c {
        ColorRef::Rgb(rgb) => *rgb,
        ColorRef::Index(i) => match i {
            1 => [255, 0, 0],
            2 => [200, 160, 0],
            3 => [0, 150, 0],
            4 => [0, 150, 170],
            5 => [0, 0, 255],
            6 => [200, 0, 200],
            8 => [128, 128, 128],
            9 => [160, 160, 160],
            _ => [0, 0, 0],
        },
        ColorRef::ByLayer | ColorRef::ByBlock => [0, 0, 0],
    };
    v.map(|c| num(f64::from(c) / 255.0))
}

/// A curve stroked: a line as itself, anything curved as fine chords.
fn stroke(page: &mut String, c: &Curve2D) {
    let points = match c {
        Curve2D::Line { a, b } => vec![*a, *b],
        _ => c.tessellate(CHORD),
    };
    let Some(first) = points.first() else {
        return;
    };
    let _ = write!(page, "{} {} m", x(first), y(first));
    for p in &points[1..] {
        let _ = write!(page, " {} {} l", x(p), y(p));
    }
    page.push_str(if c.is_closed() { " h S\n" } else { " S\n" });
}

/// DXF's escapes as the characters they stand for.
fn plain(s: &str) -> String {
    s.replace("%%c", "Ø")
        .replace("%%d", "°")
        .replace("%%p", "±")
        .replace("%%%", "%")
        .replace("%%", "%")
}

/// Text as its letters' outlines, filled by the non-zero rule — the font's
/// own, which marks a counter by winding it the other way.
fn text(page: &mut String, t: &Text) {
    if t.content.trim().is_empty() {
        return;
    }
    // Laid out in millimetres, where `f32` holds a letter's curves finely;
    // a capital stands seven tenths of the em.
    let size_mm = (t.height * 1e3 / 0.7) as f32;
    let align = match t.align {
        TextAlign::Left => kreate_mesh::text::Align::Left,
        TextAlign::Center => kreate_mesh::text::Align::Center,
        TextAlign::Right => kreate_mesh::text::Align::Right,
    };
    let Ok(layout) =
        kreate_mesh::text::layout(kreate_fonts::DEFAULT_FONT, &t.content, size_mm, align, 1.2)
    else {
        return;
    };
    let (sin, cos) = t.rotation.sin_cos();
    let place = |p: kreate_math::Vec2| {
        let (u, v) = (f64::from(p.x) * 1e-3, f64::from(p.y) * 1e-3);
        t.at + DVec2::new(u * cos - v * sin, u * sin + v * cos)
    };
    let mut any = false;
    for contour in &layout.contours {
        let Some(first) = contour.first() else {
            continue;
        };
        let p = place(*first);
        let _ = write!(page, "{} {} m", x(&p), y(&p));
        for q in &contour[1..] {
            let q = place(*q);
            let _ = write!(page, " {} {} l", x(&q), y(&q));
        }
        page.push_str(" h\n");
        any = true;
    }
    if any {
        page.push_str("f\n");
    }
}

/// The file: catalog, pages, the page, its compressed contents, and the
/// table that finds them.
fn document(content: &str, width_pt: f64, height_pt: f64) -> Vec<u8> {
    let mut deflate = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = deflate.write_all(content.as_bytes());
    let stream = deflate.finish().unwrap_or_default();
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = Vec::new();
    let mut object = |out: &mut Vec<u8>, body: &[u8]| {
        offsets.push(out.len());
        let n = offsets.len();
        out.extend_from_slice(format!("{n} 0 obj\n").as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };
    object(&mut out, b"<< /Type /Catalog /Pages 2 0 R >>");
    object(&mut out, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    object(
        &mut out,
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Contents 4 0 R /Resources << >> >>",
            num(width_pt),
            num(height_pt)
        )
        .as_bytes(),
    );
    let mut body = format!(
        "<< /Length {} /Filter /FlateDecode >>\nstream\n",
        stream.len()
    )
    .into_bytes();
    body.extend_from_slice(&stream);
    body.extend_from_slice(b"\nendstream");
    object(&mut out, &body);
    object(&mut out, b"<< /Producer (Kreate) >>");
    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes(),
    );
    for o in &offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R /Info 5 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            offsets.len() + 1
        )
        .as_bytes(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;

    fn contents(pdf: &[u8]) -> String {
        let start = pdf
            .windows(7)
            .position(|w| w == b"stream\n")
            .expect("a stream")
            + 7;
        let end = pdf
            .windows(10)
            .rposition(|w| w == b"\nendstream")
            .expect("its end");
        let mut text = String::new();
        flate2::read::ZlibDecoder::new(&pdf[start..end])
            .read_to_string(&mut text)
            .expect("inflates");
        text
    }

    /// A page the size asked, its table pointing at each object, lines in
    /// points, a dashed layer dashed, and text — Hangul too — drawn as
    /// filled outlines.
    #[test]
    fn a_drawing_is_a_page_of_strokes_and_filled_letters() {
        let mut d = Drawing::new();
        let dashed = d.add_layer("HIDDEN");
        if let Some(l) = d.layers.iter_mut().find(|l| l.id == dashed) {
            l.linetype = "DASHED".into();
        }
        d.add_curve(Curve2D::Line {
            a: DVec2::new(0.01, 0.01),
            b: DVec2::new(0.1, 0.01),
        });
        d.current_layer = dashed;
        d.add_curve(Curve2D::Circle {
            c: DVec2::new(0.05, 0.05),
            r: 0.01,
        });
        d.add(EntityKind::Text(Text {
            at: DVec2::new(0.02, 0.08),
            height: 0.005,
            rotation: 0.0,
            content: "도면 A".into(),
            align: TextAlign::Left,
            font: String::new(),
        }));
        let pdf = write(&d, 0.297, 0.21);
        assert!(pdf.starts_with(b"%PDF-1.4"));
        let text = String::from_utf8_lossy(&pdf);
        assert!(
            text.contains("/MediaBox [0 0 841.89 595.276]"),
            "A4 landscape in points"
        );
        // Each object where the table says.
        let xref = text.rfind("xref\n").unwrap();
        let table = &text[xref..];
        for (i, line) in table.lines().skip(3).take(5).enumerate() {
            let at: usize = line[..10].parse().unwrap();
            assert!(
                text[at..].starts_with(&format!("{} 0 obj", i + 1)),
                "object {}",
                i + 1
            );
        }
        let page = contents(&pdf);
        // 10 mm is 28.346 pt.
        assert!(
            page.contains("28.346 28.346 m 283.465 28.346 l S"),
            "{page}"
        );
        assert!(page.contains("] 0 d"), "the circle is dashed");
        assert!(page.matches(" h\nf\n").count() >= 1, "letters are filled");
        assert!(
            page.matches(" h\n").count() > 4,
            "each letter's outlines: {page}"
        );
    }

    #[test]
    fn dxfs_escapes_print_as_their_signs() {
        assert_eq!(plain("%%c12%%p0.1 45%%d"), "Ø12±0.1 45°");
    }
}
