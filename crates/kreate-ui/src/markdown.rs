//! Enough Markdown for an assistant's answers: headings, lists, quotes, code
//! blocks and the inline marks (`code`, **strong**, *emphasis*, [links]).
//!
//! Assistants write Markdown whether asked to or not, and shown as plain text
//! it is a page of asterisks and back-ticks. This is not a Markdown
//! implementation — no nesting beyond list indent, no HTML, no references —
//! only what reads badly when left raw. A line break inside a paragraph is
//! kept, as a chat reader expects, not joined as a document's would be.

use egui::text::LayoutJob;
use egui::{Color32, FontId, TextFormat, Ui};

/// One block of an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    /// Running text; its lines are kept.
    Paragraph(String),
    /// `#` to `######`, and its text.
    Heading(u8, String),
    /// `- `, `* `, `+ ` or `1. `: how far in (in steps of two spaces), the
    /// marker as it should be shown, and the text.
    Item {
        indent: usize,
        marker: String,
        text: String,
    },
    /// `> `.
    Quote(String),
    /// A fenced block: its language, if named, and the code.
    Code { lang: String, code: String },
    /// A table's rows, shown as they were written, in a fixed-width face.
    Table(String),
    /// `---`.
    Rule,
}

/// A run of text with one look.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub strong: bool,
    pub emphasis: bool,
    pub code: bool,
    pub link: bool,
}

impl Span {
    fn plain(text: String) -> Self {
        Self {
            text,
            strong: false,
            emphasis: false,
            code: false,
            link: false,
        }
    }
}

/// Splits an answer into blocks.
pub fn blocks(source: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut paragraph: Vec<&str> = Vec::new();
    let mut lines = source.lines().peekable();
    let flush = |paragraph: &mut Vec<&str>, out: &mut Vec<Block>| {
        if !paragraph.is_empty() {
            out.push(Block::Paragraph(paragraph.join("\n")));
            paragraph.clear();
        }
    };
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if let Some(fence) = trimmed.strip_prefix("```") {
            flush(&mut paragraph, &mut out);
            let mut code = Vec::new();
            for line in lines.by_ref() {
                if line.trim_start().starts_with("```") {
                    break;
                }
                code.push(line);
            }
            out.push(Block::Code {
                lang: fence.trim().to_string(),
                code: code.join("\n"),
            });
            continue;
        }
        if trimmed.is_empty() {
            flush(&mut paragraph, &mut out);
            continue;
        }
        if let Some(heading) = heading(trimmed) {
            flush(&mut paragraph, &mut out);
            out.push(heading);
            continue;
        }
        if is_rule(trimmed) {
            flush(&mut paragraph, &mut out);
            out.push(Block::Rule);
            continue;
        }
        if trimmed.starts_with('|') {
            flush(&mut paragraph, &mut out);
            let mut rows = vec![line.trim()];
            while let Some(next) = lines.peek() {
                if !next.trim_start().starts_with('|') {
                    break;
                }
                rows.push(next.trim());
                lines.next();
            }
            out.push(Block::Table(table(&rows)));
            continue;
        }
        if let Some(quote) = trimmed.strip_prefix('>') {
            flush(&mut paragraph, &mut out);
            let mut text = vec![quote.trim_start()];
            while let Some(next) = lines.peek() {
                let Some(more) = next.trim_start().strip_prefix('>') else {
                    break;
                };
                text.push(more.trim_start());
                lines.next();
            }
            out.push(Block::Quote(text.join("\n")));
            continue;
        }
        let indent = (line.len() - trimmed.len()) / 2;
        if let Some((marker, text)) = item(trimmed) {
            flush(&mut paragraph, &mut out);
            out.push(Block::Item {
                indent,
                marker,
                text: text.to_string(),
            });
            continue;
        }
        // A line under an item, indented, carries the item on.
        if indent > 0 && paragraph.is_empty() {
            if let Some(Block::Item { text, .. }) = out.last_mut() {
                text.push('\n');
                text.push_str(trimmed);
                continue;
            }
        }
        paragraph.push(line);
    }
    flush(&mut paragraph, &mut out);
    out
}

fn heading(line: &str) -> Option<Block> {
    let level = line.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let text = line[level..].strip_prefix(' ')?;
    Some(Block::Heading(level as u8, text.trim().to_string()))
}

fn is_rule(line: &str) -> bool {
    let line = line.trim();
    line.len() >= 3
        && ["-", "*", "_"].iter().any(|m| {
            line.chars()
                .filter(|c| !c.is_whitespace())
                .all(|c| c.to_string() == *m)
        })
}

fn item(line: &str) -> Option<(String, &str)> {
    for bullet in ["- ", "* ", "+ "] {
        if let Some(text) = line.strip_prefix(bullet) {
            // i18n-exempt: a list bullet
            return Some(("•".to_string(), text));
        }
    }
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 && digits < 4 {
        let rest = &line[digits..];
        if let Some(text) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some((format!("{}.", &line[..digits]), text));
        }
    }
    None
}

/// A table's rows lined up in columns, its `|---|` line dropped.
fn table(rows: &[&str]) -> String {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            row.trim_matches('|')
                .split('|')
                .map(|c| strip_marks(c.trim()))
                .collect()
        })
        .filter(|cells: &Vec<String>| {
            !cells
                .iter()
                .all(|c| !c.is_empty() && c.chars().all(|ch| matches!(ch, '-' | ':' | ' ')))
        })
        .collect();
    let columns = cells.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..columns)
        .map(|i| {
            cells
                .iter()
                .filter_map(|r| r.get(i))
                .map(|c| display_width(c))
                .max()
                .unwrap_or(0)
        })
        .collect();
    cells
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, c)| {
                    let pad = widths[i].saturating_sub(display_width(c));
                    format!("{c}{}", " ".repeat(pad))
                })
                .collect::<Vec<_>>()
                .join("  ")
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Columns a cell takes in a fixed-width face: Hangul and other wide
/// characters take two.
fn display_width(text: &str) -> usize {
    text.chars()
        .map(|c| if (c as u32) >= 0x1100 { 2 } else { 1 })
        .sum()
}

fn strip_marks(text: &str) -> String {
    spans(text).into_iter().map(|s| s.text).collect()
}

/// The inline marks of one block's text.
pub fn spans(text: &str) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut plain = String::new();
    let mut strong = false;
    let mut emphasis = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let push = |out: &mut Vec<Span>, plain: &mut String, strong: bool, emphasis: bool| {
        if !plain.is_empty() {
            out.push(Span {
                strong,
                emphasis,
                ..Span::plain(std::mem::take(plain))
            });
        }
    };
    let find = |from: usize, needle: &[char]| -> Option<usize> {
        (from..chars.len().saturating_sub(needle.len() - 1))
            .find(|&j| chars[j..j + needle.len()] == *needle)
    };
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() && chars[i + 1].is_ascii_punctuation() {
            plain.push(chars[i + 1]);
            i += 2;
            continue;
        }
        if c == '`' {
            if let Some(end) = find(i + 1, &['`']) {
                push(&mut out, &mut plain, strong, emphasis);
                out.push(Span {
                    code: true,
                    ..Span::plain(chars[i + 1..end].iter().collect())
                });
                i = end + 1;
                continue;
            }
        }
        if (c == '*' || c == '_') && i + 1 < chars.len() && chars[i + 1] == c {
            let closing = [c, c];
            if strong || find(i + 2, &closing).is_some() {
                push(&mut out, &mut plain, strong, emphasis);
                strong = !strong;
                i += 2;
                continue;
            }
        }
        // A single `*` emphasises when it opens before a word and closes
        // after one; `_` only at a word's edge, so snake_case stays itself.
        if c == '*' || c == '_' {
            let before = i.checked_sub(1).map(|j| chars[j]);
            let after = chars.get(i + 1).copied();
            let opens = !emphasis
                && after.is_some_and(|a| !a.is_whitespace())
                && (c == '*' || before.is_none_or(|b| !b.is_alphanumeric()))
                && find(i + 1, &[c]).is_some();
            let closes = emphasis
                && before.is_some_and(|b| !b.is_whitespace())
                && (c == '*' || after.is_none_or(|a| !a.is_alphanumeric()));
            if opens || closes {
                push(&mut out, &mut plain, strong, emphasis);
                emphasis = !emphasis;
                i += 1;
                continue;
            }
        }
        if c == '[' {
            if let Some(close) = find(i + 1, &[']']) {
                if chars.get(close + 1) == Some(&'(') {
                    if let Some(end) = find(close + 2, &[')']) {
                        push(&mut out, &mut plain, strong, emphasis);
                        out.push(Span {
                            link: true,
                            strong,
                            emphasis,
                            ..Span::plain(chars[i + 1..close].iter().collect())
                        });
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        plain.push(c);
        i += 1;
    }
    push(&mut out, &mut plain, strong, emphasis);
    out
}

/// Draws an answer. `copy` names the menu item that copies a code block.
pub fn show(ui: &mut Ui, theme: &crate::Theme, source: &str, copy: &str) {
    let body = egui::TextStyle::Body.resolve(ui.style());
    let mono = egui::TextStyle::Monospace.resolve(ui.style());
    let width = ui.available_width();
    for (index, block) in blocks(source).into_iter().enumerate() {
        if index > 0 {
            ui.add_space(4.0);
        }
        match block {
            Block::Paragraph(text) => {
                label(ui, job(&text, &body, &mono, theme.text, theme, width));
            }
            Block::Heading(level, text) => {
                let scale = match level {
                    1 => 1.3,
                    2 => 1.18,
                    _ => 1.06,
                };
                let font = FontId::new(body.size * scale, body.family.clone());
                ui.add_space(2.0);
                label(ui, job(&text, &font, &mono, theme.text, theme, width));
            }
            Block::Item {
                indent,
                marker,
                text,
            } => {
                ui.horizontal_top(|ui| {
                    ui.add_space(4.0 + indent as f32 * 14.0);
                    let marker_width = if marker == "•" { 10.0 } else { 18.0 };
                    ui.add_sized(
                        [marker_width, body.size + 2.0],
                        // i18n-exempt: a list marker
                        egui::Label::new(egui::RichText::new(marker).color(theme.text_dim)),
                    );
                    let width = ui.available_width();
                    label(ui, job(&text, &body, &mono, theme.text, theme, width));
                });
            }
            Block::Quote(text) => {
                egui::Frame::new()
                    .inner_margin(egui::Margin {
                        left: 10,
                        right: 0,
                        top: 2,
                        bottom: 2,
                    })
                    .show(ui, |ui| {
                        let rect = ui.max_rect();
                        ui.painter().line_segment(
                            [
                                rect.left_top() - egui::vec2(8.0, 2.0),
                                rect.left_bottom() - egui::vec2(8.0, -2.0),
                            ],
                            egui::Stroke::new(2.0, theme.border_strong),
                        );
                        let width = ui.available_width();
                        label(ui, job(&text, &body, &mono, theme.text_dim, theme, width));
                    });
            }
            Block::Code { code, .. } | Block::Table(code) if !code.is_empty() => {
                let response = egui::Frame::new()
                    .fill(theme.sunken)
                    .stroke(egui::Stroke::new(1.0, theme.border))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let font = FontId::new(mono.size * 0.92, mono.family.clone());
                        // i18n-exempt: code, as the assistant wrote it
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&code).font(font).color(theme.text),
                            )
                            .wrap(),
                        );
                    })
                    .response;
                response.context_menu(|ui| {
                    if ui.button(copy).clicked() {
                        ui.ctx().copy_text(code.clone());
                        ui.close();
                    }
                });
            }
            Block::Code { .. } | Block::Table(_) => {}
            Block::Rule => {
                ui.separator();
            }
        }
    }
}

fn label(ui: &mut Ui, job: LayoutJob) {
    ui.add(egui::Label::new(job).wrap());
}

fn job(
    text: &str,
    font: &FontId,
    mono: &FontId,
    color: Color32,
    theme: &crate::Theme,
    width: f32,
) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.wrap.max_width = width;
    for span in spans(text) {
        let format = if span.code {
            TextFormat {
                font_id: FontId::new(mono.size * 0.92, mono.family.clone()),
                color: theme.text,
                background: theme.sunken,
                ..Default::default()
            }
        } else {
            TextFormat {
                font_id: font.clone(),
                // No bold face is loaded: strong text stands out by colour.
                color: if span.link || span.strong {
                    theme.accent
                } else {
                    color
                },
                italics: span.emphasis,
                underline: if span.link {
                    egui::Stroke::new(1.0, theme.accent)
                } else {
                    egui::Stroke::NONE
                },
                ..Default::default()
            }
        };
        job.append(&span.text, 0.0, format);
    }
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answer_splits_into_its_blocks() {
        let answer = "## Plan\nFirst line\nsecond line\n\n- one\n  more of one\n- two\n3. three\n\n```rust\nlet x = 1;\n```\n> quoted\n---\nend";
        assert_eq!(
            blocks(answer),
            vec![
                Block::Heading(2, "Plan".into()),
                Block::Paragraph("First line\nsecond line".into()),
                Block::Item {
                    indent: 0,
                    marker: "•".into(),
                    text: "one\nmore of one".into()
                },
                Block::Item {
                    indent: 0,
                    marker: "•".into(),
                    text: "two".into()
                },
                Block::Item {
                    indent: 0,
                    marker: "3.".into(),
                    text: "three".into()
                },
                Block::Code {
                    lang: "rust".into(),
                    code: "let x = 1;".into()
                },
                Block::Quote("quoted".into()),
                Block::Rule,
                Block::Paragraph("end".into()),
            ]
        );
    }

    #[test]
    fn inline_marks_become_spans_and_snake_case_stays() {
        let s = spans("Use **object.add_cube** with `size=2` and *care*, see [docs](https://x).");
        let pick = |t: &str| s.iter().find(|s| s.text == t).cloned().unwrap();
        assert!(pick("object.add_cube").strong);
        assert!(pick("size=2").code);
        assert!(pick("care").emphasis);
        assert!(pick("docs").link);
        let text: String = s.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(text, "Use object.add_cube with size=2 and care, see docs.");
        assert_eq!(spans("a_b_c * 2")[0].text, "a_b_c * 2");
    }

    #[test]
    fn a_table_lines_up_without_its_rule() {
        let t = blocks("| name | 크기 |\n|---|---|\n| **Cube** | 2 |");
        assert_eq!(t, vec![Block::Table("name  크기\nCube  2".into())]);
    }
}
