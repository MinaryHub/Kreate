//! Wrapping prose to a width, for languages that do not use spaces.
//!
//! # Why this exists at all
//!
//! `epaint` picks its line breaks at whitespace, at CJK ideographs and at kana.
//! Its own source says what is missing:
//!
//! ```text
//! // TODO(bigfarts): Add support for Korean Hangul.
//! ```
//!
//! So a Korean sentence offers a layout engine *no* break opportunity — Korean
//! puts spaces between phrases, not between the places a line may break — and a
//! long one in a narrow properties panel overflows instead of wrapping. This is
//! not a prediction: it is what the character-class test in `epaint` does.
//!
//! # Why not a zero-width space
//!
//! The obvious fix is to insert U+200B at each break opportunity and let the
//! layout engine do the rest. It cannot work here: `epaint` tests
//! `char::is_whitespace()`, and no zero-width character has the Unicode
//! White_Space property. The hint would be invisible in both senses.
//!
//! # Why not `break_anywhere`
//!
//! `epaint` can break after any character, and for Korean alone that is nearly
//! right. But a modelling panel is full of mixed text — `Bevel 폭`, `UV 시임` —
//! and breaking anywhere splits the Latin words down the middle. Correct for one
//! language at the cost of the other is not a trade worth making when the real
//! rule is available.
//!
//! So the text is broken here, at UAX #14 opportunities, measured with the
//! actual font, and handed to the layout engine as lines it cannot argue with.

use egui::{Color32, FontId, Ui};

/// Wraps `text` into lines no wider than `width`, breaking per UAX #14.
///
/// `measure` returns the rendered width of a candidate line. Passed in rather
/// than taken from a `Ui` so that the interesting part — where the breaks land —
/// is testable without a window, which is where the bugs are.
///
/// Greedy, and measured per candidate line rather than by summing piece widths:
/// summing assumes the shaper adds no kerning across a boundary, which is true
/// for Hangul and not for Latin.
pub fn wrap_to_width(text: &str, width: f32, mut measure: impl FnMut(&str) -> f32) -> String {
    if text.is_empty() || width <= 0.0 {
        return text.to_string();
    }
    // Existing newlines are the author's, and are honoured: wrapping a
    // deliberately broken paragraph into one run would lose their structure.
    //
    // Behind a `dyn` boundary because the recursion is what needs it: passing
    // `&mut measure` down generically gives every level a new type, and the
    // compiler instantiates them until it gives up.
    let measure: &mut dyn FnMut(&str) -> f32 = &mut measure;
    if text.contains('\n') {
        return text
            .split('\n')
            .map(|line| wrap_one(line, width, measure))
            .collect::<Vec<_>>()
            .join("\n");
    }
    wrap_one(text, width, measure)
}

/// Wraps a single paragraph — one with no newlines of its own.
fn wrap_one(text: &str, width: f32, measure: &mut dyn FnMut(&str) -> f32) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut stops = kreate_i18n::korean::line_break_offsets(text);
    stops.push(text.len());

    let mut lines: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    // The furthest opportunity that still fit on the line being built.
    let mut last_fit: Option<usize> = None;

    while index < stops.len() {
        let stop = stops[index];
        if stop <= start {
            index += 1;
            continue;
        }
        if measure(&text[start..stop]) <= width {
            last_fit = Some(stop);
            index += 1;
            continue;
        }

        // Too wide. Break at the last opportunity that fit; with none, let this
        // piece overflow rather than loop — a single word longer than the panel
        // has to go somewhere, and cutting it mid-character is worse.
        //
        // Trimmed at the end only. A UAX #14 opportunity falls *after* the space
        // that created it, so a break never leaves one at the start of the next
        // line — only at the end of the one just finished.
        let brk = last_fit.unwrap_or(stop);
        lines.push(text[start..brk].trim_end());
        // `index` is deliberately not advanced: the same opportunity is
        // reconsidered against the new, shorter line. A forced break sets
        // `start` to `stop`, and the guard at the top of the loop then moves
        // past it — which is what makes this terminate without a special case.
        debug_assert!(brk > start, "every break has to make progress");
        start = brk;
        last_fit = None;
    }

    if start < text.len() {
        lines.push(text[start..].trim_end());
    }
    if lines.is_empty() {
        return String::new();
    }
    lines.join("\n")
}

/// Wraps prose to a width using the interface's own body font.
pub fn wrap(ui: &Ui, text: &str, width: f32) -> String {
    let font = egui::TextStyle::Body.resolve(ui.style());
    wrap_with_font(ui, text, width, &font)
}

/// The same, with an explicit font.
pub fn wrap_with_font(ui: &Ui, text: &str, width: f32, font: &FontId) -> String {
    wrap_to_width(text, width, |line| {
        ui.ctx().fonts_mut(|f| {
            f.layout_no_wrap(line.to_owned(), font.clone(), Color32::WHITE)
                .size()
                .x
        })
    })
}

/// A label wrapped to the space actually available.
///
/// What every piece of prose in a panel should use. `ui.label` alone relies on
/// the layout engine's own wrapping, which is where the Korean problem lives.
pub fn wrapped_label(ui: &mut Ui, text: &str) {
    let width = ui.available_width();
    let wrapped = wrap(ui, text, width);
    ui.label(wrapped);
}

/// Prose for a tooltip, wrapped to a comfortable reading width.
///
/// Tooltips are not constrained by a panel, so they need a width of their own —
/// long enough not to break a short sentence into a column, short enough to
/// read. `epaint` applies a default of its own but only wraps at whitespace,
/// which a Korean sentence does not offer.
pub fn wrap_tooltip(ui: &Ui, text: &str) -> String {
    wrap(ui, text, TOOLTIP_WIDTH)
}

/// Reading width for tooltip prose, in points.
pub const TOOLTIP_WIDTH: f32 = 320.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// A measure that gives every character the same width.
    ///
    /// Enough to test where the breaks land, which is the part that has nothing
    /// to do with fonts.
    fn per_char(width: f32) -> impl FnMut(&str) -> f32 {
        move |s: &str| s.chars().count() as f32 * width
    }

    /// Korean, with no spaces at any of its break points.
    const KOREAN: &str = "선택한면을돌출시킬거리입니다";

    #[test]
    fn korean_without_spaces_still_wraps() {
        // The whole reason this module exists. A layout engine that breaks only
        // at whitespace leaves this on one line and lets it overflow the panel.
        let wrapped = wrap_to_width(KOREAN, 5.0, per_char(1.0));
        let lines: Vec<&str> = wrapped.lines().collect();
        assert!(
            lines.len() >= 3,
            "expected several lines, got {lines:?} from {} characters",
            KOREAN.chars().count()
        );
        for line in &lines {
            assert!(
                line.chars().count() <= 5,
                "line {line:?} is wider than the limit"
            );
        }
    }

    #[test]
    fn wrapping_loses_no_text() {
        // A wrap that drops a character is worse than one that overflows.
        let wrapped = wrap_to_width(KOREAN, 4.0, per_char(1.0));
        let rejoined: String = wrapped.lines().collect();
        assert_eq!(rejoined, KOREAN);
    }

    #[test]
    fn latin_words_are_not_split() {
        // The reason `break_anywhere` is not the answer: it is nearly right for
        // Korean and wrong for everything else.
        let text = "Distance to extrude the selected faces";
        let wrapped = wrap_to_width(text, 12.0, per_char(1.0));
        for line in wrapped.lines() {
            for word in line.split_whitespace() {
                assert!(
                    text.split_whitespace().any(|w| w == word),
                    "{word:?} is not a whole word of the original"
                );
            }
        }
    }

    #[test]
    fn mixed_text_breaks_in_both_scripts() {
        // What a modelling panel is actually full of.
        let text = "Bevel 폭을 지정합니다 width in metres";
        let wrapped = wrap_to_width(text, 8.0, per_char(1.0));
        assert!(wrapped.lines().count() >= 3, "{wrapped:?}");
        let strip = |s: &str| -> String { s.chars().filter(|c| *c != '\n' && *c != ' ').collect() };
        assert_eq!(strip(&wrapped), strip(text), "characters went missing");
    }

    #[test]
    fn a_line_does_not_begin_with_a_space() {
        // The space that ended the previous line must not indent the next one.
        let wrapped = wrap_to_width("aaa bbb ccc ddd", 7.0, per_char(1.0));
        for line in wrapped.lines() {
            assert!(!line.starts_with(' '), "{line:?} starts with a space");
            assert!(!line.ends_with(' '), "{line:?} ends with a space");
        }
    }

    #[test]
    fn a_word_longer_than_the_width_overflows_rather_than_looping() {
        // A single unbreakable run wider than the panel has to go somewhere.
        // Cutting it mid-character would be worse, and looping would be fatal.
        let wrapped = wrap_to_width("supercalifragilistic", 5.0, per_char(1.0));
        assert_eq!(wrapped, "supercalifragilistic");
    }

    #[test]
    fn an_authored_newline_survives() {
        // Wrapping a deliberately broken paragraph into one run loses the
        // author's structure.
        let wrapped = wrap_to_width("first line\nsecond line", 100.0, per_char(1.0));
        assert_eq!(wrapped, "first line\nsecond line");
    }

    #[test]
    fn each_authored_line_is_wrapped_on_its_own() {
        let wrapped = wrap_to_width("aaa bbb\nccc ddd", 4.0, per_char(1.0));
        assert_eq!(wrapped, "aaa\nbbb\nccc\nddd");
    }

    #[test]
    fn an_empty_string_stays_empty() {
        assert_eq!(wrap_to_width("", 10.0, per_char(1.0)), "");
    }

    #[test]
    fn a_zero_width_returns_the_text_unchanged() {
        // A panel dragged shut has no width. Returning the text is better than
        // returning one character per line, and far better than not returning.
        assert_eq!(wrap_to_width(KOREAN, 0.0, per_char(1.0)), KOREAN);
    }

    #[test]
    fn text_that_already_fits_is_left_alone() {
        let text = "short enough";
        assert_eq!(wrap_to_width(text, 1000.0, per_char(1.0)), text);
    }

    #[test]
    fn breaks_land_on_character_boundaries() {
        // Hangul is three bytes per syllable. Slicing at a byte offset that is
        // not a boundary panics, and a wrap routine that panics on a tooltip is
        // a crash on hover.
        for width in 1..20 {
            let wrapped = wrap_to_width(KOREAN, width as f32, per_char(1.0));
            let rejoined: String = wrapped.lines().collect();
            assert_eq!(rejoined, KOREAN, "at width {width}");
        }
    }
}
