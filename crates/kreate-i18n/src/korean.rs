//! Korean-specific text handling.
//!
//! Two things about Korean cannot be expressed by a `printf`-style translation
//! system, and both of them are visible to a user immediately.
//!
//! **Particles depend on the preceding sound.** 은/는, 이/가, 을/를, 와/과 and
//! 으로/로 are chosen by whether the noun before them ends in a consonant
//! (받침). A translation that hardcodes one form produces text that reads as
//! machine output the moment the noun is a placeholder — and in a modelling tool
//! nearly every message has an object name in it.
//!
//! **Line breaking has no spaces to work with.** Korean prose does not put
//! spaces at the points where a line may break, so a layout engine that only
//! breaks on whitespace will overflow a narrow panel rather than wrap.
//!
//! Both are handled here rather than left to translators to work around.

use unicode_normalization::UnicodeNormalization as _;

/// First and last Hangul syllable code points in the precomposed block.
const HANGUL_BASE: u32 = 0xAC00;
const HANGUL_LAST: u32 = 0xD7A3;
/// Number of final-consonant (jongseong) slots per syllable, including "none".
const JONGSEONG_COUNT: u32 = 28;
/// Jongseong index of ㄹ, which several particles treat as a special case.
const JONGSEONG_RIEUL: u32 = 8;

/// The final consonant of a syllable, to the extent particles care about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Batchim {
    /// Ends in a vowel.
    None,
    /// Ends in ㄹ, which 으로/로 and 이랑/랑 treat like a vowel.
    Rieul,
    /// Ends in any other consonant.
    Other,
}

impl Batchim {
    pub fn is_present(self) -> bool {
        self != Batchim::None
    }
}

/// Determines the final consonant of the last meaningful character of `text`.
///
/// Handles precomposed Hangul, Latin letters and digits, because a modelling
/// tool's messages are full of names like `Cube.001` and `Mesh_2`.
pub fn batchim_of(text: &str) -> Batchim {
    // Trailing punctuation and closing brackets are not pronounced, so they do
    // not decide the particle. `"Cube (1)"` reads as "일".
    let last = text
        .chars()
        .rev()
        .find(|c| c.is_alphanumeric())
        .unwrap_or(' ');

    let cp = last as u32;
    if (HANGUL_BASE..=HANGUL_LAST).contains(&cp) {
        return match (cp - HANGUL_BASE) % JONGSEONG_COUNT {
            0 => Batchim::None,
            JONGSEONG_RIEUL => Batchim::Rieul,
            _ => Batchim::Other,
        };
    }

    // Digits are read aloud, and how they are read decides the particle.
    if let Some(d) = last.to_digit(10) {
        return match d {
            // 영, 삼, 육 end in ㅇ/ㅁ/ㄱ
            0 | 3 | 6 => Batchim::Other,
            // 일, 칠, 팔 end in ㄹ
            1 | 7 | 8 => Batchim::Rieul,
            // 이, 사, 오, 구 end in a vowel
            _ => Batchim::None,
        };
    }

    // Latin letters, read as their Korean names. Only the final sound matters.
    if last.is_ascii_alphabetic() {
        return match last.to_ascii_lowercase() {
            // 엘(ㄹ), 알(ㄹ)
            'l' | 'r' => Batchim::Rieul,
            // 엔, 엠, 빈(ㄴ/ㅁ), 케이크… the consonant-final names
            'n' | 'm' => Batchim::Other,
            // 에이, 비, 시, 디, 이, 에프… vowel-final
            _ => Batchim::None,
        };
    }

    // Unknown script: assume no 받침, which is the less jarring of the two
    // possible mistakes because it matches the more common form.
    Batchim::None
}

/// Chooses the correct form of a Korean particle for the noun before it.
///
/// `pair` is written `"<with-받침>/<without-받침>"`, matching how these are
/// conventionally written: `"은/는"`, `"이/가"`, `"을/를"`, `"과/와"`,
/// `"으로/로"`.
///
/// ㄹ is treated as a vowel for 으로/로 and 이랑/랑 — 파일**로**, not
/// 파일**으로** — and as a consonant everywhere else.
pub fn particle<'a>(noun: &str, pair: &'a str) -> &'a str {
    let (with, without) = match pair.split_once('/') {
        Some(p) => p,
        // A pair with no slash is not a choice; hand it back unchanged rather
        // than mangling a legitimate literal.
        None => return pair,
    };

    let batchim = batchim_of(noun);
    // The particles where ㄹ behaves like a vowel. Checked by the *without*
    // form, because that is the part that is stable across how a translator
    // spells the pair.
    let rieul_is_vowel = matches!(without, "로" | "랑");

    match batchim {
        Batchim::None => without,
        Batchim::Rieul if rieul_is_vowel => without,
        _ => with,
    }
}

/// Formats a noun followed by its particle, as one string.
pub fn with_particle(noun: &str, pair: &str) -> String {
    format!("{noun}{}", particle(noun, pair))
}

/// Normalises text to NFC.
///
/// Required on every text input path. Hangul arriving as decomposed jamo — from
/// a paste, a macOS filename, or some IME output — renders as separate marks
/// rather than as syllables in a layout engine that does not compose, and
/// compares unequal to the same word typed normally.
pub fn normalize(text: &str) -> String {
    text.nfc().collect()
}

/// Whether `text` is already NFC, so a caller can skip the allocation.
pub fn is_normalized(text: &str) -> bool {
    unicode_normalization::is_nfc(text)
}

/// Byte offsets at which a line may be broken, per UAX #14.
///
/// Needed because Korean has no spaces at its break points: a layout engine that
/// only breaks on whitespace overflows a narrow panel instead of wrapping. The
/// offsets are the *starts* of the pieces after each opportunity.
pub fn line_break_offsets(text: &str) -> Vec<usize> {
    unicode_linebreak::linebreaks(text)
        .filter(|(offset, _)| *offset < text.len())
        .map(|(offset, _)| offset)
        .collect()
}

/// Wraps `text` to at most `columns` display cells per line.
///
/// A deliberately simple width model — CJK counts as two cells, everything else
/// as one — because the real measurement happens in the layout engine with the
/// actual font. This exists so the *break opportunities* are correct; a UI that
/// asks its text shaper to break on whitespace will get Korean wrong no matter
/// how good its metrics are.
pub fn wrap(text: &str, columns: usize) -> Vec<String> {
    if columns == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut line_start = 0usize;
    let mut last_opportunity: Option<usize> = None;

    for (offset, _) in unicode_linebreak::linebreaks(text) {
        let piece = &text[line_start..offset];
        if display_width(piece) <= columns {
            last_opportunity = Some(offset);
            continue;
        }
        // This piece is too wide. Break at the previous opportunity if there was
        // one; otherwise emit the overlong piece rather than looping.
        let brk = last_opportunity
            .filter(|b| *b > line_start)
            .unwrap_or(offset);
        lines.push(text[line_start..brk].trim_end().to_string());
        line_start = brk;
        last_opportunity = Some(offset);
    }
    if line_start < text.len() {
        lines.push(text[line_start..].trim_end().to_string());
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Display cells a string occupies, counting CJK as two.
pub fn display_width(text: &str) -> usize {
    text.chars()
        .map(|c| {
            let cp = c as u32;
            let wide = (0x1100..=0x115F).contains(&cp)          // Hangul jamo
                || (0x2E80..=0xA4CF).contains(&cp)              // CJK radicals..Yi
                || (HANGUL_BASE..=HANGUL_LAST).contains(&cp)    // Hangul syllables
                || (0xF900..=0xFAFF).contains(&cp)              // CJK compatibility
                || (0xFF00..=0xFF60).contains(&cp); // fullwidth forms
            if wide {
                2
            } else {
                1
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchim_is_detected_in_hangul() {
        assert_eq!(batchim_of("큐브"), Batchim::None); // 브: vowel-final
        assert_eq!(batchim_of("원통"), Batchim::Other); // 통: ㅇ
        assert_eq!(batchim_of("파일"), Batchim::Rieul); // 일: ㄹ
        assert_eq!(batchim_of("객체"), Batchim::None);
        assert_eq!(batchim_of("점"), Batchim::Other); // ㅁ
    }

    #[test]
    fn object_particles_match_the_noun() {
        // The whole reason this module exists: a hardcoded particle reads as
        // machine output as soon as the noun is a placeholder.
        assert_eq!(particle("큐브", "을/를"), "를");
        assert_eq!(particle("원통", "을/를"), "을");
        assert_eq!(particle("큐브", "은/는"), "는");
        assert_eq!(particle("원통", "은/는"), "은");
        assert_eq!(particle("큐브", "이/가"), "가");
        assert_eq!(particle("원통", "이/가"), "이");
    }

    #[test]
    fn rieul_counts_as_a_vowel_for_ro_but_not_for_eul() {
        // 파일로, not 파일으로. But 파일을, not 파일를.
        assert_eq!(particle("파일", "으로/로"), "로");
        assert_eq!(particle("파일", "을/를"), "을");
        // A different final consonant does take 으로.
        assert_eq!(particle("원통", "으로/로"), "으로");
        // And a vowel-final noun takes the plain form.
        assert_eq!(particle("큐브", "으로/로"), "로");
    }

    #[test]
    fn digits_are_read_aloud_to_pick_the_particle() {
        // "Cube.001" is read as 공공일, ending in ㄹ.
        assert_eq!(particle("Cube.001", "으로/로"), "로");
        assert_eq!(particle("Cube.001", "을/를"), "을");
        // 2 is 이, vowel-final.
        assert_eq!(particle("Mesh_2", "을/를"), "를");
        // 3 is 삼, ㅁ-final.
        assert_eq!(particle("Mesh_3", "을/를"), "을");
        // 6 is 육, ㄱ-final.
        assert_eq!(particle("LOD6", "이/가"), "이");
    }

    #[test]
    fn latin_names_are_read_by_their_korean_pronunciation() {
        // "Cube" ends in 브, vowel-final.
        assert_eq!(particle("Cube", "을/를"), "를");
        // "Cylinder" ends in 알(ㄹ)? No — 'r' is read 알, so ㄹ.
        assert_eq!(particle("Cylinder", "으로/로"), "로");
        // 'n' is 엔, consonant-final.
        assert_eq!(particle("Plain", "을/를"), "을");
    }

    #[test]
    fn trailing_punctuation_does_not_decide_the_particle() {
        // "Cube (1)" is read ending in 일, so ㄹ.
        assert_eq!(particle("Cube (1)", "을/를"), "을");
        assert_eq!(particle("큐브!", "을/를"), "를");
    }

    #[test]
    fn a_pair_without_a_slash_is_returned_unchanged() {
        // A translator writing a literal must not have it mangled.
        assert_eq!(particle("큐브", "입니다"), "입니다");
    }

    #[test]
    fn with_particle_joins_noun_and_particle() {
        assert_eq!(with_particle("큐브", "을/를"), "큐브를");
        assert_eq!(with_particle("원통", "을/를"), "원통을");
    }

    #[test]
    fn empty_and_symbol_only_input_does_not_panic() {
        assert_eq!(batchim_of(""), Batchim::None);
        assert_eq!(batchim_of("!!!"), Batchim::None);
        assert_eq!(particle("", "을/를"), "를");
    }

    // --- normalisation ----------------------------------------------------

    #[test]
    fn decomposed_hangul_is_recomposed() {
        // What a paste from macOS or some IMEs delivers. Without this it renders
        // as separate jamo and compares unequal to the same word typed normally.
        let decomposed = "\u{1100}\u{1161}"; // ㄱ + ㅏ
        assert_eq!(decomposed.chars().count(), 2);
        let normalized = normalize(decomposed);
        assert_eq!(normalized, "가");
        assert_eq!(normalized.chars().count(), 1);
        assert!(!is_normalized(decomposed));
        assert!(is_normalized(&normalized));
    }

    #[test]
    fn normalisation_leaves_already_composed_text_alone() {
        for s in ["큐브", "Cube", "Cube.001", ""] {
            assert_eq!(normalize(s), s);
            assert!(is_normalized(s));
        }
    }

    #[test]
    fn a_decomposed_name_compares_equal_after_normalising() {
        let typed = "가";
        let pasted = "\u{1100}\u{1161}";
        assert_ne!(typed, pasted, "the raw strings differ");
        assert_eq!(normalize(typed), normalize(pasted));
    }

    // --- line breaking ----------------------------------------------------

    #[test]
    fn korean_has_break_opportunities_without_spaces() {
        // The failure this exists to prevent: a layout engine that breaks only on
        // whitespace finds nowhere to break and overflows the panel.
        let text = "선택된정점을이동합니다";
        assert!(
            !text.contains(' '),
            "the point is that there are no spaces to break on"
        );
        assert!(
            line_break_offsets(text).len() > 3,
            "expected several break opportunities, found {:?}",
            line_break_offsets(text)
        );
    }

    #[test]
    fn korean_wraps_instead_of_overflowing() {
        let text = "선택된정점을이동합니다";
        let lines = wrap(text, 8);
        assert!(lines.len() > 1, "should have wrapped, got {lines:?}");
        for line in &lines {
            assert!(
                display_width(line) <= 8,
                "line {line:?} is {} cells wide",
                display_width(line)
            );
        }
        // Nothing may be lost or duplicated by wrapping.
        assert_eq!(lines.concat(), text);
    }

    #[test]
    fn english_still_wraps_on_words() {
        let lines = wrap("move the selected vertices along the axis", 16);
        assert!(lines.len() > 1);
        for line in &lines {
            assert!(display_width(line) <= 16, "{line:?}");
        }
        // Words must not be split mid-way when there is a space to use.
        assert!(lines.iter().all(|l| !l.starts_with(' ')));
    }

    #[test]
    fn a_word_wider_than_the_column_is_emitted_rather_than_looping() {
        let lines = wrap("aaaaaaaaaaaaaaaaaaaaaaaa", 5);
        assert_eq!(lines.len(), 1, "no break opportunity exists inside it");
        assert_eq!(lines[0], "aaaaaaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn cjk_counts_as_two_cells() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("큐브"), 4);
        assert_eq!(display_width("큐브abc"), 7);
    }

    #[test]
    fn wrapping_handles_degenerate_input() {
        assert_eq!(wrap("", 10), vec![""]);
        assert_eq!(
            wrap("abc", 0),
            vec!["abc"],
            "a zero column budget cannot wrap"
        );
    }
}
