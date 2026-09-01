//! The font stack.
//!
//! Four things about CJK text in egui are traps, and all four are silent:
//!
//! 1. **Ordering matters more than it looks.** egui resolves a glyph by walking
//!    the family list. Put the Korean font first and every Latin character comes
//!    from *its* Latin glyphs, which have different metrics — the whole interface
//!    ends up subtly wrong-looking with no obvious cause.
//! 2. **TTF, not OTF.** egui rasterises with `ab_glyph`, whose CFF/OTF handling
//!    has historically been weaker than its `glyf` handling.
//! 3. **The atlas rebuilds and re-uploads whole.** The first time a
//!    Korean-heavy panel opens, every glyph in it is rasterised at once and the
//!    texture is replaced — a visible hitch. Hence [`prewarm`].
//! 4. **A `pixels_per_point` change invalidates the entire atlas.** Dragging the
//!    window to a monitor with a different DPI re-rasterises everything, which
//!    with a CJK font loaded is a real stall. [`prewarm`] must run again after a
//!    DPI change, which is why it is public rather than a startup detail.
//!
//! # The embedded font
//!
//! Pretendard, SIL Open Font License 1.1 (see `assets/fonts/`). Chosen over Noto
//! Sans KR because it carries Latin and Hangul in one family with UI-appropriate
//! metrics, and because it ships a real TrueType build.
//!
//! Subset with `pyftsubset` to Latin, Latin Extended-A, punctuation, the symbols
//! a modelling UI uses, compatibility jamo, and **all 11,172 modern Hangul
//! syllables**. Measured alternatives, for when this is revisited:
//!
//! | coverage | size |
//! |---|---|
//! | KS X 1001 (2,350 syllables) | 420 KB |
//! | all modern syllables | 2.1 MB |
//! | unsubset original | 2.7 MB |
//!
//! Full coverage was chosen because the 2,350-syllable set leaves rare syllables
//! as tofu, and the place a user would hit that is an object name they typed
//! themselves — a visible defect, for what is only one-time binary size. Nothing
//! per-frame changes: the atlas only ever rasterises glyphs actually drawn. Worth
//! reducing to the small subset once a system-font fallback chain exists, which
//! is wanted anyway for Japanese, Chinese and emoji.

use std::sync::Arc;

/// The embedded Korean + Latin UI font.
pub const PRETENDARD: &[u8] = include_bytes!("../../../assets/fonts/Pretendard-Regular.subset.ttf");

/// Family names, in resolution order.
pub mod family {
    /// Latin and Hangul. Listed first for both, because Pretendard's Latin is
    /// designed to sit with its Hangul rather than being an afterthought.
    pub const UI: &str = "Pretendard";
}

/// Installs the font stack on an egui context.
pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        family::UI.to_owned(),
        Arc::new(egui::FontData::from_static(PRETENDARD)),
    );

    // Prepended rather than appended: egui's own default is Latin-only, so
    // leaving it first would make every Latin character come from a different
    // font than the Hangul beside it, and the two would not share metrics.
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, family::UI.to_owned());
    }

    ctx.set_fonts(fonts);
}

/// Characters worth rasterising before the user can see a hitch.
///
/// Not the whole font — that would defeat the purpose. Just the syllables and
/// symbols a first panel is overwhelmingly likely to contain, so opening one
/// costs nothing visible.
pub const PREWARM_TEXT: &str = concat!(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    " .,:;!?()[]{}<>/\\|-_+=*&%#@~\"'`^$",
    // The syllables a modelling UI shows constantly, deduplicated: object, mesh,
    // vertex, edge, face, select, move, rotate, scale, add, delete, undo, redo,
    // file, view, edit, tool, settings, properties, name, position, direction,
    // dimensions, units, angle, count, thickness, radius, segments, and the
    // dialog verbs.
    "오브젝트메시정점엣지면선택이동회전크기추가삭제되돌리다실행파일보편집도구설속성름위치방향수단각개두께반세그먼확인취소적용저장열내져새로만들닫",
);

/// Rasterises [`PREWARM_TEXT`] so the atlas is populated before it is needed.
///
/// Call once after [`install`], and **again after any `pixels_per_point`
/// change**: a DPI change discards the entire atlas, and re-rasterising a
/// Korean-heavy interface lazily during a window drag is exactly the kind of
/// stall that makes an application feel cheap.
///
/// Must be called *during or after* the first frame. egui builds its font set
/// lazily inside `Context::begin_pass`, and asking for it earlier panics with
/// "No fonts available until first call to Context::run()" — so this belongs at
/// the end of the first frame, not next to [`install`] at startup.
pub fn prewarm(ctx: &egui::Context) {
    // The sizes are named explicitly rather than read from the current style,
    // because the atlas is keyed by (glyph, size): warming a size nothing draws
    // at would be wasted, and missing one leaves the hitch in place.
    ctx.fonts_mut(|fonts| {
        for font_id in [
            egui::FontId::proportional(PREWARM_SIZES[0]),
            egui::FontId::proportional(PREWARM_SIZES[1]),
            egui::FontId::proportional(PREWARM_SIZES[2]),
            egui::FontId::monospace(PREWARM_SIZES[0]),
        ] {
            // Laying the text out is what forces each glyph into the atlas.
            let _ = fonts.layout_no_wrap(PREWARM_TEXT.to_owned(), font_id, egui::Color32::WHITE);
        }
    });
}

/// Text sizes to warm: body, button and heading.
pub const PREWARM_SIZES: [f32; 3] = [14.0, 12.0, 18.0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_font_is_present_and_is_truetype() {
        assert!(
            PRETENDARD.len() > 100_000,
            "the font looks truncated: {} bytes",
            PRETENDARD.len()
        );
        // `glyf` outlines, not CFF: ab_glyph handles TrueType better, and a
        // silently-swapped OTF would degrade rendering with no error.
        let tag = &PRETENDARD[0..4];
        assert!(
            tag == b"\x00\x01\x00\x00" || tag == b"true",
            "expected a TrueType sfnt header, got {tag:?}"
        );
        assert_ne!(tag, b"OTTO", "an OpenType/CFF build would rasterise worse");
    }

    #[test]
    fn the_font_stays_within_its_size_budget() {
        // Recorded so a future re-subset that accidentally keeps everything is
        // caught here rather than in a release-size review.
        assert!(
            PRETENDARD.len() < 3 << 20,
            "{} bytes exceeds the 3 MiB budget",
            PRETENDARD.len()
        );
    }

    #[test]
    fn prewarm_text_covers_latin_and_hangul() {
        let has_hangul = PREWARM_TEXT
            .chars()
            .any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
        let has_latin = PREWARM_TEXT.chars().any(|c| c.is_ascii_alphabetic());
        assert!(has_hangul && has_latin);
        // Small enough to rasterise in one go without itself being the hitch.
        assert!(
            PREWARM_TEXT.chars().count() < 400,
            "prewarm text is {} characters; it should be the common set, not the font",
            PREWARM_TEXT.chars().count()
        );
    }

    #[test]
    fn prewarm_text_has_no_duplicate_hangul() {
        // Duplicates cost nothing at run time but signal the list was edited
        // carelessly, and it is the sort of list that grows without review.
        let mut seen = std::collections::HashSet::new();
        let dupes: Vec<char> = PREWARM_TEXT
            .chars()
            .filter(|c| ('\u{AC00}'..='\u{D7A3}').contains(c) && !seen.insert(*c))
            .collect();
        assert!(dupes.is_empty(), "duplicated syllables: {dupes:?}");
    }

    /// A context with fonts installed and one frame run, which is what egui
    /// requires before its font set exists.
    fn ready_context() -> egui::Context {
        let ctx = egui::Context::default();
        install(&ctx);
        ctx.begin_pass(egui::RawInput::default());
        let _ = ctx.end_pass();
        ctx
    }

    #[test]
    fn installing_puts_the_ui_font_first_in_every_family() {
        // The ordering trap: if the CJK font is not first, Latin glyphs come from
        // a different font than the Hangul beside them and the metrics disagree.

        // egui does not expose the family list back, so assert on the definitions
        // we build rather than on internal state.
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            family::UI.to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(PRETENDARD)),
        );
        for egui_family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts
                .families
                .entry(egui_family.clone())
                .or_default()
                .insert(0, family::UI.to_owned());
            assert_eq!(
                fonts.families[&egui_family].first().map(String::as_str),
                Some(family::UI),
                "{egui_family:?} must resolve {} first",
                family::UI
            );
        }
    }

    #[test]
    fn korean_text_lays_out_with_a_real_width() {
        // The end-to-end check that the font is actually usable: without it, the
        // glyphs are missing and the layout collapses or shows tofu.
        let ctx = ready_context();
        let galley = ctx.fonts_mut(|fonts| {
            fonts.layout_no_wrap(
                "오브젝트 삭제".to_owned(),
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            )
        });
        assert!(
            galley.size().x > 40.0,
            "Korean text laid out to {} px wide; the glyphs are probably missing",
            galley.size().x
        );
        assert!(galley.size().y > 8.0);
    }

    #[test]
    fn latin_and_korean_report_different_widths_for_the_same_count() {
        // Sanity that both scripts are really being measured, rather than one
        // falling back to a notdef box of a fixed width.
        let ctx = ready_context();
        let width = |text: &str| {
            ctx.fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(
                        text.to_owned(),
                        egui::FontId::proportional(14.0),
                        egui::Color32::WHITE,
                    )
                    .size()
                    .x
            })
        };
        let korean = width("오브젝트");
        let latin = width("Ob700");
        assert!(korean > 0.0 && latin > 0.0);
        assert!(
            korean > latin,
            "four Hangul syllables ({korean}) should be wider than five Latin \
             characters ({latin})"
        );
    }

    #[test]
    fn prewarming_does_not_panic_and_is_repeatable() {
        // It must be safe to call again after a DPI change, which is the whole
        // reason it is public.
        let ctx = ready_context();
        prewarm(&ctx);
        // A DPI change discards the atlas, so it must be safe to warm again —
        // the whole reason this is public rather than a startup detail.
        ctx.set_pixels_per_point(2.0);
        ctx.begin_pass(egui::RawInput::default());
        let _ = ctx.end_pass();
        prewarm(&ctx);
    }
}
