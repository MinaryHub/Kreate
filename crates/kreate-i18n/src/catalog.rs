//! The translation catalog.
//!
//! # Why Fluent
//!
//! Not `printf`-style formatting, and not a flat key-to-string map. Korean needs
//! more than either can express: word order differs from English, so a
//! translator must be able to *reorder* the placeables in a message, and
//! particles depend on the noun before them, so the message itself has to make
//! that choice. Fluent gives both — reordering by construction, and the
//! particle choice through the custom [`PARTICLE`] function registered here.
//!
//! # Keys, not strings
//!
//! Every user-visible string goes through a key. The fallback when a key is
//! missing is the key itself rather than a panic or an empty string, so a
//! partially translated build is usable and a missing key is obvious on screen
//! rather than silent. A CI gate (see `kreate-testkit`) forbids bare string
//! literals reaching UI label calls, because retrofitting i18n once a few
//! hundred panels exist is a classic project-killer.
//!
//! # Three separate toggles
//!
//! UI text, tooltips and new-data names are translated independently. Many Korean
//! professionals want an English interface with Korean tooltips: the menu names
//! are what they read in tutorials and remember as shortcuts, while the
//! explanations are what they actually want in their own language.

use std::borrow::Cow;

use fluent_bundle::{FluentArgs, FluentBundle, FluentResource, FluentValue};
use unic_langid::LanguageIdentifier;

use crate::korean;

/// Which categories of text to translate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranslationOptions {
    /// Menus, panel titles, operator names, button labels.
    pub ui: bool,
    /// Tooltips and longer descriptions.
    pub tooltips: bool,
    /// Names given to newly created objects and datablocks.
    ///
    /// Off by default: a translated default name ends up inside saved files and
    /// exported assets, where it is a liability rather than a convenience.
    pub new_data_names: bool,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        Self {
            ui: true,
            tooltips: true,
            new_data_names: false,
        }
    }
}

/// What kind of text is being asked for, so the toggles can apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextKind {
    Ui,
    Tooltip,
    NewDataName,
    /// Always translated regardless of the toggles: errors and validation
    /// messages are the text a user most needs in their own language.
    Message,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("`{0}` is not a valid language identifier")]
    BadLocale(String),
    #[error("failed to parse the translation resource: {0}")]
    Parse(String),
}

/// One locale's messages.
struct Locale {
    id: LanguageIdentifier,
    bundle: FluentBundle<FluentResource>,
    /// Message ids, captured while parsing.
    ///
    /// `fluent-bundle` offers no way to enumerate a bundle's messages, so the
    /// ids are taken from the resource's syntax tree before it is handed over.
    /// Without them there is no way to answer "which keys is this translation
    /// missing", and a translation that cannot be audited quietly rots.
    keys: Vec<String>,
}

/// A stack of locales, most preferred first.
pub struct Catalog {
    locales: Vec<Locale>,
    pub options: TranslationOptions,
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            locales: Vec::new(),
            options: TranslationOptions::default(),
        }
    }

    /// Adds a resource for a locale, appended as the least preferred.
    ///
    /// Call order is the fallback order: add the user's locale first and the
    /// source language last, so an untranslated key falls back to English rather
    /// than to the raw key.
    pub fn add(&mut self, locale: &str, ftl: &str) -> Result<(), CatalogError> {
        let lang: LanguageIdentifier = locale
            .parse()
            .map_err(|_| CatalogError::BadLocale(locale.to_string()))?;
        let resource = FluentResource::try_new(ftl.to_string())
            .map_err(|(_, errs)| CatalogError::Parse(format!("{errs:?}")))?;
        let keys = message_ids(&resource);

        // Isolating marks wrap interpolated values in Unicode bidi controls.
        // Correct for mixed-direction text, but they show up as stray characters
        // in a terminal and in a texture atlas that has not been built for them,
        // and Kreate ships no right-to-left locale yet.
        let mut bundle = FluentBundle::new(vec![lang.clone()]);
        bundle.set_use_isolating(false);
        register_functions(&mut bundle);

        bundle
            .add_resource(resource)
            .map_err(|errs| CatalogError::Parse(format!("{errs:?}")))?;
        self.locales.push(Locale {
            id: lang,
            bundle,
            keys,
        });
        Ok(())
    }

    pub fn locales(&self) -> impl Iterator<Item = &LanguageIdentifier> {
        self.locales.iter().map(|l| &l.id)
    }

    pub fn is_empty(&self) -> bool {
        self.locales.is_empty()
    }

    /// Looks up a key, falling back to the key itself.
    pub fn get(&self, key: &str) -> String {
        self.format(key, TextKind::Ui, &[])
    }

    /// Looks up a key with arguments.
    pub fn get_args(&self, key: &str, args: &[(&str, &str)]) -> String {
        self.format(key, TextKind::Ui, args)
    }

    /// Looks up a key of a specific kind, honouring the toggles.
    pub fn format(&self, key: &str, kind: TextKind, args: &[(&str, &str)]) -> String {
        if !self.wants(kind) {
            return key.to_string();
        }
        for locale in &self.locales {
            if let Some(text) = format_from(&locale.bundle, key, args) {
                return text;
            }
        }
        // The key itself. Visible and obviously wrong on screen, which is what a
        // missing translation should be — an empty string or a panic is worse.
        key.to_string()
    }

    fn wants(&self, kind: TextKind) -> bool {
        match kind {
            TextKind::Ui => self.options.ui,
            TextKind::Tooltip => self.options.tooltips,
            TextKind::NewDataName => self.options.new_data_names,
            TextKind::Message => true,
        }
    }

    /// Whether a key resolves in any loaded bundle. For a coverage report rather
    /// than for control flow.
    pub fn has(&self, key: &str) -> bool {
        self.locales.iter().any(|l| l.bundle.has_message(key))
    }

    /// Keys present in the first bundle but missing from a later one.
    ///
    /// Turns "is the Korean translation complete" into a question with an answer,
    /// which is what keeps a translation from silently rotting as keys are added.
    pub fn missing_in(&self, locale: &str) -> Vec<String> {
        let Ok(wanted) = locale.parse::<LanguageIdentifier>() else {
            return Vec::new();
        };
        let Some(target) = self.locales.iter().find(|l| l.id == wanted) else {
            return Vec::new();
        };
        let mut missing: Vec<String> = self
            .locales
            .iter()
            .flat_map(|l| l.keys.iter().cloned())
            .filter(|k| !target.bundle.has_message(k))
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Message ids declared by a resource, read from its syntax tree.
fn message_ids(resource: &FluentResource) -> Vec<String> {
    resource
        .entries()
        .filter_map(|entry| match entry {
            fluent_syntax::ast::Entry::Message(m) => Some(m.id.name.to_string()),
            _ => None,
        })
        .collect()
}

fn format_from(
    bundle: &FluentBundle<FluentResource>,
    key: &str,
    args: &[(&str, &str)],
) -> Option<String> {
    let message = bundle.get_message(key)?;
    let pattern = message.value()?;
    let mut fluent_args = FluentArgs::new();
    for (name, value) in args {
        fluent_args.set(*name, FluentValue::from(*value));
    }
    let mut errors = Vec::new();
    let text = bundle.format_pattern(pattern, Some(&fluent_args), &mut errors);
    if !errors.is_empty() {
        // A formatting error means the message and the call site disagree — a
        // missing argument, usually. Report it rather than shipping mangled text.
        tracing::warn!(key, ?errors, "translation formatting problem");
    }
    Some(text.into_owned())
}

/// Registers Kreate's Fluent functions on a bundle.
fn register_functions(bundle: &mut FluentBundle<FluentResource>) {
    // PARTICLE($noun, "을/를") -> the correct form for that noun.
    //
    // Twenty lines, and it is the difference between text that reads as
    // translated and text that reads as written by a Korean speaker. Without it
    // every message containing an object name is visibly wrong half the time.
    let _ = bundle.add_function("PARTICLE", |positional, _named| {
        let noun = match positional.first() {
            Some(FluentValue::String(s)) => s.as_ref().to_string(),
            Some(FluentValue::Number(n)) => n.value.to_string(),
            _ => return FluentValue::Error,
        };
        let pair = match positional.get(1) {
            Some(FluentValue::String(s)) => s.as_ref().to_string(),
            _ => return FluentValue::Error,
        };
        FluentValue::String(Cow::Owned(korean::particle(&noun, &pair).to_string()))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const EN: &str = r#"
op-object-delete = Delete Objects
deleted-object = Deleted { $name }.
selected-count = { $count } selected
"#;

    // Note the reordering: Korean puts the object before the verb, which a
    // printf-style system cannot express without renumbering arguments.
    const KO: &str = r#"
op-object-delete = 오브젝트 삭제
deleted-object = { $name }{ PARTICLE($name, "을/를") } 삭제했습니다.
"#;

    fn catalog() -> Catalog {
        let mut c = Catalog::new();
        c.add("ko-KR", KO).expect("ko");
        c.add("en-US", EN).expect("en");
        c
    }

    #[test]
    fn a_key_resolves_from_the_preferred_locale() {
        assert_eq!(catalog().get("op-object-delete"), "오브젝트 삭제");
    }

    #[test]
    fn an_untranslated_key_falls_back_to_the_next_locale() {
        // `selected-count` exists only in English. Falling back to English is far
        // better than showing the raw key.
        let text = catalog().get_args("selected-count", &[("count", "3")]);
        assert_eq!(text, "3 selected");
    }

    #[test]
    fn a_missing_key_shows_itself_rather_than_vanishing() {
        // Visible and obviously wrong beats silent: an empty label looks like a
        // layout bug and gets misdiagnosed for weeks.
        assert_eq!(catalog().get("no-such-key"), "no-such-key");
    }

    #[test]
    fn korean_particles_are_chosen_by_the_interpolated_noun() {
        // The payoff. The same message, two nouns, two different particles —
        // which no amount of translator care could achieve with printf.
        let c = catalog();
        assert_eq!(
            c.get_args("deleted-object", &[("name", "큐브")]),
            "큐브를 삭제했습니다."
        );
        assert_eq!(
            c.get_args("deleted-object", &[("name", "원통")]),
            "원통을 삭제했습니다."
        );
    }

    #[test]
    fn particles_work_for_latin_and_numeric_names() {
        let c = catalog();
        // "Cube" ends in 브, so 를.
        assert_eq!(
            c.get_args("deleted-object", &[("name", "Cube")]),
            "Cube를 삭제했습니다."
        );
        // "Cube.001" is read ending in 일, so 을.
        assert_eq!(
            c.get_args("deleted-object", &[("name", "Cube.001")]),
            "Cube.001을 삭제했습니다."
        );
    }

    #[test]
    fn the_ui_toggle_turns_off_ui_translation_only() {
        // The arrangement Korean professionals frequently ask for: English menus,
        // Korean explanations.
        let mut c = catalog();
        c.options.ui = false;
        assert_eq!(
            c.format("op-object-delete", TextKind::Ui, &[]),
            "op-object-delete",
            "UI translation is off, so the key comes back"
        );
        assert_eq!(
            c.format("op-object-delete", TextKind::Tooltip, &[]),
            "오브젝트 삭제",
            "tooltips are still translated"
        );
    }

    #[test]
    fn messages_are_translated_regardless_of_the_toggles() {
        // Errors and validation text are what a user most needs in their own
        // language, so they do not follow the UI toggle.
        let mut c = catalog();
        c.options.ui = false;
        c.options.tooltips = false;
        assert_eq!(
            c.format("op-object-delete", TextKind::Message, &[]),
            "오브젝트 삭제"
        );
    }

    #[test]
    fn new_data_names_are_not_translated_by_default() {
        // A translated default name ends up inside saved files and exported
        // assets, where it is a liability.
        let c = catalog();
        assert!(!c.options.new_data_names);
        assert_eq!(
            c.format("op-object-delete", TextKind::NewDataName, &[]),
            "op-object-delete"
        );
    }

    #[test]
    fn missing_keys_can_be_reported_per_locale() {
        // Turns "is the Korean translation complete" into a question with an
        // answer, so it cannot rot silently as keys are added.
        let c = catalog();
        assert_eq!(c.missing_in("ko-KR"), vec!["selected-count"]);
        assert!(c.missing_in("en-US").is_empty());
    }

    #[test]
    fn an_invalid_locale_is_rejected_with_a_clear_error() {
        let mut c = Catalog::new();
        assert!(matches!(
            c.add("not a locale!!", EN),
            Err(CatalogError::BadLocale(_))
        ));
    }

    #[test]
    fn a_malformed_resource_is_rejected_rather_than_partially_loaded() {
        let mut c = Catalog::new();
        assert!(c.add("en-US", "= this is not valid ftl").is_err());
        assert!(c.is_empty(), "nothing should have been loaded");
    }

    #[test]
    fn an_empty_catalog_returns_keys_unchanged() {
        // The state at startup before any resource is loaded; the application
        // must be usable, in English-ish key form, rather than blank.
        let c = Catalog::new();
        assert!(c.is_empty());
        assert_eq!(c.get("op-object-delete"), "op-object-delete");
    }

    #[test]
    fn interpolated_values_carry_no_bidi_control_characters() {
        // Fluent wraps placeables in isolating marks by default. Correct for
        // mixed-direction text, but they render as stray boxes in a terminal and
        // in an atlas built only for Latin and Hangul.
        let text = catalog().get_args("deleted-object", &[("name", "Cube")]);
        assert!(
            !text.chars().any(|c| matches!(c, '\u{2068}' | '\u{2069}')),
            "found isolating marks in {text:?}"
        );
    }

    #[test]
    fn locales_are_listed_in_preference_order() {
        let c = catalog();
        let names: Vec<String> = c.locales().map(|l| l.to_string()).collect();
        assert_eq!(names, vec!["ko-KR", "en-US"]);
    }
}
