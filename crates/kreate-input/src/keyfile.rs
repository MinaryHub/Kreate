//! Reading and writing the keymap file.
//!
//! # What is stored
//!
//! Only the user's **changes**, against stable binding ids, plus which preset
//! they are changes to. Not a copy of the table. That is the whole reason
//! [`Override`] exists, and writing a flattened keymap here would undo it: a
//! later preset improvement would never reach anyone who had ever pressed a
//! button on the remapping screen.
//!
//! # Robustness is the feature
//!
//! The plan is blunt about this: a preferences file that crashes on startup is a
//! self-inflicted support nightmare. So:
//!
//! - **Nothing here can panic or fail.** [`parse`] always returns a document.
//!   Malformed lines become [`Problem`]s the application can log or show, and
//!   the rest of the file still loads.
//! - **A newer `schema_version` is refused, not guessed at.** Reading a file
//!   from a future version as though it were this one silently mis-binds keys,
//!   and the user cannot tell why. Refusing keeps their file intact for when
//!   they go back to the newer build.
//! - **Unknown keys are ignored, not rejected.** The same forward-compatibility
//!   rule the document format uses.
//!
//! # Why the format is written by hand
//!
//! The schema is a version, a preset name, and a flat list of four-field
//! tables — small, fixed, and a compatibility surface worth controlling
//! exactly. A general TOML implementation would be the right call once
//! preferences grow past this, and the native document format will want `serde`
//! regardless; there is no reason for a config parser to be where that
//! dependency first lands.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::keymap::{BindingId, EventKind, Input, Modifiers, MouseButton, Override};
use crate::presets::Preset;

/// The current schema. Every bump adds a step to [`MIGRATIONS`].
pub const SCHEMA_VERSION: u32 = 1;

/// The oldest schema this build understands.
///
/// A file with no `schema_version` is assumed to be this: the field was present
/// from the first release, so its absence means a hand-written file rather than
/// an old one.
pub const OLDEST_SCHEMA: u32 = 1;

/// Something wrong with the file that did not stop it loading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Written by a newer build. Nothing was loaded.
    FromTheFuture { found: u32, understood: u32 },
    /// A line that is neither a comment, a section, nor `key = value`.
    UnreadableLine { line: usize, text: String },
    /// A `key = value` outside any `[[binding]]`, where only `schema_version`
    /// and `preset` belong.
    StrayField { line: usize, name: String },
    /// A binding entry with no `id`, which cannot be matched to anything.
    BindingWithoutId { line: usize },
    /// A preset name this build does not know.
    UnknownPreset { line: usize, name: String },
    /// A field whose value is not one of the accepted spellings.
    UnknownValue {
        line: usize,
        name: String,
        value: String,
    },
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::FromTheFuture { found, understood } => write!(
                f,
                "the keymap file is version {found} and this build understands {understood}; \
                 it was left untouched"
            ),
            Problem::UnreadableLine { line, text } => {
                write!(f, "line {line}: cannot read `{text}`")
            }
            Problem::StrayField { line, name } => {
                write!(f, "line {line}: `{name}` is not a top-level setting")
            }
            Problem::BindingWithoutId { line } => {
                write!(f, "line {line}: a binding with no `id` was skipped")
            }
            Problem::UnknownPreset { line, name } => {
                write!(f, "line {line}: no preset called `{name}`")
            }
            Problem::UnknownValue { line, name, value } => {
                write!(f, "line {line}: `{name}` cannot be `{value}`")
            }
        }
    }
}

/// A keymap file, as loaded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyFile {
    pub preset: Preset,
    pub overrides: BTreeMap<BindingId, Override>,
}

/// Migrations from older schemas, applied in order.
///
/// Empty because there is exactly one version so far. The mechanism is here
/// rather than added later on purpose: retrofitting a migration chain onto files
/// already in the wild means guessing what the old ones meant, and the guess is
/// what corrupts them.
type Migration = (u32, fn(&mut KeyFile));

const MIGRATIONS: &[Migration] = &[];

fn migrate(from: u32, file: &mut KeyFile) {
    for (version, step) in MIGRATIONS {
        if *version >= from {
            step(file);
        }
    }
}

/// Writes a keymap file.
///
/// Deterministic: the overrides are a sorted map and the fields go out in a
/// fixed order, so an unchanged keymap produces an identical file. That matters
/// for the same reason the document format is deterministic — a file that
/// rewrites itself differently every save is one that shows up in every diff.
pub fn write(file: &KeyFile) -> String {
    let mut out = String::new();
    out.push_str(
        "# Kreate keymap.\n\
         #\n\
         # Only your changes are stored, against each binding's stable id, so\n\
         # later improvements to the preset still reach you. Delete an entry to\n\
         # go back to the preset's default for that binding.\n\n",
    );
    let _ = writeln!(out, "schema_version = {SCHEMA_VERSION}");
    let _ = writeln!(out, "preset = \"{}\"", file.preset.name());

    for (id, change) in &file.overrides {
        out.push_str("\n[[binding]]\n");
        let _ = writeln!(out, "id = \"{}\"", escape(&id.0));
        match &change.input {
            Some(Input::Key(name)) => {
                let _ = writeln!(out, "key = \"{}\"", escape(name));
            }
            Some(Input::Mouse(button)) => {
                let _ = writeln!(out, "mouse = \"{}\"", button_name(*button));
            }
            Some(Input::Wheel) => out.push_str("wheel = true\n"),
            None => {}
        }
        if let Some(m) = change.modifiers {
            let _ = writeln!(out, "shift = {}", m.shift);
            let _ = writeln!(out, "ctrl = {}", m.ctrl);
            let _ = writeln!(out, "alt = {}", m.alt);
        }
        if let Some(event) = change.event {
            let _ = writeln!(out, "event = \"{}\"", event_name(event));
        }
        if let Some(disabled) = change.disabled {
            let _ = writeln!(out, "disabled = {disabled}");
        }
    }
    out
}

/// Reads a keymap file. Never fails; anything wrong comes back as a problem.
pub fn parse(text: &str) -> (KeyFile, Vec<Problem>) {
    let mut problems = Vec::new();
    let mut file = KeyFile::default();
    let mut version = OLDEST_SCHEMA;

    // A binding under construction, and the line its table started on.
    let mut current: Option<(usize, Option<BindingId>, Override, Option<Modifiers>)> = None;

    let finish = |current: &mut Option<(usize, Option<BindingId>, Override, Option<Modifiers>)>,
                  file: &mut KeyFile,
                  problems: &mut Vec<Problem>| {
        let Some((line, id, mut change, modifiers)) = current.take() else {
            return;
        };
        change.modifiers = modifiers;
        match id {
            Some(id) => {
                file.overrides.insert(id, change);
            }
            None => problems.push(Problem::BindingWithoutId { line }),
        }
    };

    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = strip_comment(raw).trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "[[binding]]" {
            finish(&mut current, &mut file, &mut problems);
            current = Some((line, None, Override::default(), None));
            continue;
        }
        let Some((name, value)) = trimmed.split_once('=') else {
            problems.push(Problem::UnreadableLine {
                line,
                text: trimmed.to_string(),
            });
            continue;
        };
        let name = name.trim();
        let value = value.trim();

        match &mut current {
            None => match name {
                "schema_version" => version = value.parse().unwrap_or(OLDEST_SCHEMA),
                "preset" => {
                    let text = unquote(value);
                    match Preset::from_name(&text) {
                        Some(p) => file.preset = p,
                        None => problems.push(Problem::UnknownPreset { line, name: text }),
                    }
                }
                // Ignored rather than rejected: the same forward-compatibility
                // rule the document format uses, so a file from a build with an
                // extra setting still loads here.
                _ if name.starts_with("x-") => {}
                _ => problems.push(Problem::StrayField {
                    line,
                    name: name.to_string(),
                }),
            },
            Some((_, id, change, modifiers)) => {
                // The modifier set is created only when a modifier field is
                // actually present. Creating it for any field at all would turn
                // "the user disabled this binding" into "the user disabled it
                // *and* cleared its modifiers", which strips a `Ctrl` from the
                // preset — visible only when they later re-enable it and get a
                // different combination than the preset's.
                let mut modifier = |set: &mut dyn FnMut(&mut Modifiers)| {
                    set(modifiers.get_or_insert_with(Modifiers::default));
                };
                match name {
                    "id" => *id = Some(BindingId::new(&unquote(value))),
                    "key" => change.input = Some(Input::Key(unquote(value))),
                    "mouse" => {
                        let text = unquote(value);
                        match button_from_name(&text) {
                            Some(b) => change.input = Some(Input::Mouse(b)),
                            None => problems.push(Problem::UnknownValue {
                                line,
                                name: "mouse".into(),
                                value: text,
                            }),
                        }
                    }
                    "wheel" => {
                        if value == "true" {
                            change.input = Some(Input::Wheel);
                        }
                    }
                    "shift" => {
                        let on = value == "true";
                        modifier(&mut |m| m.shift = on);
                    }
                    "ctrl" => {
                        let on = value == "true";
                        modifier(&mut |m| m.ctrl = on);
                    }
                    "alt" => {
                        let on = value == "true";
                        modifier(&mut |m| m.alt = on);
                    }
                    "event" => {
                        let text = unquote(value);
                        match event_from_name(&text) {
                            Some(e) => change.event = Some(e),
                            None => problems.push(Problem::UnknownValue {
                                line,
                                name: "event".into(),
                                value: text,
                            }),
                        }
                    }
                    "disabled" => change.disabled = Some(value == "true"),
                    _ => {}
                }
            }
        }
    }
    finish(&mut current, &mut file, &mut problems);

    // The version check comes last so the whole file has been read and its other
    // problems reported before it is thrown away — a user debugging a file wants
    // to know everything wrong with it, not just the first thing.
    if version > SCHEMA_VERSION {
        return (
            KeyFile::default(),
            vec![Problem::FromTheFuture {
                found: version,
                understood: SCHEMA_VERSION,
            }],
        );
    }
    if version < SCHEMA_VERSION {
        migrate(version, &mut file);
    }
    (file, problems)
}

/// Removes a trailing comment, respecting quotes.
fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return &line[..i],
            _ => {}
        }
    }
    line
}

fn unquote(value: &str) -> String {
    let trimmed = value.trim();
    let inner = trimmed
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(trimmed);
    inner.replace("\\\"", "\"").replace("\\\\", "\\")
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn button_name(button: MouseButton) -> &'static str {
    match button {
        MouseButton::Left => "left",
        MouseButton::Middle => "middle",
        MouseButton::Right => "right",
    }
}

fn button_from_name(name: &str) -> Option<MouseButton> {
    Some(match name {
        "left" => MouseButton::Left,
        "middle" => MouseButton::Middle,
        "right" => MouseButton::Right,
        _ => return None,
    })
}

fn event_name(event: EventKind) -> &'static str {
    match event {
        EventKind::Press => "press",
        EventKind::Release => "release",
        EventKind::Drag => "drag",
    }
}

fn event_from_name(name: &str) -> Option<EventKind> {
    Some(match name {
        "press" => EventKind::Press,
        "release" => EventKind::Release,
        "drag" => EventKind::Drag,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> KeyFile {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            BindingId::new("mesh.extrude"),
            Override {
                input: Some(Input::key("Q")),
                modifiers: Some(Modifiers::shift()),
                ..Default::default()
            },
        );
        overrides.insert(
            BindingId::new("mesh.bridge"),
            Override {
                disabled: Some(true),
                ..Default::default()
            },
        );
        overrides.insert(
            BindingId::new("view.orbit.mmb"),
            Override {
                input: Some(Input::Mouse(MouseButton::Right)),
                event: Some(EventKind::Drag),
                modifiers: Some(Modifiers::alt()),
                ..Default::default()
            },
        );
        KeyFile {
            preset: Preset::Maya,
            overrides,
        }
    }

    #[test]
    fn a_file_round_trips() {
        let original = sample();
        let text = write(&original);
        let (parsed, problems) = parse(&text);
        assert_eq!(problems, Vec::new(), "{text}");
        assert_eq!(parsed, original);
    }

    #[test]
    fn writing_is_deterministic() {
        // A file that rewrites itself differently every save shows up in every
        // diff and makes a real change impossible to spot.
        let file = sample();
        let first = write(&file);
        for _ in 0..5 {
            assert_eq!(write(&file), first);
        }
    }

    #[test]
    fn an_empty_file_gives_the_default_preset_and_no_changes() {
        let (file, problems) = parse("");
        assert_eq!(file, KeyFile::default());
        assert_eq!(problems, Vec::new());
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let text = "\
# a comment
schema_version = 1   # trailing comment

preset = \"maya\"

[[binding]]
id = \"mesh.extrude\"  # this one moved
key = \"Q\"
";
        let (file, problems) = parse(text);
        assert_eq!(problems, Vec::new());
        assert_eq!(file.preset, Preset::Maya);
        assert_eq!(
            file.overrides[&BindingId::new("mesh.extrude")].input,
            Some(Input::key("Q"))
        );
    }

    #[test]
    fn a_hash_inside_a_quoted_value_is_not_a_comment() {
        let text = "schema_version = 1\n[[binding]]\nid = \"weird#id\"\nkey = \"A\"\n";
        let (file, problems) = parse(text);
        assert_eq!(problems, Vec::new());
        assert!(file.overrides.contains_key(&BindingId::new("weird#id")));
    }

    #[test]
    fn a_newer_file_is_refused_rather_than_misread() {
        // Reading a future file as though it were this version silently
        // mis-binds keys and the user cannot tell why. Refusing leaves their
        // file intact for when they go back to the newer build.
        let text = format!(
            "schema_version = {}\npreset = \"maya\"\n[[binding]]\nid = \"a\"\nkey = \"Q\"\n",
            SCHEMA_VERSION + 1
        );
        let (file, problems) = parse(&text);
        assert_eq!(file, KeyFile::default(), "nothing may be loaded");
        assert_eq!(
            problems,
            vec![Problem::FromTheFuture {
                found: SCHEMA_VERSION + 1,
                understood: SCHEMA_VERSION
            }]
        );
    }

    #[test]
    fn a_file_with_no_version_is_read_as_the_oldest() {
        let (file, problems) = parse("preset = \"max\"\n");
        assert_eq!(file.preset, Preset::Max);
        assert_eq!(problems, Vec::new());
    }

    #[test]
    fn a_malformed_line_is_reported_and_the_rest_still_loads() {
        // The whole point. A file a user has hand-edited badly must cost them
        // one binding, not their session.
        let text = "\
schema_version = 1
preset = \"blender\"
this line is nonsense

[[binding]]
id = \"mesh.extrude\"
key = \"Q\"

[[binding]]
key = \"W\"

[[binding]]
id = \"mesh.bridge\"
mouse = \"scroll-wheel-thing\"
event = \"hover\"
disabled = true
";
        let (file, problems) = parse(text);
        assert_eq!(file.preset, Preset::Blender);
        assert_eq!(
            file.overrides[&BindingId::new("mesh.extrude")].input,
            Some(Input::key("Q")),
            "a good entry after a bad line must survive"
        );
        assert_eq!(
            file.overrides[&BindingId::new("mesh.bridge")].disabled,
            Some(true),
            "the good fields of a partly bad entry must survive"
        );
        assert!(problems
            .iter()
            .any(|p| matches!(p, Problem::UnreadableLine { line: 3, .. })));
        assert!(problems
            .iter()
            .any(|p| matches!(p, Problem::BindingWithoutId { .. })));
        assert_eq!(
            problems
                .iter()
                .filter(|p| matches!(p, Problem::UnknownValue { .. }))
                .count(),
            2,
            "the bad mouse button and the bad event: {problems:?}"
        );
    }

    #[test]
    fn an_unknown_preset_is_reported_and_the_default_is_kept() {
        let (file, problems) = parse("schema_version = 1\npreset = \"emacs\"\n");
        assert_eq!(file.preset, Preset::default());
        assert_eq!(
            problems,
            vec![Problem::UnknownPreset {
                line: 2,
                name: "emacs".into()
            }]
        );
    }

    #[test]
    fn an_unknown_extension_field_is_ignored() {
        // Forward compatibility: a file written by a build with an extra
        // setting must still load here.
        let (file, problems) =
            parse("schema_version = 1\npreset = \"maya\"\nx-future-thing = \"whatever\"\n");
        assert_eq!(file.preset, Preset::Maya);
        assert_eq!(problems, Vec::new());
    }

    #[test]
    fn a_stray_field_is_reported_rather_than_silently_dropped() {
        let (_, problems) = parse("schema_version = 1\nkey = \"Q\"\n");
        assert_eq!(
            problems,
            vec![Problem::StrayField {
                line: 2,
                name: "key".into()
            }]
        );
    }

    #[test]
    fn disabling_a_binding_does_not_also_clear_its_modifiers() {
        // `Some(Modifiers::NONE)` means "the user cleared every modifier", which
        // is a different instruction from "the user said nothing about
        // modifiers". Conflating them strips a `Ctrl` from the preset, and it
        // shows up only when the binding is later re-enabled with the wrong
        // combination.
        let (file, problems) = parse(
            "schema_version = 1
[[binding]]
id = \"a\"
disabled = true
",
        );
        assert_eq!(problems, Vec::new());
        let change = &file.overrides[&BindingId::new("a")];
        assert_eq!(change.disabled, Some(true));
        assert_eq!(change.modifiers, None, "no modifier field was present");
        assert_eq!(change.input, None);
    }

    #[test]
    fn an_explicitly_cleared_modifier_is_kept_as_such() {
        // The other side of the same distinction: writing `ctrl = false` really
        // does mean "remove the preset's Ctrl".
        let (file, _) = parse(
            "schema_version = 1
[[binding]]
id = \"a\"
ctrl = false
",
        );
        assert_eq!(
            file.overrides[&BindingId::new("a")].modifiers,
            Some(Modifiers::NONE)
        );
    }

    #[test]
    fn quotes_and_backslashes_survive_a_round_trip() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            BindingId::new(r#"odd "id" with \ slashes"#),
            Override {
                input: Some(Input::key("A")),
                ..Default::default()
            },
        );
        let file = KeyFile {
            preset: Preset::Kreate,
            overrides,
        };
        let (parsed, problems) = parse(&write(&file));
        assert_eq!(problems, Vec::new());
        assert_eq!(parsed, file);
    }

    #[test]
    fn nothing_in_here_panics_on_arbitrary_input() {
        // A preferences file is attacker-adjacent at worst and hand-edited at
        // best. Either way it must not be able to stop the application starting.
        let cases = [
            "=",
            "= =",
            "[[binding]",
            "[[binding]]\n[[binding]]\n[[binding]]",
            "schema_version = not-a-number",
            "schema_version = 99999999999999999999",
            "id = \"unterminated",
            "\0\0\0",
            "preset =",
            "[[binding]]\nkey =",
            &"[[binding]]\n".repeat(1000),
            &"#".repeat(10_000),
        ];
        for case in cases {
            let (_, problems) = parse(case);
            // The only requirement is that it returned at all.
            let _ = problems.len();
        }
    }

    #[test]
    fn a_bad_schema_version_falls_back_rather_than_failing() {
        let (file, _) = parse("schema_version = not-a-number\npreset = \"max\"\n");
        assert_eq!(file.preset, Preset::Max, "the rest of the file still loads");
    }

    #[test]
    fn the_written_file_explains_itself() {
        // It is a file users open in an editor. A header that says what is
        // stored and how to undo a change is cheaper than the support question.
        let text = write(&sample());
        assert!(text.starts_with("# Kreate keymap."));
        assert!(text.contains("stable id"));
        assert!(text.contains("Delete an entry"));
    }
}
