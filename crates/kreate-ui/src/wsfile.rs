//! Reading and writing the workspace file.
//!
//! Follows `kreate_input::keyfile` in shape and in reasoning, because the
//! reasoning has not changed: a preferences file that crashes on startup is a
//! self-inflicted support nightmare, so
//!
//! - **nothing here can panic or fail.** [`parse`] always returns a set of
//!   workspaces. Anything wrong becomes a [`Problem`] the application can log,
//!   and the rest of the file still loads.
//! - **a newer `schema_version` is refused, not guessed at.** Reading a file
//!   from a future build as though it were this one hands the user a layout
//!   they never made, and they cannot tell why. Refusing keeps their file intact
//!   for when they go back to the newer build.
//! - **unknown fields are ignored, not rejected** — the same
//!   forward-compatibility rule the document format uses.
//!
//! # What is stored
//!
//! A name and a flattened tree per workspace, and which one was in use. Not the
//! area ids, and therefore not any per-area state: see [`NodeSpec`]. A UV
//! editor's zoom does not survive a restart, which is worth less than a format
//! that cannot be got wrong.
//!
//! # Why the tree is a list of lines
//!
//! `Screen::to_nodes` gives a pre-order sequence, and a pre-order sequence of a
//! full binary tree needs no nesting to be read back exactly. So the file needs
//! no brackets, no indentation and no recursive parser — one repeated `node`
//! field, in order, is the whole layout.

use std::fmt::Write as _;

use crate::area::{Axis, EditorId, NodeSpec, Screen};
use crate::workspace::{Workspace, Workspaces};

/// The current schema. Every bump adds a step to [`MIGRATIONS`].
pub const SCHEMA_VERSION: u32 = 1;

/// The oldest schema this build understands.
pub const OLDEST_SCHEMA: u32 = 1;

/// Something wrong with the file that did not stop it loading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Written by a newer build. Nothing was loaded.
    FromTheFuture { found: u32, understood: u32 },
    /// A line that is neither a comment, a section, nor `key = value`.
    UnreadableLine { line: usize, text: String },
    /// A field outside any `[[workspace]]`, where only `schema_version` and
    /// `active` belong.
    StrayField { line: usize, name: String },
    /// A workspace with no name, which cannot be labelled or switched to.
    WorkspaceWithoutName { line: usize },
    /// A `node` value that is neither `area <editor>` nor `split <axis>
    /// <ratio>`.
    UnreadableNode { line: usize, text: String },
    /// A workspace whose nodes are not exactly one tree. It is dropped: a
    /// partial tree is not a smaller layout, it is a broken one.
    BrokenLayout { line: usize, name: String },
    /// The file named no workspaces at all, so the shipped set was used.
    NoWorkspaces,
    /// `active` named a workspace the file does not contain.
    UnknownActive { name: String },
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::FromTheFuture { found, understood } => write!(
                f,
                "the workspace file is version {found} and this build understands \
                 {understood}; it was left untouched"
            ),
            Problem::UnreadableLine { line, text } => {
                write!(f, "line {line}: cannot read `{text}`")
            }
            Problem::StrayField { line, name } => {
                write!(f, "line {line}: `{name}` is not a top-level setting")
            }
            Problem::WorkspaceWithoutName { line } => {
                write!(f, "line {line}: a workspace with no `name` was skipped")
            }
            Problem::UnreadableNode { line, text } => {
                write!(f, "line {line}: cannot read the layout node `{text}`")
            }
            Problem::BrokenLayout { line, name } => write!(
                f,
                "line {line}: the layout of `{name}` is incomplete, so it was skipped"
            ),
            Problem::NoWorkspaces => {
                write!(f, "no workspaces in the file; the shipped ones were used")
            }
            Problem::UnknownActive { name } => {
                write!(f, "no workspace called `{name}`; the first one was used")
            }
        }
    }
}

/// Migrations from older schemas, applied in order.
///
/// Empty because there is exactly one version so far. The mechanism is here
/// rather than added later on purpose: retrofitting a migration chain onto files
/// already in the wild means guessing what the old ones meant, and the guess is
/// what corrupts them.
type Migration = (u32, fn(&mut Workspaces));

const MIGRATIONS: &[Migration] = &[];

fn migrate(from: u32, spaces: &mut Workspaces) {
    for (version, step) in MIGRATIONS {
        if *version >= from {
            step(spaces);
        }
    }
}

/// Writes a workspace file.
///
/// Deterministic: the workspaces go out in order, the fields in a fixed order,
/// and the ratios to four decimals. An unchanged layout produces an identical
/// file — which matters for the same reason the document format is
/// deterministic, and rather more here: a file that rewrites itself differently
/// every save is one the user's backup tool reports as changed every day.
///
/// Four decimals rather than full precision because the extra digits are
/// noise: a boundary is placed by dragging it, and a ten-thousandth of a window
/// is a fortieth of a pixel.
///
/// **Not for stability.** An earlier version of this comment claimed the file
/// would drift on every save without the rounding, and a sweep removing it
/// showed that is not so — `f32`'s own formatting is the shortest form that
/// reads back exactly, so it is already idempotent. The rounding buys a file a
/// person can read and a diff without seven-digit noise in it, and nothing
/// else.
pub fn write(spaces: &Workspaces) -> String {
    let mut out = String::new();
    out.push_str(
        "# Kreate workspaces.\n\
         #\n\
         # One entry per named layout. Each `node` line is one node of the\n\
         # layout tree, in the order a walk from the root visits them: a\n\
         # `split` is followed by its two children.\n\
         #\n\
         # Delete this file to go back to the layouts Kreate ships with.\n\n",
    );
    let _ = writeln!(out, "schema_version = {SCHEMA_VERSION}");
    let _ = writeln!(out, "active = \"{}\"", escape(&spaces.active().name));
    // The workspaces Kreate ships, as this build knows them. A later build
    // that ships another adds it on load; one named here and missing from the
    // list below was deleted by the user and stays deleted. `x-` so a build
    // that predates the field reads the file without a complaint.
    for shipped in Workspaces::default().iter() {
        let _ = writeln!(out, "x-shipped = \"{}\"", escape(&shipped.name));
    }

    for space in spaces.iter() {
        out.push_str("\n[[workspace]]\n");
        let _ = writeln!(out, "name = \"{}\"", escape(&space.name));
        for node in space.screen.to_nodes() {
            match node {
                NodeSpec::Area { editor } => {
                    let _ = writeln!(out, "node = \"area {}\"", editor.as_str());
                }
                NodeSpec::Split { axis, ratio } => {
                    let _ = writeln!(out, "node = \"split {} {ratio:.4}\"", axis.tag());
                }
            }
        }
        // The panels floating over it: editor, then the window's corner and
        // size in points. A build that predates the field ignores it.
        for f in space.screen.floating() {
            let r = f.rect;
            let _ = writeln!(
                out,
                "float = \"{} {:.0} {:.0} {:.0} {:.0}\"",
                f.editor.as_str(),
                r.min.x,
                r.min.y,
                r.width(),
                r.height()
            );
        }
    }
    out
}

/// A `float` value back: `outliner 900 40 300 400`.
fn float_from_text(text: &str) -> Option<(EditorId, egui::Rect)> {
    let mut parts = text.split_whitespace();
    let editor = EditorId::parse(parts.next()?);
    let mut n = || parts.next()?.parse::<f32>().ok().filter(|v| v.is_finite());
    let (x, y, w, h) = (n()?, n()?, n()?, n()?);
    (w > 0.0 && h > 0.0).then(|| {
        (
            editor,
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h)),
        )
    })
}

/// Reads a workspace file. Never fails; anything wrong comes back as a problem.
pub fn parse(text: &str) -> (Workspaces, Vec<Problem>) {
    let mut problems = Vec::new();
    let mut version = OLDEST_SCHEMA;
    let mut active_name: Option<String> = None;
    let mut loaded: Vec<Workspace> = Vec::new();
    // The shipped workspaces the file's writer knew. `None` for a file written
    // before the field: everything shipped since is new to it.
    let mut shipped: Option<Vec<String>> = None;

    // A workspace under construction: the line its table started on, its name,
    // and the nodes seen so far.
    #[allow(clippy::type_complexity)]
    let mut current: Option<(
        usize,
        Option<String>,
        Vec<NodeSpec>,
        Vec<(EditorId, egui::Rect)>,
    )> = None;

    #[allow(clippy::type_complexity)]
    let finish = |current: &mut Option<(
        usize,
        Option<String>,
        Vec<NodeSpec>,
        Vec<(EditorId, egui::Rect)>,
    )>,
                  loaded: &mut Vec<Workspace>,
                  problems: &mut Vec<Problem>| {
        let Some((line, name, nodes, floats)) = current.take() else {
            return;
        };
        let Some(name) = name else {
            problems.push(Problem::WorkspaceWithoutName { line });
            return;
        };
        // A layout that is not one tree is dropped rather than repaired. The
        // alternative is handing the user an arrangement they never saved, which
        // they then have to work out how to undo.
        match Screen::from_nodes(&nodes) {
            Some(mut screen) => {
                for (editor, rect) in floats {
                    screen.add_floating(editor, rect);
                }
                loaded.push(Workspace { name, screen })
            }
            None => problems.push(Problem::BrokenLayout { line, name }),
        }
    };

    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = strip_comment(raw).trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "[[workspace]]" {
            finish(&mut current, &mut loaded, &mut problems);
            current = Some((line, None, Vec::new(), Vec::new()));
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
                "active" => active_name = Some(unquote(value)),
                "x-shipped" => shipped.get_or_insert_with(Vec::new).push(unquote(value)),
                // Ignored rather than rejected, so a file from a build with an
                // extra setting still loads here.
                _ if name.starts_with("x-") => {}
                _ => problems.push(Problem::StrayField {
                    line,
                    name: name.to_string(),
                }),
            },
            Some((_, space_name, nodes, floats)) => match name {
                "name" => *space_name = Some(unquote(value)),
                // A floating panel that does not read is dropped with a word:
                // the rest of the layout is still the user's.
                "float" => {
                    let text = unquote(value);
                    match float_from_text(&text) {
                        Some(f) => floats.push(f),
                        None => problems.push(Problem::UnreadableNode { line, text }),
                    }
                }
                "node" => {
                    let text = unquote(value);
                    match node_from_text(&text) {
                        Some(node) => nodes.push(node),
                        None => problems.push(Problem::UnreadableNode { line, text }),
                    }
                }
                _ => {}
            },
        }
    }
    finish(&mut current, &mut loaded, &mut problems);

    // The version check comes last so the whole file has been read and its other
    // problems reported before it is thrown away — a user debugging a file wants
    // to know everything wrong with it, not just the first thing.
    if version > SCHEMA_VERSION {
        return (
            Workspaces::default(),
            vec![Problem::FromTheFuture {
                found: version,
                understood: SCHEMA_VERSION,
            }],
        );
    }

    if loaded.is_empty() {
        problems.push(Problem::NoWorkspaces);
        let mut spaces = Workspaces::default();
        if version < SCHEMA_VERSION {
            migrate(version, &mut spaces);
        }
        return (spaces, problems);
    }

    let mut spaces = Workspaces::single(loaded[0].name.clone(), loaded[0].screen.clone());
    for space in &loaded[1..] {
        spaces.add(space.name.clone(), space.screen.clone());
    }
    // `add` makes each new one active, so the intended one is chosen at the end.
    // By name rather than by index: the file is text a user can reorder, and an
    // index would then select whichever layout moved into that position.
    match active_name {
        Some(wanted) => match loaded.iter().position(|space| space.name == wanted) {
            Some(index) => {
                spaces.switch_to(index);
            }
            None => {
                spaces.switch_to(0);
                problems.push(Problem::UnknownActive { name: wanted });
            }
        },
        None => {
            spaces.switch_to(0);
        }
    }

    if version < SCHEMA_VERSION {
        migrate(version, &mut spaces);
    }
    add_new_shipped(&mut spaces, shipped.as_deref());
    (spaces, problems)
}

/// Appends the shipped workspaces the file's writer did not know.
///
/// A file saved before a build added the Scripting workspace would otherwise
/// keep it out for good: the file lists what existed then, and what is in the
/// file is what loads. One the writer did know and the file does not list was
/// deleted, and is not brought back.
///
/// A file from before the record holds none of it, and is read by what it
/// contains: one built from the shipped set — any shipped name in it — gains
/// what shipped since; one of the user's own layouts alone is left alone.
fn add_new_shipped(spaces: &mut Workspaces, known: Option<&[String]>) {
    let defaults = Workspaces::default();
    if known.is_none()
        && !spaces
            .iter()
            .any(|s| defaults.iter().any(|d| d.name == s.name))
    {
        return;
    }
    let active = spaces.active_index();
    for space in defaults.iter() {
        let present = spaces.iter().any(|s| s.name == space.name);
        let deleted = known.is_some_and(|k| k.contains(&space.name));
        if !present && !deleted {
            spaces.add(space.name.clone(), space.screen.clone());
        }
    }
    // `add` switches to what it adds; the user's choice is kept.
    spaces.switch_to(active);
}

/// One `node` value: `area <editor>` or `split <axis> <ratio>`.
fn node_from_text(text: &str) -> Option<NodeSpec> {
    let mut parts = text.split_whitespace();
    match parts.next()? {
        "area" => {
            // An id this build does not know is *kept*, not refused. Refusing it
            // discarded the whole layout — one unknown area and every workspace in
            // the file was lost — and falling back to a viewport would have the
            // user's arrangement come back subtly wrong. See `EditorId`.
            let editor = EditorId::parse(parts.next()?);
            parts.next().is_none().then_some(NodeSpec::Area { editor })
        }
        "split" => {
            let axis = Axis::from_tag(parts.next()?)?;
            let ratio: f32 = parts.next()?.parse().ok()?;
            // Out of range is not refused here: `Screen::from_nodes` brings any
            // ratio into range, and refusing the whole layout over a number
            // that can be fixed would lose the user their arrangement.
            parts
                .next()
                .is_none()
                .then_some(NodeSpec::Split { axis, ratio })
        }
        _ => None,
    }
}

/// Removes a trailing comment, respecting quotes **and the escapes inside
/// them**.
///
/// The second half is the part that took a test to find. A workspace name is
/// the one piece of free text in the file, and a name containing a quote is
/// written escaped — so a stripper that toggled on every quote would count the
/// `\"` as a closing one, decide it was outside the string, and cut the line at
/// the next `#`. The name `a" # b` came back as `a`.
///
/// `kreate_input::keyfile` has the same helper without this, and it is latent
/// there rather than fixed: everything quoted in a keymap file is an identifier
/// this build wrote — a binding id, a preset name — and none of them contain a
/// quote. It is worth knowing about before that stops being true.
fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, c) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_quotes => escaped = true,
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

#[cfg(test)]
mod shipped_tests {
    use super::*;

    const OLD_FILE: &str = "schema_version = 1
active = \"Modeling\"

        [[workspace]]
name = \"Layout\"
node = \"area viewport\"

        [[workspace]]
name = \"Modeling\"
node = \"area viewport\"
";

    fn names(spaces: &Workspaces) -> Vec<String> {
        spaces.iter().map(|s| s.name.clone()).collect()
    }

    #[test]
    fn a_file_from_before_a_workspace_shipped_gains_it_and_keeps_its_active_one() {
        // Saved when only these two existed: Scripting and the rest came
        // later, and never showed up for anyone who had a file.
        let (spaces, problems) = parse(OLD_FILE);
        assert!(problems.is_empty(), "{problems:?}");
        let got = names(&spaces);
        assert!(got.contains(&"Scripting".to_string()), "{got:?}");
        assert_eq!(
            &got[..2],
            ["Layout", "Modeling"],
            "the file's own come first, in its order"
        );
        assert_eq!(spaces.active().name, "Modeling");
    }

    #[test]
    fn a_file_of_the_users_own_layouts_alone_is_left_alone() {
        let text = "schema_version = 1

[[workspace]]
name = \"Mine\"
node = \"area viewport\"
";
        let (spaces, _) = parse(text);
        assert_eq!(names(&spaces), ["Mine"]);
    }

    #[test]
    fn a_shipped_workspace_the_user_deleted_stays_deleted() {
        let mut spaces = Workspaces::default();
        let scripting = names(&spaces)
            .iter()
            .position(|n| n == "Scripting")
            .expect("shipped");
        assert!(spaces.remove(scripting));
        let (back, problems) = parse(&write(&spaces));
        assert!(problems.is_empty(), "{problems:?}");
        assert!(!names(&back).contains(&"Scripting".to_string()));
        assert_eq!(names(&back), names(&spaces));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_problem_says_what_it_is() {
        // These go to a log and to the console, where they are the only thing
        // between a user and a preferences file they cannot explain. A variant
        // added without a message would fall through to the derived `Debug`,
        // which is not prose.
        let problems = [
            Problem::FromTheFuture {
                found: 9,
                understood: 1,
            },
            Problem::UnreadableLine {
                line: 3,
                text: "junk".into(),
            },
            Problem::StrayField {
                line: 4,
                name: "colour".into(),
            },
            Problem::WorkspaceWithoutName { line: 5 },
            Problem::UnreadableNode {
                line: 6,
                text: "leaf x".into(),
            },
            Problem::BrokenLayout {
                line: 7,
                name: "Mine".into(),
            },
            Problem::NoWorkspaces,
            Problem::UnknownActive {
                name: "Gone".into(),
            },
        ];
        let mut seen: Vec<String> = Vec::new();
        for problem in &problems {
            let text = problem.to_string();
            assert!(text.len() > 20, "{problem:?} says only {text:?}");
            assert!(
                !text.contains('{') && !text.contains("Problem"),
                "{problem:?} reads as a debug dump: {text}"
            );
            assert!(!seen.contains(&text), "two problems read alike: {text}");
            seen.push(text);
        }
    }

    #[test]
    fn the_shipped_set_survives_a_round_trip() {
        // Names and which one is in use come back exactly. The layouts come back
        // to within the rounding the format applies on purpose, so they are
        // compared as *where the areas end up* rather than bit for bit.
        let mut original = Workspaces::default();
        original.switch_to(2);

        let (loaded, problems) = parse(&write(&original));
        assert_eq!(problems, vec![], "a file we wrote should have no problems");
        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.active_index(), original.active_index());
        assert_eq!(loaded.active().name, original.active().name);

        let space = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 900.0));
        for (a, b) in loaded.iter().zip(original.iter()) {
            assert_eq!(a.name, b.name);
            let (la, lb) = (a.screen.layout(space), b.screen.layout(space));
            assert_eq!(la.areas.len(), lb.areas.len(), "{}", a.name);
            for (x, y) in la.areas.iter().zip(&lb.areas) {
                assert_eq!(x.editor, y.editor, "{} moved an editor", a.name);
                let moved = (x.rect.min - y.rect.min).length() + (x.rect.max - y.rect.max).length();
                assert!(moved < 1.0, "{} moved an area by {moved}", a.name);
            }
        }
    }

    #[test]
    fn writing_is_deterministic_and_idempotent() {
        // A file that rewrites itself differently every save shows up in every
        // diff and in every backup. And the second write has to equal the first,
        // or the rounding is not stable and the file drifts on each save.
        let spaces = Workspaces::default();
        let once = write(&spaces);
        assert_eq!(once, write(&spaces), "two writes of one value differ");

        let (loaded, _) = parse(&once);
        assert_eq!(write(&loaded), once, "the file drifts on every save");
    }

    #[test]
    fn a_file_from_the_future_is_refused_rather_than_guessed_at() {
        // Reading a newer file as though it were this one hands the user a
        // layout they never made and cannot explain. Their file is left intact
        // for when they go back to the newer build.
        let text = format!(
            "schema_version = {}\n\n[[workspace]]\nname = \"Mine\"\nnode = \"area viewport\"\n",
            SCHEMA_VERSION + 1
        );
        let (spaces, problems) = parse(&text);
        assert_eq!(
            problems,
            vec![Problem::FromTheFuture {
                found: SCHEMA_VERSION + 1,
                understood: SCHEMA_VERSION,
            }]
        );
        assert_eq!(spaces, Workspaces::default(), "the file was partly read");
    }

    #[test]
    fn an_unreadable_file_still_loads_what_it_can() {
        // The whole robustness claim. Every line here is wrong in a different
        // way, and the one good workspace still arrives.
        let text = "\
schema_version = 1
active = \"Good\"
nonsense
colour = \"blue\"
x-future-setting = 3

[[workspace]]
name = \"Good\"
node = \"area viewport\"
node = \"gibberish\"
unknown_field = 7

[[workspace]]
node = \"area outliner\"

[[workspace]]
name = \"Broken\"
node = \"split v 0.5\"
node = \"area viewport\"
";
        let (spaces, problems) = parse(text);

        assert_eq!(spaces.len(), 1, "the good workspace did not survive alone");
        assert_eq!(spaces.active().name, "Good");
        assert_eq!(spaces.screen().area_count(), 1);

        // Each failure reported once, and the forward-compatible field silently
        // ignored rather than complained about.
        assert!(problems.iter().any(|p| matches!(
            p,
            Problem::UnreadableLine { text, .. } if text == "nonsense"
        )));
        assert!(problems.iter().any(|p| matches!(
            p,
            Problem::StrayField { name, .. } if name == "colour"
        )));
        assert!(problems.iter().any(|p| matches!(
            p,
            Problem::UnreadableNode { text, .. } if text == "gibberish"
        )));
        assert!(problems
            .iter()
            .any(|p| matches!(p, Problem::WorkspaceWithoutName { .. })));
        assert!(problems.iter().any(|p| matches!(
            p,
            Problem::BrokenLayout { name, .. } if name == "Broken"
        )));
        assert!(
            !problems.iter().any(|p| matches!(
                p,
                Problem::StrayField { name, .. } if name.starts_with("x-")
            )),
            "a forward-compatible field was rejected"
        );
    }

    #[test]
    fn an_empty_file_gives_the_shipped_layouts() {
        // Missing, empty or all-comments. A user who deletes the file is asking
        // for the defaults back, and a build that answered with a blank window
        // would be a build they cannot recover.
        for text in ["", "\n\n", "# just a comment\n", "schema_version = 1\n"] {
            let (spaces, problems) = parse(text);
            assert_eq!(spaces, Workspaces::default(), "{text:?}");
            assert!(
                problems.contains(&Problem::NoWorkspaces),
                "{text:?} said nothing about having no workspaces"
            );
        }
    }

    #[test]
    fn active_is_matched_by_name_so_reordering_the_file_is_safe() {
        // By name rather than by index, because the file is text a user can
        // reorder — and an index would then select whichever layout moved into
        // that position, which is not the one they were using.
        let text = "\
active = \"Second\"

[[workspace]]
name = \"First\"
node = \"area viewport\"

[[workspace]]
name = \"Second\"
node = \"area outliner\"
";
        let (spaces, problems) = parse(text);
        assert_eq!(problems, vec![]);
        assert_eq!(spaces.active().name, "Second");
        assert_eq!(spaces.active_index(), 1);

        // Swapped in the file, and the same workspace is still active.
        let swapped = "\
active = \"Second\"

[[workspace]]
name = \"Second\"
node = \"area outliner\"

[[workspace]]
name = \"First\"
node = \"area viewport\"
";
        let (spaces, _) = parse(swapped);
        assert_eq!(spaces.active().name, "Second");
        assert_eq!(spaces.active_index(), 0, "the index followed the name");
    }

    #[test]
    fn an_active_name_that_is_not_there_falls_back_to_the_first() {
        let text = "\
active = \"Gone\"

[[workspace]]
name = \"Here\"
node = \"area viewport\"
";
        let (spaces, problems) = parse(text);
        assert_eq!(spaces.active().name, "Here");
        assert!(problems.contains(&Problem::UnknownActive {
            name: "Gone".into()
        }));
    }

    #[test]
    fn a_name_with_quotes_hashes_and_backslashes_survives() {
        // The one thing in the file that is free text, so the one thing that can
        // break the format — and the user typed it.
        //
        // The first version of this test used `My "best\ layout`, which
        // survives whether or not anything is escaped: a sweep writing the name
        // raw passed it. `a" # b` is the one that matters, and it found a real
        // defect in `strip_comment` — see its own comment. Every awkward
        // character is listed rather than one representative, because each is
        // handled by a different line.
        for awkward in [
            r#"a" # b"#,
            r#"plain # hash"#,
            r#"quote" only"#,
            r#"back\slash"#,
            r#"trailing""#,
            r#"# leading hash"#,
        ] {
            let mut spaces = Workspaces::single(awkward, Screen::single(EditorId::VIEWPORT));
            spaces.add("Plain", Screen::single(EditorId::OUTLINER));
            spaces.switch_to(0);

            let (loaded, problems) = parse(&write(&spaces));
            assert_eq!(problems, vec![], "{awkward:?}");
            assert_eq!(loaded.len(), 2, "{awkward:?} lost a workspace");
            assert_eq!(loaded.active().name, awkward, "the name came back changed");
        }
    }

    #[test]
    fn a_node_line_is_all_or_nothing() {
        // Each of these is a way a hand-edited line can be wrong, and every one
        // has to be refused rather than half-read: a `split` with a missing
        // ratio that defaulted to something would move the user's boundary.
        for bad in [
            "area",
            "area viewport extra",
            "split",
            "split v",
            "split x 0.5",
            "split v notanumber",
            "split v 0.5 extra",
            "",
            "leaf viewport",
        ] {
            assert_eq!(node_from_text(bad), None, "{bad:?} was accepted");
        }
        // And the two good forms, so this is not a test that everything fails.
        assert_eq!(
            node_from_text("area viewport"),
            Some(NodeSpec::Area {
                editor: EditorId::VIEWPORT
            })
        );

        // An editor id this build does not know is **not** one of the bad
        // forms, and that changed deliberately. Refusing it used to fail the
        // whole node, which failed the whole file — one unknown area and every
        // workspace in it was replaced by the shipped defaults. A layout saved
        // while an add-on was installed is not a corrupt layout.
        //
        // Kept rather than mapped to a viewport, too: the area is drawn as
        // missing, with the id it wants, so the user can install the add-on
        // instead of rebuilding an arrangement that was never broken.
        assert_eq!(
            node_from_text("area com.example.graph"),
            Some(NodeSpec::Area {
                editor: EditorId::parse("com.example.graph")
            })
        );
        assert_eq!(
            node_from_text("split h 0.25"),
            Some(NodeSpec::Split {
                axis: Axis::Horizontal,
                ratio: 0.25,
            })
        );
    }

    #[test]
    fn a_ratio_out_of_range_in_a_file_is_brought_back_rather_than_refused() {
        // The layout is worth more than the number. Refusing the whole workspace
        // over a ratio that can be clamped would lose the user their
        // arrangement; `Screen::from_nodes` brings it into range.
        let text = "\
[[workspace]]
name = \"Odd\"
node = \"split v 7.5\"
node = \"area viewport\"
node = \"area outliner\"
";
        let (spaces, problems) = parse(text);
        assert_eq!(problems, vec![]);
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces.screen().area_count(), 2);

        let space = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        for area in spaces.screen().layout(space).areas {
            assert!(
                space.expand(1.0).contains_rect(area.rect),
                "{:?} is outside the screen",
                area.rect
            );
        }
    }

    #[test]
    fn a_floating_panel_reads_back_and_a_broken_one_is_dropped() {
        let text = "schema_version = 1\nactive = \"A\"\n\n[[workspace]]\nname = \"A\"\n\
                    node = \"split v 0.7000\"\nnode = \"area viewport\"\nnode = \"area properties\"\n\
                    float = \"outliner 900 40 300 400\"\nfloat = \"outliner nine\"\n";
        let (spaces, problems) = parse(text);
        let screen = &spaces.iter().next().unwrap().screen;
        assert_eq!(screen.area_count(), 2);
        assert_eq!(screen.floating().len(), 1);
        let f = &screen.floating()[0];
        assert_eq!(f.editor, EditorId::OUTLINER);
        assert_eq!(f.rect.min, egui::pos2(900.0, 40.0));
        assert_eq!(f.rect.size(), egui::vec2(300.0, 400.0));
        assert_eq!(problems.len(), 1, "{problems:?}");
        // Written again, the same.
        assert!(write(&spaces).contains("float = \"outliner 900 40 300 400\""));
    }
}
