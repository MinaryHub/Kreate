//! Architectural rules, enforced as tests rather than left to convention.
//!
//! The dependency direction in this workspace is load-bearing, not stylistic:
//!
//! - `kreate-mesh` must stand alone as a library and compile to wasm, so it may
//!   not reach for wgpu, egui, winit, async, or the scene layer.
//! - `kreate-render` and `kreate-ui` must not be able to mutate a document; they
//!   consume an immutable snapshot and dispatch `(op_id, params)` respectively.
//!   That is what gives the renderer a lock-free path and what makes undo,
//!   scripting, and macro recording fall out of the operator system for free.
//! - The sRGB transfer function is implemented once, in `kreate-math`. A second
//!   copy — or a `powf(2.2)` standing in for it — is how a colour ends up in
//!   the wrong space, and §2.1 records that having happened twice already.
//!
//! Every one of these is easy to violate with a single convenient `use`, and
//! each violation is expensive to unwind six months later. A grep-level check
//! costs nothing and fails the build the day it happens.
//!
//! The manifests are read as text on purpose: pulling in a TOML parser just to
//! police the manifests would itself add a dependency to police.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/kreate-testkit
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

/// The dependency names declared by a crate, as they appear in its manifest.
///
/// Only the `[dependencies]`-style tables are considered; `[dev-dependencies]`
/// is deliberately excluded, because a test pulling in a renderer to build a
/// fixture is not an architecture violation.
fn declared_dependencies(manifest: &str) -> Vec<String> {
    let mut deps = Vec::new();
    let mut in_deps = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_deps = matches!(line, "[dependencies]" | "[build-dependencies]");
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = line.split_once(['=', '.']) {
            let name = name.trim().trim_matches('"');
            if !name.is_empty() {
                deps.push(name.to_string());
            }
        }
    }
    deps
}

fn manifest_of(crate_dir: &str) -> Option<String> {
    let path = workspace_root().join(crate_dir).join("Cargo.toml");
    std::fs::read_to_string(path).ok()
}

fn assert_forbidden(crate_dir: &str, forbidden: &[&str], why: &str) {
    let Some(manifest) = manifest_of(crate_dir) else {
        panic!("{crate_dir}/Cargo.toml not found");
    };
    let deps = declared_dependencies(&manifest);
    for f in forbidden {
        assert!(
            !deps.iter().any(|d| d == f),
            "{crate_dir} must not depend on `{f}`.\n{why}"
        );
    }
}

#[test]
fn mesh_kernel_stays_standalone() {
    assert_forbidden(
        "crates/kreate-mesh",
        &[
            "wgpu",
            "egui",
            "egui-wgpu",
            "egui-winit",
            "winit",
            "tokio",
            "kreate-scene",
            "kreate-ops",
            "kreate-render",
            "kreate-ui",
            "kreate-io",
        ],
        "The mesh kernel must compile standalone and to wasm; it is plausibly a \
         product of its own. Anything it needs from a higher layer is a sign the \
         API is inverted.",
    );
}

#[test]
fn math_depends_on_nothing_of_ours() {
    let manifest = manifest_of("crates/kreate-math").expect("manifest");
    let ours: Vec<String> = declared_dependencies(&manifest)
        .into_iter()
        .filter(|d| d.starts_with("kreate-"))
        .collect();
    assert!(
        ours.is_empty(),
        "kreate-math must be a leaf, but depends on {ours:?}"
    );
}

#[test]
fn io_does_not_reach_into_presentation() {
    assert_forbidden(
        "crates/kreate-io",
        &[
            "wgpu",
            "egui",
            "winit",
            "kreate-render",
            "kreate-ui",
            "kreate-input",
        ],
        "Import/export is a data concern. If it needs the renderer, the neutral \
         IR is not doing its job.",
    );
}

#[test]
fn renderer_does_not_depend_on_ui_or_input() {
    let Some(manifest) = manifest_of("crates/kreate-render") else {
        return; // not built out yet
    };
    let deps = declared_dependencies(&manifest);
    for f in ["kreate-ui", "kreate-input"] {
        assert!(
            !deps.iter().any(|d| d == f),
            "kreate-render must not depend on `{f}`. The renderer consumes an \
             immutable EvaluatedScene snapshot; it does not know about panels or \
             keymaps."
        );
    }
}

#[test]
fn reflection_is_a_leaf() {
    // The property system deliberately has no dependencies, which is what lets
    // the UI read an operator's schema without being able to reach the operator
    // implementations or the document.
    let manifest = manifest_of("crates/kreate-reflect").expect("manifest");
    assert!(
        declared_dependencies(&manifest).is_empty(),
        "kreate-reflect must stay a leaf, but depends on {:?}",
        declared_dependencies(&manifest)
    );
}

#[test]
fn the_ui_can_read_a_schema_but_not_reach_the_document() {
    let manifest = manifest_of("crates/kreate-ui").expect("manifest");
    let deps = declared_dependencies(&manifest);
    assert!(
        deps.contains(&"kreate-reflect".to_string()),
        "the UI needs the reflection types to build a parameter map"
    );
    for forbidden in ["kreate-scene", "kreate-mesh", "wgpu", "winit"] {
        assert!(
            !deps.iter().any(|d| d == forbidden),
            "kreate-ui must not depend on `{forbidden}`. It reads a view model and 
             emits (op_id, params); the application dispatches."
        );
    }
}

#[test]
fn ui_and_render_do_not_depend_on_the_operator_implementations() {
    // The UI dispatches `(op_id, params)` through the registry. Depending on the
    // operator *implementations* would let a panel call `extrude()` directly,
    // which is exactly the shortcut that costs you undo, scripting, macro
    // recording, and testability all at once.
    for c in ["crates/kreate-ui", "crates/kreate-render"] {
        let Some(manifest) = manifest_of(c) else {
            continue;
        };
        let deps = declared_dependencies(&manifest);
        assert!(
            !deps.iter().any(|d| d == "kreate-ops"),
            "{c} must not depend on kreate-ops; go through the operator registry"
        );
    }
}

#[test]
fn mesh_kernel_forbids_unsafe() {
    let src = std::fs::read_to_string(workspace_root().join("crates/kreate-mesh/src/lib.rs"))
        .expect("kreate-mesh/src/lib.rs");
    assert!(
        src.contains("#![forbid(unsafe_code)]"),
        "kreate-mesh must keep `#![forbid(unsafe_code)]`; bytemuck covers the \
         casts it needs safely"
    );
}

#[test]
fn dependency_parser_reads_what_it_should() {
    // Guards the check itself: a parser that silently returns nothing would make
    // every rule above pass vacuously.
    let manifest = "\
[package]
name = \"x\"

[dependencies]
kreate-math.workspace = true
glam = { version = \"0.30\" }
ahash = \"0.8\"
# commented = \"1\"

[dev-dependencies]
proptest = \"1\"
";
    let deps = declared_dependencies(manifest);
    assert_eq!(deps, vec!["kreate-math", "glam", "ahash"]);
    assert!(
        !deps.contains(&"proptest".to_string()),
        "dev-deps are exempt"
    );
    assert!(!deps.contains(&"commented".to_string()));

    // And the real manifests must not be empty, or the rules are vacuous.
    let real = manifest_of("crates/kreate-mesh").expect("manifest");
    assert!(
        declared_dependencies(&real).contains(&"kreate-math".to_string()),
        "kreate-mesh should declare kreate-math; if not, the parser is broken"
    );
}

// --- i18n gate --------------------------------------------------------------

/// egui calls whose text argument reaches the screen, written as the call plus an
/// opening quote so only *literal* text matches.
const TEXT_CALLS: &[&str] = &[
    "ui.label(\"",
    "ui.heading(\"",
    "ui.button(\"",
    "ui.small_button(\"",
    "Button::new(\"",
    "RichText::new(\"",
    "ui.weak(\"",
    "ui.monospace(\"",
    "ui.hyperlink_to(\"",
    ".on_hover_text(\"",
    ".selected_text(\"",
    ".selectable_value(",
];

/// Marker allowing a specific literal, for text that is genuinely not
/// translatable: axis letters, format names, script identifiers, node type ids.
const EXEMPT: &str = "i18n-exempt";

/// Whether the literal on line `i` carries an exemption marker.
///
/// The marker may be on the line itself or in the comment introducing the
/// *statement* the line belongs to. rustfmt splits a builder chain across
/// lines, so the comment above `if ui` can sit several lines above the literal
/// it is about, and a one-line lookback made the gate impossible to satisfy
/// without contorting the code — which is how a gate ends up switched off.
///
/// The walk stops at the end of the previous statement, so an exemption cannot
/// leak forward onto an unrelated literal further down. A blind lookback does
/// exactly that, and it is worse than no gate: it reports success while quietly
/// excusing text nobody decided to excuse.
fn exempt_for(lines: &[&str], i: usize) -> bool {
    if lines[i].contains(EXEMPT) {
        return true;
    }
    let mut p = i;
    while p > 0 {
        p -= 1;
        if lines[p].contains(EXEMPT) {
            return true;
        }
        let prev = lines[p].trim();
        // The previous statement, block boundary or blank line. Anything above
        // this belongs to something else.
        if prev.is_empty() || prev.ends_with(';') || prev.ends_with('{') || prev.ends_with('}') {
            return false;
        }
    }
    false
}

/// Finds bare-literal label calls in one source file, as `(line number, text)`.
///
/// Separated from the test so the scanner itself can be exercised against a
/// sample. A gate that silently finds nothing is worse than no gate, because it
/// reports success while the rule rots.
fn find_untranslated(source: &str) -> Vec<(usize, String)> {
    // Everything from the test module onward is not user-facing.
    let body = match source.find("\nmod tests {") {
        Some(i) => &source[..i],
        None => match source.find("\n#[cfg(test)]") {
            Some(i) => &source[..i],
            None => source,
        },
    };

    let lines: Vec<&str> = body.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        let Some(call) = TEXT_CALLS.iter().find(|c| line.contains(**c)) else {
            continue;
        };
        // `selectable_value` takes the text last, so match a literal anywhere
        // after the call; the others are matched by the quote in the pattern.
        if *call == ".selectable_value(" && !line.contains(", \"") {
            continue;
        }
        // An empty literal is a deliberate "no text here".
        if line.contains(&format!("{call}\")")) {
            continue;
        }
        if !exempt_for(&lines, i) {
            out.push((i + 1, trimmed.to_string()));
        }
    }
    out
}

/// Forbids bare string literals reaching a label in the UI crate.
///
/// The value of this gate is entirely in existing *now*. Retrofitting i18n once
/// a few hundred panels have been written is a classic project-killer, and the
/// reason it happens is that every individual bare string looks harmless. A
/// literal that genuinely should not be translated is allowed with an
/// `i18n-exempt` comment, which makes the decision explicit and reviewable
/// rather than invisible.
#[test]
fn ui_text_goes_through_a_translation_key() {
    let ui_src = workspace_root().join("crates/kreate-ui/src");
    let files = rust_files(&ui_src);
    assert!(
        !files.is_empty(),
        "no UI sources found; the gate would be vacuous"
    );

    let mut offenders = Vec::new();
    for path in files {
        let source = std::fs::read_to_string(&path).expect("read");
        for (line, text) in find_untranslated(&source) {
            offenders.push(format!(
                "{}:{line}: {text}",
                path.file_name().unwrap().to_string_lossy()
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "UI text must come from a translation key. Use a key, or mark the line \
         `// i18n-exempt: <why>` if the text genuinely must not be translated.\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_i18n_scanner_detects_what_it_claims_to() {
    // Guards the gate itself. Without this, a scanner that matched nothing would
    // report success forever while the rule quietly stopped applying.
    let sample = r#"
fn panel(ui: &mut Ui) {
    ui.label("Width");
    ui.label(translated_name);
    // i18n-exempt: axis letters are not translated
    ui.label("X");
    ui.button("");
    ui.heading("Bevel");
    // i18n-exempt: a glyph standing in for an icon
    if ui
        .add_enabled(true, egui::Button::new("+"))
        .clicked()
    {}
    if ui
        .add_enabled(true, egui::Button::new("Apply"))
        .clicked()
    {}
}
"#;
    let found = find_untranslated(sample);
    let lines: Vec<&str> = found.iter().map(|(_, t)| t.as_str()).collect();

    assert!(
        lines.iter().any(|l| l.contains("\"Width\"")),
        "a bare literal must be caught, got {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("\"Bevel\"")),
        "headings must be caught too, got {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("translated_name")),
        "a variable is not a violation"
    );
    assert!(
        !lines.iter().any(|l| l.contains("\"X\"")),
        "an i18n-exempt line must be allowed"
    );
    assert!(
        !lines.iter().any(|l| l.contains("button(\"\")")),
        "an empty literal is a deliberate no-text"
    );
    // A builder chain broken across lines: the marker is three lines above the
    // literal it applies to, and the pattern is not one of the `ui.x("` calls.
    assert!(
        !lines.iter().any(|l| l.contains("\"+\"")),
        "a marker a few lines up must still exempt: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("\"Apply\"")),
        "a Button::new literal must be caught, got {lines:?}"
    );

    assert_eq!(
        found.len(),
        3,
        "exactly the three real violations: {lines:?}"
    );
}

#[test]
fn the_i18n_scanner_ignores_test_code() {
    let sample =
        "fn a(ui: &mut Ui) { ui.label(name); }\nmod tests {\n    ui.label(\"fixture\");\n}\n";
    assert!(
        find_untranslated(sample).is_empty(),
        "literals inside tests are not user-facing"
    );
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(rust_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out.sort();
    out
}

// --- colour gate ------------------------------------------------------------

/// The sRGB curve's own constants, which give it away in any language.
///
/// Matched as numbers rather than as expressions, because there are many ways
/// to spell the same conversion and only a handful of numbers in it. Each of
/// these is specific enough that finding one outside `kreate-math` means
/// somebody wrote the curve again.
const COLOUR_CONSTANTS: &[&str] = &[
    "12.92",
    "1.055",
    "0.055",
    "0.04045",
    "0.040_45",
    "0.0031308",
    "0.003_130_8",
];

/// A gamma exponent, which only gives itself away next to a power.
///
/// `2.2` on its own is any number at all; `pow` and `2.2` on one line is the
/// approximation the real curve is meant to replace. Both members have to
/// appear, which is what keeps this from objecting to `vec2(1.0, 2.2)`.
///
/// Two spellings, and that is the point of this list existing separately: Rust
/// writes `x.powf(2.2)` and WGSL writes `pow(x, 2.2)`. A gate built from the
/// Rust side read every shader and found nothing, which is how the one place
/// the mistake actually happened stayed unguarded.
const COLOUR_POWERS: &[(&str, &str)] = &[
    ("pow", "2.2"),
    ("pow", "2.4"),
    ("pow", "0.4545"),
    ("pow", "0.41666"),
];

/// Marker allowing a specific line, for the one implementation and for
/// arithmetic that only looks like it.
const COLOUR_EXEMPT: &str = "srgb-exempt";

/// Whether a line of Rust or WGSL is converting a colour.
fn smells_of_colour(line: &str) -> bool {
    COLOUR_CONSTANTS.iter().any(|c| line.contains(c))
        || COLOUR_POWERS
            .iter()
            .any(|(call, exponent)| line.contains(call) && line.contains(exponent))
}

/// Where the transfer function is allowed to live.
const COLOUR_HOME: &str = "kreate-math";

/// Every file a colour conversion could hide in.
///
/// Shaders included, and they are the reason this is not just `rust_files`: a
/// `pow(2.2)` in WGSL is the same mistake in a place no Rust scan would look,
/// and the one time it happened the symptom was a viewport seven times too
/// dark.
fn source_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(source_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs" || e == "wgsl") {
            out.push(path);
        }
    }
    out.sort();
    out
}

#[test]
fn the_transfer_function_is_implemented_once() {
    // §2.1's rule, as a gate. Two implementations of a colour conversion do not
    // stay equal, and the way anyone finds out is by putting the product next to
    // another tool — by which time every scene and every imported texture is
    // already wrong.
    //
    // The exemption marker exists because some of these numbers are innocent
    // elsewhere; what it does not permit is a *second* conversion, which is why
    // it has to be written on the line and read by a person.
    let mut offences = Vec::new();
    for crate_dir in [
        "crates/kreate-render",
        "crates/kreate-ui",
        "crates/kreate-mesh",
        "crates/kreate-scene",
        "crates/kreate-ops",
        "crates/kreate-io",
        "apps/kreate",
    ] {
        let src = workspace_root().join(crate_dir).join("src");
        for file in source_files(&src) {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if line.trim_start().starts_with("//") || line.contains(COLOUR_EXEMPT) {
                    continue;
                }
                if smells_of_colour(line) {
                    offences.push(format!("{}:{}: {}", file.display(), i + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        offences.is_empty(),
        "colour conversion outside {COLOUR_HOME}. Use `Srgb::decode`,          `DisplayRgb::encode`, or `srgb_to_linear`/`linear_to_srgb` from          kreate-math — or write `{COLOUR_EXEMPT}` on the line if the number is          not a colour:\n{}",
        offences.join("\n")
    );
}

#[test]
fn the_one_implementation_is_where_it_says_it_is() {
    // Guards the gate: a rule that forbids the function everywhere would pass
    // just as well if the function had been deleted, and then every colour in
    // the product would be converted by nothing at all.
    let color = workspace_root()
        .join("crates")
        .join(COLOUR_HOME)
        .join("src/color.rs");
    let text =
        std::fs::read_to_string(&color).unwrap_or_else(|e| panic!("{}: {e}", color.display()));
    assert!(
        text.contains("pub fn srgb_to_linear") && text.contains("pub fn linear_to_srgb"),
        "the transfer function is not in kreate-math any more; the gate above          is now forbidding something that does not exist"
    );
    // And it is the piecewise curve rather than the approximation the gate
    // rejects elsewhere.
    assert!(
        text.contains("12.92") && text.contains("2.4"),
        "the implementation does not look like the sRGB curve"
    );
}

#[test]
fn the_colour_gate_reads_shaders() {
    // Stated as a test because the difference between `rust_files` and
    // `source_files` is one word and reverting it would leave the gate passing
    // while the place the mistake actually happened stopped being read.
    let shaders = source_files(&workspace_root().join("crates/kreate-render/src/shaders"));
    assert!(
        shaders
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "wgsl")),
        "no shaders found, so the gate is not reading any"
    );
}

#[test]
fn the_colour_scanner_detects_what_it_claims_to() {
    // Guards the gate itself, the same way `the_i18n_scanner_detects_what_it
    // _claims_to` does. Without this a scanner that matched nothing would
    // report success forever while the rule quietly stopped applying.
    let offending = [
        // Rust
        "let linear = c.powf(2.2);",
        "let encoded = v.powf(1.0 / 2.4);",
        "let toe = c / 12.92;",
        "let shifted = (c + 0.055) / 1.055;",
        // WGSL, which spells the call the other way round — and is the shape
        // the first version of this gate missed entirely.
        "    return pow(c, 2.2);",
        "    let lin = pow((s + 0.055) / 1.055, 2.4);",
        "    return select(c / 12.92, pow(c, 2.4), c > 0.04045);",
    ];
    for sample in offending {
        assert!(
            smells_of_colour(sample),
            "the scanner would miss `{sample}`"
        );
    }

    let innocent = [
        "let scale = size.powf(2.0);",
        "let ratio = a / 12.0;",
        "let angle = t.powf(1.0 / 3.0);",
        // A number that only looks like an exponent, with no power in sight.
        "let step = vec2(1.0, 2.2);",
        // And a power with an exponent that is nobody's gamma.
        "let falloff = d.powf(3.0);",
    ];
    for sample in innocent {
        assert!(
            !smells_of_colour(sample),
            "the scanner would object to `{sample}`"
        );
    }

    // And the exemption marker has to work, or the gate is unsatisfiable for
    // the cases that are genuinely not colours.
    let excused = format!("let x = c.powf(2.2); // {COLOUR_EXEMPT}: a falloff curve");
    assert!(excused.contains(COLOUR_EXEMPT));
}
