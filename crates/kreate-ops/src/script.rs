//! Reading a script line back into an operator call, and running a script.
//!
//! The history already writes itself as a script: every entry renders as
//! `kreate.ops.mesh.translate(translation=(1.4, 0.0, 0.0))`, and the status bar
//! shows the last one. Until this existed nothing could *read* that line, so
//! the script was a record and not a language. This is the inverse of
//! `PropertyMap::to_script_call`, and it is held to being one: a test renders
//! every operator's parameters and parses them back.
//!
//! **The schema is the grammar's type system.** A parsed name has to become the
//! `&'static str` a `PropertyMap` keys on, and a parsed number has to become an
//! enum index, an integer or a float — none of which the text says. Both come
//! from the operator's own `PropertyDesc`s, which is also what makes a typo in
//! a parameter name an error naming the parameter rather than a call that runs
//! with a default the user did not ask for.

use crate::history::History;
use crate::operator::{Context, OpError, OpResult, OperatorRegistry};
use kreate_reflect::{PropertyKind, PropertyMap, PropertyValue};
use kreate_scene::Document;

/// A parsed operator call: the id, as the registry spells it, and its
/// parameters, typed against the operator's schema.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub op_id: &'static str,
    pub params: PropertyMap,
}

/// Why a line could not be read or run.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScriptError {
    #[error("expected `kreate.ops.<operator>(...)`: {0}")]
    Syntax(String),
    #[error("no operator called `{0}`")]
    UnknownOperator(String),
    #[error("`{op}` has no parameter `{name}`")]
    UnknownParameter { op: String, name: String },
    #[error("`{name}` wants {wanted}, not `{got}`")]
    WrongType {
        name: String,
        wanted: &'static str,
        got: String,
    },
    #[error("{0}")]
    Operator(String),
}

impl From<OpError> for ScriptError {
    fn from(e: OpError) -> Self {
        ScriptError::Operator(e.to_string())
    }
}

/// The prefix a rendered line carries. Optional on the way in, so a user can
/// type `mesh.extrude_region(offset=1.0)` and be understood.
const PREFIX: &str = "kreate.ops.";

/// Reads one line.
pub fn parse_call(registry: &OperatorRegistry, line: &str) -> Result<Call, ScriptError> {
    let line = line.trim();
    let body = line.strip_prefix(PREFIX).unwrap_or(line);
    let open = body
        .find('(')
        .ok_or_else(|| ScriptError::Syntax("missing `(`".into()))?;
    if !body.ends_with(')') {
        return Err(ScriptError::Syntax("missing `)` at the end".into()));
    }
    let id = body[..open].trim();
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_')
    {
        return Err(ScriptError::Syntax(format!("`{id}` is not an operator id")));
    }
    let info = registry
        .get(id)
        .ok_or_else(|| ScriptError::UnknownOperator(id.to_string()))?;
    let args = &body[open + 1..body.len() - 1];

    let mut params = PropertyMap::new();
    for arg in split_args(args)? {
        let (name, raw) = arg
            .split_once('=')
            .ok_or_else(|| ScriptError::Syntax(format!("`{arg}` is not `name=value`")))?;
        let name = name.trim();
        let desc = info
            .schema
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| ScriptError::UnknownParameter {
                op: id.to_string(),
                name: name.to_string(),
            })?;
        let value = parse_value(raw.trim())?;
        let value = coerce(value, &desc.kind, desc.name)?;
        params.set(desc.name, value);
    }
    Ok(Call {
        op_id: info.id,
        params,
    })
}

/// The comma-separated arguments, respecting parentheses, brackets and quotes.
fn split_args(args: &str) -> Result<Vec<&str>, ScriptError> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut start = 0;
    for (i, c) in args.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&args[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if in_string {
        return Err(ScriptError::Syntax("unterminated string".into()));
    }
    if depth != 0 {
        return Err(ScriptError::Syntax("unbalanced brackets".into()));
    }
    let last = &args[start..];
    if !last.trim().is_empty() {
        out.push(last);
    }
    Ok(out.into_iter().map(str::trim).filter(|a| !a.is_empty()).collect())
}

/// A value as the text has it, before the schema says what it is for.
#[derive(Clone, Debug, PartialEq)]
enum Raw {
    Bool(bool),
    Int(i64),
    Float(f64),
    Tuple(Vec<f64>),
    List(Vec<i64>),
    /// A bracketed list with a number that is not a whole one in it.
    FloatList(Vec<f64>),
    Text(String),
}

fn parse_value(raw: &str) -> Result<Raw, ScriptError> {
    match raw {
        "True" | "true" => return Ok(Raw::Bool(true)),
        "False" | "false" => return Ok(Raw::Bool(false)),
        _ => {}
    }
    if let Some(inner) = raw.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        let parts: Result<Vec<f64>, _> = inner
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| {
                p.parse::<f64>()
                    .map_err(|_| ScriptError::Syntax(format!("`{p}` is not a number")))
            })
            .collect();
        return Ok(Raw::Tuple(parts?));
    }
    if let Some(inner) = raw.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let items: Vec<&str> = inner.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
        if items.iter().any(|p| p.parse::<i64>().is_err()) && items.iter().all(|p| p.parse::<f64>().is_ok()) {
            return Ok(Raw::FloatList(items.iter().map(|p| p.parse::<f64>().unwrap_or(0.0)).collect()));
        }
        let parts: Result<Vec<i64>, _> = inner
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| {
                p.parse::<i64>()
                    .map_err(|_| ScriptError::Syntax(format!("`{p}` is not an integer")))
            })
            .collect();
        return Ok(Raw::List(parts?));
    }
    if raw.starts_with('"') {
        return unquote(raw).map(Raw::Text);
    }
    if let Ok(i) = raw.parse::<i64>() {
        return Ok(Raw::Int(i));
    }
    if let Ok(f) = raw.parse::<f64>() {
        return Ok(Raw::Float(f));
    }
    Err(ScriptError::Syntax(format!("`{raw}` is not a value")))
}

/// The inverse of `{:?}` on a string: double quotes, backslash escapes.
fn unquote(raw: &str) -> Result<String, ScriptError> {
    let inner = raw
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .ok_or_else(|| ScriptError::Syntax(format!("`{raw}` is not a string")))?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('u') => {
                // `\u{1F600}`, as Rust's Debug writes it.
                let rest: String = chars.by_ref().take_while(|c| *c != '}').collect();
                let hex = rest.trim_start_matches('{');
                let code = u32::from_str_radix(hex, 16)
                    .map_err(|_| ScriptError::Syntax(format!("bad escape `\\u{rest}`")))?;
                out.push(
                    char::from_u32(code)
                        .ok_or_else(|| ScriptError::Syntax(format!("bad escape `\\u{rest}`")))?,
                );
            }
            other => {
                return Err(ScriptError::Syntax(format!(
                    "bad escape `\\{}`",
                    other.map(String::from).unwrap_or_default()
                )))
            }
        }
    }
    Ok(out)
}

/// What the schema says the value is for, and whether the text can be that.
fn coerce(raw: Raw, kind: &PropertyKind, name: &str) -> Result<PropertyValue, ScriptError> {
    let wrong = |wanted: &'static str, raw: &Raw| ScriptError::WrongType {
        name: name.to_string(),
        wanted,
        got: format!("{raw:?}"),
    };
    Ok(match (kind, raw) {
        (PropertyKind::Bool, Raw::Bool(b)) => PropertyValue::Bool(b),
        (PropertyKind::Int, Raw::Int(i)) => PropertyValue::Int(i),
        // A whole-number float for an integer parameter is what a user types
        // as `cuts=2.0`; refusing it would be pedantry.
        (PropertyKind::Int, Raw::Float(f)) if f == f.trunc() => PropertyValue::Int(f as i64),
        (PropertyKind::Float, Raw::Float(f)) => PropertyValue::Float(f),
        (PropertyKind::Float, Raw::Int(i)) => PropertyValue::Float(i as f64),
        (PropertyKind::Enum { variants }, Raw::Int(i)) => {
            if i < 0 || i as usize >= variants.len() {
                return Err(ScriptError::WrongType {
                    name: name.to_string(),
                    wanted: "an index into the variants",
                    got: i.to_string(),
                });
            }
            PropertyValue::Enum(i as u32)
        }
        // A variant by name, which is what a person types.
        (PropertyKind::Enum { variants }, Raw::Text(t)) => {
            let index = variants
                .iter()
                .position(|v| *v == t)
                .ok_or_else(|| ScriptError::WrongType {
                    name: name.to_string(),
                    wanted: "one of the variants",
                    got: t.clone(),
                })?;
            PropertyValue::Enum(index as u32)
        }
        (PropertyKind::Vec3 | PropertyKind::Color, Raw::Tuple(v)) if v.len() == 3 => {
            PropertyValue::Vec3([v[0], v[1], v[2]])
        }
        (PropertyKind::Text, Raw::Text(t)) => PropertyValue::Text(t),
        (PropertyKind::Ints, Raw::List(v)) => {
            PropertyValue::Ints(v.into_iter().map(|i| i.max(0) as u32).collect())
        }
        (PropertyKind::Ints, Raw::Tuple(v)) => {
            PropertyValue::Ints(v.into_iter().map(|f| f.max(0.0) as u32).collect())
        }
        (PropertyKind::Floats, Raw::FloatList(v) | Raw::Tuple(v)) => PropertyValue::Floats(v),
        (PropertyKind::Floats, Raw::List(v)) => PropertyValue::Floats(v.into_iter().map(|i| i as f64).collect()),
        (kind, raw) => {
            return Err(wrong(
                match kind {
                    PropertyKind::Bool => "True or False",
                    PropertyKind::Int => "an integer",
                    PropertyKind::Float => "a number",
                    PropertyKind::Vec3 => "three numbers in parentheses",
                    PropertyKind::Color => "three numbers in parentheses",
                    PropertyKind::Text => "a quoted string",
                    PropertyKind::Enum { .. } => "a variant index or name",
                    PropertyKind::Ints => "a list of integers",
                    PropertyKind::Floats => "a list of numbers",
                },
                &raw,
            ))
        }
    })
}

/// What happened to one line of a script.
#[derive(Clone, Debug, PartialEq)]
pub struct LineOutcome {
    /// One-based, as an editor counts.
    pub line: usize,
    pub text: String,
    pub result: Result<OpResult, ScriptError>,
}

impl History {
    /// Runs a script, one operator call per line, through the history.
    ///
    /// Each call is its own undo step, exactly as though the user had done it —
    /// the whole reason the history *is* a script. Blank lines and lines
    /// starting with `#` are skipped. A line that fails is reported and does
    /// not stop the rest: a console user wants to see every complaint, and a
    /// macro that half-ran leaves a document the undo stack can take back one
    /// line at a time.
    pub fn run_script(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        mut sessions: Option<&mut crate::EditSessions>,
        ctx: &Context,
        source: &str,
    ) -> Vec<LineOutcome> {
        let mut out = Vec::new();
        for (index, text) in source.lines().enumerate() {
            let trimmed = text.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let result = parse_call(registry, trimmed).and_then(|call| {
                self.dispatch_in_session(
                    registry,
                    doc,
                    sessions.as_deref_mut(),
                    ctx,
                    call.op_id,
                    &call.params,
                )
                .map_err(ScriptError::from)
            });
            out.push(LineOutcome {
                line: index + 1,
                text: trimmed.to_string(),
                result,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> OperatorRegistry {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        r
    }

    #[test]
    fn a_rendered_line_reads_back_as_the_call_that_wrote_it() {
        // The inverse relationship the module exists for, held over every
        // kind of value at once.
        let r = registry();
        let mut params = PropertyMap::new();
        params.set("translation", PropertyValue::Vec3([1.4, 0.0, -2.5]));
        let line = params.to_script_call("mesh.translate");
        let call = parse_call(&r, &line).expect("parses");
        assert_eq!(call.op_id, "mesh.translate");
        assert_eq!(call.params, params);
    }

    #[test]
    fn every_builtin_operators_defaults_round_trip() {
        // Each operator's schema, rendered with a representative value of
        // every kind and parsed back. If a kind is added that the parser does
        // not read, this is where it shows.
        let r = registry();
        for info in r.iter() {
            let mut params = PropertyMap::new();
            for desc in info.schema {
                let value = match &desc.kind {
                    PropertyKind::Bool => PropertyValue::Bool(true),
                    PropertyKind::Int => PropertyValue::Int(3),
                    PropertyKind::Float => PropertyValue::Float(0.25),
                    PropertyKind::Vec3 | PropertyKind::Color => PropertyValue::Vec3([1.0, 0.5, 0.0]),
                    PropertyKind::Text => PropertyValue::Text("a \"quoted\" name\nwith a break".into()),
                    PropertyKind::Enum { .. } => PropertyValue::Enum(0),
                    PropertyKind::Ints => PropertyValue::Ints(vec![0, 7, 42]),
                    PropertyKind::Floats => PropertyValue::Floats(vec![0.5, -1.0, 3.0, 1e-7]),
                };
                params.set(desc.name, value);
            }
            let line = params.to_script_call(info.id);
            let call = parse_call(&r, &line).unwrap_or_else(|e| panic!("{}: {e}\n{line}", info.id));
            assert_eq!(call.op_id, info.id);
            assert_eq!(call.params, params, "{}", info.id);
        }
    }

    #[test]
    fn the_prefix_is_optional_and_a_variant_can_be_named() {
        let r = registry();
        let call = parse_call(&r, r#"object.add_primitive(shape="torus")"#).expect("parses");
        let shape = call.params.get("shape").expect("shape");
        assert_eq!(*shape, PropertyValue::Enum(7), "torus is the eighth shape");
    }

    #[test]
    fn mistakes_are_named_rather_than_run_with_defaults() {
        let r = registry();
        assert!(matches!(
            parse_call(&r, "mesh.nothing()"),
            Err(ScriptError::UnknownOperator(id)) if id == "mesh.nothing"
        ));
        assert!(matches!(
            parse_call(&r, "mesh.translate(offset=(1.0, 0.0, 0.0))"),
            Err(ScriptError::UnknownParameter { name, .. }) if name == "offset"
        ));
        assert!(matches!(
            parse_call(&r, "mesh.translate(translation=1.0)"),
            Err(ScriptError::WrongType { name, .. }) if name == "translation"
        ));
        assert!(matches!(parse_call(&r, "mesh.translate"), Err(ScriptError::Syntax(_))));
        assert!(matches!(
            parse_call(&r, r#"object.add_primitive(shape="banana")"#),
            Err(ScriptError::WrongType { .. })
        ));
    }

    #[test]
    fn a_script_runs_line_by_line_and_reports_each() {
        let r = registry();
        let mut doc = Document::new();
        let mut h = History::new();
        let ctx = Context::default();
        let outcomes = h.run_script(
            &r,
            &mut doc,
            None,
            &ctx,
            "# two cubes and a mistake\nobject.add_primitive(shape=1)\n\nobject.add_primitive(shape=1)\nmesh.nothing()\n",
        );
        assert_eq!(outcomes.len(), 3, "blank and comment lines are skipped");
        assert_eq!(outcomes[0].line, 2);
        assert!(outcomes[0].result.is_ok());
        assert!(outcomes[1].result.is_ok());
        assert!(matches!(outcomes[2].result, Err(ScriptError::UnknownOperator(_))));
        assert_eq!(doc.object_count(), 2, "the good lines ran; the bad one did not stop them");
        // One undo step per line, as though the user had done each.
        assert!(h.undo(&mut doc));
        assert_eq!(doc.object_count(), 1);
    }

    #[test]
    fn the_history_replays_itself() {
        // Do things, print the script, run it in a fresh document: the same
        // scene. This is what "the history is a script" promises.
        let r = registry();
        let ctx = Context::default();
        let mut a = Document::new();
        let mut h = History::new();
        let mut p = PropertyMap::new();
        p.set("shape", PropertyValue::Enum(5));
        h.dispatch(&r, &mut a, &ctx, "object.add_primitive", &p).unwrap();
        let mut t = PropertyMap::new();
        t.set("translation", PropertyValue::Vec3([2.0, 0.0, 0.0]));
        h.dispatch(&r, &mut a, &ctx, "object.translate", &t).unwrap();
        let script = h.to_script();

        let mut b = Document::new();
        let mut h2 = History::new();
        let outcomes = h2.run_script(&r, &mut b, None, &ctx, &script);
        assert!(outcomes.iter().all(|o| o.result.is_ok()), "{outcomes:?}");
        assert_eq!(b.object_count(), 1);
        let (id_a, _) = a.objects().next().unwrap();
        let (id_b, _) = b.objects().next().unwrap();
        assert_eq!(
            a.world_transform(id_a).translation,
            b.world_transform(id_b).translation
        );
    }
}
