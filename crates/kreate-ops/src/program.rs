//! A script with names and loops, unrolled into the one-call-a-line script
//! the history records and [`crate::script`] runs.
//!
//! An assistant laying out six holes wrote six lines that differed in one
//! number, and a bolt circle was trigonometry done by hand in the line. This
//! is what it can write instead:
//!
//! ```text
//! d = 6mm
//! pitch = 20mm
//! for i in range(6):
//!     kreate.ops.object.add_primitive(shape="cylinder", size=d, location=(i * pitch, 0, 0))
//! ```
//!
//! - **Names.** `name = value`: a number (units welcome — `6mm`, `90deg` —
//!   and kept in metres and radians), arithmetic of numbers and earlier
//!   names (the expression language drivers use: `+ - * / ** %`,
//!   parentheses, `sin`, `sqrt`, `pi`…), a tuple of those, or a quoted text.
//! - **Loops.** `for name in range(n):`, `range(start, stop)` or
//!   `range(start, stop, step)`, or a list `[a, b, c]` — the block is the
//!   indented lines under it. Loops nest.
//! - **Values.** In an operator line, a value that uses a name or does
//!   arithmetic is worked out; a tuple's or a list's elements each. A plain
//!   literal (`20mm`, `"text"`, `True`) is passed through untouched, so the
//!   parameter's own unit check still reads it.
//! - **`$last`** is passed through: the runner puts the name of the last
//!   object made in its place when the line runs.
//!
//! What it is not: no `if`, no functions, no reading the document. The
//! result is plain lines, each its own undo step, exactly as if they had
//! been written out — which is what the history then shows.

use std::collections::HashMap;

/// Lines it will not unroll past: a loop that runs away is a mistake, and a
/// million undo steps would be a second one.
pub const MOST_LINES: usize = 5000;

/// What a name or scene query holds.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Tuple(Vec<f64>),
    Text(String),
}

/// Callback answering scene queries such as `bbox("Cube")`, `size("Cube")`,
/// `pos("Cube")`, or `param("thickness")`.
pub type SceneQuery<'a> = dyn Fn(&str, &str) -> Option<Value> + 'a;

impl Value {
    /// As it is written in a script line.
    fn script(&self) -> String {
        match self {
            Value::Number(x) => number(*x),
            Value::Tuple(xs) => format!(
                "({})",
                xs.iter().map(|x| number(*x)).collect::<Vec<_>>().join(", ")
            ),
            Value::Text(t) => format!("{t:?}"),
        }
    }

    /// As interpolated inside `f"..."` or `str(...)`.
    fn display(&self) -> String {
        match self {
            Value::Number(x) if *x == x.trunc() && x.abs() < 1e15 => format!("{}", *x as i64),
            Value::Number(x) => format!("{x}"),
            Value::Tuple(xs) => format!(
                "({})",
                xs.iter()
                    .map(|x| Value::Number(*x).display())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Text(t) => t.clone(),
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Value::Number(x) => *x != 0.0,
            Value::Tuple(xs) => !xs.is_empty(),
            Value::Text(t) => !t.is_empty(),
        }
    }
}

/// A number as a script reads it back exactly.
fn number(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e15 {
        format!("{x:.1}")
    } else {
        format!("{x:?}")
    }
}

/// Whether `source` uses anything this module adds: plain scripts go
/// through untouched, while loops, conditionals, assignments, f-strings,
/// scene queries, or inline arithmetic (`location=(30mm * cos(45deg), 0, 0)`)
/// are unrolled.
pub fn needs_unrolling(source: &str) -> bool {
    source.lines().any(|l| {
        let t = l.trim();
        if t.starts_with('#') || t.is_empty() {
            return false;
        }
        t.starts_with("for ")
            || t.starts_with("if ")
            || assignment(t).is_some()
            || t.contains("f\"")
            || t.contains("bbox(")
            || t.contains("size(")
            || t.contains("pos(")
            || t.contains("param(")
            || inline_expression(t)
    })
}

/// Whether an operator call line contains arithmetic or trig/math calls
/// outside string literals that plain `script::parse_call` cannot parse alone.
fn inline_expression(line: &str) -> bool {
    let Some((_, args)) = line.split_once('(') else {
        return false;
    };
    let mut in_quotes = false;
    let bytes = args.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_quotes = !in_quotes;
            continue;
        }
        if in_quotes {
            continue;
        }
        if matches!(b, b'*' | b'/' | b'%') {
            return true;
        }
        // Binary `+` or `-` (not a leading sign after `=`, `(`, `,`, `[`, or `e`/`E`).
        if matches!(b, b'+' | b'-') && i > 0 {
            let prev = args[..i]
                .trim_end()
                .as_bytes()
                .last()
                .copied()
                .unwrap_or(b'(');
            if !matches!(prev, b'=' | b'(' | b',' | b'[' | b'e' | b'E') {
                return true;
            }
        }
    }
    false
}

/// `name = value`, where the name is a name and the line is not an
/// operator call or a conditional.
fn assignment(line: &str) -> Option<(&str, &str)> {
    if line.contains("kreate.ops.")
        || line.starts_with('#')
        || line.starts_with("if ")
        || line.starts_with("elif ")
        || line.starts_with("else")
    {
        return None;
    }
    let (name, value) = line.split_once('=')?;
    if name.ends_with(['!', '<', '>']) || value.starts_with('=') {
        return None;
    }
    let name = name.trim();
    let ok = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    ok.then_some((name, value.trim()))
}

/// The script with its names, loops, conditionals and arithmetic worked out:
/// one operator call a line. An error says the line it is about.
pub fn unroll(source: &str) -> Result<Vec<String>, String> {
    unroll_with(source, &|_, _| None)
}

/// [`unroll`] with access to live scene queries (`bbox("Name")`, `size("Name")`,
/// `pos("Name")`, `param("name")`).
pub fn unroll_with(source: &str, query: &SceneQuery<'_>) -> Result<Vec<String>, String> {
    let lines: Vec<(usize, &str)> = source
        .lines()
        .map(|l| (indent(l), l.trim_end()))
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .collect();
    let mut out = Vec::new();
    let mut names = HashMap::new();
    block(&lines, &mut names, query, &mut out)?;
    Ok(out)
}

fn indent(line: &str) -> usize {
    line.chars()
        .take_while(|c| c.is_whitespace())
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

fn indented_span(
    lines: &[(usize, &str)],
    header_idx: usize,
    depth: usize,
    head_line: &str,
) -> Result<usize, String> {
    let end = lines[header_idx + 1..]
        .iter()
        .position(|(d, _)| *d <= depth)
        .map_or(lines.len(), |p| header_idx + 1 + p);
    if end == header_idx + 1 {
        return Err(format!(
            "`{head_line}`: block has no indented lines under it"
        ));
    }
    Ok(end)
}

fn block(
    lines: &[(usize, &str)],
    names: &mut HashMap<String, Value>,
    query: &SceneQuery<'_>,
    out: &mut Vec<String>,
) -> Result<(), String> {
    let mut i = 0;
    while i < lines.len() {
        let (depth, raw) = lines[i];
        let line = raw.trim();
        if let Some(head) = line.strip_prefix("for ") {
            let head = head
                .strip_suffix(':')
                .ok_or_else(|| format!("`{line}`: a for line ends with a colon"))?;
            let (name, what) = head
                .split_once(" in ")
                .ok_or_else(|| format!("`{line}`: for <name> in range(...) or a [list]"))?;
            let name = name.trim().to_string();
            let items = iterate(what.trim(), names, query).map_err(|e| format!("`{line}`: {e}"))?;
            let end = indented_span(lines, i, depth, line)?;
            for item in items {
                names.insert(name.clone(), item);
                block(&lines[i + 1..end], names, query, out)?;
            }
            i = end;
            continue;
        }
        if let Some(first_cond) = line.strip_prefix("if ") {
            let first_cond = first_cond
                .strip_suffix(':')
                .ok_or_else(|| format!("`{line}`: an if line ends with a colon"))?;
            let mut branches: Vec<(Option<&str>, usize, usize, &str)> = Vec::new();
            let first_end = indented_span(lines, i, depth, line)?;
            branches.push((Some(first_cond.trim()), i + 1, first_end, line));
            let mut cursor = first_end;
            let mut saw_else = false;
            while cursor < lines.len() && lines[cursor].0 == depth {
                let next_line = lines[cursor].1.trim();
                if let Some(elif_cond) = next_line.strip_prefix("elif ") {
                    if saw_else {
                        return Err(format!("`{next_line}`: elif cannot follow else"));
                    }
                    let elif_cond = elif_cond
                        .strip_suffix(':')
                        .ok_or_else(|| format!("`{next_line}`: an elif line ends with a colon"))?;
                    let b_end = indented_span(lines, cursor, depth, next_line)?;
                    branches.push((Some(elif_cond.trim()), cursor + 1, b_end, next_line));
                    cursor = b_end;
                } else if let Some(rest) = next_line.strip_prefix("else") {
                    if rest.trim() != ":" {
                        break;
                    }
                    if saw_else {
                        return Err(format!("`{next_line}`: duplicate else branch"));
                    }
                    saw_else = true;
                    let b_end = indented_span(lines, cursor, depth, next_line)?;
                    branches.push((None, cursor + 1, b_end, next_line));
                    cursor = b_end;
                } else {
                    break;
                }
            }
            for (cond, start, end, header) in branches {
                let take = match cond {
                    Some(c) => {
                        eval_condition(c, names, query).map_err(|e| format!("`{header}`: {e}"))?
                    }
                    None => true,
                };
                if take {
                    block(&lines[start..end], names, query, out)?;
                    break;
                }
            }
            i = cursor;
            continue;
        }
        if line.starts_with("elif ") || line == "else:" {
            return Err(format!("`{line}`: unexpected branch without matching if"));
        }
        if let Some((name, value)) = assignment(line) {
            let v = value_of(value, names, query).map_err(|e| format!("`{line}`: {e}"))?;
            names.insert(name.to_string(), v);
        } else {
            out.push(call(line, names, query).map_err(|e| format!("`{line}`: {e}"))?);
            if out.len() > MOST_LINES {
                return Err(format!(
                    "the loops make more than {MOST_LINES} lines: make fewer at once"
                ));
            }
        }
        i += 1;
    }
    Ok(())
}

/// Evaluates a boolean condition (`or`, `and`, `not`, `== != <= >= < >`, or truthiness).
fn eval_condition(
    cond: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<bool, String> {
    let cond = cond.trim();
    if let Some((a, b)) = split_top_keyword(cond, " or ") {
        return Ok(eval_condition(a, names, query)? || eval_condition(b, names, query)?);
    }
    if let Some((a, b)) = split_top_keyword(cond, " and ") {
        return Ok(eval_condition(a, names, query)? && eval_condition(b, names, query)?);
    }
    if let Some(rest) = cond.strip_prefix("not ") {
        return Ok(!eval_condition(rest, names, query)?);
    }
    if cond == "True" {
        return Ok(true);
    }
    if cond == "False" {
        return Ok(false);
    }
    for op in ["==", "!=", "<=", ">=", "<", ">"] {
        if let Some((lhs, rhs)) = split_top_keyword(cond, op) {
            let lv = value_of(lhs, names, query)?;
            let rv = value_of(rhs, names, query)?;
            return match (op, &lv, &rv) {
                ("==", a, b) => Ok(a == b),
                ("!=", a, b) => Ok(a != b),
                (_, Value::Number(a), Value::Number(b)) => Ok(match op {
                    "<=" => a <= b,
                    ">=" => a >= b,
                    "<" => a < b,
                    ">" => a > b,
                    _ => unreachable!(),
                }),
                _ => Err(format!("cannot compare `{lhs}` {op} `{rhs}`")),
            };
        }
    }
    Ok(value_of(cond, names, query)?.truthy())
}

fn split_top_keyword<'a>(text: &'a str, needle: &str) -> Option<(&'a str, &'a str)> {
    let bytes = text.as_bytes();
    let nbytes = needle.as_bytes();
    let mut depth = 0i32;
    let mut quoted = false;
    let mut i = 0;
    while i + nbytes.len() <= bytes.len() {
        let b = bytes[i];
        if b == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            quoted = !quoted;
        } else if !quoted {
            match b {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                _ => {}
            }
            if depth == 0 && &bytes[i..i + nbytes.len()] == nbytes {
                // Avoid matching `<` inside `<=`, `>` inside `>=`, etc.
                if (needle == "<" || needle == ">") && bytes.get(i + 1).copied() == Some(b'=') {
                    i += 2;
                    continue;
                }
                return Some((text[..i].trim(), text[i + nbytes.len()..].trim()));
            }
        }
        i += 1;
    }
    None
}

/// What a for line runs over.
fn iterate(
    what: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<Vec<Value>, String> {
    if let Some(args) = what
        .strip_prefix("range(")
        .and_then(|r| r.strip_suffix(')'))
    {
        let n: Vec<f64> = split(args)
            .iter()
            .map(|a| number_of(a, names, query))
            .collect::<Result<_, _>>()?;
        let (start, stop, step) = match n.as_slice() {
            [stop] => (0.0, *stop, 1.0),
            [start, stop] => (*start, *stop, 1.0),
            [start, stop, step] if *step != 0.0 => (*start, *stop, *step),
            _ => return Err("range takes a stop, a start and a stop, or those and a step".into()),
        };
        let count = ((stop - start) / step).ceil().max(0.0);
        if count > MOST_LINES as f64 {
            return Err(format!("range of {count} is more than {MOST_LINES}"));
        }
        return Ok((0..count as usize)
            .map(|k| Value::Number(start + step * k as f64))
            .collect());
    }
    if let Some(items) = what.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        return split(items)
            .iter()
            .map(|v| value_of(v, names, query))
            .collect();
    }
    match value_of(what, names, query) {
        Ok(Value::Tuple(xs)) => Ok(xs.into_iter().map(Value::Number).collect()),
        _ => Err("a for line runs over range(...) or a [list]".into()),
    }
}

/// Resolves a scene query call like `bbox("Cube")` or `param("w")`.
fn scene_call(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<Option<Value>, String> {
    for fn_name in ["bbox", "size", "pos", "param"] {
        if let Some(rest) = raw.strip_prefix(fn_name) {
            let rest = rest.trim_start();
            if let Some(inner) = rest.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                let arg_val = value_of(inner.trim(), names, query)?;
                let arg_str = arg_val.display();
                return query(fn_name, &arg_str)
                    .map(Some)
                    .ok_or_else(|| format!("{fn_name}({arg_str:?}) matched nothing in the scene"));
            }
        }
    }
    Ok(None)
}

/// Evaluates `f"..."` interpolation.
fn format_string(
    body: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<String, String> {
    let mut out = String::new();
    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            if i + 1 < chars.len() && chars[i + 1] == '{' {
                out.push('{');
                i += 2;
                continue;
            }
            let close = chars[i + 1..]
                .iter()
                .position(|&c| c == '}')
                .ok_or_else(|| format!("unclosed `{{` in f\"{body}\""))?
                + i
                + 1;
            let expr: String = chars[i + 1..close].iter().collect();
            let v = value_of(expr.trim(), names, query)?;
            out.push_str(&v.display());
            i = close + 1;
        } else if chars[i] == '}' && i + 1 < chars.len() && chars[i + 1] == '}' {
            out.push('}');
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    Ok(out)
}

/// Evaluates trailing `[index]` on a tuple expression (`pt[0]`, `size("Cube")[2]`).
fn indexed_value(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<Option<Value>, String> {
    if !raw.ends_with(']') || raw.starts_with('[') {
        return Ok(None);
    }
    let bytes = raw.as_bytes();
    let mut depth = 0i32;
    let mut open = None;
    for (i, &b) in bytes.iter().enumerate().rev() {
        match b {
            b']' => depth += 1,
            b'[' => {
                depth -= 1;
                if depth == 0 {
                    open = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(open) = open else {
        return Ok(None);
    };
    if open == 0 {
        return Ok(None);
    }
    let base = raw[..open].trim();
    let idx_expr = raw[open + 1..raw.len() - 1].trim();
    let base_val = value_of(base, names, query)?;
    let idx_f = number_of(idx_expr, names, query)?;
    let Value::Tuple(xs) = base_val else {
        return Err(format!("`{base}` is not a tuple and cannot be indexed"));
    };
    let resolved = if idx_f < 0.0 {
        (xs.len() as isize + idx_f as isize) as usize
    } else {
        idx_f as usize
    };
    xs.get(resolved)
        .copied()
        .map(|x| Some(Value::Number(x)))
        .ok_or_else(|| {
            format!(
                "index {idx_f} is out of range for tuple of length {}",
                xs.len()
            )
        })
}

/// A value on the right of `=`.
fn value_of(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<Value, String> {
    let raw = raw.trim();
    if let Some(body) = raw.strip_prefix("f\"").and_then(|s| s.strip_suffix('"')) {
        return format_string(body, names, query).map(Value::Text);
    }
    if let Some(inner) = raw.strip_prefix("str(").and_then(|s| s.strip_suffix(')')) {
        return Ok(Value::Text(value_of(inner, names, query)?.display()));
    }
    if let Some(inner) = raw.strip_prefix("len(").and_then(|s| s.strip_suffix(')')) {
        return match value_of(inner, names, query)? {
            Value::Tuple(xs) => Ok(Value::Number(xs.len() as f64)),
            Value::Text(t) => Ok(Value::Number(t.chars().count() as f64)),
            Value::Number(_) => Err(format!("`{inner}` has no length")),
        };
    }
    if raw.len() >= 2
        && raw.starts_with('"')
        && raw.ends_with('"')
        && !raw[1..raw.len() - 1].contains('"')
    {
        return Ok(Value::Text(raw[1..raw.len() - 1].to_string()));
    }
    // String concatenation (`"Bolt_" + str(i)`).
    if let Some((lhs, rhs)) = split_top_keyword(raw, "+") {
        if lhs.starts_with('"')
            || lhs.starts_with("f\"")
            || lhs.starts_with("str(")
            || rhs.starts_with('"')
            || rhs.starts_with("f\"")
            || rhs.starts_with("str(")
        {
            let lv = value_of(lhs, names, query)?;
            let rv = value_of(rhs, names, query)?;
            if matches!(lv, Value::Text(_)) || matches!(rv, Value::Text(_)) {
                return Ok(Value::Text(format!("{}{}", lv.display(), rv.display())));
            }
        }
    }
    if let Some(v) = scene_call(raw, names, query)? {
        return Ok(v);
    }
    if let Some(v) = indexed_value(raw, names, query)? {
        return Ok(v);
    }
    if let Some(inner) = raw.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        let parts = split(inner);
        if parts.len() > 1 {
            return parts
                .iter()
                .map(|p| number_of(p, names, query))
                .collect::<Result<_, _>>()
                .map(Value::Tuple);
        }
    }
    if let Some(name) = names.get(raw) {
        return Ok(name.clone());
    }
    number_of(raw, names, query).map(Value::Number)
}

/// Expands `expr[idx]`, `len(...)`, and `bbox(...)`/`size(...)`/`pos(...)`/`param(...)`
/// sub-expressions before handing arithmetic to `kreate_scene::expression::evaluate`.
fn preprocess_numeric(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<String, String> {
    let trimmed = raw.trim();
    if let Some(Value::Number(x)) = scene_call(trimmed, names, query)? {
        return Ok(format!("({x:?})"));
    }
    if let Some(Value::Number(x)) = indexed_value(trimmed, names, query)? {
        return Ok(format!("({x:?})"));
    }
    // Replace inline `ident[idx]` occurrences such as `pt[0] + 5mm`.
    let mut out = String::new();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            if i < chars.len() && chars[i] == '[' {
                let mut depth = 0i32;
                let mut end = None;
                for (k, &ch) in chars.iter().enumerate().skip(i) {
                    if ch == '[' {
                        depth += 1;
                    } else if ch == ']' {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(k);
                            break;
                        }
                    }
                }
                if let Some(end) = end {
                    let token: String = chars[start..=end].iter().collect();
                    if let Some(Value::Number(x)) = indexed_value(&token, names, query)? {
                        out.push_str(&format!("({x:?})"));
                        i = end + 1;
                        continue;
                    }
                }
            }
            let ident: String = chars[start..i].iter().collect();
            out.push_str(&ident);
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    Ok(out)
}

/// A number: arithmetic of numbers with units and of names holding numbers.
fn number_of(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<f64, String> {
    let expanded = preprocess_numeric(raw, names, query)?;
    let text = with_units(expanded.trim());
    kreate_scene::expression::evaluate(&text, &|name| match names.get(name) {
        Some(Value::Number(x)) => Some(*x),
        _ => None,
    })
    .map_err(|e| format!("{raw}: {e}"))
}

/// `6mm` as `(6*0.001)`, and so on for the units a script takes, so the
/// expression language — which knows numbers only — can read them.
fn with_units(raw: &str) -> String {
    const UNITS: [(&str, f64); 9] = [
        ("mm", 0.001),
        ("cm", 0.01),
        ("km", 1000.0),
        ("in", 0.0254),
        ("ft", 0.3048),
        ("deg", std::f64::consts::PI / 180.0),
        ("rad", 1.0),
        ("°", std::f64::consts::PI / 180.0),
        ("m", 1.0),
    ];
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let starts_number = c.is_ascii_digit()
            && (i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_'));
        if !starts_number {
            out.push(c);
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
            j += 1;
        }
        // An exponent: 1e-3.
        if j < chars.len() && (chars[j] == 'e' || chars[j] == 'E') {
            let mut k = j + 1;
            if k < chars.len() && (chars[k] == '-' || chars[k] == '+') {
                k += 1;
            }
            if k < chars.len() && chars[k].is_ascii_digit() {
                j = k;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
            }
        }
        let digits: String = chars[i..j].iter().collect();
        let rest: String = chars[j..].iter().collect();
        let unit = UNITS.iter().find(|(u, _)| {
            rest.starts_with(u)
                && !rest[u.len()..]
                    .chars()
                    .next()
                    .is_some_and(|n| n.is_alphanumeric() || n == '_')
        });
        match unit {
            Some((u, scale)) => {
                out.push_str(&format!("({digits}*{scale:?})"));
                i = j + u.chars().count();
            }
            None => {
                out.push_str(&digits);
                i = j;
            }
        }
    }
    out
}

/// The parts of a comma-separated list, respecting brackets and quotes.
fn split(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut quoted, mut part) = (0i32, false, String::new());
    for c in text.chars() {
        match c {
            '"' => quoted = !quoted,
            '(' | '[' if !quoted => depth += 1,
            ')' | ']' if !quoted => depth -= 1,
            ',' if !quoted && depth == 0 => {
                out.push(std::mem::take(&mut part).trim().to_string());
                continue;
            }
            _ => {}
        }
        part.push(c);
    }
    if !part.trim().is_empty() {
        out.push(part.trim().to_string());
    }
    out
}

/// An operator line with its values worked out.
fn call(
    line: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<String, String> {
    let (Some(open), true) = (line.find('('), line.ends_with(')')) else {
        return Ok(line.to_string());
    };
    let head = &line[..open];
    let args = split(&line[open + 1..line.len() - 1]);
    let mut done = Vec::new();
    for arg in args {
        let Some((key, raw)) = arg.split_once('=') else {
            done.push(arg);
            continue;
        };
        done.push(format!(
            "{}={}",
            key.trim(),
            argument(raw.trim(), names, query)?
        ));
    }
    Ok(format!("{head}({})", done.join(", ")))
}

/// One value of an operator line: worked out when it uses a name or does
/// arithmetic, passed through when it is a plain literal.
fn argument(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<String, String> {
    if raw.starts_with("f\"") || raw.starts_with("str(") {
        return Ok(value_of(raw, names, query)?.script());
    }
    if raw.starts_with('"') && !raw.contains('+') {
        return Ok(raw.to_string());
    }
    if raw == "True" || raw == "False" || raw.contains("$last") {
        return Ok(raw.to_string());
    }
    let open_close = |o: char, c: char| raw.starts_with(o) && raw.ends_with(c);
    if open_close('(', ')') || open_close('[', ']') {
        let (o, c) = (&raw[..1], &raw[raw.len() - 1..]);
        let inner = split(&raw[1..raw.len() - 1]);
        // A single parenthesised value is arithmetic, not a tuple.
        if o == "(" && inner.len() == 1 {
            return scalar(raw, names, query);
        }
        let parts: Vec<String> = inner
            .iter()
            .map(|p| argument(p, names, query))
            .collect::<Result<_, _>>()?;
        return Ok(format!("{o}{}{c}", parts.join(", ")));
    }
    scalar(raw, names, query)
}

fn scalar(
    raw: &str,
    names: &HashMap<String, Value>,
    query: &SceneQuery<'_>,
) -> Result<String, String> {
    if let Some(v) = names.get(raw) {
        return Ok(v.script());
    }
    if raw.contains('"') {
        return Ok(value_of(raw, names, query)?.script());
    }
    let uses_name = raw
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|w| names.contains_key(w) || matches!(w, "bbox" | "size" | "pos" | "param" | "len"));
    let does_arithmetic = raw
        .char_indices()
        .any(|(i, c)| matches!(c, '+' | '*' | '/' | '%' | '(' | '[') || (c == '-' && i > 0));
    if uses_name || does_arithmetic {
        value_of(raw, names, query).map(|v| v.script())
    } else {
        Ok(raw.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_lines_pass_through() {
        let s = "kreate.ops.object.add_primitive(shape=\"cube\", size=20mm, location=(0, 0, 1))";
        assert!(!needs_unrolling(s));
        assert_eq!(unroll(s).unwrap(), vec![s.to_string()]);
    }

    #[test]
    fn names_loops_and_arithmetic_unroll_into_lines() {
        let s = "d = 6mm\n\
                 pitch = 2 * 10mm\n\
                 for i in range(3):\n\
                 \x20   kreate.ops.object.add_primitive(shape=\"cylinder\", size=d, location=(i * pitch, 0, 0))";
        assert!(needs_unrolling(s));
        let lines = unroll(s).unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[2],
            "kreate.ops.object.add_primitive(shape=\"cylinder\", size=0.006, location=(0.04, 0, 0))"
        );
    }

    #[test]
    fn a_bolt_circle_and_nested_loops() {
        let s = "r = 30mm\n\
                 for k in range(4):\n\
                 \x20   a = k * 90deg\n\
                 \x20   kreate.ops.object.add_primitive(location=(r * cos(a), r * sin(a), 0))";
        let lines = unroll(s).unwrap();
        assert_eq!(lines.len(), 4);
        assert!(
            lines[1].contains("location=(1.8369701987210297e-18, 0.03, 0.0)")
                || lines[1].contains("0.03")
        );
        let grid = "for x in [0, 1]:\n    for y in range(1, 4, 2):\n        kreate.ops.object.add_primitive(location=(x, y, 0))";
        let lines = unroll(grid).unwrap();
        assert_eq!(lines.len(), 4);
        assert_eq!(
            lines[3],
            "kreate.ops.object.add_primitive(location=(1.0, 3.0, 0))"
        );
    }

    #[test]
    fn literals_and_last_are_left_for_the_runner() {
        let s = "n = 2\nkreate.ops.object.set_transform(object=\"$last\", scale=(n, n, 1), location=(5mm, 0, 0))";
        let lines = unroll(s).unwrap();
        assert_eq!(
            lines[0],
            "kreate.ops.object.set_transform(object=\"$last\", scale=(2.0, 2.0, 1), location=(5mm, 0, 0))"
        );
    }

    #[test]
    fn mistakes_say_their_line() {
        assert!(unroll("for i in range(3)\n    x")
            .unwrap_err()
            .contains("colon"));
        assert!(unroll("x = 1 +").unwrap_err().contains("x = 1 +"));
        assert!(unroll("for i in range(1000000):\n    kreate.ops.a()").is_err());
        assert!(unroll("for i in range(2):\nkreate.ops.a()")
            .unwrap_err()
            .contains("no indented"));
    }

    #[test]
    fn conditionals_fstrings_tuple_indexing_and_scene_queries_work() {
        let script = "\
dims = size(\"Base\")
top_z = bbox(\"Base\")[5]
for i in range(4):
    if i == 0:
        kreate.ops.object.add_primitive(name=f\"Corner_{i}\", location=(0, 0, top_z))
    elif i % 2 == 1 and dims[0] >= 0.1:
        kreate.ops.object.add_primitive(name=\"Odd_\" + str(i), location=(dims[0], 0, top_z))
    else:
        kreate.ops.object.add_primitive(name=f\"Even_{i}\", location=(0, dims[1], top_z))";

        let lines = unroll_with(script, &|fn_name, arg| match (fn_name, arg) {
            ("size", "Base") => Some(Value::Tuple(vec![0.2, 0.3, 0.05])),
            ("bbox", "Base") => Some(Value::Tuple(vec![-0.1, -0.15, 0.0, 0.1, 0.15, 0.05])),
            _ => None,
        })
        .unwrap();

        assert_eq!(lines.len(), 4);
        assert_eq!(
            lines[0],
            "kreate.ops.object.add_primitive(name=\"Corner_0\", location=(0, 0, 0.05))"
        );
        assert_eq!(
            lines[1],
            "kreate.ops.object.add_primitive(name=\"Odd_1\", location=(0.2, 0, 0.05))"
        );
        assert_eq!(
            lines[2],
            "kreate.ops.object.add_primitive(name=\"Even_2\", location=(0, 0.3, 0.05))"
        );
        assert_eq!(
            lines[3],
            "kreate.ops.object.add_primitive(name=\"Odd_3\", location=(0.2, 0, 0.05))"
        );
    }
}
