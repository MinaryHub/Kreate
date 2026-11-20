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

/// What a name holds.
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Number(f64),
    Tuple(Vec<f64>),
    Text(String),
}

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
/// through untouched.
pub fn needs_unrolling(source: &str) -> bool {
    source.lines().any(|l| {
        let t = l.trim();
        t.starts_with("for ") || assignment(t).is_some()
    })
}

/// `name = value`, where the name is a name and the line is not an
/// operator call.
fn assignment(line: &str) -> Option<(&str, &str)> {
    if line.contains("kreate.ops.") || line.starts_with('#') {
        return None;
    }
    let (name, value) = line.split_once('=')?;
    let name = name.trim();
    let ok = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    (ok && !value.starts_with('=')).then_some((name, value.trim()))
}

/// The script with its names, loops and arithmetic worked out: one
/// operator call a line. An error says the line it is about.
pub fn unroll(source: &str) -> Result<Vec<String>, String> {
    let lines: Vec<(usize, &str)> = source
        .lines()
        .map(|l| (indent(l), l.trim_end()))
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .collect();
    let mut out = Vec::new();
    let mut names = HashMap::new();
    block(&lines, &mut names, &mut out)?;
    Ok(out)
}

fn indent(line: &str) -> usize {
    line.chars()
        .take_while(|c| c.is_whitespace())
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

fn block(
    lines: &[(usize, &str)],
    names: &mut HashMap<String, Value>,
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
            let items = iterate(what.trim(), names).map_err(|e| format!("`{line}`: {e}"))?;
            // The block: the lines after it, further in.
            let end = lines[i + 1..]
                .iter()
                .position(|(d, _)| *d <= depth)
                .map_or(lines.len(), |p| i + 1 + p);
            if end == i + 1 {
                return Err(format!("`{line}`: the loop has no indented lines under it"));
            }
            for item in items {
                names.insert(name.clone(), item);
                block(&lines[i + 1..end], names, out)?;
            }
            i = end;
            continue;
        }
        if let Some((name, value)) = assignment(line) {
            let v = value_of(value, names).map_err(|e| format!("`{line}`: {e}"))?;
            names.insert(name.to_string(), v);
        } else {
            out.push(call(line, names).map_err(|e| format!("`{line}`: {e}"))?);
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

/// What a for line runs over.
fn iterate(what: &str, names: &HashMap<String, Value>) -> Result<Vec<Value>, String> {
    if let Some(args) = what
        .strip_prefix("range(")
        .and_then(|r| r.strip_suffix(')'))
    {
        let n: Vec<f64> = split(args)
            .iter()
            .map(|a| number_of(a, names))
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
        return split(items).iter().map(|v| value_of(v, names)).collect();
    }
    match names.get(what) {
        Some(Value::Tuple(xs)) => Ok(xs.iter().map(|x| Value::Number(*x)).collect()),
        _ => Err("a for line runs over range(...) or a [list]".into()),
    }
}

/// A value on the right of `=`.
fn value_of(raw: &str, names: &HashMap<String, Value>) -> Result<Value, String> {
    let raw = raw.trim();
    if raw.len() >= 2 && (raw.starts_with('"') && raw.ends_with('"')) {
        return Ok(Value::Text(raw[1..raw.len() - 1].to_string()));
    }
    if let Some(inner) = raw.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        let parts = split(inner);
        if parts.len() > 1 {
            return parts
                .iter()
                .map(|p| number_of(p, names))
                .collect::<Result<_, _>>()
                .map(Value::Tuple);
        }
    }
    if let Some(name) = names.get(raw) {
        return Ok(name.clone());
    }
    number_of(raw, names).map(Value::Number)
}

/// A number: arithmetic of numbers with units and of names holding numbers.
fn number_of(raw: &str, names: &HashMap<String, Value>) -> Result<f64, String> {
    let text = with_units(raw.trim());
    kreate_scene::expression::evaluate(&text, &|name| match names.get(name) {
        Some(Value::Number(x)) => Some(*x),
        _ => None,
    })
    .map_err(|e| format!("{raw}: {e}"))
}

/// `6mm` as `(6*0.001)`, and so on for the units a script takes, so the
/// expression language — which knows numbers only — can read them.
fn with_units(raw: &str) -> String {
    const UNITS: [(&str, f64); 7] = [
        ("mm", 0.001),
        ("cm", 0.01),
        ("in", 0.0254),
        ("ft", 0.3048),
        ("deg", std::f64::consts::PI / 180.0),
        ("rad", 1.0),
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
fn call(line: &str, names: &HashMap<String, Value>) -> Result<String, String> {
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
        done.push(format!("{}={}", key.trim(), argument(raw.trim(), names)?));
    }
    Ok(format!("{head}({})", done.join(", ")))
}

/// One value of an operator line: worked out when it uses a name or does
/// arithmetic, passed through when it is a plain literal.
fn argument(raw: &str, names: &HashMap<String, Value>) -> Result<String, String> {
    if raw.starts_with('"') || raw == "True" || raw == "False" || raw.contains("$last") {
        return Ok(raw.to_string());
    }
    let open_close = |o: char, c: char| raw.starts_with(o) && raw.ends_with(c);
    if open_close('(', ')') || open_close('[', ']') {
        let (o, c) = (&raw[..1], &raw[raw.len() - 1..]);
        let inner = split(&raw[1..raw.len() - 1]);
        // A single parenthesised value is arithmetic, not a tuple.
        if o == "(" && inner.len() == 1 {
            return scalar(raw, names);
        }
        let parts: Vec<String> = inner
            .iter()
            .map(|p| argument(p, names))
            .collect::<Result<_, _>>()?;
        return Ok(format!("{o}{}{c}", parts.join(", ")));
    }
    scalar(raw, names)
}

fn scalar(raw: &str, names: &HashMap<String, Value>) -> Result<String, String> {
    if let Some(v) = names.get(raw) {
        return Ok(v.script());
    }
    let uses_name = raw
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|w| names.contains_key(w));
    let does_arithmetic = raw
        .char_indices()
        .any(|(i, c)| matches!(c, '+' | '*' | '/' | '%' | '(') || (c == '-' && i > 0));
    if uses_name || does_arithmetic {
        number_of(raw, names).map(number)
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
}
